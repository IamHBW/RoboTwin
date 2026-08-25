#!/usr/bin/env python3
"""Single-host dynamic dispatcher for RoboTwin task-phase work items."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


BUSY_TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 3
POLL_INTERVAL_S = 0.2
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "run_root",
        "adapter",
        "tasks",
        "phases",
        "workers",
        "result_contract",
    ):
        if key not in payload:
            raise ValueError(f"Dispatch manifest missing `{key}`: {path}")
    if SAFE_ID.fullmatch(str(payload["run_id"])) is None:
        raise ValueError(f"Unsafe run ID: {payload['run_id']!r}")
    if not isinstance(payload["tasks"], list) or not payload["tasks"]:
        raise ValueError("`tasks` must be a non-empty ordered list.")
    if not isinstance(payload["phases"], list) or not payload["phases"]:
        raise ValueError("`phases` must be a non-empty ordered list.")
    if any(not isinstance(value, str) or not value for value in payload["tasks"]):
        raise ValueError("Every task must be a non-empty string.")
    if any(not isinstance(value, str) or not value for value in payload["phases"]):
        raise ValueError("Every phase must be a non-empty string.")
    if any(SAFE_ID.fullmatch(value) is None for value in payload["phases"]):
        raise ValueError("Every phase must be safe for worker log filenames.")
    if len(set(payload["tasks"])) != len(payload["tasks"]):
        raise ValueError("`tasks` contains duplicates.")
    if len(set(payload["phases"])) != len(payload["phases"]):
        raise ValueError("`phases` contains duplicates.")
    if not isinstance(payload["workers"], list) or any(
        not isinstance(worker, dict) for worker in payload["workers"]
    ):
        raise ValueError("`workers` must be a list of objects.")
    worker_ids = [str(worker.get("worker_id", "")) for worker in payload["workers"]]
    if not worker_ids or any(SAFE_ID.fullmatch(worker_id) is None for worker_id in worker_ids):
        raise ValueError("Every worker requires a safe non-empty `worker_id`.")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("`workers` contains duplicate worker IDs.")
    episodes = int(payload.get("episodes_per_task_phase", 20))
    if episodes <= 0:
        raise ValueError("`episodes_per_task_phase` must be positive.")
    payload["episodes_per_task_phase"] = episodes
    payload["candidate_seed_start"] = int(payload.get("candidate_seed_start", 100_000))
    if not isinstance(payload["result_contract"], dict) or not payload["result_contract"]:
        raise ValueError("`result_contract` must freeze at least one episode field.")
    payload["_manifest_path"] = str(path.resolve())
    return payload


def dispatch_db_path(run_id: str) -> Path:
    # ponytail: single-host SQLite only; replace the coordinator before multi-host workers.
    return Path("/var/tmp/robotwin-dispatch") / run_id / "dispatch.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return connection


@contextmanager
def _database(path: Path):
    connection = _connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(manifest: dict[str, Any], database: Path) -> None:
    run_id = str(manifest["run_id"])
    expected = {
        (task, phase): (task_index, phase_index)
        for task_index, task in enumerate(manifest["tasks"])
        for phase_index, phase in enumerate(manifest["phases"])
    }
    with _database(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work_items(
              run_id TEXT NOT NULL,
              task_index INTEGER NOT NULL,
              task_name TEXT NOT NULL,
              phase_index INTEGER NOT NULL,
              phase TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'done', 'failed')),
              owner_worker TEXT,
              claim_token TEXT,
              attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT,
              updated_at REAL NOT NULL,
              PRIMARY KEY (run_id, task_name, phase)
            )
            """
        )
        now = time.time()
        connection.executemany(
            """
            INSERT OR IGNORE INTO work_items(
              run_id, task_index, task_name, phase_index, phase, state, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            [
                (run_id, task_index, task, phase_index, phase, now)
                for (task, phase), (task_index, phase_index) in expected.items()
            ],
        )
        rows = connection.execute(
            "SELECT task_name, phase, task_index, phase_index FROM work_items WHERE run_id=?",
            (run_id,),
        ).fetchall()
        actual = {
            (row["task_name"], row["phase"]): (row["task_index"], row["phase_index"])
            for row in rows
        }
        if actual != expected:
            raise ValueError("Existing dispatch database does not match the frozen manifest.")
    reconcile_from_results(manifest, database)


def _worker_root(manifest: dict[str, Any]) -> Path:
    return Path(str(manifest["run_root"])).expanduser().resolve() / "client" / "workers"


def load_episode_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, int], str] = {}
    expected = {(task, phase) for task in manifest["tasks"] for phase in manifest["phases"]}
    for path in sorted(_worker_root(manifest).glob("*/episodes.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (
                        str(record["task"]),
                        str(record["phase"]),
                        int(record["episode_index"]),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid episode record at {path}:{line_number}") from exc
                if key[:2] not in expected:
                    raise ValueError(f"Episode record is outside the frozen manifest: {key}")
                for field, expected_value in manifest["result_contract"].items():
                    if record.get(field) != expected_value:
                        raise ValueError(
                            f"Episode contract mismatch at {path}:{line_number}: "
                            f"{field}={record.get(field)!r}, expected {expected_value!r}"
                        )
                source = f"{path}:{line_number}"
                if key in seen:
                    raise ValueError(f"Duplicate episode key {key}: {seen[key]} and {source}")
                seen[key] = source
                records.append(record)
    return records


def item_progress(
    manifest: dict[str, Any], records: list[dict[str, Any]], task: str, phase: str
) -> tuple[int, int, int]:
    total = int(manifest["episodes_per_task_phase"])
    start_seed = int(manifest["candidate_seed_start"])
    selected = sorted(
        (
            record
            for record in records
            if record["task"] == task
            and record["phase"] == phase
            and record.get("error") is None
        ),
        key=lambda record: int(record["episode_index"]),
    )
    indices = [int(record["episode_index"]) for record in selected]
    if indices != list(range(len(selected))) or len(selected) > total:
        raise ValueError(f"Invalid episode prefix for {task}/{phase}: {indices}")
    seeds = [int(record["candidate_seed"]) for record in selected]
    accepted_seeds = [int(record["accepted_seed"]) for record in selected]
    if accepted_seeds != seeds:
        raise ValueError(f"Candidate/accepted seed mismatch for {task}/{phase}")
    if seeds and (seeds[0] < start_seed or any(a >= b for a, b in zip(seeds, seeds[1:]))):
        raise ValueError(f"Invalid candidate seed sequence for {task}/{phase}: {seeds}")
    seed_offset = 0 if not seeds else seeds[-1] - start_seed + 1
    return len(selected), total - len(selected), seed_offset


def reconcile_from_results(manifest: dict[str, Any], database: Path) -> None:
    records = load_episode_records(manifest)
    run_id = str(manifest["run_id"])
    with _database(database) as connection:
        for task in manifest["tasks"]:
            for phase in manifest["phases"]:
                _, remaining, _ = item_progress(manifest, records, task, phase)
                if remaining == 0:
                    connection.execute(
                        """
                        UPDATE work_items SET state='done', owner_worker=NULL, claim_token=NULL,
                          last_error=NULL, updated_at=?
                        WHERE run_id=? AND task_name=? AND phase=?
                        """,
                        (time.time(), run_id, task, phase),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE work_items SET state='pending', owner_worker=NULL, claim_token=NULL,
                          updated_at=?
                        WHERE run_id=? AND task_name=? AND phase=? AND state='done'
                        """,
                        (time.time(), run_id, task, phase),
                    )


def claim_one(database: Path, run_id: str, worker_id: str) -> dict[str, Any] | None:
    while True:
        connection = _connect(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT task_name, phase, attempts FROM work_items
                WHERE run_id=? AND state='pending'
                ORDER BY task_index, phase_index LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE work_items SET state='running', owner_worker=?, claim_token=?,
                  attempts=attempts+1, updated_at=?
                WHERE run_id=? AND task_name=? AND phase=? AND state='pending'
                """,
                (worker_id, token, time.time(), run_id, row["task_name"], row["phase"]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("Atomic claim did not update exactly one row.")
            connection.commit()
            return {
                "task": row["task_name"],
                "phase": row["phase"],
                "attempt": int(row["attempts"]) + 1,
                "claim_token": token,
            }
        except sqlite3.OperationalError as exc:
            connection.rollback()
            if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY and "locked" not in str(exc):
                raise
            time.sleep(0.05)
        finally:
            connection.close()


def _finish_claim(
    database: Path,
    run_id: str,
    worker_id: str,
    claim: dict[str, Any],
    *,
    complete: bool,
    error: str | None = None,
) -> None:
    state = "done" if complete else ("failed" if claim["attempt"] >= MAX_ATTEMPTS else "pending")
    with _database(database) as connection:
        cursor = connection.execute(
            """
            UPDATE work_items SET state=?, owner_worker=NULL, claim_token=NULL,
              last_error=?, updated_at=?
            WHERE run_id=? AND task_name=? AND phase=? AND state='running'
              AND owner_worker=? AND claim_token=?
            """,
            (
                state,
                None if complete else error,
                time.time(),
                run_id,
                claim["task"],
                claim["phase"],
                worker_id,
                claim["claim_token"],
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Claim completion lost ownership.")


def _state_counts(database: Path, run_id: str) -> dict[str, int]:
    with _database(database) as connection:
        rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM work_items WHERE run_id=? GROUP BY state",
            (run_id,),
        ).fetchall()
    return {row["state"]: int(row["count"]) for row in rows}


def write_worker_contract(manifest: dict[str, Any], worker_id: str) -> Path:
    worker = next(
        (item for item in manifest["workers"] if str(item["worker_id"]) == worker_id),
        None,
    )
    if worker is None:
        raise ValueError(f"Worker `{worker_id}` is not in the frozen manifest.")
    worker_dir = _worker_root(manifest) / worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    pid_path = worker_dir / "worker.pid"
    if pid_path.is_file():
        existing_pid = int(pid_path.read_text(encoding="ascii").strip())
        if existing_pid != os.getpid() and Path(f"/proc/{existing_pid}").exists():
            raise RuntimeError(f"Worker `{worker_id}` is already live with pid {existing_pid}")
    payload = dict(worker)
    payload.update(
        {
            "run_id": manifest["run_id"],
            "contract_id": manifest.get("contract_id", manifest["run_id"]),
            "run_root": str(Path(str(manifest["run_root"])).expanduser().resolve()),
            "manifest": manifest["_manifest_path"],
            "adapter_config": manifest.get("adapter_config", {}),
            "episodes_jsonl": str(worker_dir / "episodes.jsonl"),
            "summary_json": str(worker_dir / "results_summary.json"),
            "results_lock": str(worker_dir / "results.lock"),
            "claims_jsonl": str(worker_dir / "claims.jsonl"),
            "log_dir": str(worker_dir / "logs"),
        }
    )
    path = worker_dir / "worker.json"
    _atomic_json(path, payload)
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    return path


def _append_claim(contract: dict[str, Any], claim: dict[str, Any]) -> None:
    payload = dict(claim, claimed_at=time.time())
    with Path(contract["claims_jsonl"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit_results(manifest: dict[str, Any]) -> dict[str, Any]:
    records = load_episode_records(manifest)
    errors = [record for record in records if record.get("error") is not None]
    if errors:
        raise ValueError(f"Episode results contain {len(errors)} non-null errors.")
    per_phase = {phase: 0 for phase in manifest["phases"]}
    for task in manifest["tasks"]:
        for phase in manifest["phases"]:
            completed, remaining, _ = item_progress(manifest, records, task, phase)
            if remaining:
                raise ValueError(f"Incomplete results for {task}/{phase}: {completed} complete")
            per_phase[phase] += completed
    expected = len(manifest["tasks"]) * len(manifest["phases"]) * int(
        manifest["episodes_per_task_phase"]
    )
    if len(records) != expected:
        raise ValueError(f"Expected {expected} episode records, found {len(records)}")
    return {"episodes": len(records), "per_phase": per_phase}


def run_worker(manifest: dict[str, Any], database: Path, worker_id: str) -> int:
    contract_path = write_worker_contract(manifest, worker_id)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    adapter = Path(str(manifest["adapter"])).expanduser().resolve()
    if not adapter.is_file() or not os.access(adapter, os.X_OK):
        raise ValueError(f"Policy adapter is not executable: {adapter}")
    run_id = str(manifest["run_id"])
    while True:
        claim = claim_one(database, run_id, worker_id)
        if claim is None:
            counts = _state_counts(database, run_id)
            if counts.get("pending", 0) or counts.get("running", 0):
                time.sleep(POLL_INTERVAL_S)
                continue
            if counts.get("failed", 0):
                return 1
            audit_results(manifest)
            return 0

        _append_claim(contract, claim)
        try:
            records = load_episode_records(manifest)
            start_episode, remaining, seed_offset = item_progress(
                manifest, records, claim["task"], claim["phase"]
            )
            if remaining == 0:
                _finish_claim(database, run_id, worker_id, claim, complete=True)
                continue
            log_dir = Path(contract["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / (
                f"{adapter.stem}-{claim['phase']}-{worker_id}-attempt{claim['attempt']}.log"
            )
            command = [
                str(adapter),
                "--worker-contract",
                str(contract_path),
                "--task",
                claim["task"],
                "--phase",
                claim["phase"],
                "--start-episode",
                str(start_episode),
                "--remaining",
                str(remaining),
                "--seed-offset",
                str(seed_offset),
            ]
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                adapter_pid = Path(contract["episodes_jsonl"]).parent / "adapter.pid"
                adapter_pid.write_text(f"{process.pid}\n", encoding="ascii")
                return_code = process.wait()
                adapter_pid.unlink(missing_ok=True)
            records = load_episode_records(manifest)
            _, remaining_after, _ = item_progress(
                manifest, records, claim["task"], claim["phase"]
            )
            if remaining_after == 0:
                _finish_claim(database, run_id, worker_id, claim, complete=True)
            else:
                _finish_claim(
                    database,
                    run_id,
                    worker_id,
                    claim,
                    complete=False,
                    error=f"adapter exit={return_code}, remaining={remaining_after}",
                )
        except Exception as exc:
            _finish_claim(
                database, run_id, worker_id, claim, complete=False, error=f"{type(exc).__name__}: {exc}"
            )


def recover_dead_worker(manifest: dict[str, Any], database: Path, worker_id: str) -> int:
    pid_path = _worker_root(manifest) / worker_id / "worker.pid"
    if not pid_path.is_file():
        raise ValueError(f"Cannot prove worker death without {pid_path}")
    pid = int(pid_path.read_text(encoding="ascii").strip())
    if Path(f"/proc/{pid}").exists():
        raise RuntimeError(f"Worker {worker_id} is still alive with pid {pid}")
    adapter_pid_path = pid_path.with_name("adapter.pid")
    if adapter_pid_path.is_file():
        adapter_pid = int(adapter_pid_path.read_text(encoding="ascii").strip())
        if Path(f"/proc/{adapter_pid}").exists():
            raise RuntimeError(
                f"Worker {worker_id} adapter is still alive with pid {adapter_pid}"
            )
    with _database(database) as connection:
        cursor = connection.execute(
            """
            UPDATE work_items SET state='pending', owner_worker=NULL, claim_token=NULL,
              last_error=?, updated_at=?
            WHERE run_id=? AND state='running' AND owner_worker=?
            """,
            (f"recovered after confirmed worker death (pid {pid})", time.time(), manifest["run_id"], worker_id),
        )
    return cursor.rowcount


def finalize(manifest_path: Path, manifest: dict[str, Any], database: Path) -> dict[str, Any]:
    counts = _state_counts(database, str(manifest["run_id"]))
    if counts.get("done", 0) != len(manifest["tasks"]) * len(manifest["phases"]):
        raise ValueError(f"Dispatch queue is not complete: {counts}")
    audit = audit_results(manifest)
    with _database(database) as connection:
        snapshot = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, task_index, task_name, phase_index, phase, state,
                  owner_worker, claim_token, attempts, last_error, updated_at
                FROM work_items WHERE run_id=? ORDER BY task_index, phase_index
                """,
                (manifest["run_id"],),
            )
        ]
    run_root = Path(str(manifest["run_root"])).expanduser().resolve()
    snapshot_path = run_root / "dispatch_queue_snapshot.json"
    _atomic_json(snapshot_path, snapshot)
    claim_hashes = {
        str(worker["worker_id"]): _sha256(path)
        for worker in manifest["workers"]
        if (path := _worker_root(manifest) / str(worker["worker_id"]) / "claims.jsonl").is_file()
    }
    resource_path = run_root / "dispatch_workers.json"
    _atomic_json(resource_path, manifest["workers"])
    finalization = {
        "completed_at": time.time(),
        "audit": audit,
        "queue_snapshot": str(snapshot_path),
        "queue_snapshot_sha256": _sha256(snapshot_path),
        "claim_logs_sha256": claim_hashes,
        "resource_mapping": str(resource_path),
        "resource_mapping_sha256": _sha256(resource_path),
    }
    persisted = {key: value for key, value in manifest.items() if not key.startswith("_")}
    persisted["dispatch_finalization"] = finalization
    _atomic_json(manifest_path, persisted)
    return finalization


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "worker", "recover", "finalize", "status"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    database = dispatch_db_path(str(manifest["run_id"]))
    if args.command == "init":
        initialize(manifest, database)
        return 0
    if args.command == "status":
        print(json.dumps(_state_counts(database, str(manifest["run_id"])), sort_keys=True))
        return 0
    if args.command in {"worker", "recover"} and not args.worker_id:
        parser.error(f"{args.command} requires --worker-id")
    if args.command == "worker":
        return run_worker(manifest, database, args.worker_id)
    if args.command == "recover":
        print(recover_dead_worker(manifest, database, args.worker_id))
        return 0
    print(json.dumps(finalize(args.manifest, manifest, database), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
