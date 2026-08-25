import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from script.distributed_eval_dispatcher import (
    _database,
    _finish_claim,
    audit_results,
    claim_one,
    finalize,
    initialize,
    load_manifest,
    recover_dead_worker,
    run_worker,
    write_worker_contract,
)


def _claim_once(database, run_id, worker_id, queue):
    claim = claim_one(Path(database), run_id, worker_id)
    queue.put(None if claim is None else (claim["task"], claim["phase"]))


def _drain_queue(database, run_id, worker_id, queue):
    database = Path(database)
    while True:
        claim = claim_one(database, run_id, worker_id)
        if claim is None:
            return
        queue.put((claim["task"], claim["phase"], time.monotonic()))
        _finish_claim(database, run_id, worker_id, claim, complete=True)


def _claim_and_wait(database, run_id, worker_id, worker_dir):
    worker_dir = Path(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "worker.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    claim_one(Path(database), run_id, worker_id)
    while True:
        time.sleep(1)


class DistributedEvalDispatcherTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_root = self.root / "run"
        self.adapter = self.root / "dummy_adapter.py"
        self.adapter.write_text(
            """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--worker-contract', type=Path, required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--phase', required=True)
parser.add_argument('--start-episode', type=int, required=True)
parser.add_argument('--remaining', type=int, required=True)
parser.add_argument('--seed-offset', type=int, required=True)
args = parser.parse_args()
contract = json.loads(args.worker_contract.read_text())
attempt_marker = Path(contract['run_root']) / 'dummy_attempted'
count = args.remaining if attempt_marker.exists() else min(2, args.remaining)
attempt_marker.parent.mkdir(parents=True, exist_ok=True)
attempt_marker.touch()
with Path(contract['episodes_jsonl']).open('a', encoding='utf-8') as stream:
    for offset in range(count):
        episode = args.start_episode + offset
        seed = 100000 + args.seed_offset + offset
        stream.write(json.dumps({
            'run_id': contract['run_id'],
            'task': args.task,
            'phase': args.phase,
            'episode_index': episode,
            'candidate_seed': seed,
            'accepted_seed': seed,
            'success': bool(episode % 2),
            'error': None,
        }, sort_keys=True) + '\\n')
        stream.flush()
        os.fsync(stream.fileno())
raise SystemExit(0 if count == args.remaining else 7)
""",
            encoding="utf-8",
        )
        self.adapter.chmod(0o755)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_manifest(self, tasks=None, phases=None, workers=1, episodes=3):
        payload = {
            "run_id": "test-run",
            "run_root": str(self.run_root),
            "adapter": str(self.adapter),
            "tasks": tasks or ["task_0"],
            "phases": phases or ["clean"],
            "episodes_per_task_phase": episodes,
            "candidate_seed_start": 100000,
            "result_contract": {"run_id": "test-run"},
            "workers": [
                {"worker_id": f"worker{index:02d}", "gpu_id": index % 8}
                for index in range(workers)
            ],
            "adapter_config": {"frozen": True},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, load_manifest(path)

    def test_32_processes_claim_unique_single_item(self):
        _, manifest = self.make_manifest(workers=32)
        database = self.root / "dispatch.sqlite3"
        initialize(manifest, database)
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_claim_once,
                args=(str(database), manifest["run_id"], f"worker{index:02d}", queue),
            )
            for index in range(32)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        results = [queue.get(timeout=1) for _ in processes]
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_32_processes_claim_100_items_once_with_sub_five_second_p95_gap(self):
        tasks = [f"task_{index:02d}" for index in range(50)]
        _, manifest = self.make_manifest(
            tasks=tasks, phases=["clean", "randomized"], workers=32, episodes=1
        )
        database = self.root / "dispatch.sqlite3"
        initialize(manifest, database)
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_drain_queue,
                args=(str(database), manifest["run_id"], f"worker{index:02d}", queue),
            )
            for index in range(32)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            self.assertEqual(process.exitcode, 0)
        claims = [queue.get(timeout=1) for _ in range(100)]
        keys = [(task, phase) for task, phase, _ in claims]
        self.assertEqual(len(keys), len(set(keys)))
        times = sorted(timestamp for _, _, timestamp in claims)
        gaps = sorted(right - left for left, right in zip(times, times[1:]))
        self.assertLess(gaps[int(len(gaps) * 0.95)], 5.0)

    def test_worker_resumes_strict_prefix_after_adapter_failure(self):
        _, manifest = self.make_manifest(episodes=3)
        database = self.root / "dispatch.sqlite3"
        initialize(manifest, database)
        self.assertEqual(run_worker(manifest, database, "worker00"), 0)
        records = audit_results(manifest)
        self.assertEqual(records, {"episodes": 3, "per_phase": {"clean": 3}})
        claims = [
            json.loads(line)
            for line in (self.run_root / "client/workers/worker00/claims.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([claim["attempt"] for claim in claims], [1, 2])
        episodes = [
            json.loads(line)
            for line in (self.run_root / "client/workers/worker00/episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([record["episode_index"] for record in episodes], [0, 1, 2])
        self.assertEqual([record["candidate_seed"] for record in episodes], [100000, 100001, 100002])

    def test_killed_worker_claim_requires_proven_death_before_requeue(self):
        _, manifest = self.make_manifest(workers=2, episodes=1)
        database = self.root / "dispatch.sqlite3"
        initialize(manifest, database)
        worker_dir = self.run_root / "client/workers/worker00"
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_claim_and_wait,
            args=(str(database), manifest["run_id"], "worker00", str(worker_dir)),
        )
        process.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with _database(database) as connection:
                state = connection.execute(
                    "SELECT state FROM work_items WHERE run_id=?",
                    (manifest["run_id"],),
                ).fetchone()["state"]
            if state == "running":
                break
            time.sleep(0.05)
        else:
            self.fail("worker did not claim item")
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            recover_dead_worker(manifest, database, "worker00")
        process.terminate()
        process.join(10)
        self.assertFalse(process.is_alive())
        (worker_dir / "adapter.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "adapter is still alive"):
            recover_dead_worker(manifest, database, "worker00")
        (worker_dir / "adapter.pid").unlink()
        self.assertEqual(recover_dead_worker(manifest, database, "worker00"), 1)
        replacement = claim_one(database, manifest["run_id"], "worker01")
        self.assertIsNotNone(replacement)

    def test_finalizer_exports_snapshot_and_hashes(self):
        manifest_path, manifest = self.make_manifest(episodes=3)
        database = self.root / "dispatch.sqlite3"
        initialize(manifest, database)
        self.assertEqual(run_worker(manifest, database, "worker00"), 0)
        payload = finalize(manifest_path, manifest, database)
        self.assertEqual(payload["audit"]["episodes"], 3)
        self.assertTrue(Path(payload["queue_snapshot"]).is_file())
        self.assertEqual(set(payload["claim_logs_sha256"]), {"worker00"})
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["dispatch_finalization"], payload)

    def test_canonical_50_by_2_by_20_audit(self):
        tasks = [f"task_{index:02d}" for index in range(50)]
        _, manifest = self.make_manifest(
            tasks=tasks, phases=["clean", "randomized"], workers=32, episodes=20
        )
        path = self.run_root / "client/workers/worker00/episodes.jsonl"
        path.parent.mkdir(parents=True)
        records = [
            {
                "run_id": "test-run",
                "task": task,
                "phase": phase,
                "episode_index": episode,
                "candidate_seed": 100000 + episode,
                "accepted_seed": 100000 + episode,
                "success": False,
                "error": None,
            }
            for task in tasks
            for phase in ("clean", "randomized")
            for episode in range(20)
        ]
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        self.assertEqual(
            audit_results(manifest),
            {"episodes": 2000, "per_phase": {"clean": 1000, "randomized": 1000}},
        )

    def test_dispatcher_has_no_policy_name_branch(self):
        source = (Path(__file__).parent / "distributed_eval_dispatcher.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("fastwam", source.lower())

    def test_manifest_rejects_path_traversal_ids(self):
        path, _ = self.make_manifest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_id"] = "../escape"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe run ID"):
            load_manifest(path)

    def test_duplicate_live_worker_writer_is_rejected(self):
        _, manifest = self.make_manifest()
        worker_dir = self.run_root / "client/workers/worker00"
        worker_dir.mkdir(parents=True)
        (worker_dir / "worker.pid").write_text("1\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "already live"):
            write_worker_contract(manifest, "worker00")


if __name__ == "__main__":
    unittest.main()
