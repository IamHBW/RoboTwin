import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


envs = types.ModuleType("envs")
envs.CONFIGS_PATH = ""
create_actor = types.ModuleType("envs.utils.create_actor")
create_actor.UnStableError = RuntimeError
sys.modules.setdefault("envs", envs)
sys.modules.setdefault("envs.utils", types.ModuleType("envs.utils"))
sys.modules.setdefault("envs.utils.create_actor", create_actor)
sys.modules.setdefault(
    "generate_episode_instructions",
    types.ModuleType("generate_episode_instructions"),
)

from script.eval_policy import record_rgbdwam_eval_result


class EvalResultRecordingTest(unittest.TestCase):
    def test_records_episode_instruction_and_exact_model_prompt(self):
        task_env = types.SimpleNamespace(test_num=3, suc=2)
        model = types.SimpleNamespace(
            last_instruction="Put the red block into the basket.",
            last_model_prompt=(
                "A video recorded from a robot's point of view executing the "
                "following instruction: Put the red block into the basket."
            ),
        )
        args = {
            "task_name": "place_object_basket",
            "task_config": "demo_clean",
            "ckpt_setting": "step_030000",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            detail_path = Path(temp_dir) / "results_detailed.jsonl"
            with patch.dict(
                os.environ,
                {
                    "RGBDWAM_RESULTS_DETAILED_JSONL": str(detail_path),
                    "RGBDWAM_RESULTS_SUMMARY_JSON": str(
                        Path(temp_dir) / "results_summary.json"
                    ),
                    "RGBDWAM_RESULTS_LOCK": str(Path(temp_dir) / "results.lock"),
                },
            ):
                record_rgbdwam_eval_result(
                    {}, args, task_env, model, 100002, True, "episode2.mp4"
                )

            record = json.loads(detail_path.read_text(encoding="utf-8"))

        self.assertEqual(record["task"], "place_object_basket")
        self.assertEqual(record["episode_index"], 2)
        self.assertEqual(record["seed"], 100002)
        self.assertTrue(record["success"])
        self.assertEqual(record["instruction"], model.last_instruction)
        self.assertEqual(record["model_prompt"], model.last_model_prompt)

    def test_rejects_missing_model_text(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"RGBDWAM_RESULTS_DETAILED_JSONL": str(Path(temp_dir) / "result.jsonl")},
        ):
            with self.assertRaisesRegex(RuntimeError, "episode instruction"):
                record_rgbdwam_eval_result(
                    {}, {}, types.SimpleNamespace(), types.SimpleNamespace(), 0, False, None
                )


if __name__ == "__main__":
    unittest.main()
