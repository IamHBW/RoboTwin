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

from script import eval_policy as eval_policy_module
from script.eval_policy import record_rgbdwam_eval_result


class EvalResultRecordingTest(unittest.TestCase):
    def test_video_size_matches_written_head_camera_frames(self):
        args = {"camera": {"head_camera_type": "D435", "collect_wrist_camera": True}}
        with patch.object(
            eval_policy_module, "get_camera_config", return_value={"w": 320, "h": 240}
        ):
            self.assertEqual(eval_policy_module.get_eval_video_size(args), "320x240")

    def test_eval_policy_resumes_episode_and_seed_offsets(self):
        class FakeEnv:
            def __init__(self):
                self.setup_calls = []

            def setup_demo(self, now_ep_num, seed, **_kwargs):
                self.setup_calls.append((now_ep_num, seed))
                self.plan_success = True
                self.eval_success = False
                self.eval_video_path = None
                self.render_freq = 0
                self.take_action_cnt = 0
                self.step_lim = 1

            def play_once(self):
                return {"info": {}}

            def check_success(self):
                return True

            def close_env(self, **_kwargs):
                pass

            def set_instruction(self, instruction):
                self.instruction = instruction

            def get_obs(self):
                return None

        task_env = FakeEnv()

        def policy_function(_policy_name, function_name):
            if function_name == "eval":
                return lambda env, _model, _observation: setattr(env, "eval_success", True)
            return lambda _model: None

        with patch.object(
            eval_policy_module,
            "eval_function_decorator",
            side_effect=policy_function,
        ), patch.object(
            eval_policy_module,
            "generate_episode_descriptions",
            return_value=[{"unseen": ["instruction"]}],
            create=True,
        ):
            next_seed, successes = eval_policy_module.eval_policy(
                "click_bell",
                task_env,
                {
                    "task_name": "click_bell",
                    "task_config": "demo_clean",
                    "ckpt_setting": "remote_step_030000",
                    "policy_name": "fastwam_policy",
                    "clear_cache_freq": 10,
                    "render_freq": 0,
                },
                types.SimpleNamespace(),
                100000,
                test_num=1,
                start_episode=14,
                start_seed_offset=9,
                instruction_type="unseen",
            )

        self.assertEqual(task_env.setup_calls, [(14, 100009), (14, 100009)])
        self.assertEqual(task_env.test_num, 15)
        self.assertEqual(next_seed, 100010)
        self.assertEqual(successes, 1)

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
                    "ROBOTWIN_RUN_ID": "run-1",
                    "ROBOTWIN_CONTRACT_ID": "contract-1",
                },
            ):
                record_rgbdwam_eval_result(
                    {}, args, task_env, model, 100002, True, "episode2.mp4"
                )

            record = json.loads(detail_path.read_text(encoding="utf-8"))

        self.assertEqual(record["task"], "place_object_basket")
        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["contract_id"], "contract-1")
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
