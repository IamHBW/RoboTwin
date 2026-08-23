import sys
import os
import subprocess
import fcntl
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _read_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _update_results_summary(summary_path, records):
    groups = {}
    for record in records:
        layer = str(record.get("w2a_layer_idx", "unknown"))
        keys = [
            ("overall", "all", "all", "all"),
            (record.get("modality", ""), "all", "all", "all"),
            (record.get("modality", ""), layer, "all", "all"),
            (record.get("modality", ""), layer, record.get("task", ""), record.get("task_config", "")),
        ]
        for key in keys:
            bucket = groups.setdefault(
                "|".join(key),
                {
                    "modality": key[0],
                    "w2a_layer_idx": key[1],
                    "task": key[2],
                    "task_config": key[3],
                    "episodes": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                },
            )
            bucket["episodes"] += 1
            bucket["successes"] += int(bool(record.get("success", False)))

    for bucket in groups.values():
        episodes = bucket["episodes"]
        bucket["success_rate"] = bucket["successes"] / episodes if episodes else 0.0

    overall = groups.get("overall|all|all|all", {"episodes": 0, "successes": 0, "success_rate": 0.0})
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": overall["episodes"],
        "successes": overall["successes"],
        "success_rate": overall["success_rate"],
        "groups": sorted(
            groups.values(),
            key=lambda item: (item["modality"], str(item["w2a_layer_idx"]), item["task"], item["task_config"]),
        ),
    }
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, summary_path)


def record_rgbdwam_eval_result(usr_args, args, task_env, model, episode_seed, success, video_path):
    detail_file = os.environ.get("RGBDWAM_RESULTS_DETAILED_JSONL", "")
    if not detail_file:
        return

    instruction = getattr(model, "last_instruction", None)
    model_prompt = getattr(model, "last_model_prompt", None)
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("Policy did not record a non-empty episode instruction.")
    if not isinstance(model_prompt, str) or not model_prompt.strip():
        raise RuntimeError("Policy did not record a non-empty episode model prompt.")

    detail_path = Path(detail_file)
    summary_file = os.environ.get("RGBDWAM_RESULTS_SUMMARY_JSON", "")
    summary_path = Path(summary_file) if summary_file else detail_path.with_name("results_summary.json")
    lock_path = Path(os.environ.get("RGBDWAM_RESULTS_LOCK", str(detail_path) + ".lock"))
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    w2a_layer_idx = str(usr_args.get("w2a_layer_idx", os.environ.get("WAM_W2A_LAYER_IDX", "")))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modality": str(usr_args.get("model_type", "")),
        "w2a_layer_idx": w2a_layer_idx,
        "task": str(args.get("task_name", "")),
        "task_config": str(args.get("task_config", "")),
        "ckpt_setting": str(args.get("ckpt_setting", "")),
        "instruction": instruction,
        "model_prompt": model_prompt,
        "episode_index": int(max(0, getattr(task_env, "test_num", 1) - 1)),
        "seed": int(episode_seed),
        "success": bool(success),
        "success_int": int(bool(success)),
        "successes_so_far": int(getattr(task_env, "suc", 0)),
        "episodes_so_far": int(getattr(task_env, "test_num", 0)),
        "success_rate_so_far": float(getattr(task_env, "suc", 0) / max(1, getattr(task_env, "test_num", 0))),
        "video_path": str(video_path) if video_path else "",
        "eval_save_dir": str(usr_args.get("eval_save_dir", "")),
        "wam_checkpoint": str(usr_args.get("wam_checkpoint", "")),
        "v2w_checkpoint": str(usr_args.get("v2w_checkpoint", "")),
        "vae_checkpoint": str(usr_args.get("vae_checkpoint", "")),
        "server_url": str(usr_args.get("server_url", "")),
        "actions_per_eval": usr_args.get("actions_per_eval", ""),
        "action_per_frame": usr_args.get("action_per_frame", ""),
        "video_frame_stride": usr_args.get("video_frame_stride", ""),
        "learn_row": usr_args.get("learn_row", ""),
    }

    try:
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with detail_path.open("a", encoding="utf-8") as detail_handle:
                detail_handle.write(json.dumps(record, sort_keys=True) + "\n")
                detail_handle.flush()
                os.fsync(detail_handle.fileno())
            _update_results_summary(summary_path, _read_jsonl(detail_path))
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        print(f"[rgbdwam-results] failed to record eval result: {exc}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    usr_args["eval_save_dir"] = str(save_dir)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("eval_num_episodes", 100))
    if test_num <= 0:
        raise ValueError(f"eval_num_episodes must be positive, got {test_num}")
    eval_start_episode = int(usr_args.get("eval_start_episode", 0))
    eval_start_seed_offset = int(usr_args.get("eval_start_seed_offset", eval_start_episode))
    if eval_start_episode < 0 or eval_start_seed_offset < 0:
        raise ValueError(
            f"eval_start_episode/eval_start_seed_offset must be non-negative, "
            f"got {eval_start_episode}/{eval_start_seed_offset}"
        )
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   usr_args=usr_args,
                                   test_num=test_num,
                                   start_episode=eval_start_episode,
                                   start_seed_offset=eval_start_seed_offset,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                usr_args=None,
                test_num=100,
                start_episode=0,
                start_seed_offset=0,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    usr_args = usr_args or {}

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = int(start_episode)

    now_id = int(start_episode)
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed + int(start_seed_offset)
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                if os.environ.get("ROBOTWIN_DEBUG_EXCEPTIONS", "0") == "1":
                    traceback.print_exc()
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1
        video_path = None
        if TASK_ENV.eval_video_path is not None:
            video_path = Path(TASK_ENV.eval_video_path) / f"episode{TASK_ENV.test_num - 1}.mp4"
        record_rgbdwam_eval_result(usr_args, args, TASK_ENV, model, now_seed, succ, video_path)

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
