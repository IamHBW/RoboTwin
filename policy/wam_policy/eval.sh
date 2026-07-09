#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: bash policy/wam_policy/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <rgb|depth> [wam_checkpoint] [v2w_checkpoint] [vae_checkpoint] [backend] [camera] [eval_num_episodes] [learn_row] [eval_start_episode] [eval_start_seed_offset]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_name="wam_policy"
task_name="$1"
task_config="$2"
ckpt_setting="$3"
seed="$4"
gpu_id="$5"
model_type="${6:-${WAM_MODEL_TYPE:-depth}}"
wam_checkpoint="${7:-${WAM_CHECKPOINT:-}}"
v2w_checkpoint="${8:-${WAM_V2W_CHECKPOINT:-}}"
vae_checkpoint="${9:-${WAM_VAE_CHECKPOINT:-}}"
backend="${10:-${WAM_BACKEND:-local}}"
camera="${11:-${WAM_CAMERA:-front_camera}}"
eval_num_episodes="${12:-${EVAL_NUM_EPISODES:-1}}"
learn_row="${13:-${WAM_LEARN_ROW:-false}}"
eval_start_episode="${14:-${EVAL_START_EPISODE:-0}}"
eval_start_seed_offset="${15:-${EVAL_START_SEED_OFFSET:-${eval_start_episode}}}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "policy: ${policy_name}"
echo "task: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "model_type: ${model_type}"
echo "backend: ${backend}"
echo "camera: ${camera}"
echo "eval_num_episodes: ${eval_num_episodes}"
echo "learn_row: ${learn_row}"
echo "eval_start_episode: ${eval_start_episode}"
echo "eval_start_seed_offset: ${eval_start_seed_offset}"

cd "${ROBOTWIN_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
  --overrides \
  --task_name "${task_name}" \
  --task_config "${task_config}" \
  --ckpt_setting "${ckpt_setting}" \
  --seed "${seed}" \
  --policy_name "${policy_name}" \
  --model_type "${model_type}" \
  --backend "${backend}" \
  --camera "${camera}" \
  --wam_checkpoint "${wam_checkpoint}" \
  --v2w_checkpoint "${v2w_checkpoint}" \
  --vae_checkpoint "${vae_checkpoint}" \
  --rgbdwam_root "${RGBDWAM_ROOT:-/mnt/data/users/tianyu/workspace/code/rgbdwam}" \
  --inference_module "${WAM_INFERENCE_MODULE:-}" \
  --inference_factory "${WAM_INFERENCE_FACTORY:-}" \
  --server_url "${WAM_SERVER_URL:-}" \
  --server_timeout_s "${WAM_SERVER_TIMEOUT_S:-600}" \
  --actions_per_eval "${WAM_ACTIONS_PER_EVAL:-4}" \
  --action_per_frame "${WAM_ACTION_PER_FRAME:-4}" \
  --video_frame_stride "${WAM_VIDEO_FRAME_STRIDE:-1}" \
  --eval_num_episodes "${eval_num_episodes}" \
  --learn_row "${learn_row}" \
  --eval_start_episode "${eval_start_episode}" \
  --eval_start_seed_offset "${eval_start_seed_offset}"
