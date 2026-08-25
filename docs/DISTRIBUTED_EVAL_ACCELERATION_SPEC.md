# RoboTwin 1.0 通用分布式评测加速规格

- 状态：实现与隔离验证完成；活动评测结束后再合入传输补丁并做实机吞吐验收
- 日期：2026-08-25
- 范围：通用动态任务领取、远程 policy 无损传输优化
- 首个参考 profile：FastWAM
- 参考运行：`rgbroi-seed-full8-rp24-scale-20260825T044107Z`

## 1. 目标

本规格服务于所有 RoboTwin 1.0 policy，不把调度器绑定到 FastWAM、某种模型输入或某种 RPC 协议。

只实现两项改进：

1. policy-agnostic dynamic dispatcher：空闲 worker 动态领取 task-phase。
2. capability-gated transport optimization：远程 policy 在证明模型输入完全一致后，按各自 profile 减少传输开销。

FastWAM 是第一个实现和性能验收对象，不是 dispatcher 的内置特例。以后增加 policy 时，只新增一个薄的启动 adapter；只有确有传输瓶颈时才新增 transport profile，不复制 dispatcher。

## 2. 固定边界

### 2.1 每次运行必须冻结

run manifest 必须记录并冻结：

- benchmark、任务列表、phase 列表和每个 task-phase 的有效 rollout 数。
- instruction type、candidate seed 起点与递增规则、专家可行性过滤和成功判定。
- policy 名称、代码 commit、checkpoint 及 policy-specific 参数。
- worker 数、worker 到 GPU 的映射，以及可选的 policy slot/server endpoint。
- adapter 路径、结果目录、结果主键和完成审计规则。

标准 canonical full run 仍为 50 tasks × clean/randomized × 20 = 2,000 个有效 rollout。调度方式不得改变运行 manifest 中的评测语义。

### 2.2 Dispatcher 不得感知

dispatcher 不得 import policy 代码，也不得假设：

- policy 是本地还是远程运行；
- observation 包含 RGB、depth 或其他 modality；
- action 是单步还是 chunk；
- server 数量、端口或模型 batch size；
- FastWAM 的 checkpoint、replan、denoise、Seed 或预处理逻辑。

dispatcher 只理解 worker、task、phase、resume state、子进程退出状态和标准结果记录。

### 2.3 当前基础设施边界

- 当前 dispatcher 只支持所有 worker 位于同一台 client 主机。
- worker 数及每个 worker 的资源绑定由 run manifest 决定；当前参考拓扑为单机 8 张 RTX 5090、每卡 4 个 worker，共 32 个。
- worker 在整个 run 内保持自己的 GPU 和可选 endpoint 绑定，只动态更换 task-phase。
- 不加入任务耗时预测、重任务优先或其他优先级策略；工作项按 manifest 固定顺序领取。

## 3. Policy adapter 契约

每个 policy 只需提供一个可执行 adapter，不新增 class hierarchy 或插件注册系统。dispatcher 以 argv 调用：

```text
<policy_adapter> \
  --worker-contract <worker.json> \
  --task <task_name> \
  --phase <phase> \
  --start-episode <n> \
  --remaining <n> \
  --seed-offset <n>
```

`worker.json` 至少包含 `worker_id` 和 `gpu_id`；远程 policy 可额外包含 `policy_slot`、`host`、`port`，本地 policy 不需要这些字段。

adapter 的责任：

- 把通用参数转换为现有 policy evaluator 命令和配置。
- 使用 dispatcher 提供的 worker-specific JSONL、summary、lock 和日志路径。
- 保持 manifest 冻结的 policy 参数不变。
- 返回 evaluator 子进程退出码。
- 如有远程 transport profile，在启动前完成 server health/capability 协商。

dispatcher 不以 adapter 的退出码直接判断 task-phase 完成；完成事实只由标准 episode JSONL 审计确定。一个 run 只绑定一个 policy adapter；不同 policy 使用不同 run ID。

## 4. P1：通用动态任务领取

### 4.1 工作项

唯一工作项为：

```text
(run_id, task_name, phase)
```

canonical full run 共 100 项。task-phase 内仍由现有 evaluator 顺序产生 20 个有效 rollout，不拆成 episode 级任务，从而复用现有 seed、resume、结果记录和完成审计。

### 4.2 队列与锁

在 client 单机上使用 Python 标准库 `sqlite3`。数据库必须位于该 client 主机的本地文件系统：

```text
/var/tmp/robotwin-dispatch/<run_id>/dispatch.sqlite3
```

实际 `litchi_data_001:/mnt/data` 是 NFSv3，并带有 `nolock,local_lock=all`；它继续保存 episode JSONL 和最终产物，但不承担调度互斥。队列丢失时从 JSONL 重建，因此 SQLite 不是评测结果的唯一持久化来源。

最小表结构：

```text
work_items(
  run_id,
  task_index,
  task_name,
  phase_index,
  phase,
  state,          -- pending | running | done | failed
  owner_worker,
  claim_token,
  attempts,
  last_error,
  updated_at,
  PRIMARY KEY (run_id, task_name, phase)
)
```

所有 worker 连接同一个本地数据库并设置 `busy_timeout=30000`。领取使用单个短事务：

1. `BEGIN IMMEDIATE`。
2. 按 `task_index, phase_index` 选择第一个 pending 项。
3. 使用 `WHERE state='pending'` 更新为 running，并写入 owner、唯一 claim token 和时间戳。
4. 必须断言恰好更新一行；否则 rollback，不能启动 adapter。
5. COMMIT 成功后才能启动 adapter；遇到 `SQLITE_BUSY` 重新领取，不能使用旧查询结果。

`BEGIN IMMEDIATE` 在 SELECT 前取得写事务。一个 worker 持有事务时，其他 worker 只能等待；提交后等待者重新查询，因此多个 worker 不能成功领取同一工作项。

该锁仅适用于当前单 client 主机拓扑。未来若使用多机 worker，必须另行设计跨机协调，不能把此 SQLite 文件移到 NFS 后继续使用。

### 4.3 Worker 生命周期

每个长驻 worker 循环执行：

1. 原子领取一个 pending 工作项。
2. 聚合全 run 的 episode JSONL，计算该 task-phase 的严格连续前缀、下一 candidate seed offset 和 remaining。
3. remaining 为 0 时直接置为 done；否则调用当前 run 的 policy adapter。
4. adapter 退出后重新审计；只有 20 个唯一、有效且 episode index 为 `0..19` 的记录才置为 done。
5. 未完成则增加 attempts；少于 3 次时重新 pending，达到 3 次时置为 failed，保留有效前缀并继续处理其他任务。
6. pending 暂时为空但仍有 running 时等待，不退出；仅在全部工作项 done 且最终审计通过后统一结束。

### 4.4 结果所有权与故障恢复

- 同一 task-phase 任意时刻最多一个 live owner。
- resume helper 聚合 `<run_root>/client/workers/*/episodes.jsonl`，不能只读取当前 worker 文件。
- 结果主键统一为 `(task_name, phase, episode_index)`；finalizer 拒绝重复主键。
- 每个 worker 只写自己的 `claims.jsonl`、`episodes.jsonl` 和日志，禁止多个进程并发 append 同一个 NFS 文件。
- adapter 的临时 config 和日志名包含 policy、phase 与 worker ID，避免跨 phase 或跨 worker 覆盖。
- adapter 失败后按现有有效前缀 resume，不删除或重写历史记录。
- worker/systemd unit 失败后，controller 必须先确认进程已经死亡，才能把其 running claim 重新置为 pending；不能只靠超时抢占 live owner。
- dispatcher 重启时先审计 JSONL，再重建/修正 SQLite；JSONL 是进度事实来源。

### 4.5 P1 验收

- dispatcher 源码和测试不 import、匹配或分支判断任何 policy 名称。
- 用 dummy adapter 即可完成队列、resume、失败重试和 finalizer 集成测试。
- 在本地 ext4 上用 32 个独立进程同时领取唯一一个 pending 项，必须恰好一个成功。
- 用 32 个进程领取 100 个 pending 项，每个主键必须恰好成功领取一次。
- 模拟杀死一个 worker 后，其 task-phase 能被其他 worker 从严格前缀接管。
- pending 非空时，健康 worker 的领取空档 p95 小于 5 秒。
- canonical run 最终每个 task-phase 恰好 20 条唯一有效记录，总数 2,000，clean/randomized 各 1,000，无 non-null error。
- finalizer 单进程导出最终队列 snapshot，并在 manifest 记录队列、per-worker claim log 和资源映射的 SHA-256。

## 5. P2：通用远程传输优化

### 5.1 通用规则

P2 只适用于经网络传输 observation 的 policy；本地 policy 自动跳过。通用基础设施只负责 capability 协商、指标采集和 SSH 链路，不对 observation 做递归 dtype 转换。

每个 policy 默认使用 identity transport。只有满足以下条件才能启用 optimized profile：

1. server health 明确声明 profile ID 和期望 schema。
2. server 先上线，并向后兼容该 policy 的 legacy client。
3. client 只在 capability 与 run manifest 的期望值完全匹配时启用；否则回退 identity 并记录原因。
4. profile 明确列出允许变化的字段、旧/新 wire dtype、服务端原本的第一次转换和验证程序。
5. 新旧路径抵达模型前的最终输入必须逐元素一致；成功率近似一致不能替代此证明。
6. 禁止没有独立 profile 的全局 cast，也禁止 FP16/BF16、量化和有损图像编码。

建议 health 字段：

```json
{
  "policy_backend": "<name>",
  "action_transport": "<policy-specific>",
  "observation_transport_profiles": ["identity", "<profile-id>"]
}
```

每个优化请求携带实际 profile ID；server 必须校验 schema、shape 和 dtype，不匹配立即报错，不能静默猜测。

### 5.2 Profile 验收模板

每个新增 profile 都必须记录并验证：

```text
profile_id
applicable_policy_and_commit
affected_observation_fields
legacy_wire_dtype -> optimized_wire_dtype
server_first_conversion
legacy/optimized model-input equality check
wire-byte reduction
fallback behavior
```

对同一批原始 observation 同时执行：

```text
legacy wire/decode -> legacy server preprocessing -> model input
optimized wire/decode -> optimized server preprocessing -> model input
```

验收必须覆盖正常值、边界值及该 policy 原本支持的 NaN/Inf；比较所有模型输入的 key、shape、dtype 和数值。若预处理结果本应完全相同，使用 `torch.equal`/`np.array_equal(equal_nan=True)`，不能放宽为经验 tolerance。

转换必须写入新的 transport payload，不得原地修改 simulator observation。未列入 profile 的 RGB、state、instruction、action 等字段保持原 dtype 和内容。

### 5.3 FastWAM 参考 profile

FastWAM 的首个 profile 为 `fastwam_robotwin_depth_fp32_v1`。它不是通用 dispatcher 的内置逻辑，代码放在 FastWAM client/server adapter 中。

允许变化的字段仅为：

```text
observation["observation"]["head_camera"]["depth"]
observation["observation"]["left_camera"]["depth"]
observation["observation"]["right_camera"]["depth"]
```

legacy wire dtype 为 float64，optimized wire dtype 为 C-contiguous float32。只有部署中的 FastWAM server commit 确认 `_depth_to_normalized_chw` 在任何算术、裁剪、缩放和归一化之前执行：

```python
depth = np.asarray(depth, dtype=np.float32)
```

才允许启用此 profile。client 使用 `np.ascontiguousarray(depth, dtype=np.float32)`；NaN/Inf 保持原值，并由既有 server `nan_to_num` 处理。若后续 server 在 cast 前增加任何算术，必须撤销 capability 并重新验证。

三路 `320x240` depth 从 float64 改为 float32，理论上可使相机原始载荷下降约 36.4%；最终以实际 wire bytes 为准。FastWAM 的 RGB、low-dimensional state、instruction、action chunk 及 `chunk_v1` action transport 均不改变。

### 5.4 SSH 链路

SSH 优化适用于所有使用相同 relay 的远程 policy，不依赖 observation profile：

1. 用相同 policy、任务切片和并发度记录当前 relay 基线。
2. 两段 relay 均启用原生无损 `-C` 压缩，比较连续稳定窗口。
3. CPU 资源充足且吞吐提高至少 5%、错误率不增加时保留；否则恢复无压缩。
4. 如果仍有证据表明单条 ControlMaster 存在 head-of-line blocking，再按 policy slot 拆分 ControlMaster；没有 slot 概念的 policy 按独立 endpoint 拆分。

拆分只能由以下证据触发：worker 健康且仍有 pending、server queue 未饱和、aggregate wire throughput 触顶或单 TCP 流受限。第一版不引入应用层 zlib、新 framing 或新依赖。

### 5.5 P2 可观测性与验收

远程 client 每次 policy RPC 记录：

- 编码前后 payload bytes；
- encode time 和完整 socket round-trip time；
- policy、transport profile、worker 和 endpoint；
- fallback、decode error 和连接重置。

server 每连接首次记录解码后 schema/dtype/shape，并保留已有 decode、queue 和 model timing。

通用验收条件：

- optimized 与 identity 路径的模型输入通过该 profile 声明的逐元素一致性测试。
- profile 未授权字段没有变化。
- decode error、意外 fallback、连接重置均为 0。
- SSH `-C` 或 ControlMaster 拆分只有在固定窗口吞吐提高至少 5% 时保留。

FastWAM 参考验收额外要求：

- camera payload bytes 至少下降 35%。
- 固定 `8x4` 下 action chunk 吞吐至少达到 161 chunks/min 基线的 1.10 倍，目标为 180–200 chunks/min。
- 参考 workload 约 28,611 chunks；无外部服务故障时，2,000 episodes 目标不超过 3 小时。

其他 policy 必须建立自己的 payload、吞吐和 wall-time 基线；不能直接继承 FastWAM 的 2–3 小时结论。

## 6. 实施顺序与回退

1. 实现通用 dispatcher 和 dummy adapter 测试，不加入任何 policy 分支。
2. 用薄 FastWAM adapter 接入现有命令，保持 identity transport，验证 P1 完整 resume 和 2,000 条审计。
3. 实现 FastWAM client/server 的 capability 和 `fastwam_robotwin_depth_fp32_v1`，通过模型输入逐元素一致性测试。
4. 对 SSH `-C` 做固定窗口 A/B；有证据时再拆 ControlMaster。
5. 后续 policy 只提供自己的启动 adapter；默认 identity transport，测出传输瓶颈后才增加独立 profile。

回退方式：

- P1：停止 dispatcher，从 episode JSONL 重建剩余项；必要时可生成静态 shard 并用原 policy adapter 续跑。
- P2：capability 不匹配或校验失败时回退 identity；SSH 恢复无压缩或原 ControlMaster 布局。
- 回退必须保留 run ID、policy/checkpoint、seed、已有有效 rollout 和审计主键。

## 7. 溯源

- FastWAM 基准记录：`/mnt/data/users/bowen/workspace/outputs/robotwin_distributed/rgbroi-seed-full8-rp24-scale-20260825T044107Z/fastwam/THROUGHPUT_SCALE.md`
- 当前静态 launcher：同一 run root 下的 `client/launch_client_group.sh`
- FastWAM client checkout：`/mnt/data/users/bowen/workspace/code/rgbdwam-robotwin-5090-client`
- RoboTwin checkout：`/mnt/data/users/bowen/workspace/code/RoboTwin`
- FastWAM HTrain server checkout：`/home/bowen/workspace/code/rgbdwam-vlm-guidance-rgbd`
- 参考运行依赖的原始 action stats、depth VAE 等 `/mnt/data/users/tianyu/` 路径已记录在该 run 的 `server_manifest.json`；实际运行继续只使用 bowen-owned copy，本规格不增加新的跨用户运行依赖。

## 8. 实现入口

通用 dispatcher 为 `script/distributed_eval_dispatcher.py`，生产 CLI 固定把 SQLite 放在
`/var/tmp/robotwin-dispatch/<run_id>/dispatch.sqlite3`：

```bash
python script/distributed_eval_dispatcher.py init --manifest <run_manifest.json>
python script/distributed_eval_dispatcher.py worker --manifest <run_manifest.json> --worker-id <worker_id>
python script/distributed_eval_dispatcher.py recover --manifest <run_manifest.json> --worker-id <confirmed_dead_worker>
python script/distributed_eval_dispatcher.py finalize --manifest <run_manifest.json>
```

manifest 的 dispatcher 必需字段如下；`tasks` 必须展开为冻结的有序列表，不能只写数量或另指一份可变 registry：

```json
{
  "contract_id": "<frozen evaluator contract>",
  "run_id": "<safe unique run id>",
  "run_root": "<NFS result root>",
  "adapter": "<absolute executable adapter path>",
  "tasks": ["<ordered task names>"],
  "phases": ["clean", "randomized"],
  "episodes_per_task_phase": 20,
  "candidate_seed_start": 100000,
  "result_contract": {
    "run_id": "<same run id>",
    "contract_id": "<same contract id>",
    "model": "<policy name>",
    "checkpoint": "<frozen checkpoint path>",
    "instruction_type": "unseen"
  },
  "workers": [
    {"worker_id": "worker00", "gpu_id": 0, "policy_slot": 0, "host": "127.0.0.1", "port": 35853}
  ],
  "adapter_config": {}
}
```

FastWAM 薄 adapter 为 `experiments/robotwin/fastwam_dispatch_adapter.py`。其
`adapter_config` 冻结 `python`、`client_root`、`robotwin_root`、`checkpoint`、
`output_dir`、`observation_transport_profile` 和 policy-specific `hydra_overrides`；动态 task、phase、
episode/seed offset、GPU、endpoint 和 writer 路径只能由 dispatcher 注入。
