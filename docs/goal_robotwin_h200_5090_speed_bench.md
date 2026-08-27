你是执行者，本文是唯一任务；拿不准写 `BLOCKED.md`，继续别项。
断点续跑先读 run 根目录的 `PROGRESS.md`；每项完成即更新，不重做。
把 RoboTwin Joint Double-VAE 的 5090 分布式评测加速到聚合每 2–3 秒 1 条、全量 2,000 条不超 3 小时；纯 H200 只作粗参考。
冲突时按「评测有效 > 吞吐 > 结果完整 > 对照严谨」让步。
「只允许/不许」是死规矩；「建议」可替代，但在 `PROGRESS.md` 记原因。

## 我替领导拍的板

- 模型定为报告中 `joint_dv_on`=Joint（Double-VAE, VLM-on）；checkpoint 为 `/mnt/data/users/bowen/workspace/ckpt/rgbdwam/runs/dblvae-c50-r3/20260804-140501/checkpoints/weights/step_030000.pt`，SHA-256 `f3639116d472f9d96d91ab930784af8893afafdb90d78c9d0695b72c5559ba6c`。
- bench 用冻结的 canonical 50 tasks×clean/randomized×2 条有效 rollout=200（取「一两百」上限），保留全量的 100 个 task-phase 工作项。
- 绝对目标：「2–3 秒一条」指实际 clients 的聚合节拍，client 数可调、不作验收；bench 从 manifest 所列 clients 全 ready 到第 200 条的时长/200≤3.0s；最终 2,000 条从 HTrain job `Running` 到 `EVAL_COMPLETE`≤10,800s，目标 7,200s。
- 仍报 H200 粗参考、client 总数/卡映射、启动/wall time、eps/h、action-steps/s、chunks/min 和加速比，但不设加速比门槛。
- 预算：1 次纯 H200、1 次初始跨云、最多 2 次优化复测、1 次最终 2,000 条；共享语义代码改了可补跑 H200，最多 6 个单节点 8-GPU job/18 node-hours（猜的）。

## 界限

- 只在 `bowen` 目录建隔离 worktree；只改 RoboTwin evaluator/dispatcher、FastWAM server/client、bench 审计相关文件；产物只写 `/mnt/data/users/bowen/workspace/outputs/robotwin_speed_bench/<RUN_ID>` 和 `/mnt/data/users/bowen/workspace/ckpt/rgbdwam/dependencies/<RUN_ID>`。
- 原 checkout/历史结果只读；不覆盖已有改动，不碰别的 HTrain/JDCloud 任务，不删 job/数据，不改密钥、SSH、权限或共享环境。
- 禁止直接用 `tianyu/luzheng` runtime 资产；复制或复用 bowen-owned 副本，manifest 记录原来源和 SHA-256。
- 两路同 checkpoint/config、unseen instruction、horizon 32、replan 24、10 denoise steps 和视频设置；case/seed 无需逐条配对，起跑无需同步，禁止用历史结果顶替本次 bench。

## 现状与任务 0

- 2026-08-26 实测：报告在九章云 `/home/bowen/workspace/code/rgbdwam/evaluate_results/robotwin/five_model_summary_20260807`；历史纯 H200 8 workers，clean 1000 条约 3.40h、randomized 1000 条约 4.67h。现役 8×4 5090 参考是 161 chunks/min。
- 九章 `submit --status` 的 `GPU-H800E-8卡` 即 H200；job 内仍记录 `nvidia-smi -L`。当时 JDCloud 002/003/005/006/007 空闲；启动前用 `$jdcloud-cluster` 串行重查 001–016。
- 基线检查已实测为 client 14 tests、dispatcher 9 tests、server 10 tests，全部 0 failure/error。当前 server wrapper 会把 manifest 硬编码成 `50×2×20`；必须先让它消费冻结 evaluator manifest/hash，否则 200 bench 属于假绿。
- 代码锚点：九章模型 `rgbdwam-vlm-guidance-rgbd@a8151b2`、server infra `rgbdwam-robotwin-5090-eval@39a47c2`；JD client `rgbdwam-robotwin-5090-client@66c25a4`；RoboTwin `fcb220c` 原 checkout 已脏，只读。
- 先读 `$robotwin-eval`、`$robotwin-5090-eval`、`$jdcloud-cluster`；核对身份、AGENTS.md、repo HEAD/status、checkpoint/config/hash、资产 owner、50-task 三方交集、HTrain 配额和活动任务。冻结 `evaluator_manifest.json` 与 `bench_cases.json`，再把目标/顺序/最大风险≤10 行写入 `PROGRESS.md`；任一关键事实不符就停相关路径并记 `BLOCKED.md`。

## 任务 1：跑粗参考与吞吐 bench

用 bowen-owned 资产生成 run-local config；先 `DRY_RUN=1`、严格 checkpoint/normalization load、1条 rollout 和并发 smoke。A 用 1 台 8×H200 跑模型+仿真；B 用另 1 台 8×H200 的 8 policy processes 加 1 台 8×5090。A/B 尽量在相近时段并行，谁先 ready 谁先跑，不设共同屏障、不等另一边。B 用不计时短 sweep 选稳定≤3s的 client 总数/卡数并记映射，冻结后跑正式 bench。每个 client 唯一 session/slot；可共享模型/GPU lock，不得互相 reset/消费 action。两路各得 200 条后，用 run-local `audit.py` 生成 `benchmark_report.json`；审计须拒绝缺口、重复、error、checkpoint/config/hash 错配和非 H200/5090 硬件。

## 任务 2：不达标就修

若 B 的聚合节拍>3.0s，先用已记录的 client sim/Seed/编码与字节、RPC RTT、server queue/decode/VLM/model forward、视频/写盘时间定位最大瓶颈，再在共享根因处做最小修复。建议先核对 manifest 内 clients 全运行、chunk/replan=24、无 slot 串扰和动态领取尾巴，再看 transport/SSH；没数据不引入新依赖/架构。每个修复留 1 个回归检查，同一 200-case 复测；提升<10% 或完整性变差就回滚。bench 达标后冻结最佳配置，另跑 canonical `50×2×20=2,000`，从 HTrain `Running` 计时到严格审计；超 3h 仍是未达标，在剩余轮次内继续修。

## 规矩

不许 skip/todo、放宽断言/门槛、mock 真实评测对象、删测试、改 case/seed/模型参数、关视频、`|| true`或把 infra 失败记成策略失败；测试数只许≥基线。审计器必须先用缺一条/重复一条/改 checkpoint 的副本各制造一次红灯，再还原为绿。同一验收连败 3 次换项；跑分变差就回滚。不新增权限、流程或依赖，确必须时记 `BLOCKED.md`。

## 完成条件

1. `audit.py` 输出 `BENCH_OK`：A/B 各 200 条有效记录、0 重复、0 error、各自合同/hash 正确；B 从 manifest 所列 clients 全 ready 到第 200 条的时长/200≤3.0s，A 只作粗参考。
2. 最佳 B 的 canonical 审计为 2,000/2,000、clean/randomized 各1,000、0 重复、0 error，HTrain `Running`→`EVAL_COMPLETE`≤10,800s；基线 14+9+10 与针对性检查全绿，原 checkout 无新改，job/client/端口/隧道全释放。
每条都在对话贴实际命令输出，含审计器红→绿；只说做完不算。`BLOCKED.md` 随交付，空也写「无」。或已用完两轮优化/上述资源预算：满轮即停，不得冒充达标，交最佳结果、瓶颈证据和还差什么。
