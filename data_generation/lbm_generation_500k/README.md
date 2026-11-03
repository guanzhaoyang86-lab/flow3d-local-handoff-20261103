# 50 万条单视角 Taichi-LBM3D 轨迹

此独立包装器复用既有源码的计划器和单案例生成命令，不修改 `flow3d-current` 或 LBM 算法。正式数据固定为 **3125 个独立流场案例 × 160 条粒子轨迹 × 1 个 xz 投影 = 500000 条单视角轨迹**。每条轨迹有 20 个观测时刻，合计 10000000 个二维观测点。

配置沿用既有生成范围：32³ 网格，lid speed 0.03–0.06，上游 `niu` 0.12–0.21，snapshot/warmup 600–1200 步，观测时长 120，RK4、每区间 4 子步，粒子边距 2，候选倍数 32，随机选择有效粒子，最小路径长度 0，计划种子 20260918。上游必须是干净的固定提交 `fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c`。

这是固定几何、固定 +z 顶盖方向、零速初始态产生的三分量腔流快照；粒子在冻结的速度快照中平流。变更粒子种子不创建新物理流场，warmup 是快照步数。`niu` 原样传入 `set_viscosity()`，此包不宣称其雷诺数解释或真实湍流已验证。增加轨迹总数不会自动增加物理类型。

## 当前运行：单个 16 小时连续作业

2026-11-02 起，已有 ARM 环境和数据采用 `arm_env/generate_continuous.slurm` 续跑：在同一张 GPU、同一个 16 小时 allocation 内顺序处理所有未完成块，再执行全量校验。已完成块保留原记录；每个新增案例仍由原生成器校验、落盘，支持中断后复用。`progress.json` 在作业开始、每块结束和退出时刷新；退出后的最终 Slurm 完成状态可再运行 `refresh_progress.py --write` 获取。

当前作业以 `arm_env/submission/submission.json` 为准，其中 `execution_mode=continuous`，`remaining_chunks` 与 `finalize` 指向同一个作业。历史数组提交保存在 `previous_record`，旧取消作业不计入当前失败状态。后续恢复须先确认当前连续作业已终止，再使用连续作业脚本并更新提交记录；下面原四阶段入口的 `--resume-failed` 不适用于这种共享作业 ID 的记录。

## 初次构建与原四阶段提交路径

主路径使用 DeltaAI 的 aarch64 主机和 GH200 GPU，账户 `biup-dtai-gh`。构建与环境验证使用 `ghx4-interactive`，正式生成及最终检查使用 `ghx4`。Taichi 使用实验性 [LLVM20 + ARM 分支](https://github.com/taichi-dev/taichi/pull/8735)，固定源码提交 `93c190f603b676ec1f8871e4f46cf5d5be89548e`，配套 LLVM 20.1.8、独立 `arm_env/venv` 和站点 PyTorch 2.10.0。它不是官方 Taichi 1.7.4 的现成 Linux ARM 发行包，**尚待实际构建、CUDA 内核和 32³ LBM 案例验证**；CPU 计划检查或调度资源检查不能替代这些验证。

在当前 DeltaAI 登录终端执行：

```bash
bash /u/zguan2/projects/flow3d_runs/lbm_generation_500k/arm_env/submit_arm.sh
```

入口先在 CPU 上检查完整生成计划，并检查两个分区的资源请求，再提交以下依赖链。后续阶段仅在前一阶段成功后运行：

| 阶段 | 作用 | 资源与时限 |
| --- | --- | --- |
| `build.slurm` | 从固定实验源码构建并安装 ARM Taichi，随后实测 CUDA 内核与一个原始 32³ LBM pilot，记录 `validated_env.json`；不生成正式第 0 块 | `ghx4-interactive`，1 GH200 GPU、8 CPU、48G、2 小时 |
| `validate_gpu.slurm` | 复核已验证环境，再生成正式第 0 块的 125 例 | `ghx4`，1 GH200 GPU、8 CPU、32G、2 小时 |
| `generate_arm.slurm` | 数组 `1–24%1`，每块 125 例、最多 1 个任务运行 | `ghx4`，每任务 1 GH200 GPU、8 CPU、32G、2 小时 |
| `finalize_arm.slurm` | 验证全部数据、构建 manifest 和校验和，发布 `COMPLETE.json` | `ghx4`，1 GH200 GPU、8 CPU、32G、30 分钟 |

站点不接受 0 GPU 的构建资源请求，因此编译阶段也申请一张 GPU，并在同一作业内接着完成实际 GPU 验证。`ghx4-interactive` 每用户最多提交和运行一个作业，本链只在该分区提交一个构建验证作业。2 小时是作业申请上限，不是实测耗时。独立 pilot 用于检验环境；正式集合仍是 25 块 × 125 例 × 160 条轨迹，不能把 pilot 另计入 50 万条。构建或验证失败时，依赖的正式生成阶段不会运行。

默认数据目录为 `/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023`。提交记录和 Slurm 日志分别位于 `arm_env/submission/`、`arm_env/logs/`；完整流水线记录为 `arm_env/submission/submission.json`。数据目录下的 `chunks/` 和 `attempts/` 保存块进度与逐例日志。

失败恢复使用同一入口的 `--resume-failed`。它要求四类旧作业均不在队列中，且 accounting 确认全部终止；旧提交记录保存在 `arm_env/submission/attempts/`，随后建立新依赖链。生成器会严格核对环境、计划与已有归档，再跳过已完成案例。存在运行中作业、部分提交记录或进行中标记时拒绝自动重提。恢复时须保留原代码、环境和验证清单；清单内容或哈希改变会触发核心恢复校验拒绝。

## 备用路径：Delta / A100

Delta x86_64 的既有 `~/envs/flow3d-lbm3d` 环境仍可作为备用。完整包放到 Delta 可读目录后，在 Delta 登录终端执行 `bash /path/to/lbm_generation_500k/submit_delta.sh`；账户为 `biup-delta-gpu`，分区为 `gpuA100x4`，申请单张 A100。该入口直接生成第 0 块，再运行数组 `1–24%1`，最后校验全量数据；不经过 ARM 构建阶段。

备用路径默认数据目录是 `/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_20261023`，环境前缀由 `LBM500K_ENV_PREFIX` 指定，默认为 `~/envs/flow3d-lbm3d`。两条路径使用各自的提交记录、运行环境和数据目录，不能将 ARM 数据集静默切换到 x86 环境恢复。

## 接口

Python 底层接口为兼容既有流程仍默认使用 `delta-x86` profile，需要 **Linux x86_64** 环境；既有环境为 Python 3.11、PyTorch 2.5.1、Taichi 1.7.4。主路径的 ARM 作业脚本会显式选择 `deltaai-arm-llvm20`。正式生成必须处于 Slurm 作业内，恰好可见一张 CUDA GPU。具体 GPU 的 Taichi CUDA 可用性必须由实际作业验证。粒子平流沿用既有 PyTorch CPU 实现，Taichi CFD 使用 CUDA。

DeltaAI 的 ARM/aarch64 需要另外构建并实测；此包仅开放显式 `deltaai-arm-llvm20` profile，限定 Taichi 源提交 `93c190f603b676ec1f8871e4f46cf5d5be89548e` 和实际链接的 LLVM 20.1.8，并要求匹配当前解释器和 GPU 的实测环境标记。默认 profile 继续拒绝 ARM；缺少实测标记也拒绝运行 ARM profile。

先用本地环境只检查全量计划，不导入 torch/Taichi、不启动 CUDA 或模拟，也不创建数据文件；`--check-plan` 是 `--dry-run` 的同义选项：

```bash
python -B run_generation.py \
  --source-root /absolute/frozen/flow3d-source \
  --upstream-repo /absolute/taichi_LBM3D \
  --output-dir /work/project/user/datasets/lbm500k \
  --dry-run
```

在外层已分配 GPU 的 Slurm 作业内，顺序运行全部 25 块并完成最终验证：

```bash
python -B run_generation.py \
  --source-root /absolute/frozen/flow3d-source \
  --upstream-repo /absolute/taichi_LBM3D \
  --output-dir /work/project/user/datasets/lbm500k
```

默认每块 125 例、20000 条单视角轨迹。进度逐例落盘并打印；每例模拟输出写入 `attempts/case_XXXXX_JOBID_TIMESTAMP.log`。原生成器以临时 NPZ 加原子替换保存，不会把半写文件当成已有样本。

同一命令加 `--resume` 可恢复未完成集合。恢复严格比较包装器和源文件 SHA256、源代码路径、上游路径/提交、运行解释器、Python/依赖版本、CUDA runtime、GPU 型号、全部生成参数、分块数量与完整计划。已有归档经元数据、数组形状、有限值、粒子数量、完整观测掩码及 xz 投影一致性验证后跳过；只检查本次涉及的块。

需要分多次作业或数组运行时，先在 GPU 作业内用同样参数加 `--init-only` 建立计划，再分别用 `--resume --chunk-index I` 运行 0–24 号块。不同块可以并行，同一块并发写入会被锁拒绝。所有块结束后，用同样参数加 `--resume --finalize` 执行最终验证。`--num-chunks N` 可以在首次初始化时调整块数，恢复时必须保持相同。

也可用 `bash run.sh ...` 启动；该脚本使用 `${FLOW3D_PYTHON:-python3}`，只负责调用 Python 入口。账户、分区、GPU 类型、时限、日志目录与提交动作由外层 Slurm 脚本负责。

## 显式 ARM 环境验证门槛

ARM 生成、初始化、续跑和最终检查都必须额外提供：

```bash
  --runtime-profile deltaai-arm-llvm20 \
  --environment-manifest /absolute/arm_env/validated_env.json
```

该 JSON 由外层环境验证流程在真实 CUDA 内核和原始生成脚本的 32³ pilot 均通过后写入，不能用构建成功替代这两项实测。必须有以下字段；值来自实际运行环境，示意中的版本/GPU/路径须替换：

```json
{
  "schema_version": 1,
  "profile": "deltaai-arm-llvm20",
  "system": "Linux",
  "machine": "aarch64",
  "python_executable": "/absolute/arm-venv/bin/python",
  "python_version": "3.12.9",
  "taichi_version": "1.8.0",
  "taichi_commit": "93c190f603b676ec1f8871e4f46cf5d5be89548e",
  "taichi_source_commit": "93c190f603b676ec1f8871e4f46cf5d5be89548e",
  "torch_version": "2.10.0+cu129",
  "cuda_runtime": "12.9",
  "gpu_name": "实际 torch.cuda.get_device_name(0) 返回值",
  "llvm_version": "20.1.8",
  "compiler_patch_sha256": "实际编译器补丁的64位小写SHA256",
  "cuda_kernel_validated": true,
  "lbm32_pilot_validated": true,
  "validated_at_utc": "2026-10-23T23:00:00+00:00"
}
```

`python_executable` 必须直接记录 `sys.executable`，保留 venv 路径，不能把符号链接解析到系统 Python。`taichi_version` 必须等于 `importlib.metadata.version('taichi')`；其基础版本也必须匹配 native core 版本，允许同基础版本的开发/本地版本后缀。`taichi_commit` 必须等于 `taichi._lib.core.get_commit_hash()` 的实际返回值；可以是固定提交的 7–40 位前缀。`llvm_version` 同时与 native core 的 `get_llvm_target_support()` 返回值核对。torch 版本直接使用 `torch.__version__`。

标记文件完整 JSON、原始字节的 SHA256 和路径全部写入生成配置；额外验证证据字段也会保留。任何修改都会使续跑配置不匹配。ARM 支持仅适用于经过上述验证的这一构建和环境，不表示通用 ARM Taichi 支持或已完成当前环境的实际验证。数据的冻结腔流科学范围保持不变。

## 完成条件和文件

每块成功只表示该块完成。全量完成必须存在 `COMPLETE.json` 且 `status=completed`，并且 Slurm 作业退出码为零。最后会重新检查恰好 3125 个计划内 NPZ、3125 个物理流场组、500000 条三维物理轨迹和 500000 条 xz 轨迹，调用原有 manifest 工具检查数据契约并按物理流场组划分 train/validation/test，随后生成实际文件 SHA256 清单。默认划分为 2499/313/313 例。

主要文件为 `generation.json`、完整 `plan.json`、逐块 `chunks/chunk_XXX.json`、逐例日志、`case_00000.npz` 至 `case_03124.npz`、`collection.json`、`manifest.json`、`SHA256SUMS`、`COMPLETE.json`。已完成集合不可继续生成；可在数据目录运行 `sha256sum -c SHA256SUMS` 验证校验和。

生成通过证明数据数量和接口契约通过，不表示物理收敛、分辨率充分或重建科研质量通过。每例原始元数据保留收敛检查点变化量、各速度分量能量和求解耗时供后续分析。

## CPU 小型检查

以下测试只构造 3 个 8³ 合成 NPZ、每例 2 条轨迹，调用真实 manifest 工具验证分组和数据接口，不启动 LBM、Taichi 或 GPU。生产常量保持 3125 × 160。

```bash
FLOW3D_TEST_SOURCE_ROOT=/absolute/frozen/flow3d-source \
  python -B -m unittest -v test_generation.py
```

覆盖生产精确计数/唯一计划/块边界、严格恢复拒绝不一致、初始化中断恢复、轨迹投影错误、意外文件拒绝，以及真实 manifest 的最终输出流程。ARM 门槛测试使用 CPU mock 检查默认拒绝 ARM、保留 venv、全量环境匹配、GPU/pilot 标记要求、原生提交/LLVM 限制和包版本混装拒绝；mock 测试不构成实际 ARM/CUDA 验证。
