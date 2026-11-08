# Flow3D 本地接续包 · 2026-11-03

这是服务器实验的代码与结果快照，适用于把后续工作交给本地 Codex。服务器负责 DiT3D、Tensor-DiT、Flow3D-DPS，覆盖 JHTDB、FS-SPH 和 LBM。LBM 50 万轨迹已完成生成与校验，修订后的适配和模型作业已提交排队。原 8 个 baseline 作业已取消转本地，已有结果保留。最新分工见 [work_assignment.json](results/work_assignment.json)。

**2026-11-08 本地结果更新。** JHTDB 的 DeepONet、UNet、FNO 和 UNet diffusion 已各完成 100 轮训练和 313 个独立 test 案例。[查看训练记录与 test 指标](results/local_jhtdb_s31_20261108/README.md)。

## 从这里开始

```powershell
git -c core.autocrlf=false clone https://github.com/guanzhaoyang86-lab/flow3d-local-handoff-20261103.git
cd flow3d-local-handoff-20261103
git config core.autocrlf false
python local_support/local_baselines.py list
```

然后阅读 [本地运行说明](local_support/README.txt) 和 [Codex 接续约定](AGENTS.md)。上述四项 JHTDB 训练与 test 已在 Windows / RTX 5090 Laptop GPU 上实际完成。Tensor-DiT 的冻结准备代码使用 Linux 文件锁，需 WSL2/Linux 的独立本地入口。

## 包含的模型

| 目录 | 内容 |
| --- | --- |
| [models/baseline](models/baseline) | Mean、POD、RBF、CNN、DeepONet、UNet、FNO、UNet diffusion |
| [models/flow3d-current](models/flow3d-current) | 三种条件扩散架构：UNet3D、DiT3D、Tensor-DiT |
| [models/flow3d-dps](models/flow3d-dps) | Flow3D-DPS：无条件流场 prior 训练、轨迹约束后验采样 |
| [experiments](experiments) | 本轮 baseline/JHTDB 修复版/Tensor-DiT 的冻结源码、参数、适配代码和协议 |
| [data_generation](data_generation) | LBM 数据生成与连续作业相关代码、配置 |
| [server_operations](server_operations) | 本轮服务器 DiT3D、Tensor-DiT、DPS 的冻结适配源码、配置、检查与提交记录 |

本轮 FS-SPH/JHTDB baseline 应使用对应冻结实验版本，不能把三个目录里的同名 Python 包混装。后续训练只用种子 **31**；历史 32/33 的完成结果保留。源文件清单与校验值见 [source_inventory.json](source_inventory.json)。

## 已有结果与服务器分工

[结果摘要](results/summary.json)、[训练记录 CSV](results/current_training_objectives.csv)、[完整结果索引](results/results_inventory.json) 保存 2026-11-03 交接快照的证据。训练 loss、校准分数、旧数据集的验证结果分别标注，不能视为本轮统一的最终测试集对比。

| 实验 | 导出时状态 |
| --- | --- |
| FS-SPH Mean/POD 拟合 | 完成 |
| FS-SPH CNN、DeepONet、UNet、FNO、UNet diffusion，seed31 | 100 epochs 完成 |
| JHTDB Mean/POD 拟合与 CNN seed31 | 完成；CNN 为 100 epochs |
| JHTDB 其余四个神经 baseline | 已取消服务器作业，交本地继续 |
| 两个数据集的 POD/RBF 校准 | 已取消服务器作业，交本地继续 |
| LBM 50 万轨迹 | 3,125 个案例、500,000 条轨迹已完成并校验；训练/验证/测试为 2499/313/313 |
| 本轮 Tensor-DiT | 服务器负责；本地不重复运行 |
| 本轮 DiT3D / Flow3D-DPS | 服务器负责；本地不重复运行 |

实际作业见带时间戳的 [server_queue.json](results/server_queue.json)。已明确分配本地的任务允许旧快照，但仍阻止已完成或服务器活跃的重复任务。experiments/ 中的旧队列和三种子计划是历史证据。

本轮服务器顺序和评估范围见 [execution_policy.json](server_operations/server_own_models_20261103/execution_policy.json)。DPS 分为训练、validation 校准和 held-out test 三个作业；DiT3D、Tensor-DiT 本轮安排训练与 validation，held-out 预测评估留待后续。因此作业数量包含数据准备和评估阶段，不等于模型数量。

## 权重和数据如何取回

Git 中上传代码、配置、训练记录与小型结果文件。**约 10.26 GiB 的 244 个 checkpoint 和原始/适配数据集未放入普通 Git。** 权重库存包含历次 epoch 存档，不代表 244 个独立模型。

使用 [资产清单](results/asset_manifest.json) 和 [下载脚本](results/transfer_assets.py) 从你已有 SSH 权限的服务器按需取回。脚本默认只打印计划，显式添加 `--execute` 才复制；需要 WSL2/Linux 的 rsync 和可用的 SSH 别名。

```bash
python results/transfer_assets.py --help
python results/transfer_assets.py --list
```

完整复制适配数据目录，保持 manifest 与 NPZ 的相对布局；不要修改 manifest 内容或冻结源文件的换行。下载目标应是本仓库根目录，权重放在 `results/` 对应运行目录，数据放在 `data/`；这些下载文件和本地新输出已加入忽略规则。

## 给本地 Codex 的任务

以下是初始交接计划，四项 JHTDB 训练和 test 的最新状态见上方的本地结果更新。

> 阅读仓库根目录 AGENTS.md 和 local_support/README.txt，核对已完成结果与服务器分工。
> 先配置本机 CUDA、按需取回适配数据和 checkpoint，完成独立的单案例 validation smoke。
> 本地接手 JHTDB 的 DeepONet、UNet、FNO、UNet diffusion，以及两个数据集的 POD/RBF 校准，只用 seed31。已完成训练不重跑。
> 正式校准入口 local_support/local_calibration.py 保留3个sigma、前8个validation案例、1000步；显式 --execute 才运行，--resume 需相同 --run-name。
> DiT3D、Tensor-DiT、DPS 留服务器，不重复启动。所有本地新输出写入 local_runs/。
> 没有实际跑过的步骤应明确记录，不能用训练 loss 代替最终测试集指标。

这次上传是时间点快照，服务器后续产生的新结果不会自动同步到 GitHub。


LBM 2026-11-04 修复：保留原始数据及划分，将适配器、DiT3D、Tensor-DiT、DPS 的数量检查与数据绑定改为实际发布版本。新的冻结入口位于 [lbm_split_fix_20261104](server_operations/lbm_split_fix_20261104/)，旧 LBM 入口仅保留作历史记录。作业替换证据见 [lbm_split_fix.json](results/lbm_split_fix.json)，实际进度以时间戳队列为准；CPU 预检通过不代表 GPU 训练完成。
