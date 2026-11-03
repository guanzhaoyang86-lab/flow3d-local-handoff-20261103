# Flow3D 本地接续包 · 2026-11-03

这是服务器实验的代码与结果快照，适用于把后续工作交给本地 Codex。服务器已提交的任务继续按原计划执行。本次导出没有提交、取消或重启任何计算任务。

## 从这里开始

```powershell
git -c core.autocrlf=false clone https://github.com/guanzhaoyang86-lab/flow3d-local-handoff-20261103.git
cd flow3d-local-handoff-20261103
python local_support/local_baselines.py list
```

然后阅读 [本地运行说明](local_support/README.txt) 和 [Codex 接续约定](AGENTS.md)。这份代码在服务器端做过 CPU 检查；尚未在你的 Windows / RTX 5090 笔记本上做 CUDA 实测。Tensor-DiT 的冻结准备代码使用 Linux 文件锁，需 WSL2/Linux 的独立本地入口。

## 包含的模型

| 目录 | 内容 |
| --- | --- |
| [models/baseline](models/baseline) | Mean、POD、RBF、CNN、DeepONet、UNet、FNO、UNet diffusion |
| [models/flow3d-current](models/flow3d-current) | 三种条件扩散架构：UNet3D、DiT3D、Tensor-DiT |
| [models/flow3d-dps](models/flow3d-dps) | Flow3D-DPS：无条件流场 prior 训练、轨迹约束后验采样 |
| [experiments](experiments) | 本轮 baseline/JHTDB 修复版/Tensor-DiT 的冻结源码、参数、适配代码和协议 |
| [data_generation](data_generation) | LBM 数据生成与连续作业相关代码、配置 |

本轮 FS-SPH/JHTDB baseline 应使用对应冻结实验版本，不能把三个目录里的同名 Python 包混装。后续训练只用种子 **31**；历史 32/33 的完成结果保留。源文件清单与校验值见 [source_inventory.json](source_inventory.json)。

## 已有结果与服务器分工

[结果摘要](results/summary.json)、[训练记录 CSV](results/current_training_objectives.csv)、[完整结果索引](results/results_inventory.json) 保存导出时的证据。训练 loss、校准分数、旧数据集的验证结果分别标注，不能视为本轮统一的最终测试集对比。

| 实验 | 导出时状态 |
| --- | --- |
| FS-SPH Mean/POD 拟合 | 完成 |
| FS-SPH CNN、DeepONet、UNet、FNO、UNet diffusion，seed31 | 100 epochs 完成 |
| JHTDB Mean/POD 拟合与 CNN seed31 | 完成；CNN 为 100 epochs |
| JHTDB 其余四个神经 baseline | 继续由服务器排队执行 |
| 两个数据集的 POD/RBF 校准 | 继续由服务器排队执行 |
| LBM 50 万轨迹 | 已完成 1,000/3,125 案例、160,000/500,000 轨迹；连续生成任务仍由服务器负责 |
| 本轮 Tensor-DiT | 原提交已取消，尚无本轮正式结果；可由本地 Codex 准备独立入口 |
| 本轮 DiT3D / Flow3D-DPS | 已提供代码；尚无本轮正式完成记录 |

导出时共 **9 个 PENDING 作业**，详见带时间戳的 [server_queue.json](results/server_queue.json)。这是快照；启动本地正式任务前必须更新状态并检查服务器结果，不能把“已不在队列”直接理解成“尚未运行”。旧 LBM 条件扩散结果单独归档；旧提交计划不等于所有实验完成。

## 权重和数据如何取回

Git 中上传代码、配置、训练记录与小型结果文件。**约 10.26 GiB 的 244 个 checkpoint 和原始/适配数据集未放入普通 Git。** 权重库存包含历次 epoch 存档，不代表 244 个独立模型。

使用 [资产清单](results/asset_manifest.json) 和 [下载脚本](results/transfer_assets.py) 从你已有 SSH 权限的服务器按需取回。脚本默认只打印计划，显式添加 `--execute` 才复制；需要 WSL2/Linux 的 rsync 和可用的 SSH 别名。

```bash
python results/transfer_assets.py --help
python results/transfer_assets.py --list
```

完整复制适配数据目录，保持 manifest 与 NPZ 的相对布局；不要修改 manifest 内容或冻结源文件的换行。下载目标应是本仓库根目录，权重放在 `results/` 对应运行目录，数据放在 `data/`；这些下载文件和本地新输出已加入忽略规则。

## 给本地 Codex 的任务

> 阅读仓库根目录 AGENTS.md 和 local_support/README.txt，核对已完成结果与服务器分工。
> 先配置本机 CUDA、按需取回适配数据和 checkpoint，完成独立的单案例 validation smoke。
> 保留服务器排队任务，后续只用 seed31。优先准备尚未提交的模型实验；所有本地输出写入 local_runs/。
> Tensor-DiT 需要在 WSL2 中增加独立准备入口，保持冻结源码、Tucker rank gate 和来源校验；
> DiT3D/DPS 不能直接把 FS 的时变 window manifest 当作静态流场 manifest。
> 没有实际跑过的步骤应明确记录，不能用训练 loss 代替最终测试集指标。

这次上传是时间点快照，服务器后续产生的新结果不会自动同步到 GitHub。

