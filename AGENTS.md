# 本地 Codex 接续约定

本文件位于新公开仓库 [flow3d-local-handoff-20261103](https://github.com/guanzhaoyang86-lab/flow3d-local-handoff-20261103) 的根目录。以下路径均相对于仓库根目录，不再有 handoff_20261103/ 外层目录。

- 先读 README.md、local_support/README.txt、results/summary.json、results/server_queue.json。
- 用户明确要求：服务器已提交的任务保持执行；本地接续没有完成且未由服务器负责的工作。当前仅用训练种子 31。
- 所有模型源码位于 models/；本轮 baseline 和 Tensor-DiT 使用 experiments/ 内的对应冻结版本。不要把不同版本同名包混装。
- experiments/ 内 frozen_manifest.json 覆盖的文件保持原始字节，不全局替换服务器路径，不修改换行。新入口放 local_support/，新输出放 local_runs/。
- Windows 使用独立 x86-64 Python/CUDA 环境，不能复制 ARM 服务器环境。先实际运行 CUDA 前向/反向与单案例 smoke，再讨论正式训练。
- Tensor-DiT 的冻结数据准备程序依赖 fcntl 和 Slurm 门禁；WSL2 下新增 wrapper 调用库接口，不伪造 SLURM_JOB_ID，不修改冻结源。
- 先查服务器队列和结果。队列快照过期需要更新；任务从队列消失可能表示已经完成，不能据此重新训练。不得自行取消服务器任务或删除已有结果。用户后续明确变更分工时，以用户指令为准。
- results/ 中已完成/历史额外种子/旧数据集结果要区分。训练 loss、validation 校准和 test 指标不能混为一谈；不要使用 test 调参。
- checkpoint 和适配数据未随 Git 下载。用 results/asset_manifest.json、transfer_assets.py 或用户已有方式取回；保持数据 manifest 字节与目录结构。
- 仅对新建本地入口做相应检查，记录真实验证范围。服务器 CPU 测试不能证明本机 5090 训练已成功。

