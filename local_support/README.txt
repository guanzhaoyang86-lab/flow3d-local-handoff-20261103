Windows 5090 / 64 GB 本地接续说明（2026-11-03 快照）

目录与边界
  本文件位于仓库根目录下的 local_support/。以下命令均在 flow3d-local-handoff-20261103/ 仓库根目录运行。
  experiments/ 保留三个实验的冻结源、配置与协议；models/ 是其他模型代码快照。
  results/ 是上传时已有结果；大 checkpoint / 数据是否已下载，以 results/asset_manifest.json 清单为准。
  local_runs/ 是本地新输出，不能覆盖 results/ 或修改 experiments/ 的冻结文件。
  results/server_queue.json 是时间点快照。最新分工见 results/work_assignment.json：服务器跑DiT3D/Tensor-DiT/DPS并保留LBM生成；8个baseline任务已取消转本地。

先保护原始字节
  克隆仓库时关闭 Git 自动换行转换：git -c core.autocrlf=false clone https://github.com/guanzhaoyang86-lab/flow3d-local-handoff-20261103.git
  然后进入仓库根目录：cd flow3d-local-handoff-20261103
  使用仓库 .gitattributes 保留冻结文件字节；不要用编辑器批量改成 CRLF。
  FS checkpoint 会检查 launch/fs_baseline.py SHA256；冻结包也检查每个源文件。
  原始服务器路径只作为历史 provenance 保留，不应全局搜索替换。

环境（原生 Windows 可运行 baseline；Tensor-DiT 建议 WSL2）
  使用新的 x86-64 Python 3.12 环境；不能复制服务器 ARM 的 conda/venv。
  PowerShell 示例：
    py -3.12 -m venv .venv
    $py = ".venv\Scripts\python.exe"
    & $py -m pip install --upgrade pip
    & $py -m pip install numpy matplotlib pytest
  按 https://pytorch.org/get-started/locally/ 为 Windows 安装支持 RTX 5090 的 CUDA PyTorch。
  不要用 CPU-only wheel；不要把项目的 torch>=2.4 下限理解为 5090 支持保证。
  使用下面的 preflight 实际执行 CUDA 前向/反向；只看到 torch.cuda.is_available() 为真还不够。
  各入口自己添加对应 source/src，不需要把三个不同版本的 flow_observation 同时 editable-install。
  一个 Python 进程只跑一个代码版本；不要设置指向服务器或另一模型副本的 PYTHONPATH。

准备数据与 checkpoint
  JHTDB 需要完整 jhtdb_frozen32/：manifest.json、COMPLETE.json、fields/*.npz。
  FS-SPH 需要完整 fs_sph_3429811_flow32_windows_20261027/：manifest.json、COMPLETE.json、windows/*.npz。
  两个正式 manifest 使用相对路径，保持目录结构和文件内容就能迁移，不要重写 manifest。
  baseline checkpoint 的 manifest_sha256 必须匹配。FS 还核对源文件 SHA256。
  缺少数据时，只有代码不能开始训练。该工具会明确列出第一个缺失文件。
  当前清单包含 244 份 checkpoint，总计约 10.26 GiB，权重本体不放普通 Git。
  传输脚本需要 rsync + ssh；Windows 建议在 WSL 中运行以下下载命令（使用 WSL 的 Python，不复用 Windows venv）：
    python results/transfer_assets.py --list
    python results/transfer_assets.py --asset checkpoint-fs_sph-cnn_s31 --host deltaai --destination-root .
  默认只打印；加 --execute 才下载。可先执行 --help 查看参数，传输不删除源文件。
  数据 asset ID 为 data-fs_sph、data-jhtdb、data-lbm，默认恢复到 data/{fs_sph,jhtdb,lbm}。
  WSL 下载到 Windows 可访问的仓库后，可再用 Windows Python 运行 baseline；路径用各系统自己的表示。
  数据也按传输清单获取；报告存在不代表完整权重/数据已包含在 clone 中。
  checkpoint 默认查找：
    results/current_baselines/<dataset>/<mean或pod>/latest.pt
    results/current_baselines/<dataset>/<模型>_s31/latest.pt
  也可通过 --checkpoint 指向下载的独立 checkpoint。Git LFS pointer 不是实际 checkpoint。

先预检，再运行一个小案例
    & $py local_support/local_baselines.py list
    & $py local_support/local_baselines.py preflight --dataset jhtdb --data-root D:\flow_data\jhtdb_frozen32
  preflight 校验冻结代码、三个 split 的所有文件路径、manifest/COMPLETE，以及实际 CUDA 运算。
  它不一次性校验全部 NPZ 数组；原模型读取各案例时继续执行原有数据检查。
  下面仅打印检查结果和拟执行命令，不训练、不提交作业：
    & $py local_support/local_baselines.py smoke --dataset jhtdb --model rbf --data-root D:\flow_data\jhtdb_frozen32
  加 --execute 才执行，RBF 不需要 checkpoint：
    & $py local_support/local_baselines.py smoke --dataset jhtdb --model rbf --data-root D:\flow_data\jhtdb_frozen32 --execute
  复用已有 CNN checkpoint 的单案例验证：
    & $py local_support/local_baselines.py smoke --dataset fs_sph --model cnn --data-root D:\flow_data\fs_sph_3429811_flow32_windows_20261027 --checkpoint D:\flow_checkpoints\fs_cnn_latest.pt --execute
  smoke 固定 validation 单案例；默认 POD/RBF 10 优化步、diffusion 10 采样步、1 个 sample。
  它不改 manifest，输出明确标为 isolated_validation_smoke_not_scientific_result，不能混入正式指标。
  即使服务器在跑该模型的训练或全量校准，这种本地小预检也是独立任务。

正式 baseline 训练
    & $py local_support/local_baselines.py train --dataset jhtdb --model deeponet --data-root D:\flow_data\jhtdb_frozen32
  默认只打印命令；--execute 才运行。正式训练固定单种子 31、N=12、100 epochs。
  服务器已分配 PENDING/RUNNING 等状态的同一训练任务会被拒绝；已有 completed 结果也不重训。
  快照缺失或不可解析时不启动；已明确分配本地的任务允许旧快照，仍阻止已完成或服务器活跃的重复任务。
  本地预检不需要更新快照。不要取消服务器任务，也不要为了绕过检查手工伪造队列状态。
  Windows OpenSSH 配好能只读访问服务器的别名（例如 deltaai）后，可更新队列：
    & $py local_support/local_baselines.py refresh-queue --host deltaai
  该命令只执行固定 squeue 查询；SSH/解析失败保留旧快照，不取消或提交任何任务。
  BatchMode 不会在这里交互输入密码；需先配置已有 SSH agent/key 或使用可登录的连接方式。
  刷新会保留以前分配过的训练任务；任务从 squeue 消失后仍需核查 accounting/完成结果，不能据此重训。
  如果有真正未完成且服务器不再负责的任务，可用 --resume <checkpoint> 在新的 local_runs/ 继续。
  --epochs 表示总轮数；当前工具保持总计 100 轮。模型自己继续核对 config/manifest 和 checkpoint。
  正式校准入口已支持原生Windows，保留3 sigma × 前8个validation案例 × 1000步，不自动运行test：
    & $py local_support/local_calibration.py --dataset jhtdb --model rbf --data-root D:\flow_data\jhtdb_frozen32 --run-name sigma_s31 --execute
  POD用 --model pod 并先下载已有checkpoint；FS用 --dataset fs_sph 及相应数据目录。
  中断后加 --resume 并使用同一 --run-name。先验证请求及产物哈希，再复用预测，拒绝覆盖不匹配结果。

已有任务如何选择
  Mean/POD 已有结果通常直接使用。JHTDB POD 实现在 CPU 做 float64 snapshot eigensystem；64 GB RAM 足够容纳当前矩阵。
  FS POD 的训练矩阵约 1 GB，使用随机低秩分解。显存容量不等于已验证峰值。
  CNN、DeepONet、UNet、FNO、UNet diffusion 的具体服务器状态以 queue/results 快照为准。
  POD/RBF 每案例逆问题优化参数少，但 FS 每步涉及全部原生时间帧和 RK4，所以可能耗时。

其他模型代码入口（本轮服务器负责，本地不重复启动）
  JHTDB/FS-SPH 正式入口以 server_operations/server_own_models_20261103/ 为准；LBM 使用 server_operations/lbm_split_fix_20261104/ 的修订冻结入口（2499/313/313），旧 LBM 入口已被替代。
  以下 models/ 命令用于查看原始开发接口；正式观测选择、时间适配和模型标签需使用本轮冻结实现。
  1. DiT3D 条件扩散：
       & $py models/flow3d-current/scripts/train_sparse_track_diffusion.py --help
     本轮 DiT3D 使用 server_operations/ 中的 full-volume DiT 协议适配，观测种子47、训练种子31。
     必须显式提供 --manifest 和新的 --output-dir；先跑 1 epoch 检查显存和损失，再按该实验的 --resume 到 100。
     使用 N=12 不等于自动复现所有冻结 baseline 观察协议，正式比较前需核对 views/observation seeds/配置。
  2. DPS：先训练不以轨迹为条件的 flow prior，再进行 trajectory-guided posterior sampling：
       & $py models/flow3d-dps/scripts/train_flow_prior.py --help
       & $py models/flow3d-dps/scripts/sample_flow_dps.py --help
     prior 支持 --architecture dit3d 或 unet3d，建议起点 --seed 31 --batch-size 1 --amp。
     采样需要 prior checkpoint、manifest、明确 --sigma；N=12、observation seed47、XZ/view0需显式设定。
     sigma 应在 validation 选择；不要查看 test 来调参。
     以上 DiT3D/DPS 入口读取冻结 3D snapshot manifest，不能直接用 FS transient window manifest 替代。
  3. 冻结 N12 Tensor-DiT：
       experiments/tensor_dit_n12_20261027/launch/tensor_data.py
       experiments/tensor_dit_n12_20261027/launch/tensor_model.py
     该实验支持 JHTDB 与 FS 的专用适配。先 train-only Tucker/rank gate，再用 selected.pt 训练。
     真实限制：tensor_data.py import fcntl；其 prepare CLI 强制 Slurm。原生 Windows 不兼容。
     本地 Codex 应在 WSL2/Linux 新增独立 wrapper 调用原 prepare_artifacts()，保留文件锁、source/manifest 校验与 rank gate；不要伪造 SLURM_JOB_ID，不改冻结源。
     tensor_model.py 本身可接受 train --dataset ... --manifest ... --artifact ... --config ... --output-dir ... --seed 31 --epochs 100 --device cuda。
     configs/jhtdb_s31.json 与 configs/fs_sph_s31.json 为对应配置。历史 plan 有31/32/33，仅执行31。
     Tensor preparation 将源文件绝对路径记入 provenance；本地生成后不要挪动路径再尝试 resume。
     Tensor 本地 wrapper、真实 5090 CUDA 预检和首个 epoch 尚未在这次服务器端交接中完成。
  这些入口的 --help 不代表本地正式任务已启动。先检查服务器快照，保持独立输出和清晰 provenance。

给本地 Codex 的开场任务
  “阅读根目录 AGENTS.md、local_support/AGENTS.md 和 local_support/README.txt。先核验文件/下载清单及 CUDA，读取 server_queue 快照；
   完成一次独立 RBF/CNN validation smoke，然后本地接手 JHTDB 的 DeepONet/UNet/FNO/UNet diffusion，
   以及两数据集POD/RBF校准。只用seed31，已完成训练不重跑；三个自有模型留服务器。保留冻结源与已有结果。”
