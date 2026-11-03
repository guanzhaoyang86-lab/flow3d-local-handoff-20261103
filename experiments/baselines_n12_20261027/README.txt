当前执行策略（2026-10-30 芝加哥时间更新）：只使用训练种子31，每个神经模型100 epochs。
UNet、FNO、UNet diffusion的种子32/33共6个排队子任务已取消；种子31和原校准依赖保留。
CNN、DeepONet此前已完成的额外种子结果保留；当前实验仅采用种子31。
JHTDB命名冲突已在独立版本 baselines_n12_jhtdb_recovery_20261030 修复；预检3492894及9个后续种子31任务已提交。
当前策略见execution_policy.json，实时快照见progress.json。
下文及plan.json是历史三种子提交记录；冻结文件保持原样，以免破坏已排队任务的校验。

N=12 baseline 实际提交批次，2026-10-27，NCSA DeltaAI

范围：JHTDB 和 FS-SPH-GPU，单XZ，观测seed47。
只跑baseline，不提交Tensor-DiT、DiT3D或物理假设不适配的PINN。
用户已明确要求先补齐FS时变实现，再提交两套实验。

每个数据集的任务分配：
mean：1次train统计，seed31占位，epoch0，不伪造3次重复。
POD：1次train建基，epoch0；另1次验证集系数/深度拟合和sigma校准。
CNN、DeepONet、U-Net、FNO、U-Net diffusion：各3个训练任务，seed31/32/33，100epochs。
RBF：1次验证拟合/校准任务，无跨数据集预训练。
另有1个预处理、1个真实GPU数据/模型pilot。
合计每套21个逻辑allocation，两套42个；包含30次正式神经模型训练和4次统计拟合。
使用22次sbatch（3种子训练使用数组0-2%1）。

资源：每allocation 1张GPU，通常8CPU/64G；FS预处理16CPU/64G。
两套数据各有1条串行依赖链，总并发最多2；任一数据/模型pilot失败，不放行该套正式任务。
preparation -> pilot -> mean -> POD -> CNN -> DeepONet -> UNet -> FNO -> UNetDiffusion -> POD校准 -> RBF校准。
模型之间afterany允许一个方法失败后测试另一个方法，所有正式任务另有afterok(pilot)门槛。
POD校准还要求POD基成功；即使它提前取消，RBF也必须等待最后训练数组结束，保证并发上限。

申请时间上限（不是实测耗时）：
                    JHTDB        FS-SPH
预处理              4小时        12小时
GPU pilot           1小时        1小时
mean                1小时        1小时
POD建基             4小时        2小时
CNN每种子           8小时        4小时
DeepONet每种子      8小时        4小时
UNet每种子          12小时       4小时
FNO每种子           12小时       4小时
UNetDiff每种子      12小时       4小时
POD验证校准         8小时        8小时
RBF验证校准         8小时        8小时
所有任务按申请上限计算共274 GPU小时，提交前核查账户余额805 GPU小时。
实际分配耗时由作业日志记录；不会把这些上限冒充GPU基准测试结果。

本轮校准只使用validation最前8例，sigma候选0.02/0.05/0.1网格坐标单位，seed31。
各24次逐案例拟合后按平均验证速度relativeL2固定selected_config.json。
本阶段不访问test；完整测试在参数选择冻结之后另行展开，避免测试调参。

数据适配：
JHTDB发布保持原样，派生根目录：
/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32
保留2500/312/313空间划分，宏块分组，记录原始crop/source IDs。
原生场转换为网格坐标/网格速度，160粒子、20时刻、RK4、终止mask，固定32个独立probe。
合成轨迹来自冻结t=1场，不宣称真实时变DNS轨迹。
存储XZ/XY/YZ，但模型只输入XZ；其他视角仅用于预测完成后的probe评估。
全false solid mask，不制造局部周期边界或腔体壁面。

FS发布保持原样，派生根目录：
/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027
初始棒材身份保持，原生transport速度及valid标记，禁止回退momentum还标transport。
同一份参考网格剔除N96观测与固定32probe并集，所有方法/N共享。
固定N12、真实不等时间间隔，126train/18validation/18test窗口，两处9窗隔离。
保留同一物理group；这是单工况时间外推，不是跨工况泛化。

FS神经基线是明确的时变适配：共享轨迹编码器加物理时间条件，每帧调用空间解码器。
每个训练window/epoch按时间权重随机取1帧，构成无偏的masked监督目标；validation评价全部20帧。
mean为train全帧masked均值；POD为train空间基加每帧时变系数；RBF同样使用时变系数。
POD/RBF只由XZ轨迹拟合系数和未知初始Y，不读取真实XYZ、支持mask或参考速度。
仅预测固定后按物理单位和有效材料支持域评价，另比较独立原生probe速度。

FS扩散为部分监督的时变适配，按帧条件边缘分布采样，不声称联合轨迹后验。
未知区域clean潜变量用人工N(0,1)，物理loss只用有效支持；额外独立归一化、权重1的
解析nuisance score loss约束未知区域（v目标0；epsilon目标sqrt(1-alpha_bar)*x_t）。
这不是把空域当物理零速度，不向推断提供真值mask或GT支持投影。

安全提交与恢复：
launch/submit_batch.py prepare 生成plan.json、逐种子配置和frozen_manifest.json。
test-only 对22个资源请求调用sbatch --test-only；不创建作业。
submit 必须先有当前冻结哈希对应的CPU集成测试和Slurm预检报告。
每次sbatch前记录attempt，只有收到明确数字job ID后才继续；歧义停止、不盲目重复提交。
每个allocation重新验证冻结代码/配置，检查真实CUDA、数据完成标记和manifest哈希。
预处理有锁、哈希匹配恢复与原子发布；训练每epoch原子保存checkpoint。
重试会恢复未完成的学习任务，核验后跳过已完成任务；不自动requeue。
统计拟合若中断且未写出checkpoint，需检查/归档该次partial目录后重跑，不覆盖不明产物。

源代码：source/ 是本批次独立副本；原baseline、原FS工程和原始数据未改动。
CPU测试只能证明所覆盖的实现和数值检查；GPU/data pilot在队列中真正执行后才有GPU实测证据。
实际job ID和提交命令见submission/submission.json，日志见logs/，运行产物见plan.json的result_root。
