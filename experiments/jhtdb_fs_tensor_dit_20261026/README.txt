JHTDB / FS-SPH-GPU：Tensor-DiT 与 baseline 实验设置
2026-10-26，当前服务器 NCSA DeltaAI

本批次的自有模型为 Tensor-DiT 条件扩散。配置已保存，尚未提交任何作业。
settings.json 是总设置；datasets/ 是各数据集任务协议；models/ 是模型设置；
protocol/ 保存 FS 的实际窗口划分和独立 probe ID；matrix.csv 是计划矩阵。
source_lock.json 锁定审查过的实现；reports/ 保存只读核查结果。

1. 共同实验设置

输入：单个 XZ 视角，每例20个观测时刻。
观测物理粒子数：12、24、48、96。
观测/采样种子：47；训练种子：31、32、33。
每方法使用相同观测；XYZ、隐藏Y和参考速度场不进入重建输入。
每例另留32个固定probe，probe不进入模型条件或轨迹拟合。

训练：100 epochs，batch1，梯度累积1，workers0。
AdamW lr=2e-4、weight_decay=1e-4、梯度裁剪1、EMA0.999。
共同轨迹编码器 motion-residual-v2，condition_dim128，coarse4^3监督权重0.1。
扩散模型：cosine1000步、v预测、condition_dropout0.1；推断DDIM50步、eta1、CFG1，2个样本。
Tensor-DiT：hidden192、depth4、heads6、patch4、dropout0。
Tensor因子潜变量不套用voxel clip8。
轨迹、散度、边界辅助损失均先设0；不强加没有物理依据的方程或真实运动边界。
两数据集给Tensor/DiT的known-boundary mask均为空，final边界投影为空操作。
FS材料/支持/规定运动mask仅用于监督和评分，不把真实规定速度送入自有模型。

主表使用最终epoch100；支持EMA的方法使用EMA。
不把按验证训练损失选出的best.pt称为生成场误差最优；test不参与选checkpoint。
归一化、POD和Tucker都只拟合各自train；不复用旧LBM统计或rank8缓存。

2. 方法范围

既有baseline：mean、POD、确定性U-Net、FNO、ResCNN、DeepONet、U-Net条件扩散、RBF。
另保留普通DiT3D作为去掉Tucker降维的结构对照，自有方法为Tensor-DiT。
DiT3D属于次级结构对照，额外增加12次训练/数据集，两数据集合计24次。
PINN配置保留供追溯，但不进入主矩阵：现有实现使用稳态、无外力、常黏度NS残差。
单时刻受迫湍流DNS和热力耦合材料运动都不满足这组物理假设，不猜黏度后直接运行。

五种监督baseline加DiT3D/Tensor-DiT，每数据集7×4N×3seeds=84个正式训练设置。
两数据集合计168个训练设置。mean/POD的train统计各只拟合一次；
POD/RBF的逐案例系数/深度优化单独计费和统计，不冒充独立监督训练。
mean没有三个伪重复种子。当前比较完整方案，参数量和推理时长另行报告，未声称参数量已匹配。

3. JHTDB 设置

原始根目录：
/work/hdd/biup/zguan2/datasets/jhtdb_iso1024_native32_3125_t1_20261024

保留原空间划分：train2500、validation312、test313；原val映射为validation。
这些是同一t=1快照的空间留出，不是不同DNS模拟或真实时间预测。
从冻结场生成20时刻的合成轨迹；原始32^3数据不改写。
每例候选池160粒子，初始位置距裁剪边界4个原生网格。
统一候选时长为0.005216254525311773 DNS时间单位：
0.75×4×dx / train最大speed_max，dx=2pi/1024，train最大速度3.5289247035980225。
这个时长仅由train确定，仍须小规模平流与步长收敛核查，不等于已验证轨迹数据。
RK4，每观测区间4子步；越出局部裁剪后terminate并记录mask，不局部周期回绕或clamp充作有效运动。
训练观测可按共同epoch随机流重选；validation/test观测固定，probe始终独立。
不创建cavity壁面：全false solid mask、空已知边界mask、不加入lid元数据。

4. FS-SPH-GPU 设置

原始根目录：
/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_rods_full_v2

使用25990×3613完整轨迹，不使用旧160条子集。
保留单一物理组fs_sph_3429811。采用单工况时间外推协议，不能称跨工况泛化。
180个连续20帧窗口按以下分组（索引从0开始）：
train：0–125，共126窗
隔离：126–134，共9窗
validation：135–152，共18窗
隔离：153–161，共9窗
test：162–179，共18窗
末尾13帧保留在原库，不进入主20帧窗口矩阵。
隔离区不表示已证明时间统计独立；后续核对train时间相关性，所有时间模板不跨划分边界。

FS所有epoch和split固定使用发布的seed47嵌套N12/24/48/96清单，不进行epoch重新抽粒子。
probe为同一seed47排列中第96–127项，已写入protocol/fs_probe_ids_32_seed47.npy。
构建同一个参考网格时，从供体中排除N96观测ID与这32个probe的并集。
所有N和方法共享同一参考；不能每个N重建真值，也不能把25990个轨迹全部排除。

速度目标使用原始紧凑快照的native transport_velocity（已是m/s，不再除10）。
必须通过原生transport_velocity_valid及有限值检查；缺支持处标unknown，不能回退动量速度后仍标transport。
复用既有FS网格构建流程，但适配完整v2轨迹和初始棒材身份；粒子粘附后不丢失身份。
目标为时变[T,3,32,32,32]，保留真实不等时间间隔和物理单位。
监督和统计使用 material & interpolation_support & ~prescribed mask，空域/ghost/未知区另标。
未知区不是零速度或固体。独立评价包括从供体剔除的原生probe速度误差。
现有稳态模型不能通过删掉T轴替代时变任务；需要明确的时变输出与损失适配。

5. 单位与评估

派生输入统一映射到[0,31]^3，速度分量使用相同空间缩放，时间保留原生单位。
保存逆变换；报告速度误差时回到原始物理单位。
FS沿用发布库提供的全程区域框，明确声明为所有方法共同已知的域信息，
其来源是全程导出材料点范围，不伪装成只由训练拟合的统计。

主指标：物理单位下速度relativeL2、vectorRMSE；FS按有效材料支持域和真实时间加权。
辅指标：分量误差、独立probe轨迹误差/有效率、FS原生probe速度误差、覆盖率、耗时、显存、参数量。
生成模型分别报告逐样本误差均值与样本均值场误差，不混为同一个数字。
先在每个seed内平均，再跨训练seed报mean和sample std；FS仍只有一个物理工况。
两次生成只用于同预算比较，不宣称已经校准后验不确定性。

6. 代码与运行前提

Tensor/DiT使用10月20日已修复源码快照，具体文件与SHA256见source_lock.json。
当前flow3d-current工作区部分代码较旧，不能用其缺少v预测/motion-residual-v2的入口回退运行。

已识别的必需适配：
- JHTDB轨迹生成、空间manifest字段和空边界契约。
- FS v2原生速度网格、有效材料mask、时间划分loader和时变模型/基线。
- 单XZ推断与评价：旧Tensor评价器要求rank3相机三角化，不能直接给rank2的单XZ。
  新评价器须推断未知初始Y；真实XYZ只能在预测固定后的评价阶段使用。
- 分别进行train-only Tucker拟合，候选rank4/8/12/16、HOOI2轮。
  选择validation均值relativeL2<=0.05、最差<=0.20的最小秩；不通过则停止复核，不偷看test。
- 在真实Slurm GPU作业内完成数据/梯度/显存/耗时pilot。

设置校验：
/sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B validate_settings.py
要求可开始训练的校验：
/sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B validate_settings.py --require-ready

当前预期为settings_valid=true，但training_ready=false。
所有readiness gate都需要实际证据后再通过；配置文件存在不等于训练数据和模型适配已经完成。
校验器锁定的是当前未适配的准备快照；完成适配后需另行更新验证规则及源码锁，不能只把passed改成true。
这里没有sbatch提交脚本，也没有在导入或校验时自动提交任务。

7. 资源和pilot

DeltaAI / ghx4 / biup-dtai-gh：每任务1 GPU、8CPU、64G，正式初始并发上限2。
Python为站点PyTorch2.10.0环境；禁用user-site，BLAS/OMP线程各1。
pilot：U-Net、DiT3D、Tensor-DiT，N48、seed31、1epoch，最多32train/8validation例，不碰test。
pilot时限30分钟，单任务。正式初始时限上限12小时，必须以pilot测时复核。
每种启用的学习模型另需小规模前向/反向/有限损失检查；mean/POD/RBF核对统计或拟合路径。
三架构pilot用于额外测时，不替代其他模型的接口检查。适配完成后必须重新锁定和审查源码。
时限是资源申请上限，不是已经验证的实际耗时或GPU小时预算承诺。
本次只准备setting；已有排队作业保持原状态，未提交本批实验。
