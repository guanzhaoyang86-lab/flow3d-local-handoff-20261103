Tensor-DiT N=12：JHTDB / FS-SPH-GPU，2026-10-27

这是用户明确要求补充提交的自有条件扩散模型。此目录为独立源码快照，
不修改 baselines_n12_20261027 已冻结的代码、配置或 Slurm 作业。
plan.json / frozen_manifest.json 是最终提交规格；submission/submission.json
保存实际 sbatch 命令和作业编号。提交前不存在编号不代表任务已提交。

共同设置
两套数据均为 N=12、单 XZ 视角、20 个观测时刻、32 个独立 probe。
观测种子47，训练种子31/32/33，每种子100 epochs，batch1。
AdamW lr2e-4 / weight_decay1e-4 / clip1 / EMA0.999，AMP。
Tensor-DiT hidden192、depth4、heads6、condition128，motion-residual-v2编码器，
共同 coarse4^3 监督权重0.1。扩散 cosine1000、v预测、condition dropout0.1；
DDIM50、eta1、CFG1、2个样本，Tensor因子潜变量不使用voxel clip8。
主表采用最终epoch100+EMA；不以验证噪声损失挑选物理误差最优模型。

数据与比较范围
直接复用 baseline 已提交的数据准备：
JHTDB prepare3467857；FS-SPH prepare3467868。
不重复处理/改写原始数据，不使用旧LBM统计/codec/权重。
JHTDB为同一DNS快照空间留出、冻结场合成轨迹；不是实际时变湍流预测。
FS为单物理工况的时间留出，126train/18validation/18test，间隔各9窗；
原始transport velocity已为m/s，逆变换按XYZ分量还原物理单位。
FS参考供体排除统一N96观测与32probe身份并集，N12不另造参考场。
观测、划分、probe和评价单位都沿用baseline协议。

Tucker准备和防泄漏
每数据集单独拟合train-only归一化、medoid空间锚与潜变量统计。
候选rank4/8/12/16、HOOI2轮。只用validation选最小通过的秩：
每case/window物理速度relativeL2均值<=0.05，最差<=0.20。
FS每窗按原生真实时间梯形加权，各窗等权；不能通过统计更多帧偏置case权重。
所有候选都失败则退出，正式训练由afterok门阻止，不降低阈值/偷看test。
此阶段只读train/validation字段，不编码test；test保留到最终协议冻结后。
codec产物只供本批次，来源与文件SHA256绑定，写入使用原子替换并支持恢复。

FS时变适配
每个训练窗口以真实时间权重随机抽一帧，编码完整12条XZ轨迹并添加真实时间条件；
验证涵盖各窗口全部20帧。时间输入含绝对秒、窗口起点/时长、相对查询时间和单位缩放。
输出是条件逐帧边缘分布；当前不宣称建立了联合时间后验或动力学方程。
监督只在材料/插值支持/非规定运动区域成立；未知区域不是零速或固体。
Tucker是全局低秩表示，FS缺失区先以train-only均值填充，拟合的是明确的代理完整场。
潜变量v目标因此包含填充约定，不能声称缺失潜变量坐标直接对应真实速度。
另加FS专有低噪声(t<=20%)解码场masked MSE权重0.1，保证物理辅助监督只取有效支持；
JHTDB该附加权重0，其余共同coarse监督权重0.1保留。具体定义以configs和实现为准。
真值support/三维轨迹/隐藏Y/原生probe不进入推断条件；预测固定后才做物理评分。
材料mask不是已知边界速度条件，不从参考场提供规定运动速度。

提交资源与顺序
DeltaAI / ghx4 / biup-dtai-gh，每任务1GPU、8CPU、64G。
                  JHTDB                  FS-SPH
Tucker准备        1任务 x 8小时          1任务 x 8小时
真实GPU试跑       1任务 x 1小时          1任务 x 1小时
Tensor-DiT训练     3种子 x 12小时         3种子 x 4小时
共6次sbatch（两组3种子数组），展开10任务；新增上限66 GPU小时。
连已有baseline274上限合计340，最近核对账户剩余805 GPU小时。
这些是时限上限，不是实际测量耗时；试跑保存真实训练/采样用时。
GPU试跑用8train/2validation、1epoch，完整选定codec及DDIM采样路径；不碰test。

为保留此前baseline先跑及全批最多同时2GPU的设置，每个数据集的Tucker任务
等待相同数据集baseline最后训练数组及RBF校准结束，且要求原数据准备成功。
Tucker -> GPU试跑 -> Tensor-DiT三种子数组(0-2%1)，均由afterok控制。
baseline后期模型失败不阻止Tensor使用已正确生成的数据；Tucker/试跑失败会阻止正式训练。
重复提交有互斥锁与记录保护，不明返回停止，不自动重试；作业no-requeue。
每epoch保存优化器/EMA/scaler/RNG以便显式恢复，不能把试跑受限checkpoint当正式结果。
本轮提交训练和验证，不提交最终test评价。

实现与验证
source/src/flow_observation的TensorDiT、Tucker、trajectory encoder、diffusion
与10月20日修复快照5aa8b270一致；新的数据/时间适配集中在launch目录。
launch/submit_tensor.py prepare：生成计划、配置并冻结；test-only：Slurm资源预检；
submit：只有CPU集成测试与冻结证据均通过后才实际提交，status：读取提交记录。
真实CUDA检查在Slurm分配内部执行，CPU测试成功不代表GPU试跑已成功。
tests/source_math/PROVENANCE.json记录复用数学测试来源。
reports/pre_submit_validation.json、cpu_tests.xml、slurm_test_only.json保存实际证据。
