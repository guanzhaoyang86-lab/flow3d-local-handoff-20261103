# Flow3D-DPS

这是独立于当前条件模型的另一条实验路线：**无条件流场先验 + 轨迹似然引导的扩散后验采样**。当前实现面向项目中的稳态三维速度场；没有更改已提交36组条件模型训练的代码或任务。

审查结论：实现了截图2.2节方法的**稳态三维特例**，不能宣称已经实现一般时变速度场`v(x,t)`。当前训练样本及先验都是一个三维快照；时变序列会明确拒绝，不能把多帧当成批次或默认帧索引。

## 与现有模型的区别

| | 现有 DiT3D / Tensor-DiT | Flow3D-DPS |
|---|---|---|
| 训练输入 | 流场和粒子轨迹条件 | 仅训练集流场 |
| 轨迹编码器 | motion-residual-v2 | 不使用 |
| 观测进入的位置 | 条件网络与CFG | 采样中的轨迹似然梯度 |
| 先验是否依赖N | 当前每个N独立训练 | 一个先验支持不同N |
| 初始深度 | 现有多视角流程 | 单视角推断潜在深度；满秩多视角由观测确定 |

`DPS`是推断路线，不是一种新的Transformer结构。先验默认用DiT3D骨干，也支持UNet3D；两者从头做无条件训练。旧条件模型的checkpoint不能冒充无条件先验。

## 对应截图中的公式

记物理速度场为 `u`，观测投影轨迹为 `y[v,n,t,2]`，相机为 `P[v,2,3]`。默认 `transition` 似然实现：

```text
x(t) = U(u, y(0), depth)
r[v,n,j] = y[v,n,j+1] - y[v,n,j] - P[v] u(x[n,j]) * dt[j]
L(u, depth) = sum_valid ||r||^2 / (2 sigma^2)
```

默认Euler推进，支持不等时间间隔和观测缺失mask。完整3D轨迹的公式有API和端到端命令行入口：`FullTrajectoryLikelihood`及`--observation-type full3d`，在已观测3D位置上求速度并计算Euler残差。可选 `--likelihood position` 是对整段回放位置的高斯误差，**它与截图中的位移残差是不同模型，输出会明确记录**。

DPS场更新的梯度为 `d L(u_hat(x_t), depth) / d x_t`，经过先验网络、反归一化、插值和粒子推进反向传播。没有把 `u_hat` detach以后只做干净流场梯度下降。实现参考 [DPS官方代码](https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py) 和 [Chung等人的论文](https://arxiv.org/abs/2209.14687)。没有声称复现截图提到的其他2026年算法。

默认使用完整时间步的DDPM先验转移；可选DDIM是加速变体，须记录为 `DDIM-DPS`。默认目标是截图的Gaussian NLL；`residual_norm` 可选项使用白化残差范数，与官方实现的归一化思路对应。可选更新RMS上限会记录每步缩放系数。DPS本身是后验近似，不能把这些样本宣称为已证明校准的精确后验。

## 初始深度与数据隔离

- 单视角：从初始二维观测确定一条三维射线。射线与已知计算域的交集给出有界深度区间。默认采用有界均匀深度先验，并做反射边界的Langevin近似更新。这是截图未指定的工程选择，并非图中给定的唯一算法。`map`和`fixed`模式可显式选择，不能将其当作深度后验采样。
- 多视角：同一批物理粒子在初始时刻的投影矩阵满秩时，用这些二维观测解出初始三维位置，不读取真值深度。初始观测作为给定约束，互相矛盾的相机坐标会明确拒绝，不能静默取平均。
- 先验训练只读取`flow_field`和几何，归一化只拟合train，不读取轨迹。
- 默认`projected`推断只读取选中的二维轨迹、相机、时刻和已知计算域，不读取`flow_field`、`trajectories_3d`、`initial_hidden_depth`或真实边界速度元数据。明确指定`full3d`时才将`trajectories_3d`作为已观测XYZ轨迹读取；这属于更强观测场景，不能与二维输入混为同口径。
- 默认`--boundary-projection none`。`--boundary-projection known`才读取元数据中的真实壁面/运动边界速度并硬覆盖；这是额外已知条件，须在对比时单独声明。
- `--evaluate`在全部生成完成后才读参考流场和用于评估的边界元数据。probe不进入DPS似然；完整3D模式只在评估时读取probe的初始XYZ，投影模式由评估视角三角化，不能辨识时明确输出空值和原因。
- 越界回放保留误差并报告有效率，不按候选流场的有效性删除观测。当前clamp是有限域的数值近似，可能出现未观测方向越界但投影残差很小的情况；低NLL不证明物理有效。输出记录有效/无效样本数，`--require-valid-replay`可明确拒绝最终越界样本。没有为了降低错误率偷偷添加截图以外的越界惩罚。

多视角扩展额外假设各相机残差噪声独立，NLL为各视角之和。XY/XZ/YZ无噪声投影会使每个分量出现两次，因此相同sigma时不能将其NLL直接与完整3D或单视角NLL比较；sigma需按观测协议在validation上确定。

N表示观测物理粒子数量；同一个粒子的多视角投影不会增加N。默认观测seed47、epoch0、每流场第0组观测的粒子选择，与既有评估一致。初始噪声与更新随机流也沿用已有配对种子规则。

## 训练与采样

以下Python命令须在分配好的GPU计算节点执行。登录节点只用于提交和检查。所有输出使用新目录。

```bash
python scripts/train_flow_prior.py \
  --manifest /path/to/manifest.json \
  --output-dir /path/to/new_dps/prior_s31 \
  --architecture dit3d --seed 31 --epochs 100 --device cuda --amp
```

断点续训使用相同目录并增加 `--resume /path/to/new_dps/prior_s31/latest.pt`。模型、训练目标、优化器、随机状态和EMA均恢复；不允许续训时悄悄改变关键设置。无条件先验训练不需要`--num-particles`。三个训练种子只需三个先验，不用随N重复训练。

AMP遇到可恢复的梯度缩放溢出时会跳过该优化步并降低缩放，且不更新EMA；成功优化步数与溢出跳过数会写入checkpoint并随续训恢复。若整个训练没有一次成功优化，不能报告训练完成。

```bash
python scripts/sample_flow_dps.py \
  --checkpoint /path/to/new_dps/prior_s31/epoch_0100.pt \
  --manifest /path/to/manifest.json \
  --split validation --case-index 13 --num-particles 2 \
  --observation-type projected --views 0 --boundary-projection none \
  --num-probes 16 --num-samples 2 --sigma 0.05 \
  --sampler ddpm --device cuda --evaluate \
  --output /path/to/new_dps/validation_case13_N2.npz
```

这里`0.05`只是命令示例，不代表已校准的噪声强度。`sigma`使用观测坐标的单位（本项目正交投影为cells）；transition对应位移残差，position对应位置残差。必须在validation上确定似然参数、引导步长和采样设置，冻结后再跑test。不能用test调参。

只使用一个投影视角时增加 `--views 0`。与现有50步设置对比时，显式使用 `--sampler ddim --sampling-steps 50`，并记录这一加速变体。先验训练中的`--prediction-type v`是扩散预测参数化，不是物理速度场符号的另一种含义。

截图公式(2)的完整三维观测模式：将上述命令改为`--observation-type full3d`，去掉`--views`，使用真实已观测的`trajectories_3d[S,N,T,3]`及可选`observation_mask_3d[S,N,T]`。无需相机矩阵、二维轨迹或隐藏深度文件；输出的空深度轴表示该场景没有未知初始深度。

不指定`--views`时使用数据中的全部视角，是对截图单个P的多视角扩展。与当前条件模型做同口径比较时，还须明确是否统一使用其额外已知边界速度，不能在不同观测场景间直接归因架构收益。

输出包括完整三维样本、均值、方差、初始深度样本、实际观测和元数据。相邻JSON保留每步NLL与梯度缩放、随机种子、权重/输入校验和及可选评估结果。单样本误差均值与“先平均流场再算误差”分别存储。

## Slurm入口

`scripts/train_prior.slurm`已配置本项目GH200分区，seed31/32/33数组，最多1组并行；`scripts/sample_dps.slurm`用于单案例采样。仅提供脚本，创建代码时没有提交任何新训练任务。每个脚本在`srun`之后还会核验实际产物，避免空运行被误判为成功。

```bash
export FLOW3D_DPS_ROOT=/u/zguan2/projects/flow3d_runs/flow3d-dps
export FLOW3D_DPS_MANIFEST=/work/hdd/biup/zguan2/datasets/2026-10-01_115342_lbm3d-full_22594936_7966ccef/manifest.json
export FLOW3D_DPS_OUTPUT=/work/hdd/biup/zguan2/results/flow3d-dps/NEW_BATCH
sbatch "$FLOW3D_DPS_ROOT/scripts/train_prior.slurm"
```

先验训练完成后，采样脚本还需要`FLOW3D_DPS_CHECKPOINT`、`FLOW3D_DPS_OUTPUT_FILE`和`FLOW3D_DPS_SIGMA`。可选设置`FLOW3D_DPS_N`、`FLOW3D_DPS_CASE_INDEX`、`FLOW3D_DPS_VIEWS`、`FLOW3D_DPS_SPLIT`、`FLOW3D_DPS_OBSERVATION_TYPE`、`FLOW3D_DPS_BOUNDARY_PROJECTION`及`FLOW3D_DPS_EVALUATE=1`。`FLOW3D_DPS_REQUIRE_VALID_REPLAY=1`启用最终轨迹有效性要求。

## 验证

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

测试包含Euler似然的解析值、噪声尺度、DPS梯度的有限差分、单视角潜在深度恢复、随机可重复性、零引导退化为无条件采样、无轨迹先验训练、断点续训一致性、三视角/单视角端到端推断，以及更改参考真值和probe轨迹不会改变输出的隔离检查。

CPU小数据测试通过只说明代码闭环和数值检查通过，不代表已经训练好真实32³先验、验证GPU性能或获得超过当前条件模型的科研结果。基线复用文件的来源和哈希见`BASELINE_PROVENANCE.json`。
