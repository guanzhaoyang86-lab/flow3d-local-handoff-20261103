# Flow3D baselines

九类基线：训练集均值场、POD＋轨迹拟合、确定性3D U-Net、空间3D FNO、ResCNN、适配DeepONet、UNet3D条件扩散、投影轨迹PINN、RBF＋轨迹拟合。

统一任务是**稳态三维速度场重建**，输入为单视角或多视角的二维粒子轨迹、观测mask、相机、时间和计算域。不支持一般时变速度序列。模型在本任务上均有必要的接口适配；POD/PINN/RBF不能描述为未经改动的原论文复现。

当前几何范围限于没有内部固体障碍物的矩形cavity；允许外表面壁面。训练、统计拟合和推断入口统一检查三维布尔solid mask，内部障碍物会明确报错。轨迹推进和PINN内部物理配点尚未实现一般障碍物几何，不能把这类输入按普通流体域处理。

## 文件与参数

- `src/flow_baselines/`：九类方法的模型、拟合、训练、推断、核验和统计。
- `src/flow_observation/`：隔离复制的已修复公共数值代码，来源和SHA见`PROVENANCE.json`。
- `configs/defaults.json`：由`config.DEFAULTS`生成的完整默认配置。
- `configs/{model}.json`：各模型的局部配置，覆盖默认值。
- `reports/parameters.csv`、`parameters.json`、`parameters.txt`：实际统计的参数量及默认设置。
- `scripts/run_baseline.py`：统一单模型入口。
- `scripts/evaluate_baseline.py`：批量案例评估入口。
- `scripts/train.slurm`、`predict.slurm`：计算节点启动脚本，不会自动提交作业。

参数表是**已实现的默认配置，不是调参后的最优参数**。网络参数量包含共同轨迹编码器及其训练辅助头；POD阶数还要按训练集数值秩截断。RBF、PINN参数属于每案例优化，与跨数据集训练参数分开解释。

## 方法和默认值

| 方法 | 默认结构 | 训练／拟合 |
|---|---|---|
| mean | 训练集逐网格平均 | 只读train，0个梯度参数 |
| pod | 请求32个中心化POD模态 | train建基；每案例拟合模态系数＋活跃初始深度 |
| unet | base32，通道倍率1/2/4 | 直接回归归一化速度场 |
| fno | width32，4层，三轴modes8，每侧padding4 | 真正XYZ三空间维FFT |
| cnn | width32，4个双卷积FiLM残差块 | 直接回归 |
| deeponet | branch/trunk各3个128宽隐藏层，latent128 | 共同集合编码器接branch；trunk接XYZ |
| unet_diffusion | base16，通道倍率1/2/4，时间嵌入64 | cosine1000步，v预测，DDIM50步 |
| pinn | 5×128 tanh，1024个物理配点 | 每案例2000步，lr0.001；显式已知黏度 |
| rbf | 4³个Gaussian中心＋常数项，length0.35，ridge0.0001 | 每案例1000步，lr0.01 |

监督模型统一：100轮、batch1、AdamW lr0.0002、weight decay0.0001、梯度裁剪1、EMA0.999；轨迹编码器motion-residual-v2、condition_dim128、4³辅助流场监督权重0.1。训练时按seed和epoch重选粒子；验证子集固定，原理同冻结条件模型。监督模型使用相同的编码器初始化和独立解码器随机流。

扩散模型：condition dropout0.1、v预测、clip_x0=8、50步DDIM、eta1、CFG1、默认生成2个样本。与历史epsilon预测的旧UNet不是同一训练方案；这里用于更可控的骨干对比。

POD/RBF拟合默认1000步、Adam lr0.01；POD方差白化系数正则权重0.0001；深度Adam lr0.01。轨迹目标为Euler位移Gaussian NLL除以有效坐标残差数，sigma0.05仅为初始设置，单位为观测位置单位。PINN额外添加散度及稳态无外力Navier–Stokes残差，权重均为1。

默认seed31；实验重复建议31/32/33。观测及扩散初始噪声种子默认47，观测粒子N可用2/4/6/12/24/48/96，默认N2。N表示物理粒子数，不把多个相机重复计数。POD/RBF/PINN的种子是拟合初始化种子；mean没有独立训练种子。

## 信息隔离与公平比较

1. 均值场、POD基和归一化只拟合train。测试案例或物理flow_group不能与训练重合。
2. 推断只读取被选中的二维轨迹。`flow_field`、`trajectories_3d`、真实初始隐藏深度不进入预测或拟合。
3. 五种监督模型使用共同观测编码器，但权重各自训练；DeepONet是对可变粒子集合的适配版本。
4. 默认`boundary_projection=none`，不读真实运动边界速度。显式`known`才启用并记录额外边界信息。默认边界none与历史DiT的known不同，需对齐后才能仅按骨干比较。
5. `configs/matched_repaired.json`启用已知速度边界，用于与采用同一额外条件的既有模型比较。PINN拒绝硬覆盖边界，因为这会使连续物理残差与最终网格场不一致；若要比较这一物理设置，需另实现一致的连续边界模型。
6. PINN黏度默认null，必须显式提供正确物理单位下的已知运动黏度。不得随意指定数值后声称物理模型已与数据匹配；不从参考速度场或未声明元数据推断黏度。
7. 单视角未知深度在已知域内优化；满秩多视角由初始二维观测确定位置。矛盾初始视角明确拒绝。POD/PINN/RBF输出点估计，不声称后验样本。
8. clamp回放是有限域近似，低轨迹误差不保证隐藏方向未越界；每个结果都报告有效性，`--require-valid-replay`可拒绝无效最终结果。

## 运行

使用已安装PyTorch的环境，或`pip install -e .`。正式训练、真实数据POD分解和模型推断在分配的计算节点执行。登录节点只做只读数据预检查、参数计数和微型合成CPU单元测试。

```bash
python scripts/run_baseline.py train \
  --model unet --manifest /path/to/manifest.json \
  --num-particles 2 --seed 31 --output-dir /path/to/new/unet_N2_s31 \
  --device cuda

python scripts/run_baseline.py predict \
  --checkpoint /path/to/new/unet_N2_s31/latest.pt \
  --manifest /path/to/manifest.json --case-index 13 --split validation \
  --output /path/to/new/unet_case13.npz --evaluate --device cuda
```

训练模型可选`mean/pod/unet/fno/cnn/deeponet/unet_diffusion`。均值和POD是一次train拟合，epochs不适用。其他模型支持`--resume /path/to/latest.pt --epochs 200`，恢复优化器、EMA、随机状态和成功更新计数；除最终轮数外禁止修改训练配置。

PINN/RBF不先跨数据集训练：

```bash
python scripts/run_baseline.py predict \
  --model rbf --manifest /path/to/manifest.json --case-index 13 \
  --num-particles 2 --seed 31 --output /path/to/new/rbf_case13.npz \
  --evaluate --device cuda

# KNOWN_NU必须是该案例已知的、单位一致的运动黏度。
python scripts/run_baseline.py predict \
  --model pinn --viscosity "$KNOWN_NU" --manifest /path/to/manifest.json \
  --case-index 13 --output /path/to/new/pinn_case13.npz --evaluate --device cuda
```

所有设置可用`--config custom.json`局部覆盖；未知参数名会报错。`--views 0`为单视角，不指定则使用全部已存储视角。默认观察16个独立probe，probe不进入重建。

`--evaluate`仅在全部预测固定后读取参考场。流场L2以未知流体内部区域计算。为保持与既有评估一致，probe初始位置在评价时由所有存储相机三角化，probe误差也用所有存储视角；即使重建只输入一个视角，这些额外probe信息也只用于后置评价，并在`evaluation_protocol`明确记录。

## 产物与统计

训练目录保存`latest.pt/best.pt/epoch_XXXX.pt`、`history.jsonl`和带校验和的`run.json`。采样输出NPZ与同名JSON，包含完整三维预测、均值、初始深度估计、观测、协议和误差。扩散模型另保存样本方差；确定性方法不制造不确定性方差。

```bash
python scripts/run_baseline.py verify /path/to/training_directory
python scripts/run_baseline.py verify /path/to/prediction.npz
python scripts/run_baseline.py summarize \
  --inputs /path/to/predictions/*.json --output /path/to/summary.json
python scripts/run_baseline.py describe --output-dir reports
```

统计先在同一seed内平均样本／案例，再跨seed计算mean ± sample std（ddof=1）。不同N、视角、训练与推断边界、黏度、评价协议或配置分别成组；结果记录checkpoint实际epoch与选择方式。各seed必须有相同案例集合，重复案例、缺失指标或非有限指标会报错。只有一个seed时std为空，不能解释为0。mean不允许用伪种子重复计为多次独立训练。扩散单样本误差平均与先平均流场再算误差分别保存。

固定预测场的后置深度诊断最多优化100步，采用`depth_learning_rate`和`sampling_seed`，设置随结果保存；不会更新已生成的速度场。

批量评价可选择明确案例集合，中断后恢复会核验已有结果的完整性和请求是否一致：

```bash
python scripts/evaluate_baseline.py \
  --manifest /path/to/manifest.json --checkpoint /path/to/run/latest.pt \
  --split validation --case-indices all --output-dir /path/to/new/evaluation \
  --device cuda
```

只测已有的44案例时，请传入那44个案例在manifest中对应的索引列表；不能把默认全部101案例的结果冒充旧44案例同口径结果。批次只有全部案例成功才标记completed。恢复时加`--resume`，损坏或配置不匹配的结果会明确拒绝复用。

不同模型的参数量尚未对齐；目前比较的是完整重建方案。若要单独归因架构或生成机制，后续应增加参数预算和推理预算相近的控制实验。

## 验证状态

测试记录见`validation/cpu_tests.xml`和`validation/summary.json`；真实32³数据只读输入检查见`validation/real_data_preflight.json`。测试通过验证代码路径与数值检查，不证明正式GPU训练稳定、重建准确率或PINN黏度已校准。当前交付不包含九类模型的正式训练权重或科研性能结果。

2026-10-23代码审查与修复记录见`validation/code_review_20261023.txt`：修复禁用辅助损失仍计算、恢复日志超前checkpoint、训练目录混用、统计协议混组／缺失值处理、配置静默接受非法类型，以及推断诊断与元数据遗漏。独立数学检查覆盖FNO谱卷积层的周期平移等变性及复权重有限差分梯度、POD与独立SVD截断误差、PINN解析Poiseuille残差；未据无证据的推测改写数学公式。默认结构、参数量与核心超参数保持参数报告所列值。
