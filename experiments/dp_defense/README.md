# DP 防御：独立草稿与 EAGLE-3 / MTP 草稿头

独立扩展当前固定候选实验。两种部署共享同一个 DP 目标：KD 草稿在固定、
独立的 2,000 条辅助记录上向该目标重新蒸馏；member 草稿从原始基础模型
独立执行 DP 微调。目标及独立草稿采用全参数训练，EAGLE-3 / MTP 仅更新
草稿头的可训练参数、冻结 DP 目标。两次访问成员数据的训练分别计账。

Python 接口及 train/audit/sweep CLI 支持 Qwen3、Gemma 4、Qwen3 EAGLE-3、Llama 3.1 EAGLE-3
和 Qwen3.5 MTP 五类现有训练配方。使用 `plan_private_training` / `train_private`
准备模型，再通过统一的 `evaluate_main` 检验主方法；见[接口与示例](../../docs/main_method_api.md)。

| 目标 ε 上限 | member 草稿 ε 上限 | KD 部署预算上限 | member 部署预算上限 |
|---:|---:|---|---|
| 1 | 1 | (1, 5×10⁻⁶) | (2, 10⁻⁵) |
| 4 | 4 | (4, 5×10⁻⁶) | (8, 10⁻⁵) |
| 8 | 8 | (8, 5×10⁻⁶) | (16, 10⁻⁵) |

表格使用基本组合上限；产物同时保存实际 RDP 会计结果。上述预算分别针对
单个部署模型对，不是跨 ε、epoch、seed 或多个模型版本联合发布的总预算。

## 安装与入口

本轮 **Qwen、epoch 1、KD、3 数据集 × 3 seed × ε=1/4/8** 有两个独立脚本：

```bash
cd /home/mxd/lib/SD_MIA
# 不带参数也默认为 dry-run，只打印计划
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh dry-run
bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh dry-run

# 先完成 27 条件的 DP 目标训练与 KD 蒸馏
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh run --gpu 0
# 再使用上述匹配的 DP 模型完成 27 条件主方法审计
bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh run --gpu 0

# 多 GPU：训练和审计仍分开运行
CUDA_VISIBLE_DEVICES=0,1,2 bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh run --gpus 0 1 2 --workers 3
CUDA_VISIBLE_DEVICES=0,1,2 bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh run --gpus 0 1 2 --workers 3
```

第一个脚本只调用 `sweep train`，第二个只调用 `sweep audit`，不会隐式启动另一阶段。
两个脚本均固定 Qwen/epoch 1/KD，不运行基线方法。dry-run 复用通用规划器，输出的
每个条件含 train/audit 两条计划命令，但不会执行；正式 run 严格只执行各自阶段。
默认单卡顺序运行；`--gpu` 是可见设备的逻辑编号，`CUDA_VISIBLE_DEVICES=2` 时使用 `--gpu 0`。
`--gpus 0 1 2 --workers 3` 可启用动态队列，每张卡最多一个 worker，空闲后领取下一个完整条件。
`--gpu` 与 `--gpus` 互斥；worker 数默认等于所选卡数，也可指定更小值，仅使用列表前 N 张卡。
`CUDA_VISIBLE_DEVICES=2,5` 时使用 `--gpus 0 1`；每个 worker 内部统一使用 `cuda:0`，
因此换卡不会改变训练请求中的 `config.gpu`，公开 seed 也不受分配顺序影响。
仅目标模型的 FP32 CPU 梯度累加器每 worker 就约需 32 GB 主机内存；并行数量还需按内存容量选择。
可用 `--benchmarks wikitection --seeds 1919 --epsilons 4` 选择子集。
`--model-root` 修改 DP 模型根目录时，训练和审计两个脚本必须传同一路径；
`--audit-root` 修改审计结果根目录，`--reference-root` 修改参考训练根目录。
参数须使用完整名称，不能覆盖固定的模型、epoch 或草稿分支。

## DP 梯度累加加速

默认 `--accumulator-device cpu` 以 FP32 在主机上累加逐文档裁剪后的梯度。
8B 参数的累加器约占 32 GB（29.8 GiB）主机内存；每条文档都要从 GPU
搬运整模型梯度，每个优化器步还要把累加值传回 GPU 加噪。因此高内存占用、
GPU 计算间歇性空闲与这条路径相符，不能据此简单增大普通训练的 batch size。
数据已在训练前转换为 examples，不存在每步远程加载数据的问题。

新增 `--accumulator-device cuda`：FP32 累加器放在当前训练 GPU，逐文档累加时
直接按参数执行混合精度加法，省去梯度的 GPU→CPU→GPU 搬运和分块加法调用。
CPU 模式也去掉了显式 FP32 转换副本，跨设备传输保持梯度的原始 dtype。
同设备加噪直接复用即将清零的累加缓冲区，噪声临时空间仍按块限制。
这不会把累加器改为 BF16，也不会改变逐文档全局裁剪、Poisson 采样、固定分母、
空批次加噪、独立秘密随机流、隐私会计或全参数训练；物理微批次仍为一条文档。

CUDA 模式额外占用约 `4 × trainable_parameters` 字节显存，同时减少对应主机内存。
模型参数、反向梯度、优化器状态和激活还需另留空间，长文本尤其如此。
现有优化器使用 paged 8-bit AdamW，显存吃紧引起分页也可能抵消收益。
默认保留 CPU 模式；CUDA 不足时不自动改变后端或恢复半完成的优化过程。
累加设备写入 `DP_REQUEST.json.execution` 和阶段训练记录，切换设备须使用新输出目录。

当前独立工作树可复用原目录的 Python 环境和冻结参考数据，结果默认写入本工作树：

```bash
cd /home/mxd/.codex/worktrees/dp-throughput/SD_MIA
export SD_AUDIT_PYTHON=/home/mxd/lib/SD_MIA/.venv/bin/python
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh dry-run \
  --reference-root /home/mxd/lib/SD_MIA/artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3 \
  --benchmarks wikitection --seeds 1919 --epsilons 4 \
  --accumulator-device cuda --gpu 0 \
  --model-root artifacts/training/dp_cuda_accum_v1/runs
```

`dry-run` 只查看计划；待目标 GPU 空闲后，将其改成 `run` 才会训练。
单条件 CLI `experiments.dp_defense.train` 和通用 `sweep` 接受同名参数；
Python `plan_private_training` / `train_private` 使用 `accumulator_device="cuda"`。
配置也覆盖 EAGLE-3/MTP 的 DP 目标及 member head，不改变普通辅助 KD 的训练路径。

本轮只运行 CPU 回归检查，未启动 GPU 训练；去除了可重复观察到的转换副本，
尚无完整 8B 模型的加速倍数或峰值显存实测。合成数据微基准见
`python -m experiments.dp_defense.benchmark_accumulator --help`，GPU 模式应在空闲卡上手动运行。

脚本默认使用仓库 `.venv/bin/python`、离线模式；重复原命令可复用已经完成且来源一致的阶段。
训练直接读取各条件的匹配参考护照，核验 `seed == data_seed == shared.seed`，
初始化、KD 和审计使用对应公开 seed；DP 采样/噪声保留独立未公开随机流。
数据、模型和参数变化时使用新目录。运行命令及结果路径也见
[当前实验指南](../../docs/ready_experiments.md)。

每次启动有独立日志目录，`--log-root` 可覆盖；默认训练在
`artifacts/training/dp_defense_v1/executions/train/<attempt>/`，审计在
`artifacts/audits/dp_defense_v1/executions/audit/<attempt>/`。
`JOBS.json` 保存条件命令，`STATUS.json` 保存状态/seed/GPU/日志位置，每条件一个 `.log`。
普通失败记录后继续其他条件，最终返回非零；Ctrl-C/SIGTERM 清理全部活动 worker 及其子进程。

通用矩阵入口仍支持直接传入选择参数：

```bash
.venv/bin/python -m experiments.dp_defense.sweep dry-run \
  --model-pairs qwen3 --epochs 1 --draft-variants kd \
  --seeds 1919 1949 1978
```

这会规划三个数据集 × 三个 seed × 三个 ε，共 **27 个条件、54 份模型产物、27 个草稿审计**。
每个条件只训练 DP 目标和向该目标蒸馏的 KD 草稿，不训练 member 草稿，也不要求其检查点。
KD 部署只继承目标的隐私预算，不加上未执行的 member 训练预算。
这里的 epoch 1 指目标的期望遍历次数（当前 125 步）；KD 沿用参考条件的蒸馏预算，默认 384 步。

`--draft-variants kd` 同时传给训练和审计；`train`、`audit` 单条件命令也支持此参数。
Python 的 `plan_private_training` / `train_private` 使用 `draft_variants=["kd"]`，
`run_audit` 不指定时自动选择该条件实际完成的草稿。默认训练选择仍为 `kd member`。
可显式对已有双草稿 DP 模型只审计 KD；请求未训练的分支会报错。
改变训练分支必须使用新输出目录；不能把已有双分支请求改写成单分支来恢复。
这些参数也适用于现有 EAGLE-3 / MTP 家族，KD 对应其 auxiliary head。

从仓库根目录运行。Opacus 是可选依赖，旧实验不要求安装：

以下以 Qwen 独立草稿为例；更换参考训练目录即可使用已注册的其他基座与 EAGLE-3/MTP 草稿头。CLI 根据训练护照选择对应训练实现。

```bash
uv sync --locked --extra dp

DP_REF=artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3/wikitection/epoch1/seed1919
DP_RUN=artifacts/training/dp_defense_v1/runs/qwen3/epsilon4/wikitection/epoch1/seed1919
DP_AUDIT=artifacts/audits/dp_defense_v1/tasks/qwen3/epsilon4/wikitection/epoch1/seed1919

# 只读规划：读取训练护照、来源指纹，反求噪声；不加载模型、不创建结果目录。
.venv/bin/python -m experiments.dp_defense.train dry-run \
  --reference-run "$DP_REF" --output-dir "$DP_RUN" --epsilon 4

# 显式启动一个条件；复用参考条件的数据划分、基础模型 revision、lr 和训练预算。
# GPU 编号为当前进程可见的逻辑编号。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m experiments.dp_defense.train run \
  --reference-run "$DP_REF" --output-dir "$DP_RUN" --epsilon 4 --gpu 0

# 重复相同 run 命令即可恢复；也可只读检查阶段完整性。
.venv/bin/python -m experiments.dp_defense.train status \
  --reference-run "$DP_REF" --output-dir "$DP_RUN" --epsilon 4

# 两种草稿分别重新采集 B=2 观测、拟合 TCN、校准并测试。
# --include-baselines 可选：同时运行原有 11 个 target-only baseline。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m experiments.dp_defense.audit run \
  --run-dir "$DP_RUN" --output-dir "$DP_AUDIT" --device cuda:0 --include-baselines

.venv/bin/python -m experiments.dp_defense.audit summarize \
  --run-dir "$DP_RUN" --output-dir "$DP_AUDIT" --include-baselines

# 与使用当前代码和同一审计设置生成的非 DP 参考批次对齐。
# 已删除的历史 qwen_fixed_v1 不能作为当前共享生成配置的可比参考。
.venv/bin/python -m experiments.dp_defense.compare \
  --dp-root artifacts/audits/dp_defense_v1/tasks \
  --reference-root artifacts/audits/dp_reference_v1/tasks \
  --output-dir artifacts/audits/dp_defense_v1/reports
```

上述 `dp_reference_v1` 是需要先生成的参考批次，可用跨模型 CLI 的 `--output-root artifacts/audits/dp_reference_v1/tasks` 运行相同模型、数据条件与审计参数。比较器拒绝缺失或不匹配的报告。

单条件 `--max-grad-norm` 默认 1.0，可显式调整；改设置必须换新结果目录。
恢复时应使用相同参数、运行环境和源码。参考实验只提供数据合同和训练配方，
不会用其已经非 DP 微调的权重初始化 DP 训练。

默认仅选择 `qwen3`。每个模型组合的矩阵规划提供三个数据集 × 两个 epoch × 三个 seed × 三个 ε，共 54 个条件、
162 份模型产物、108 个草稿审计配置；每个条件只有一份共享目标。
以下仅打印计划：

```bash
.venv/bin/python -m experiments.dp_defense.sweep dry-run

# 五种模型组合：270 个条件、810 份模型产物、540 个草稿审计配置。
.venv/bin/python -m experiments.dp_defense.sweep dry-run \
  --model-pairs qwen3 gemma4 qwen3_8b_eagle3 llama31_8b_eagle3 qwen35_9b_mtp
```

矩阵训练目录为 `runs/<model_pair>/epsilon<ε>/<benchmark>/epoch<N>/seed<N>`，审计使用相同相对路径放入 `tasks/`，避免跨模型覆盖。旧单模型目录仍可通过单条件命令显式指定；比较器兼容旧布局。`--reference-root` 仅用于单模型覆盖，默认从共享模型注册表读取各自参考根目录。

显式将 `dry-run` 替换为 `train` 或 `audit` 才执行相应阶段。默认顺序执行于
一个 GPU；可用 `--gpus` / `--workers` 并行，Ctrl-C 停止所有活动子进程。可用 `--benchmarks`、
`--epochs`、`--seeds`、`--epsilons` 限定子集。独立条件失败会被记录并继续
处理其余条件，最终返回非零；训练矩阵不会隐式开始审计。

## 训练机制与隐私边界

每个新 DP 请求保存 `seed_policy`：**同一条件的 seed = data_seed = 冻结 split.seed =
目标公开随机 seed = 草稿公开随机 seed = KD 采样 seed = audit_seed**。
训练规划核对实际 split 内容，审计拒绝另一个 seed；1919/1949/1978 分别对应自己的数据、
目标、KD 草稿、TCN、辅助分区和 bootstrap。模型训练按阶段重新设置同一个公开 seed，
避免跳过已完成阶段后，初始化/dropout 随机状态因执行历史发生变化。

**DP 的 Poisson 采样与高斯噪声是明确例外**：它们用独立、未公开的 OS 熵初始化随机流，
不由公开 seed 推导，不在元数据中保存。公开固定噪声会破坏 DP 的保护假设。
因此同一个 condition seed 表示匹配的实验条件，不保证重新执行 DP 训练后权重逐字节相同；
已完成阶段的恢复则复用同一份已校验权重。

- 邻接关系为固定训练配方下增加/移除一个原始文档。当前共享 split 每个
  document ID 恰好对应一个 SFT 样本，拒绝重复 ID；prompt 不计入损失。
  若以后将文档拆成多个样本，必须先实现文档内梯度聚合，不能直接沿用此保证。
- 公开参考规模固定为 2,000，逻辑期望 batch 为旧配置中的 batch×accumulation
  （当前 16），采样率 q=0.008。每一步独立 Poisson 采样；epoch 1/3 对应
  125/375 个优化步骤，代表期望遍历次数，而不是旧的无放回打乱训练。
  草稿头遵循现有独立配方，固定 384 次更新、期望 batch 16、学习率 2e-5。
- 每个文档单独反向传播，对全部可训练参数的联合梯度作 L2 裁剪。累加后
  加入标准差 σC 的独立高斯噪声，除以固定期望 batch；不按实际抽中数量归一化。
  空采样批次也加噪、更新并记账，未使用的参数同样加噪。
- Opacus RDP 会计根据 q、步数和 δ 反求 σ，并检查最终 ε 不超过目标。
  文档梯度不通过普通 batch 梯度裁剪近似；不进行依赖私有损失的早停或选模。
- 使用 AdamW/原有 paged 8-bit AdamW 更新加噪梯度；独立模型更新全部参数，草稿头只更新可训练参数。
  逐文档物理 microbatch=1，FP32 累加器放在 CPU 以减少 GPU 占用；这会
  增加 CPU 内存及 CPU/GPU 传输开销。8B 的该累加器本身约需 32 GB 主机内存。
  原配置的 batch 和 accumulation 此时只定义逻辑期望 batch。
- 每个训练阶段的采样与噪声使用独立、未公开的 OS 熵初始化随机流，不受公开
  condition seed 控制，也不在检查点或日志中保存。这是有限精度 PyTorch PRNG
  的研究实现，按理想 Poisson-Gaussian 机制计算预算；没有宣称获得 CSPRNG、
  浮点侧信道或生产系统安全认证。
- 保证限于冻结公开数据准备之后的受控 SFT 成员暴露；辅助集合和公开基础模型
  在邻接数据集中保持固定。不保护预训练成员身份，也不声称数据筛选、去重和
  split 生成本身是 DP 算法。

**DP 声明只覆盖模型权重及其后处理输出。** 实验目录中的 `results.json`、
split manifest、记录 ID/成员标签、审计分数，以及含来源哈希的 `DP_REQUEST.json`
和 `DP_STAGE.json` 是可信实验元数据，不属于 DP 发布物。尤其不能将含成员
名单或其来源指纹的整个实验目录作为“DP 模型”对外发布。只发布冻结的模型权重、
公开模型/分词配置和不含私有来源指纹的预算参数；同时发布多个训练版本需要组合。

## 评估、恢复与兼容

训练数据角色仍为 2,000 member / 2,000 nonmember / 2,000 draft auxiliary /
600 audit auxiliary。审计保持 320 条检测器拟合、80 条验证、200 条独立校准，
完整 2,000+2,000 条测试记录。TCN、五维难度输入、正向 sparse 参数与 B=2
规则全部复用；每个 ε 和草稿分支有独立检测器，测试标签不参与选择。

`compare` 检查方法、草稿、审计参数和完整校准/测试 ID、标签、分区相同，
然后输出 `COMPARISON.csv/json`、按匹配 seed 计算的 `SEED_SUMMARY.csv`。
报告 DP 前后 AUC、pAUC、ROC TPR、校准 TPR 与实际 FPR 的有符号差异。
该汇总仅描述已匹配报告，不把缺少条件的结果误称为完整矩阵。攻击下降不能
替代隐私会计，也不能单独证明生成质量或 SD 性能可接受；原评分报告保留成本，
模型质量与真实生成吞吐仍需配套实测。固定候选成本不是生产 SD 加速指标。

DP 检查点采用原有 `checkpoints/target`、`draft_auxiliary_distilled`、
`draft_member_sft` 布局；草稿头使用 `heads/auxiliary_head` 和 `heads/member_head`。
预算放在独立字段，原 `Config` 无需修改。阶段通过
原子目录发布与逐文件哈希校验恢复，KD 完成标记绑定教师权重指纹。未完成
训练不发布检查点；中途停止后从基础模型重跑该阶段，不伪装成逐步优化恢复。
已经完成的目标/KD 阶段可直接复用。恢复预算只覆盖最终发布的训练结果，
不包括另行暴露的中间模型或先前训练尝试。

防御专属训练、会计与验证放在 `experiments/dp_defense/`，数据、模型加载和检测复用 `experiments/shared/`。新审计额外绑定 DP 源码及来源；历史结果保留原始指纹，代码重构后不重新签署或直接续写旧缓存。
普通检查点不能被当作 DP 检查点复用，来源/预算/模型变更必须使用新的结果目录。

```bash
.venv/bin/python -m pytest -q tests/dp_defense
.venv/bin/python -m pytest -q
```

单元/集成测试覆盖噪声会计、全局逐文档裁剪、空批次、非有限梯度、真实小型
Qwen 共享权重训练、阶段来源/哈希恢复和匹配报告比较。完整 8B GPU 训练及
防御效果需显式运行上述实验后获得，代码实现完成不代表已经有防御效果结论。

2026-09-20 验证：新增 23 项 DP 测试和全仓 286 项测试通过；真实
WikiTection/epoch1/seed1919 的 6,600 条四角色记录及共享划分哈希与参考
训练护照一致；54 条件矩阵 dry-run 与单条件噪声规划通过，未启动完整训练。

仅 KD 扩展验证：全仓 472 项测试通过；对现有 Qwen3 三个数据集的 epoch1、
1919/1949/1978 参考条件，完成 ε=1/4/8 的 27 条件只读规划，确认实际冻结 split、
数据配置和审计 seed 对应一致。CPU 小模型测试覆盖单 KD 分支、真实 DP 目标更新、
中断恢复后的公开随机状态、预算、分支缺失拒绝，以及五种模型家族的审计选择。
未执行正式 GPU DP 训练或成员推断实验。
