# Pythia 文档级 delta 验证结果

结论：本次小规模验证不支持直接沿用 SFT 中“文档平均 delta 清晰分离”的结论。GitHub 存在 delta **形状**差异，主要体现在负值比例和负尾部；Wikipedia 有较弱迹象；DM Mathematics 在所检查的统计量中没有明确差异。

## 实验口径

- 直接重新计算 Pythia 6.9B / 1.4B 的逐 token `log p` 和 `log q`，两端均用 CPU FP32，同一真实 token、同一完整前缀。
- 使用已有 MIMIR 测试分区 seed=1919；三个领域各 64 成员 + 64 非成员，共 384 篇、134,133 个评分 token。
- 文档指 MIMIR 文本片段，最多 512 token；首 token 作上下文，不追加 EOS。
- 2000 次按文档 bootstrap。数据、选中 ID、模型版本、哈希见输出目录的 `PLAN.json`。所有 GPU 保留给原有任务。
- 单 seed 子样本、先前查看过的测试集，仅是探索性机制验证。不能外推为全领域/全量/三 seed 确证。

## 1. 文档平均 delta 的跨文档分布

每篇文档的分数为 `mean(log p - log q)`；AUC 固定为数值越大越像成员。

| 领域 | 成员均值 | 非成员均值 | 均值差及 95% CI | 文档平均 delta 的 AUC 及 95% CI |
|---|---:|---:|---|---|
| GitHub | 0.1787 | 0.1282 | 0.0505 [-0.0011, 0.1045] | 0.5076 [0.4011, 0.6077] |
| Wikipedia | 0.2203 | 0.1845 | 0.0358 [0.0060, 0.0678] | 0.5862 [0.4844, 0.6863] |
| DM Mathematics | 0.0336 | 0.0308 | 0.0028 [-0.0049, 0.0119] | 0.4907 [0.3911, 0.5906] |

三组平均 delta 的 AUC 区间均包含 0.5。Wikipedia 的均值差有正向迹象，但该区间属于未作多重校正的探索性指标，且文档排序仍有较大不确定性。GitHub 的平均值差不能转化成可靠的文档排序；其两类经验分布存在交叉。

作为历史参照，原 News SFT（Qwen3、3 epoch、辅助蒸馏草稿）的 D 划分各 800 篇，文档平均 delta 为 0.8949 / 0.3524，AUC=0.9747。来源为原工作区 `artifacts/reports/figures/introduction/qwen3_newstection_epoch3_auxiliary/alternatives/document_mean_log_ratio.csv`。模型、数据和训练设置不同，这只是说明原洞察展示的现象，不是控制变量对照。

## 2. 每篇文档内的 delta 形状

每篇先计算 token delta 的 CDF，再在类别内等权平均；检验使用整篇文档标签置换，绝不把 token 视为独立样本。[-2,2] 固定网格上的 CDF 差异检验，经三个条件 Holm 校正后的 p 值分别为：

- GitHub：0.0180；检测到分布形状差异。
- Wikipedia：0.1419；此主检验没有达到 0.05。
- DM Mathematics：0.8761；未检测到形状差异。

**这个检验针对文档等权的 token delta 分布，不能称为“文档平均 delta 显著可分”。**

GitHub 的具体差异：

| 文档内统计量，再按文档平均 | 成员 | 非成员 |
|---|---:|---:|
| delta < 0 的 token 比例 | 28.14% | 35.12% |
| mean(min(delta,0)) | -0.05397 | -0.09508 |
| mean(exp(min(delta,0)))，理论接受率 | 96.57% | 94.11% |

负 delta 质量的组间差为 0.04111，逐指标文档 bootstrap 95% CI 为 [0.02315,0.05965]。文档平均理论接受率 AUC=0.7500，95% CI=[0.6592,0.8315]。这表明平均 delta 难以排序，不代表完整 delta 序列没有成员相关结构。

## 3. 对主方法的含义

- GitHub 的接受反馈仍携带信号。同一批文档的旧 GPU BF16 / B=2 接受率 AUC=0.7395，旧 TCN 主方法 AUC=0.6274；这是同一旧缓存上的描述性比较，支持继续检查条件校正和证据计算。
- 精确理论接受率的 AUC 高于平均 delta 的 AUC，并不意味着截断增加了原始序列的信息；这里比较的是两种不同的文档汇总函数。
- 后续证据设计可以重点检查负 delta/拒绝的比例、幅度和位置结构，以及 q 条件校正后这些信号保留多少；本结果没有验证任何新评分方法。
- 数学领域在原始 delta、负尾部和理论接受率上均缺乏明确差异。不能把这组表现完全归咎于 TCN 证据计算，也不能凭本次有限统计量证明整个序列同分布。
- 两个 Pythia 模型均在 The Pile 上预训练。观察到相对概率差异不等于因果证明哪个模型“记得更清楚”。

## 4. 精度边界

本次直接重算的 p/q 两端都为 FP32。原接受计数及 TCN 来自 GPU BF16，因此本次不构成原运行的逐位重放。

新 FP32 草稿概率与旧缓存的逐 token 平均绝对 log 概率差：GitHub 0.0305、Wikipedia 0.0490、数学 0.0186；最大差分别约 1.88、2.34、1.43。文档平均 logq 的平均绝对差约 0.0038、0.0051、0.0016。

额外对每领域/类别中偏差最大的文档做了 CPU BF16 与 BF16 权重 + FP32 运算检查。它们仍未逐位复现旧 GPU 缓存；这项检查没有完整隔离硬件内核、运算精度等因素，也没有重算旧 GPU BF16 的目标概率。因此不能把新理论接受率与旧接受计数的全部差异归因于 Bernoulli 采样。精度检查详情保存在 `PRECISION.json`，可运行 `check_precision.py` 复现。

当前结论应表述为：**固定 Pythia 模型版本的 FP32 机制验证中，GitHub 存在负尾部结构差异，文档平均 delta 的清晰分离未在这三个小样本领域重现。** 正式确认还需原 GPU BF16 设置、完整测试集及三个 seed。

## 文件

独立脚本位于 `standalone/pythia_delta/`。全部新结果位于工作树 `artifacts/audits/pythia_delta_pilot_v1/`：`REPORT.json/md`、`documents.csv`、`delta_distributions.png/pdf`、`PRECISION.json`，以及可恢复的逐 token p/q。

统计方向、ties、Holm 校正、delta 截断、文档 bootstrap 的基本校验已通过。完成后又核对了报告哈希、384 个唯一文档、134,133 个 token、`mean(delta)=mean(logp)-mean(logq)` 和正负质量分解恒等式。
