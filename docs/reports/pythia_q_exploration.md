# Pythia q 特征与位置证据的迭代探索

目标为 GitHub、arXiv、Wikipedia 和 WikiMIA 的 AUC；保持固定候选、原始前缀、B=2。
所有评分只接收草稿 q 特征与接受计数，禁止读取精确目标概率 p。语言模型冻结。
本轮复用观测，重新拟合的条件预测器也只使用辅助非成员。

用户明确允许本轮用开发部分的成员标签比较配置，这是对原先“成员只作最终评估”的
实验范围调整；成员标签仍不用于拟合预测器。采用统一评分规则，不逐领域追选最好方法。

## 划分与解释范围

- 关注 13-gram GitHub/arXiv/Wikipedia、7-gram GitHub/arXiv、WikiMIA 64/128，7 个条件。
- 原训练/验证/校准辅助非成员为 320/80/200，各角色保持独立。
- 对全部三个 seed 的记录，按 token hash、完整文本 hash、相同前 64 token 的连通组统一分组。
- 根据固定 hash 将原测试记录分为约 60% 开发和 40% 确认部分；不按分数或标签分配。
- 仅 seed1919 的开发部分指导本轮方案迭代。确认部分不在开发日志中输出指标；最终配置冻结后统一计算。
- 三个 seed 的 WikiMIA 测试文本相同，MIMIR 也存在重叠。因此其他 seed 是反馈和辅助划分的稳定性检验，不是独立文本验证。
- 原完整测试集历史上已被多次分析，即使本轮隔离确认部分，整体仍属于探索性结果，不能称为全新独立测试。

## 迭代方向

1. q 的熵、排名、低概率尾部，以及根据 q 预选的位置：检验信号来源，保留简单接受率和接受率+q 对照。
2. 只用非成员拟合更灵活的条件计数分布，比较其验证预测质量，检查不常见的接受反馈是否提供额外区分信息。
3. 基于开发部分选择位置聚合与有限融合规则，保留前轮有效分支，冻结一版后再做确认和七基线对照。

位置贡献只能说明统计区分力，不解释为“这个 token 被记忆的概率”。基于全位置已采集反馈做筛选仍按全位置 B=2 计费。

## 运行与产物

入口：`experiments/scripts/ablation/ablation_main_pythia_q_features_b2_cpu.sh`。
共享评分：`experiments/shared/methods/q_feature_accept_only.py`。
输出新批次：`artifacts/audits/pythia_q_exploration_v1/`。

`prepare` 冻结 SPLIT_PLAN.json；每轮保存各自源码哈希、分数、非成员标准化与开发指标。
旧来源只读，参数或数值实现变化不得改写旧轮次的来源记录。

完整新批次的运行顺序如下，已有批次不要重新 prepare。源码变化后复现应指定新的 `--output-root`：

```bash
bash experiments/scripts/ablation/ablation_main_pythia_q_features_b2_cpu.sh prepare
bash experiments/scripts/ablation/ablation_main_pythia_q_features_b2_cpu.sh round1
bash experiments/scripts/ablation/ablation_main_pythia_q_features_b2_cpu.sh round2
bash experiments/scripts/ablation/ablation_main_pythia_q_features_b2_cpu.sh round3
bash experiments/scripts/ablation/ablation_main_pythia_q_reference_confirm_b2_cpu.sh run
```

## 已冻结的规则

开发目标为七个关注条件的 AUC 等权平均，各条件相对上一版最多回退 0.005。
开发过程共记录 157 个控制、特征和组合；没有逐领域选择不同规则。
选择结果为 `qref_multiscale_negative_q50_c25`，保存于 `FROZEN_SELECTION.json`。

非成员预测器为直方图梯度提升分类器，输入 log-q、归一化熵、排名、top-1 margin、相对位置；
上下文版本再加入 7/31-token 窗口均值/波动、文档内 q 偏差和相邻 q 差值。
两个版本均只在原 320 条非成员上拟合计数 C∈{0,1,2}，以原 80 条验证非成员的文档等权 log loss 选择。
其输出 π_t(C|q) 是接受计数分布，**不是目标模型 token 概率 p**。

记 A 为全位置接受率，A_N 为预测非成员接受率最低 40% 位置的实际接受率。
用 q 的熵定义启发式标准化量 `u_t=(log q_t+H(q_t))/sqrt(max(H(q_t),0.25))`；
它不是使用精确方差的 Min-K++。A_U 为 u 最低 60% 位置的实际接受率，U 为 u 最低 20% 的均值，Q 为平均 log-q。

最终分数：

`S = (z(A)+z(A_N)+z(A_U))/3 + 0.5 min(z(Q),0) + 0.25 z(U)`。

A_N 的标准化只用 80 条验证非成员，避免预测器训练样本上的位置表现影响其尺度；
其他分量标准化使用原 400 条参考非成员。200 条校准非成员始终保留作阈值。
所有位置集合先根据 q 或 q 预测的分布确定，再聚合待测接受计数；不会根据待测位置是否接受来挑选位置。

## 实测结论与接口

三轮开发目标 AUC 为 0.6965、0.6976、0.7063，上一版为 0.6914。
确认部分 7 条件 × 3 seed 的 AUC 描述性均值由 0.6944 提高到 0.7057。
7-gram arXiv 和 WikiMIA 的改善较突出，13-gram arXiv 尚未改善。
更简单的 q-hard 消融在确认部分均值达到 0.7085，高于已冻结主候选；保留为后续假设，未据此追选主方法。
三个 seed 共享记录，均值不是 21 个独立数据集的证据，也不做跨 seed 独立样本显著性推断。

[完整结果](../../artifacts/audits/pythia_q_exploration_v1/RESULTS.md)同时提供开发、确认、原完整集描述性结果、全部指标和七基线。

可复用推理接口为 `experiments.shared.methods.q_reference_scorer.QReferenceScorer`：
从某个已完成 confirmation 条件目录加载其可信本地工件，传入同一 tokenizer 的 vocabulary_size；
`score(features, counts, lengths)` 只接收六维草稿特征、B=2 计数和文档长度，返回冻结规则的记录分数。
不需要标签或目标概率，也不在新批次上重新标准化。
