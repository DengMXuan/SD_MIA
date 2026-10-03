# Pythia 接受率尾部与有界正向补充

本轮仍使用 seed=1919、B=2、固定候选及冻结 TCN，覆盖与第一轮相同的 12 个条件。
目标是减少条件校正和无界线性融合对高接受率记录的压制，检验低误报区域的收益。所有规则在新结果计算前固定，不在本轮测试结果上选择权重或方向。

## 固定主候选

长度 L 的记录有 n=2L 次判定，接受总次数为 k。定义平滑拒绝尾部分数：

`T = -log((n-k+0.5)/(n+1))`。

它使全接受记录仍具有有限分数，且区分短记录和较长记录的全接受观察。此变换是评分规则；不假设全部 token 共享同一个 Bernoulli 参数，不将它解释成目标概率或成员后验。

与第一轮相同，只用 400 条参考非成员的文档均值/样本标准差计算 z 标准化；常数通道禁用。200 条校准非成员独立保留。
记 Q 为平均草稿 log-q，G/S 为原全局/稀疏正向证据。条件分量先分别除以 L，再标准化：

`C = 0.5 z(G/L) + 0.5 z(S/L)`。

主候选 `tail_guarded_positive`：

`z(T) + 0.5 min(softplus(z(Q)), 2) + 0.25 clip(C, 0, 1)`。

草稿分量禁用时其加分为零。总加分始终在 [0,1.25]，条件加分在 [0,0.25]；负向条件证据不能降低主通道分数。
这是对分数扰动的限制，不保证整体排序或同 FPR 召回率一定优于接受率。非成员得到加分仍可能提高最终阈值。

## 对照与统计

保留原主方法、第一轮 `preserved_multiscale`、接受率、q-only、第一轮接受率+q 与去部分校正融合，共六项控制；新增七项：

| 方法 | 目的 |
|---|---|
| smoothed_reject_tail | 只改变接受率变换 |
| tail_q_bounded | 主通道 + 草稿补充 |
| tail_guarded_positive | 预先固定主候选 |
| tail_guarded_no_q | 去草稿补充 |
| tail_guarded_no_transform | 主通道改回 z(接受率)，其余规则相同 |
| tail_signed_fusion | z(T)+0.5z(Q)+0.25C，无界且保留负贡献 |
| tail_guarded_unbounded | 保留正向贡献，但去除两个加分上限 |

结果同时报告 AUC、pAUC、测试 ROC TPR@1%/10% 和独立校准 TPR/实际 FPR。
使用同一批类别内文档 bootstrap，计算 AUC 和 ROC TPR@1% 相对原方法、上一版、接受率的配对差值区间。区间固定 TCN 和校准集，未做多重检验校正；尾部仅有少数误报时应谨慎解释。
原测试集此前已被分析，单 seed 结果仍是探索性机制检验。WikiMIA 保留时间代理标签含义。

## 运行

```bash
bash experiments/scripts/ablation/ablation_main_pythia_tail_b2_cpu.sh dry-run --benchmark mimir13
bash experiments/scripts/ablation/ablation_main_pythia_tail_b2_cpu.sh run --benchmark mimir13
bash experiments/scripts/ablation/ablation_main_pythia_tail_b2_cpu.sh run --benchmark mimir7
bash experiments/scripts/ablation/ablation_main_pythia_tail_b2_cpu.sh run --benchmark wikimia
```

结果写入新批次 `artifacts/audits/pythia_tail_v1/<benchmark>/seed1919/`。共享 runner 的 `--suite tail` 启用本轮规则，默认仍为上一版规则；代码变化后重跑任何旧批次都必须指定新的输出目录，不能改写旧 PLAN 或来源哈希。
每个条件先回放旧分数；保存全部评分、参考标准化、配置、代码/输入哈希与完成标志。不重新训练语言模型/TCN、不重新采集反馈。

实现：`experiments/shared/methods/tail_accept_only.py`；共用实验入口：`experiments/pretraining/preserved_evidence.py`。

## seed1919 实测结果

12 个条件已完成，完整七基线、消融、配对区间及独立校准结果见
[本轮报告](../../artifacts/audits/pythia_tail_v1/RESULTS.md)，机器可读结果为同目录 `RESULTS.json` / `RESULTS.csv`。

本轮固定主候选在 13-gram GitHub 的 ROC TPR@1% 从 8.50% 提高到 24.50%，
7-gram arXiv 从 7.25% 提高到 13.00%；但 7-gram GitHub 从 37.69% 降到 33.21%。
13-gram 七领域宏平均 AUC 从 0.5773 降到 0.5738，TPR 从 2.64% 提高到 4.43%，
接近原始接受率的 0.5741 / 4.46%。WikiMIA 64 和 full-Pile 的 AUC 也下降。
因此保留为探索性候选，尚不据此替换默认方法或声称稳定优于七基线；不从消融中追选本轮主方法。
