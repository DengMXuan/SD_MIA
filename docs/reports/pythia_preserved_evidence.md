# Pythia 信息保留评分：单 seed 实验

固定候选、原始前缀、B=2、冻结目标/草稿与已拟合 TCN。复用原观测，使用新的评分批次；不重新训练语言模型或 TCN。默认 seed=1919，各 benchmark 独立运行。

本轮检验三项修改：减少条件校正造成的信息损失，保留草稿自身的边际区分信息，组合全局和稀疏条件证据。查询预算、草稿曝光和 WikiMIA 辅助数据来源保持原协议，便于归因；本轮不检验改变这三项设置的效果。

## 预先固定的评分

记 A 为文档平均接受率、μ 为 TCN 预测的平均接受率、Q 为文档平均草稿 log-q，R=A−0.5μ。
G 为原 eta={0.5,1,2}、rho=1 的全局正向评分；S 为原 eta 和 rho={0.05,0.1,0.25} 的稀疏评分。

每个分量先用原 400 条参考非成员（320 训练 + 80 验证）的文档分数估计均值和样本标准差，得到 z 标准化。
若参考分量标准差 <= 1e-12，则禁用该分量，所有待测文档在该分量上贡献零。标准化不裁剪测试极值，不重新归一化固定权重。
参考集曾用于 TCN 拟合/选检查点，不能称为独立校准集；独立的 200 条校准非成员只用于最终阈值。

预先指定的主候选为：

`preserved_multiscale = 0.5 z(R) + 0.25 z(Q) + 0.125 z(G) + 0.125 z(S)`。

主候选及全部对照的方向与权重在计算新结果前固定，不按当前测试集选择最优方法。

| 方法 | 作用 |
|---|---|
| main_fixed_sparse_positive | 原始主方法、保存分数回放控制 |
| accept_rate / q_only | 接受反馈与草稿边际信息基线 |
| partial_residual_050 / full_residual | 部分/完全条件校正 |
| dense_positive | 全局条件评分 |
| conditional_multiscale | 0.5 z(G) + 0.5 z(S) |
| accept_q_fusion | 0.5 z(A) + 0.5 z(Q) |
| accept_multiscale | 0.5 z(A) + 0.25 z(G) + 0.25 z(S) |
| preserved_multiscale | 固定主候选 |
| preserved_no_q | 0.5 z(R) + 0.25 z(G) + 0.25 z(S) |
| preserved_no_partial | 用 z(A) 替换主候选的 z(R)，其余权重固定 |

评分函数不接收标签或目标概率。独立校准仍采用含并列的上尾 conformal p 值；融合分数不是 e-value，也不是成员后验概率。增加的草稿分量可能捕获来源偏移，提升代理标签 AUC 不等于因果识别目标记忆。

## 运行与复现

```bash
bash experiments/scripts/ablation/ablation_main_pythia_preserved_b2_cpu.sh dry-run --benchmark mimir13
bash experiments/scripts/ablation/ablation_main_pythia_preserved_b2_cpu.sh run --benchmark mimir13
bash experiments/scripts/ablation/ablation_main_pythia_preserved_b2_cpu.sh run --benchmark mimir7
bash experiments/scripts/ablation/ablation_main_pythia_preserved_b2_cpu.sh run --benchmark wikimia
```

默认覆盖 MIMIR 13_gram_0.8 的七领域及独立 full-Pile、MIMIR 7_gram_0.2 的 GitHub/arXiv、WikiMIA 64/128 词，共 12 个条件 × 12 个评分。
支持 `--sources`、`--seed`、`--input-root`、`--output-root`、`--threads`（默认 1）、`--bootstrap`（默认 1000）。单次只执行一个 benchmark，不提供跨 benchmark 的 all 模式。

输出默认在 `artifacts/audits/pythia_preserved_v1/<benchmark>/seed1919/`。
PLAN/REQUEST 固定评分、源文件闭包、版本、输入路径与哈希；每个条件保存参考标准化参数、全部分数、指标和完成哈希。
先回放旧分数及 AUC/ROC/校准指标，失败则拒绝新对比。相同计划可恢复；参数、来源或实现变化必须换输出目录。dry-run 校验输入，不创建输出或执行 TCN。

AUC 及相对原方法、接受率的配对 bootstrap 区间按同一批测试文档计算；ROC TPR@1%/10% 与独立校准 TPR/FPR 分开报告。bootstrap 固定已拟合模型和校准集，未作多重检验校正。既有测试集已用于诊断，本轮属于探索性单 seed 验证，不能当作独立泛化确认。full-Pile 单独解释，WikiMIA 标签保留时间代理语义。

实现：`experiments/shared/methods/preserved_accept_only.py`。实验入口：`experiments/pretraining/preserved_evidence.py`。回归检查：`tests/pretraining/test_preserved_evidence.py`。
