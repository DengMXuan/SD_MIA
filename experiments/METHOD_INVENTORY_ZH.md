# 实现盘点与精简建议

审查日期：2026-09-17。依据当前源码、已保存汇总报告及模块导入关系；本次不重跑实验、不移动或删除实现。

计数口径：将当前 `sd_membership_sft` 的攻击、评分和校准实现归并为下表 **27 个方法族/组件**，不把随机种子、预算、网络超参数和组合矩阵的每个单元格重复算作新方法。另有 **11 个外部基线**、3 种草稿适配路径、独立预训练评估路径及公共工具。27 不是独立算法的严格数量；例如难度特征、稀疏评分、校准可以组合。

以下“移出主线”指停止作为默认实验维护，先保留可复现的归档；“删除候选”指提取公共依赖、保存源码与结果后，可从活跃源码删除的实验分支，不等于现在就能删除整个文件。未验证和协议不同均不能判定为无效。

## 1. SD 方法、变体与建议

文件名均相对于 `experiments/sd_membership_sft/`。

| 编号 | 方法族/组件及主要变体 | 实现文件 | 证据与保留/删除建议 |
|---:|---|---|---|
| 1 | 精确 p/q gap、接受率、正负方向与固定窗口评分 | `pq_gap_mia.py`, `directional_mia.py` | **保留少量机制对照**。读取精确目标概率，不能当作当前 accept-only 方法；模型前向、校准等公共函数必须保留。 |
| 2 | M1：Q/QH 条件校正，线性/MLP 条件模型及 logistic/MLP 检测器；B0/B1/B2 对照 | `m1_extract.py`, `m1_features.py`, `m1_fit.py`, `m1_evaluate.py` | **监督检测器移出当前主线**，其训练用成员标签；**原始草稿特征提取保留**，当前难度特征复用它，预训练路径也依赖它。 |
| 3 | M2：概率证据引导激活池化；BQ、H-only、Direct、M2-F、M2-U/G、概率容量对照及消融 | `m2_features.py`, `m2_models.py`, `m2_fit.py`, `m2_evaluate.py` | **独立归档为强信息、监督方法**；读精确 delta 并使用成员监督，不符合当前主线，不能以跨协议 AUC 判断优劣。 |
| 4 | Full-Delta：均值、UniformNet、Attention、TCN、Transformer；B2 对照 | `full_delta_mia.py` | **训练入口归档**；属于强信息实验。指标、划分等函数被当前代码大量引用，不能整文件直接删。 |
| 5 | Stat-Delta：S11/S18/S22 的 logistic、MLP、GBDT、triplet；分块统一/注意力/TCN/Transformer | `stat_delta_mia.py` | **训练矩阵归档**；需要真实 delta 和成员监督，保留必要对照和公共缓存读取。 |
| 6 | 历史固定 q 的 bit-only 检测：统计 logistic/MLP、bit-TCN/Transformer、重复查询 Transformer | `accept_only_mia.py` | **历史基线归档**。输入权限虽为 accept-only，但检测器训练/选择使用成员标签，与当前非成员训练不同；不作为当前候选。 |
| 7 | Low-q 多尺度接受评分：最低 q 的 10%/20%/50%，非成员标准化后取最大 | `active_importance_replay.py`, `adaptive_window_accept_only.py` | **保留为轻量基线**。同 B=2 历史协议 AUC 约 0.9022；无需把所有旧预算矩阵都留在默认流程。 |
| 8 | q 校正估计与主动 q 回放：联合似然、饱和区间、均匀/规则/重要性分配、oracle 上界 | `q_corrected_accept_only.py`, `active_importance_replay.py` | **保留回放及 q 校正基础工具；旧策略归档**。oracle 仅作诊断，不能作为可实施攻击。 |
| 9 | 非成员条件窗口、全局 surprise、与 low-q 融合、覆盖约束主动查询 | `adaptive_window_accept_only.py` | **删除候选实验分支**。K=2 的 learned-window AUC 0.8799，low-q 0.9022；没有支持替换轻量基线的证据。公共函数另行保留。 |
| 10 | EVT 尾部阈值、少样本校准扫描 | `adaptive_window_accept_only.py` | **EVT 移出默认配置，保留校准样本量研究**。例如 K=2、400参考/200校准时，EVT TPR 45.49%，但实际 FPR 1.86%，不能把名义 1% 的检出率当作严格 1% 的收益。 |
| 11 | 合成正 delta 神经证据、low-q 神经融合、神经/测量价值查询优先级 | `neural_adaptive_accept_only.py` | **历史训练/策略归档**。旧融合确有收益（B=2 AUC 0.9118），但合成成员替代训练不符合当前更严格约束，且条件 TCN 已有更强证据。仅非成员测量教师分支应单独标注，不能全部归类为合成监督。 |
| 12 | 可解释尺度门控：q-only、accept-aware、dense-max | `interpretable_scale_gate.py` | **学习门控为删除候选**。AUC 0.8285/0.8640，低于旧 max 基线 0.9018；保留简单固定尺度对照即可。 |
| 13 | 本地影子、pseudo、mixed、one-class 尺度门控 | `build_local_shadow_cache.py`, `shadow_scale_gate.py` | **影子训练链归档**。shadow AUC 0.9168，one-class 0.9114，不能说全都无效；但 shadow 需额外训练语言模型及影子 IN/OUT 标签，违背当前只用已微调好模型的选择。one-class 可保留历史对照，pseudo/mixed 为删除候选。 |
| 14 | Full-AI 查询分配：一次性、顺序、6/10 hybrid、uniform、oracle | `full_ai_query_allocation.py` | **高预算支线归档**。一次性 AI 在 K=8 有小幅正向证据（融合 pAUC +0.0048）；顺序/hybrid 无稳定增益。不能统称 AI 分配都无效，也不能直接与 B=2 主线按 AUC 排名。 |
| 15 | 影子尺度门控与主动评分组合 | `combine_shadow_active.py` | **归档组合入口**。有旧协议内增益，但继承影子训练及高查询预算，不纳入当前主线。 |
| 16 | 动态边际价值、闭环 water filling、上限约束分配 | `dynamic_marginal_query.py` | **优先删除候选实验分支**。K=2 动态融合 AUC 比均匀融合低 0.0024，95% 区间全负；K=8 同样退化，依赖影子训练。 |
| 17 | 非成员条件接受次数分布 TCN + 全局正向倾斜证据 | `conditional_accept_only.py` | **核心保留**。历史同协议 AUC 0.9118→0.9303；只用非成员训练/选检查点。它是后续改进共同基础。 |
| 18 | 条件模型的 span、固定残差融合、普通 NLL、q-only 变体 | `conditional_accept_only.py` | **span/融合退出默认路径，归档结果；q-only 与 NLL 保留诊断**。span/fusion AUC 0.9205/0.9168，弱于 global 0.9303。NLL 仍用于非成员训练与检查点选择，不能因其攻击 AUC 差而删除训练损失。 |
| 19 | 原始/截断前缀配对反事实 | `collect_counterfactual_accept_only.py`, `conditional_accept_only.py` 的 paired 分支 | **归档、停止默认采集**。同后缀 B=2 AUC 0.7686→0.7197；B=8 仍退化。只针对当前截断实现。 |
| 20 | 非成员潜在混合模型 + JS 主动选 q/位置 + 路径早停 | `active_protocol_design.py` | **实验策略归档**。同后缀 B=2 AUC 0.7123→0.6621，B=8 亦无优势；早停节省有限。潜在混合相关实现被优先实验动态导入，删除前核对函数依赖。 |
| 22 | 条件 TCN 加因果接受历史 | `priority_accept_only.py` 的 causal 分支 | **删除候选实验分支**。AUC 0.9303→0.9297，非成员验证 NLL 也没有改善。 |
| 23 | 三初始化 PMF 平均、不确定性折扣 | `priority_accept_only.py` 的 ensemble/discount 分支 | **删除候选实验分支**。AUC 约 0.9304，差值区间含零；增加拟合成本，当前折扣几乎不起作用。 |
| 24 | 草稿难度增强：预测熵、候选对数排名、top1–top2 logit 差 | `collect_draft_difficulty.py`, `priority_accept_only.py` | **核心保留，可关闭消融**。匹配 q 对照 AUC 0.9301→0.9342，六条件均提升；仅冻结草稿前向。 |
| 25 | 稀疏分散信号证据：固定稀疏比例 × 正向强度等权混合 | `priority_accept_only.py`, `combined_accept_only.py` | **核心保留，可关闭消融**。无需额外拟合/查询；组合实验 q-only AUC 0.9301→0.9342，难度特征+稀疏为 0.9353。 |
| 26 | 独立非成员校准扩充；统一、按长度、按难度分组 | `priority_accept_only.py`, `combined_accept_only.py` | **统一校准核心保留，分组作为可选项**。增强稀疏分数用 pooled1200 时 TPR/FPR=57.22%/1.00%；难度分组为54.60%/0.88%，不应默认开启。长度分组保留为低优先级消融。 |
| 27 | 预测熵分配第二次查询：一轮全覆盖，半数位置追加 | `priority_accept_only.py` 的 allocation 分支 | **删除候选实验分支**。等预算 AUC 0.9130→0.9131，区间含零，pAUC 略降。 |

`combined_accept_only.py` 的 2×2×3 共12配置，是17/24/25/26的组合验证，不额外算12个算法。建议保留整个已完成矩阵的结果及复现入口，但日常流程只运行需要的配置。

## 2. 外部基线与其他路径

`experiments/baseline/methods.py`、`run.py` 实现11个基线：**Loss、Min-K% Prob、Min-K%++、ReCaLL、ICP-MIA、PETAL、SEAD、WS、RS、BT、SaMIA**。建议整个独立基线套件保留，日常运行按 `--methods` 选择。它们服务于论文/实验比较，访问权限和成本不同于 accept-only，不能因本次主线精简就判为冗余。PETAL 等存在本仓库特定改编，保留原 README 的口径。

三种草稿路径 `drafts/plain.py`、`drafts/eagle3.py`、`drafts/mtp.py` 是模型准备/协议适配，不是三种检测评分；保留复现能力，当前实验不重训。`experiments/pretraining/` 是独立 Pythia/MIMIR 协议，保留，不与当前 SFT 指标混排。

以下属于公共基础设施，建议保留：

- 数据、划分、加载与溯源：`config.py`, `data.py`, `pools.py`, `splits.py`, `training.py`, `generalization.py`, `scoring_common.py`。
- 冻结概率采集/合并：`score_role.py`, `merge_role_logs.py`；`pq_gap_mia.py`、`logpq_distribution.py` 中的采集与一致性检查函数。
- 机制诊断：`token_signal_anatomy.py`, `q_only_position_anatomy.py`, `neural_residual_mechanism.py`, `logpq_distribution.py`；保留为诊断工具，按需运行。
- `aggregate_*`, `analyze_*`, `summarize_*` 和矩阵 runner 随对应方法一起保留或归档，不能将每个报告脚本算一个新方法。
- 历史 `results/` 下的 ABC、p/q sequence、unknown-p 报告和快照是复现资料；有些已无对应活跃入口，不因目录存在就算当前已维护算法。检查点目录内的 `.py` 可能属于模型加载代码，不是可随手清理的实验脚本。

## 3. 推荐精简后的日常矩阵

保留四个同输入、同预算的主要消融：

| 配置 | AUC | pAUC@10% | 用途 |
|---|---:|---:|---|
| q/位置 TCN + global | 0.9301 | 0.6803 | 条件模型基础对照 |
| q/位置 TCN + sparse | 0.9342 | 0.6999 | 较少特征的经济方案 |
| 难度特征 TCN + global | 0.9342 | 0.6973 | 难度特征增量对照 |
| 难度特征 TCN + sparse | 0.9353 | 0.7089 | 当前均值最好的组合候选 |

再加 low-q 轻量基线，以及按需开启的分组校准。当前建议探索性默认使用难度特征 TCN + sparse + pooled1200；组合超过单项的 AUC 差值区间包含零，不能宣称稳定最优。统一校准1200的平均 FPR=1%不保证每条件为1%。组合的400/800校准尚未验证，不能把旧评分的样本量结论直接迁移过来。

## 4. 为什么不能现在整文件删除

源码检查发现主线与旧实验存在公共函数耦合，例如：

- `conditional_accept_only.py` 从 `adaptive_window_accept_only.py` 导入划分抽样、指标、low-q 评分，从 `neural_adaptive_accept_only.py` 导入路径/JSON工具，从 `full_delta_mia.py` 导入划分。
- `priority_accept_only.py`、`combined_accept_only.py` 继续复用这些旧模块；特征采集依赖 `m1_features.py`。
- `active_importance_replay.py` 的公共回放读取又依赖 `token_signal_anatomy.py` 和 `stat_delta_mia.py`；这些模块进一步引用其他历史模块。
- 报告代码跨代复用配对抽样、指标和渲染函数；删除旧汇总模块也可能破坏新报告。

因此，建议执行顺序是：先把数据合同/划分、指标/校准、回放、路径/序列化提取为公共模块；迁移主线调用并验证冻结分数与划分不变；再将旧实验入口及专属测试移入归档。具体目录名可在真正重构时决定，本次没有创建这些公共模块。

优先清理9、12、16、22、23、27的无收益实验分支，其次归档19、20；11、13、15及2–6按训练/信息权限另放历史路径。保留报告、原始分数、划分、配置、源码版本和负结果，不应只留下表现好的结果。现有不少实现尚未纳入Git跟踪，归档前须确保有可恢复副本。

## 5. 证据来源与边界

- `results/sft_runs/directions_validation/FINDINGS_ZH.md`、`DIRECTIONS_REPORT.md`。
- `results/sft_runs/priority_validation/FINDINGS_ZH.md`、`PRIORITY_REPORT.md`。
- `results/sft_runs/combination_validation/FINDINGS_ZH.md`、`COMBINATION_REPORT.md`。
- `results/sft_runs/conditional_accept_only/COMPARISON_b2.md`。
- `results/sft_runs/accept_only_active_v2/` 下的窗口、EVT、neural、scale/shadow gate、full-AI、shadow组合、dynamic各自汇总及配对报告。
- M1、M2、Full-Delta、Stat-Delta、旧 accept-only 的实现与各自汇总；不将这些不同协议的数值合成总排名。

除另行说明外，当前主线是固定候选的本地验证器回放，不是自然串行远程接口。历史q与特征缓存q并非数值完全相同，引用增量仅限各自匹配对照。此处建议根据既有探索性结果与维护成本给出，没有新增训练或根据成员标签调整检测器参数。
