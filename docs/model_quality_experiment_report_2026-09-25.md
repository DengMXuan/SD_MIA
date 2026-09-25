# 五对微调模型的续写泛化性与 KD 草稿接受率实验报告

日期：2026-09-25  
结果批次：[`model_quality_v2`](../artifacts/evaluations/model_quality_v2/reports/SUMMARY.json)

## 摘要

本实验评估五对已有模型在三个数据集、三个条件 seed 下的 epoch 1 目标模型续写质量，以及目标模型与对应辅助数据 KD 草稿的分布匹配。45 个条件的泛化性与接受率任务共 90 项，全部完成；只读 `status` 复核为 90/90 `complete`。

- **留出样本续写质量总体提高。** 相对相同记录上的未微调基座，45 个条件的 nonmember BLEU-4 和 ROUGE-L 点估计全部提高；ROUGE-1 有 43 个提高、2 个下降。逐条件配对 bootstrap 的 95% 区间确认提升的条件数分别为 43、38、44。唯一在三 seed 均值上下降的模型—数据集指标是 Llama 3.1 EAGLE-3 / NewsTection 的 nonmember ROUGE-1（−0.004）；三个 seed 的对应区间均跨零。
- **成员差距不能概括为全部很小。** 以任一续写指标的 `|member − nonmember| ≥ 0.03` 为提示条件，9/45 个条件触发提示：Gemma4 3 个、Llama 3.1 EAGLE-3 4 个、Qwen3.5 MTP 2 个。其余 36 个通过该提示门限；“通过”不构成统计等价或无过拟合证明。
- **留出数据上的 KD 草稿分布重叠率为 0.623–0.825。** 这里的“接受率”是原文前缀下 `Σ min(p, q)` 的文档等权均值，不是实际投机解码轨迹的接受频率或加速比。三个 seed 汇总后，15 个模型—数据集组合的 KD auxiliary 均值都高于 nonmember；因此包含 auxiliary 的 overall 不能代表纯留出表现。

## 实验范围与口径

| 项目 | 设置 |
|---|---|
| 模型对 | Qwen3 8B + 1.7B 普通草稿、Gemma4 12B + E2B 普通草稿、Qwen3 8B + EAGLE-3、Llama 3.1 8B + EAGLE-3、Qwen3.5 9B + MTP。模型及修订号见[模型注册表](../experiments/shared/models/model_pairs.json)。 |
| 数据集与 seed | WikiTection、NewsTection、ArxivTection；每个数据集使用 1919、1949、1978 三个条件 seed；仅评估 epoch 1 的目标及相应 KD 草稿。 |
| 续写泛化性 | 每条件从冻结划分抽取 500 member + 500 nonmember。目标 tokenizer 下，长文档用前 256 token 作上下文、后 128 token 作参考；短文档按约 2:1 切分。基座与微调目标在相同记录上贪心续写，以 BLEU-4、ROUGE-1、ROUGE-L 计分。 |
| KD 草稿接受率 | 每条件抽取 256 member + 256 nonmember + 256 KD auxiliary。原文前缀下计算目标分布 `p` 与草稿分布 `q` 的 `Σᵥ min(p(v), q(v)) = 1 − TV(p, q)`，另记 top-1 一致率。计入响应末尾 EOS，不计提示位置；先对文档内位置求均值，再对文档等权平均。 |
| 不确定性 | 每条件使用对应 seed 做 1,000 次文档 bootstrap；基座与微调模型的质量差采用配对重采样。下表是三个 seed 的等权均值；跨 seed 样本标准差见[汇总表](../artifacts/evaluations/model_quality_v2/reports/seeds.csv)，不是置信区间。逐条件 95% 区间见[明细表](../artifacts/evaluations/model_quality_v2/reports/conditions.csv)。 |
| 推理与校验 | 正式推理为 BF16；45 项接受率任务的适配器因果性与数值检查均通过，其中 27 项执行并通过 FP32 前缀复核，18 项无需复核。FP32 只用于校验，不替换正式指标。 |

完整的抽样、短文档、词表映射及校验规则见[评估方案](model_asset_evaluation.md)。以下“留出”专指冻结划分中的 nonmember；它并非独立任务能力基准。

## 结果一：目标模型的留出续写质量

表中质量分数是微调目标在 nonmember 上的三 seed 均值；`Δ基座 = 微调目标 − 未微调基座`，正数表示在相同留出记录上续写指标提高。各模型使用其目标 tokenizer，跨模型绝对分数应谨慎比较。

| 模型对 | 数据集 | BLEU-4 | Δ基座 | ROUGE-1 | Δ基座 | ROUGE-L | Δ基座 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 普通草稿 | Wiki | 0.142 | +0.038 | 0.328 | +0.028 | 0.249 | +0.029 |
| Qwen3 普通草稿 | News | 0.067 | +0.006 | 0.320 | +0.008 | 0.204 | +0.008 |
| Qwen3 普通草稿 | Arxiv | 0.092 | +0.010 | 0.334 | +0.017 | 0.219 | +0.011 |
| Gemma4 普通草稿 | Wiki | 0.170 | +0.075 | 0.366 | +0.086 | 0.277 | +0.071 |
| Gemma4 普通草稿 | News | 0.071 | +0.021 | 0.319 | +0.050 | 0.206 | +0.024 |
| Gemma4 普通草稿 | Arxiv | 0.084 | +0.021 | 0.323 | +0.068 | 0.209 | +0.032 |
| Qwen3 EAGLE-3 | Wiki | 0.141 | +0.046 | 0.325 | +0.027 | 0.246 | +0.037 |
| Qwen3 EAGLE-3 | News | 0.063 | +0.015 | 0.316 | +0.009 | 0.201 | +0.017 |
| Qwen3 EAGLE-3 | Arxiv | 0.087 | +0.020 | 0.325 | +0.016 | 0.213 | +0.018 |
| Llama 3.1 EAGLE-3 | Wiki | 0.161 | +0.049 | 0.342 | +0.026 | 0.261 | +0.044 |
| Llama 3.1 EAGLE-3 | News | 0.073 | +0.012 | 0.328 | −0.004 | 0.206 | +0.009 |
| Llama 3.1 EAGLE-3 | Arxiv | 0.089 | +0.018 | 0.332 | +0.012 | 0.214 | +0.013 |
| Qwen3.5 MTP | Wiki | 0.164 | +0.054 | 0.358 | +0.048 | 0.273 | +0.048 |
| Qwen3.5 MTP | News | 0.067 | +0.012 | 0.313 | +0.010 | 0.204 | +0.015 |
| Qwen3.5 MTP | Arxiv | 0.094 | +0.008 | 0.339 | +0.010 | 0.220 | +0.009 |

Gemma4 的留出质量提升在三个数据集上均较大；例如 Wiki 的 ROUGE-1 提升 0.086，Arxiv 提升 0.068。这说明相对各自基座，微调确实改变并改善了本实验所测的文档续写表现。Llama 3.1 EAGLE-3 / News 的 ROUGE-1 三 seed 平均下降 0.004，但三个逐条件差值区间均跨零，不能据此判定稳定退化。

## 结果二：member 与 nonmember 的续写差距

表中 `差距 = member − nonmember`，为三个 seed 的等权均值。最后一列列出逐条件至少一个指标达到 0.03 提示门限的 seed；**门限逐 seed 判定，不能用表中的三 seed 均值替代**。

| 模型对 | 数据集 | BLEU-4 差距 | ROUGE-1 差距 | ROUGE-L 差距 | 提示 seed |
|---|---|---:|---:|---:|---|
| Qwen3 普通草稿 | Wiki | +0.009 | +0.008 | +0.010 | — |
| Qwen3 普通草稿 | News | +0.003 | −0.001 | +0.001 | — |
| Qwen3 普通草稿 | Arxiv | +0.001 | +0.004 | +0.002 | — |
| Gemma4 普通草稿 | Wiki | +0.029 | +0.029 | +0.029 | 1949、1978 |
| Gemma4 普通草稿 | News | +0.025 | +0.025 | +0.025 | 1978 |
| Gemma4 普通草稿 | Arxiv | +0.004 | +0.009 | +0.006 | — |
| Qwen3 EAGLE-3 | Wiki | +0.011 | +0.010 | +0.012 | — |
| Qwen3 EAGLE-3 | News | +0.007 | +0.003 | +0.005 | — |
| Qwen3 EAGLE-3 | Arxiv | +0.004 | +0.009 | +0.004 | — |
| Llama 3.1 EAGLE-3 | Wiki | +0.035 | +0.039 | +0.039 | 1919、1949、1978 |
| Llama 3.1 EAGLE-3 | News | +0.020 | +0.023 | +0.020 | 1978 |
| Llama 3.1 EAGLE-3 | Arxiv | +0.006 | +0.010 | +0.009 | — |
| Qwen3.5 MTP | Wiki | +0.031 | +0.034 | +0.034 | 1949、1978 |
| Qwen3.5 MTP | News | +0.018 | +0.024 | +0.018 | — |
| Qwen3.5 MTP | Arxiv | +0.010 | +0.017 | +0.011 | — |

差距最持续的是 Llama 3.1 EAGLE-3 / Wiki：三个 seed 全部触发提示，三 seed 平均 ROUGE-L 差距为 +0.039。提示主要集中在 Wiki（7/9 个失败条件），另有 News 的 2 个；Arxiv 没有条件触发 0.03 门限。按逐条件 95% bootstrap 区间，下界大于零的数量为 BLEU-4 17/45、ROUGE-1 21/45、ROUGE-L 21/45；这些是逐条件描述，未做多重比较校正或跨 seed 显著性检验。

## 结果三：KD 草稿的条件分布接受概率

表中前四列是 `Σ min(p, q)` 的三 seed 均值；`overall` 对 member、nonmember、KD auxiliary 三类等量混合。最后一列为 nonmember 的 top-1 一致率。**比较留出匹配质量应看 nonmember 列。**

| 模型对 | 数据集 | member | nonmember | KD auxiliary | overall | nonmember top-1 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3 普通草稿 | Wiki | 0.818 | 0.817 | 0.828 | 0.821 | 0.805 |
| Qwen3 普通草稿 | News | 0.758 | 0.763 | 0.778 | 0.767 | 0.740 |
| Qwen3 普通草稿 | Arxiv | 0.811 | 0.810 | 0.818 | 0.813 | 0.793 |
| Gemma4 普通草稿 | Wiki | 0.814 | 0.825 | 0.841 | 0.826 | 0.812 |
| Gemma4 普通草稿 | News | 0.751 | 0.777 | 0.798 | 0.775 | 0.760 |
| Gemma4 普通草稿 | Arxiv | 0.797 | 0.798 | 0.811 | 0.802 | 0.782 |
| Qwen3 EAGLE-3 | Wiki | 0.669 | 0.668 | 0.674 | 0.670 | 0.660 |
| Qwen3 EAGLE-3 | News | 0.621 | 0.623 | 0.634 | 0.626 | 0.607 |
| Qwen3 EAGLE-3 | Arxiv | 0.681 | 0.679 | 0.689 | 0.683 | 0.671 |
| Llama 3.1 EAGLE-3 | Wiki | 0.681 | 0.680 | 0.693 | 0.684 | 0.676 |
| Llama 3.1 EAGLE-3 | News | 0.643 | 0.646 | 0.662 | 0.651 | 0.633 |
| Llama 3.1 EAGLE-3 | Arxiv | 0.701 | 0.699 | 0.710 | 0.703 | 0.688 |
| Qwen3.5 MTP | Wiki | 0.828 | 0.824 | 0.830 | 0.827 | 0.814 |
| Qwen3.5 MTP | News | 0.763 | 0.764 | 0.773 | 0.767 | 0.743 |
| Qwen3.5 MTP | Arxiv | 0.815 | 0.811 | 0.816 | 0.814 | 0.797 |

News 是五对模型中 nonmember 接受概率最低的数据集。按模型—数据集三 seed 均值，KD auxiliary 高于 nonmember 的差值为 0.005–0.022；回到 45 个单 seed 条件，43 个为正，2 个略为负，因此这种差异不是逐条件无例外的规律。KD auxiliary 参与过草稿训练，较高分布重叠符合其数据角色，但该描述本身不证明具体机制。member 也并非总比 nonmember 高，例如 Gemma4 / News 为 0.751 对 0.777；不能仅凭本表把接受概率当作成员身份判据。

两组 EAGLE-3 的 nonmember 接受概率约为 0.623–0.699，低于本矩阵中普通草稿和 MTP 的 0.763–0.825。EAGLE-3 的原文 token 词表覆盖率在三 seed 汇总中约为 0.914–0.942，而其余三组接近 1.0；评估没有删除词表外位置。因此词表支持集是解释低值时必须考虑的因素。目标、草稿、适配器与 tokenizer 也不同，不能把跨组差额直接归因为 EAGLE-3 结构或换算为部署速度。

## 结论与解释边界

1. 在本实验的文档续写口径下，五组 epoch 1 微调目标大体改善了留出记录的质量；这支持“对当前数据分布有续写泛化收益”的结论，不等于通用能力提升。
2. 成员优势具有模型和数据集依赖性。尤其 Llama 3.1 EAGLE-3 / Wiki 的持续正差距，以及 Gemma4、Qwen3.5 MTP 在部分 seed 的提示结果，应在后续成员审计讨论中单独呈现。0.03 仅是预设提示阈值，不能据其通过或失败单独判定泄漏。
3. KD 草稿在留出原文前缀上与目标有明显分布重叠，但本实验没有运行完整投机解码，也没有测吞吐、延迟或真实轨迹接受频率。top-1 一致率和理论接受概率是不同指标。

## 数据来源与复核

- 本报告仅使用重新计算的 `model_quality_v2` 批次；旧 `model_quality_v1` 已归档，未并入这些数值。机器汇总见 [`SUMMARY.json`](../artifacts/evaluations/model_quality_v2/reports/SUMMARY.json)、[逐条件指标及区间](../artifacts/evaluations/model_quality_v2/reports/conditions.csv)、[跨 seed 均值及标准差](../artifacts/evaluations/model_quality_v2/reports/seeds.csv)。每个条件的 `REPORT.json`、`REPORT.md`、`SAMPLES.json` 和逐条分数保存在该批次的 `tasks/<model_pair>/<dataset>/epoch1/seed<seed>/<evaluation>/` 下。
- 已执行只读 `.venv/bin/python -m experiments.model_quality.cli status`：45 个条件、90 项任务，全部 `complete`。45 份泛化性抽样各有 500 member + 500 nonmember；45 份接受率抽样各有 256 member + 256 nonmember + 256 KD auxiliary。`status` 按当前源码、资产清单和输出校验和复核已有报告；本报告没有重新运行 GPU 推理。
- 三 seed 标准差反映条件间波动，不是三 seed 均值的 95% 置信区间。逐条件 bootstrap 区间只针对当次抽样；数据集之间、不同目标 tokenizer 之间的直接排序，以及成员审计风险推断，都超出本报告的直接测量范围。
