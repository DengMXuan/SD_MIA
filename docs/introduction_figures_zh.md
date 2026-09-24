# Introduction 关键洞察图

## 实验条件与数据来源

- 目标模型：Qwen3-8B-Base，WikiTection，3 epoch SFT。
- 草稿模型：Qwen3-1.7B-Base，辅助数据蒸馏草稿。
- 来源：`artifacts/archive/sft_runs/pq_directional/wikitection_epoch3/` 下真实模型产生的逐 token 概率缓存。
- 使用现有 `full_delta_protocol.json` 中冻结的 D 诊断划分：800 条成员、800 条非成员；不使用 V/C/T 划分。
- 对同一记录、同一前缀、同一候选 token 配对目标概率 p 与草稿概率 q。加载时检查记录 ID、标签、长度和模型来源一致。
- 按历史缓存生产器约定移除每条记录末尾追加的 EOS，保留原始 response token；移除后共 768,110 个 token。

**这是历史缓存上的机制分析，不是当前 `controlled_sft_v2` 审计矩阵的新结果。** 当前主线接受反馈缓存与这份历史 p/q 缓存的记录划分不同，本次没有混用两者。

## 图 (a)：联合概率密度

- 横轴：目标模型 log p；纵轴：草稿模型 log q。对数为自然对数。
- 橙色实线：成员；蓝色虚线：非成员。
- 灰色点线：p=q。其下方 p>q，理论接受概率饱和为 1；其上方 p<q，理论接受概率为 p/q。
- 使用 Seaborn `kdeplot` 绘制二维密度等高线。
- 每篇文档固定随机抽取 32 个 token；不足 32 个时全部使用。每个样本权重为该文档抽样数量的倒数，使文档等权。
- 本次每类 KDE 样本数均为 25,600；随机种子固定为 20260922。
- 两类分别拟合 KDE，使用相同带宽调整系数 1.0、相同 levels `[0.05, 0.2, 0.5, 0.8]`、相同绘图网格和坐标范围。
- Seaborn 的这些等高线等级是在截取后的评估网格上计算的 iso-proportion levels；不是置信区间，也不是分类边界。
- 坐标显示范围为 [-16, 0]。按文档等权计算，视域之外的真实联合概率样本占比为成员约 0.089%、非成员约 0.154%。KDE 抽样发生在视域截取之前。
- KDE 在 log 概率的上边界 0 附近可能存在平滑边界偏差，因此图用于展示分布形态，不从边界轮廓估计精确概率质量。

## 图 (b)：给定草稿概率的理论接受概率

**纵轴明确为理论接受概率，不是实测接受率。** 每个 token 的值由真实缓存计算：

$$
\alpha_i=\min(1,p_i/q_i)=\exp\{\min(0,\log p_i-\log q_i)\}.
$$

- 横轴 log q，区间 [-16, 0]，两类使用完全相同的 12 个等宽分箱；点放在箱的中点。
- 每篇文档先对箱内 token 的 alpha 求均值，再在有该箱 token 的文档之间等权平均。
- 这一定义避免长文档支配结果；不同箱的参与文档集合可以不同，样本量保存在 CSV 中。
- 置信区间：在成员和非成员内部，分别按文档进行 2,000 次有放回重采样；每次重采样对所有箱共用同一批文档。
- 阴影表示逐箱的 95% percentile bootstrap 区间；不是全曲线同时置信带，不包含跨模型训练种子的波动。
- 只有至少 30 条文档支持的箱才绘制；本次所有箱均满足。
- 不对不足支持的箱插值或跨空箱连线。
- 纵轴根据置信区间范围留出边距，本次显示约 80%–101%，刻度明确标为百分比，未从零起始。
- 区间外的 log q token 不计入分箱统计；按文档等权计算，其占比为成员约 0.086%、非成员约 0.107%。

这张图用于说明成员身份与条件接受概率的关联。它不是自然 SD 生成轨迹的测量，也不能独立证明所有文本难度因素已被控制。

## 文件与复现

绘图脚本：`experiments/figures/intro_membership_insight.py`。

输出目录：`artifacts/figures/introduction/qwen3_wikitection_epoch3_auxiliary/`。

| 文件 | 内容 |
| --- | --- |
| `joint_logprob_density.pdf/png/svg` | 独立图 (a) |
| `conditional_acceptance.pdf/png/svg` | 独立图 (b) |
| `intro_membership_insight.pdf/png/svg` | 两个面板的并排组合图 |
| `conditional_acceptance.csv` | 逐类逐箱均值、置信区间、文档和 token 数 |
| `document_bin_statistics.csv` | 每篇文档在每个箱内的统计量 |
| `density_samples.csv.gz` | KDE 实际使用的样本、权重与记录位置 |
| `provenance.json` | 来源文件 SHA-256、参数、软件版本、选取记录及截取比例 |

PDF 为嵌入字体的矢量图；PNG 为 400 DPI；SVG 保留文字以便编辑。所有图使用一致的配色、线型和衬线字体。

当前机器的系统 Python 环境已安装 Seaborn 0.13.2；项目 `.venv` 尚未安装 Seaborn。本次使用已有环境生成，无需运行模型或修改项目依赖。

从仓库根目录复现：

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 \
  python experiments/figures/intro_membership_insight.py
```

在其他环境使用时需安装 NumPy、Pandas、SciPy、Matplotlib 和 Seaborn（0.13 系列）。图中的实际库版本记录在 `provenance.json`。

## 建议英文图注

> **Membership-related structure in target–draft probabilities.** We analyze a frozen diagnostic subset of WikiTection containing 800 member and 800 non-member records, using a Qwen3-8B target after three SFT epochs and a Qwen3-1.7B auxiliary-distilled draft. (a) Document-balanced kernel density contours of the target and draft log-probabilities for the same response tokens under their original prefixes. The dotted diagonal indicates p=q. (b) Mean theoretical acceptance probability, alpha=min(1,p/q), conditional on shared bins of draft log-probability. Token probabilities are first averaged within each document and bin, and then across contributing documents. Shaded regions show pointwise 95% confidence intervals from 2,000 document-level bootstrap replicates. Appended EOS tokens are excluded. These results use historical teacher-forced caches; panel (b) reports theoretical probabilities rather than measured acceptance frequencies.

建议在正文中将图描述为机制证据：在该实验条件下，成员与非成员的相对预测关系存在差异，并表现为给定草稿概率时不同的理论接受概率。精确 p 仅用于机制分析，不是 accept-only 检测器的输入。

## NewsTection：3 epoch、辅助蒸馏草稿

同一绘图脚本也支持 NewsTection，保持上述 KDE、分箱、文档等权及 bootstrap 设置，输出至独立目录，不覆盖 WikiTection 图。

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 \
  python experiments/figures/intro_membership_insight.py --benchmark newstection
```

- 来源：`artifacts/archive/sft_runs/pq_directional/newstection_epoch3/`。
- 划分：该条件冻结的 D 诊断划分，成员和非成员各 800 条。
- 去除末尾追加的 EOS 后：成员 401,154 个 token，非成员 399,295 个 token。
- KDE：每类 25,600 个抽样 token；log p、log q 仍显示 [-16, 0]，条件统计仍使用 12 个相同边界的分箱。
- 按文档等权计算，联合坐标视域外的样本占比分别约为 0.086% 和 0.131%。
- 接受概率图仍为理论值，纵轴下限按照同一规则从该数据的置信区间范围确定；比较两个数据集时需注意纵轴刻度。
- 输出目录：`artifacts/figures/introduction/qwen3_newstection_epoch3_auxiliary/`，文件名和格式与 WikiTection 相同。

该组图同样属于历史缓存上的机制分析，不是当前审计矩阵的实测接受率。
