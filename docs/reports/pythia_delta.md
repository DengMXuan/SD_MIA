# Pythia 概率差诊断：协议与历史结果

## Pythia 原始 delta 的独立验证

目的：验证成员与非成员的 `delta_i = log p_i - log q_i` 分布是否存在差异，并区分：

1. 原始模型概率中的成员信息；
2. `alpha_i = exp(min(delta_i, 0))` 截断后留下的信息；
3. 已保存的 B=2 接受计数中的信息；
4. 旧主方法对同一批文档的排序能力。

这是机制诊断。精确目标概率不能输入 accept-only 攻击器。脚本完全独立，不导入或修改 `experiments`，不修改既有模型、数据、概率缓存、TCN 或实验结果。

## 本次范围

- 固定模型：Pythia 6.9B / 1.4B，使用原实验 manifest 指定的版本。
- 领域：GitHub、Wikipedia、DM Mathematics，代表之前信号较强、较弱、几乎无接受率差距的情况。
- 固定数据划分 seed=1919；每个领域从原测试集确定性抽取 64 成员 + 64 非成员。
- 子样本选择种子 20260927。先冻结样本再运行模型；不按结果筛选文档。
- 每份 MIMIR 样本最多 512 token，首 token 仅作上下文，实际评分长度为 L−1。这里的“文档”指冻结的文本片段。
- GPU 全被占用，本次在 CPU 上以低优先级运行。重新计算 p 和 q，均为 FP32；原 B=2 缓存来自 GPU BF16，所以与旧接受计数的比较不是逐位复现。
- 一份小规模、单 seed 的探索性验证，不能外推为全部 Pythia 领域或三 seed 确证。

## 运行

在当前独立工作树根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/pythia-delta-mpl \
nice -n 15 /home/mxd/lib/SD_MIA/.venv/bin/python -u \
  standalone/pythia_delta/verify.py --threads 16
```

模型和 tokenizer 只从本地已缓存版本读取；不会下载模型。默认结果写入工作树的 `artifacts/audits/pythia_delta_pilot_v1/`。

支持命令行指定领域、seed 和样本量。例如在有空闲计算资源后，扩展为完整测试集：

```bash
python standalone/pythia_delta/verify.py \
  --datasets github 'wikipedia_(en)' dm_mathematics arxiv hackernews pile_cc pubmed_central full_pile \
  --seeds 1919 1949 1978 --per-class 0 \
  --device cuda:0 --dtype bfloat16 \
  --output artifacts/audits/pythia_delta_full_bf16_v1
```

`--per-class 0` 表示读取全部冻结测试文档；每个领域/seed 分别分析，不把 seed 间重复文档合并为独立样本。改变实验计划时使用新输出目录；已有目录会拒绝不同的计划。

`--stage collect` 只收集概率；`--stage analyze` 仅分析已完成缓存。重复运行可恢复逐文档缓存；损坏缓存或 token/模型/数据不一致会报错。输出目录锁防止两个进程同时写同一结果。

## 统计定义

每篇文档先计算均值、绝对值均值、标准差、分位数、负 delta 比例、正/负 delta 质量、理论接受率、旧 B=2 接受率和旧主方法分数，然后比较两类文档。

- 均值差 = 成员均值 − 非成员均值。Cohen's d 使用文档统计量的组内合并标准差。
- AUC 固定为“分数越大越像成员”，不根据测试标签自动翻转方向。AUC < 0.5 表示反向关联，不意味着没有区别。
- 95% 区间为类别内按文档 bootstrap 2,000 次；它们是逐指标区间，次要指标仅作探索。
- 文档等权 delta CDF：每篇文档先形成经验 CDF，再在类别内平均。主检验使用 [-2,2] 上 161 个固定网格点的最大类间差，对整篇文档的标签置换 2,000 次；对所选领域/seed 条件进行 Holm 多重校正。网格检验并不覆盖全部可能的尾部差异。
- 另存文档级各统计量的 KS 结果，供诊断，未对这些次要指标作多重检验校正。
- 平均 delta 的组间差异、token 分布形状差异、文档排序 AUC 是不同问题，应结合解释。
- 当前测试集此前已被查看；结果不是新方法的独立泛化验证。未显著不等于证明同分布。
- p、q 为同一实际文本 token、同一前缀上的概率。它们不是从 q 采样的 token，因此正平均 delta 不违反概率归一化。

## 输出

- `PLAN.json`：模型版本、token 约定、原始文件哈希和固定抽样 ID。
- `RUNTIME.json`：推理精度、设备、线程数、代码哈希、时间。
- `logps/{target,draft}/*.npz`：实际 token ID 和逐 token log 概率，支持恢复。
- `documents.csv`：每篇文档的全部统计量，可独立复算。
- `REPORT.json` / `REPORT.md`：统计结果及解释范围。
- `delta_distributions.png/pdf`：文档等权 token delta CDF、文档平均 delta 的 ECDF、理论接受率 ECDF。
- `_COMPLETE.json`：最终报告、图和表的哈希，仅在分析完成后生成。

统计小检查已验证 AUC 正反方向和 ties、Holm 校正、delta 截断恒等式及常数差异的文档 bootstrap。真实推理同时检查数据文件哈希、tokenizer 哈希、测试分区、文档 ID/标签、评分长度和概率有限性。

实测结论和精度限制见同目录 `RESULTS.md`。`check_precision.py` 额外核对旧 GPU BF16 与本次 CPU 概率的差异，结果在 `PRECISION.json`；它不验证原 GPU 环境的完整 delta。

## Pythia 文档级 delta 验证结果

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
