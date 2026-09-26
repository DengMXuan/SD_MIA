# Pythia 原始 delta 的独立验证

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
