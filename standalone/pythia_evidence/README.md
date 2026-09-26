# Pythia 最小证据计算实验

目的：固定接受反馈和已训练 TCN，只检查条件校正强度、倾斜强度和聚合范围。
本目录是独立原型；`run.py` 不导入任何 `experiments.*` 模块，不修改旧代码、
检查点或报告。TCN 只做 CPU 前向，使用原检查点的标准化参数和分布核。
每次必须先复现旧分数、AUC、ROC 和校准指标，才能保存新旧对照。

首轮默认矩阵已完成，结果与解释见 [RESULTS.md](RESULTS.md)。

## 最小矩阵

- 固定模型：Pythia 6.9B / 1.4B；原始预训练模型对，非 SFT/KD。
- 默认领域：`github`（校正损失信号）、`wikipedia_(en)`（中等信号）、
  `dm_mathematics`（弱信号）。这三个领域来自已有诊断，不构成未经查看的独立验证。
- seed：1919、1949、1978，与原观测、TCN、数据划分一一对应。
- B=2，使用原文全部已采集位置，保留原始可变长度，不重新采样反馈。
- 每领域每 seed：400 成员、400 测试非成员，320/80/200 条原训练/验证/校准辅助非成员。
- 总计 9 个冻结条件 × 7 种评分 = **63 行结果**，语言模型查询数和 TCN 训练步数均为 0。

记 `ā = mean(C_t / 2)`，`μ̄ = mean(E_TCN[C_t] / 2)`。
正向倾斜证据为 `ell_t(eta) = eta*C_t - log sum_c pi_t(c)*exp(eta*c)`。
稀疏分支按 token 计算 `log(1-rho+rho*exp(ell_t))`，在文档内求和后对各
`(eta,rho)` 的分支做等权 log-mean-exp。全局分支使用 rho=1。

| 名称 | 固定配置 | 用途 |
|---|---|---|
| `main_fixed_sparse_positive` | eta={0.5,1,2}，rho={0.05,0.1,0.25} | 原方法；保存原报告的原始分数 |
| `accept_rate` | ā，lambda=0 | 无校正基线 |
| `partial_residual_050` | ā−0.5μ̄ | 部分校正 |
| `full_residual` | ā−μ̄，lambda=1 | 完全校正对照 |
| `weak_sparse_positive` | eta={0.05,0.1,0.2}，原 rho | 只减弱倾斜 |
| `strong_dense_positive` | 原 eta，rho=1 | 只改全局聚合 |
| `weak_dense_positive` | eta={0.05,0.1,0.2}，rho=1 | 弱倾斜与全局聚合组合 |

加入中间两个交叉对照是为了避免同时改变 eta 和 rho 后无法归因。
第一轮固定正向，不加入双侧、窗口/HMM、特征删减或重训，避免扩大选择空间。
方法清单和超参数写死在脚本中，无“选择最佳配置”开关；所有方法均完整报告。

## 如何判断两个假设

1. 对比 `accept_rate` / `partial_residual_050` / `full_residual`：
   如果部分校正优于完全校正，支持保留部分难度信息；若仍不如原始接受率，说明
   这套条件校正未带来净收益。单凭测试结果不能确定正式部署的最优 lambda。
2. 对比四种 tilt 配置：`weak_sparse` 对旧方法、`weak_dense` 对 `strong_dense`
   检查减弱倾斜；`strong_dense` 对旧方法、`weak_dense` 对 `weak_sparse`
   检查扩大信号覆盖范围。全局弱倾斜没有改善时，不能归因为 TCN 必然无用。
3. 所有改进同时与旧方法和简单接受率比较，避免只超过较弱的校正基线。
   数学领域用于观察评分改动能否利用弱反馈，不预设必须成功。

## 运行

无需安装新依赖，复用当前环境里的 NumPy、SciPy 和 PyTorch。
默认 CPU 单线程，条件顺序运行；不启动 GPU 任务、不调用 Transformers。
工作树可读取原工作区的缓存，结果写到本工作树的独立 artifact 目录。

```bash
cd /home/mxd/.codex/worktrees/dp-throughput/SD_MIA
export PYTHIA_EVIDENCE_PYTHON=/home/mxd/lib/SD_MIA/.venv/bin/python

# 只打印矩阵和缺失报告；不创建输出目录、不加载 TCN。
"$PYTHIA_EVIDENCE_PYTHON" -B standalone/pythia_evidence/run.py dry-run \
  --input-root /home/mxd/lib/SD_MIA/artifacts/audits/pythia_mimir_v1

# 完成最小矩阵；同命令可校验来源后复用已完成条件。
CUDA_VISIBLE_DEVICES='' "$PYTHIA_EVIDENCE_PYTHON" -B standalone/pythia_evidence/run.py run \
  --input-root /home/mxd/lib/SD_MIA/artifacts/audits/pythia_mimir_v1

# 自选领域/seed 时使用新的输出根目录。
CUDA_VISIBLE_DEVICES='' "$PYTHIA_EVIDENCE_PYTHON" -B standalone/pythia_evidence/run.py run \
  --input-root /home/mxd/lib/SD_MIA/artifacts/audits/pythia_mimir_v1 \
  --sources github 'wikipedia_(en)' --seeds 1919 \
  --output-root artifacts/audits/pythia_evidence_subset_v1

# 对已有输出重新汇总，不重算 TCN。
"$PYTHIA_EVIDENCE_PYTHON" -B standalone/pythia_evidence/run.py summarize
```

`--sources all` 可扩展到原有 8 个来源（含单独的 full_pile）；
`--threads` 默认 1，`--bootstrap` 默认 200。改变矩阵、代码、线程数、bootstrap
或输入文件须使用新的 `--output-root`，不能混入旧结果。输出不得位于输入树内。

## 输出与统计口径

默认：`artifacts/audits/pythia_evidence_minimal_v1/`。

- `PLAN.json`：固定方法、矩阵、实现哈希、依赖版本与 CPU 设置。
- `conditions/<source>/seed<seed>/REQUEST.json`：旧报告、观测、权重、分数、
  分区、数据 manifest 与冻结 records 的 SHA-256。
- 同条件的 `scores.npz`：旧划分中的同一批测试/校准文档 ID、标签、索引与 7 组分数。
- `REPORT.json`：旧分数回放误差、全部指标和相对旧方法的配对 bootstrap 区间。
- `_COMPLETE.json`：完成文件哈希；缺失标记的中断条件可重算，已完成条件严格校验后复用。
- `SUMMARY.md` / `.csv` / `.json`：逐领域的三 seed 均值和样本标准差、逐 seed 结果；
  不将 full_pile 当作领域宏平均，不输出自动挑选出的“最好方法”。

主指标：AUC 及相对旧方法的 ΔAUC。次指标：测试 ROC TPR@1%/10% FPR，
以及使用原 200 条独立非成员校准后的 TPR 和实际 FPR。ROC 指标和部署阈值指标分开报告。
所有方法使用相同测试文档的分层 bootstrap 重采样，生成每个 seed 的配对 ΔAUC 区间。
该区间没有重新拟合 TCN 或校准集，也没有做多重比较校正。

评分函数不接收标签；标签仅用于核对冻结来源和计算评估指标。
校准集只用于每种固定文档评分的阈值计算，不选择 lambda、eta、rho 或评分方向。
这里是模型化评分，不能因指数形式就称为严格 e-value。

当前测试集已经被用于诊断，三 seed 的文档还可能重叠；本轮是探索性机制对照，
不能把其中最好的配置当作经过独立验证的新方法。后续选择和确认需独立数据。
单领域 400 条测试非成员在 1% FPR 下仅约 4 个误报，200 校准样本的 p 值步长
为 1/201，低 FPR 结果需结合 AUC、10% FPR 与三个 seed 一起判断。

## 本地检查

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHIA_EVIDENCE_PYTHON" -B -m pytest -q \
  standalone/pythia_evidence/test_run.py -p no:cacheprovider
```

小型检查覆盖评分公式、并列分数、独立校准、配对区间、输入只读、来源校验、
旧分数不一致时中止、完成条件复用以及 dry-run/数据集选择。
