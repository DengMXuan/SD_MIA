# Pythia + MIMIR / Qwen + 时间切分：七个基线

独立入口，旧脚本和旧结果保留。只调度 `loss`、`min_k_prob`、`min_k_pp`、`sead`、`petal`、`recall`、`icp_mia`，不调度 SaMIA、RS、BT、WS。

默认 seed 为 **1919、1949、1978**，数据选择、辅助划分、方法随机数和 AUC bootstrap 均按条件 seed 一一对应。固定加载 manifest 中的预训练 target；不加载 draft、不微调语言模型。算法复用仓库现有 target-only 实现：SEAD 使用 50 次采样的频率估计，PETAL 使用目标词嵌入相似度，ICP 使用目标词嵌入检索。这些是此前七基线的同一适配版本，详见 `experiments/baseline/README.md`。

## 运行

```bash
cd /home/mxd/.codex/worktrees/dp-throughput/SD_MIA

# Pythia：默认全部 8 个领域 × 3 seeds = 24 个 GPU 条件，168 个方法结果
bash standalone/pretraining_baselines/run.sh mimir dry-run --gpus 0 1
bash standalone/pretraining_baselines/run.sh mimir run --gpus 0 1

# 只选择部分数据集/seed；括号所在的领域名须加引号
bash standalone/pretraining_baselines/run.sh mimir run \
  --sources github 'wikipedia_(en)' dm_mathematics --seeds 1919 1949 1978 --gpus 0 1

# Qwen：默认清理 + 长度匹配版本，3 seeds = 3 个 GPU 条件，21 个方法结果
bash standalone/pretraining_baselines/run.sh temporal dry-run --gpus 0 1
bash standalone/pretraining_baselines/run.sh temporal run --gpus 0 1

# 可同时对比只清理和清理 + 长度匹配两种数据版本
bash standalone/pretraining_baselines/run.sh temporal run \
  --variants clean_only length_matched --gpus 0 1

# 可随时重新汇总，不启动模型
bash standalone/pretraining_baselines/run.sh mimir summarize
bash standalone/pretraining_baselines/run.sh temporal summarize
```

请将示例中的 GPU 编号替换为可用设备。每个 GPU 一次运行一个“数据集/清理版本 × seed”条件，内部依次运行七个方法；该条件结束后自动取下一个。`--workers` 可减少同时运行的条件数。若父进程设置 `CUDA_VISIBLE_DEVICES`，`--gpus` 使用该可见列表的逻辑编号。调度器不会等待其他无关实验自动释放设备。

MIMIR 的领域名：`arxiv dm_mathematics github hackernews pile_cc pubmed_central 'wikipedia_(en)' full_pile`。

`dry-run` 校验 manifest、数据校验和、模型版本、seed 和已有主方法划分，不读取模型权重，不创建输出，不调用 GPU。工作进程重新检查任务冻结信息；实际推理仅使用本地已缓存模型。通过 `SD_AUDIT_PYTHON` 可指定 Python 环境，默认使用原工作区 `.venv`。

## 数据与公平比较

- MIMIR 默认数据根：`/home/mxd/lib/SD_MIA-pretraining-data/mimir/prepared`。普通领域测试 400 成员 + 400 非成员；`full_pile` 为 2000 + 2000。
- Qwen 默认数据根：独立工作树的 `artifacts/data/qwen_temporal_clean_v2`，默认 `length_matched`，测试 2000 历史 + 2000 近期文档。历史/近期标签仍是时间代理标签，不能表述为已核实的训练成员身份。
- 每个条件的 600 条辅助数据按当前主方法的 `deployment_partitions` 划为 320 train + 80 validation + 200 calibration。PETAL 回归拟合、ReCaLL 的 4 个示例、ICP 的 top-5 检索仅使用前两个角色合计 400 条。独立的 200 条只用于阈值校准。全部测试记录保留；不使用旧 M1 的子集划分。
- 沿用 `k_percent=20`、ReCaLL shots=4、ICP top-k=5/min、SEAD samples=50/temperature=1，其余参数与现有七基线一致。每种方法重置到本条件 seed；方法跳过/恢复不会改变后续方法的随机序列。
- 原始文本第 0 个 token 只作上下文，评分位置为 1..L−1；不追加 prompt/EOS。排除填充词表行并重新归一化，与当前预训练主方法保持相同文本合同。
- 模型使用 BF16/CUDA + SDPA。各方法独立测量准备、校准和测试成本，复用相同权重驻留，但不跨方法复用推理结果。

`--data-root` 可以覆盖对应数据根（MIMIR 根下直接包含领域，Qwen 根下直接包含 seed）；`--output-root` 指定新的基线任务根；`--main-root` 指定只读主方法任务根。根目录下均按 `领域或版本/seedSEED` 查找/写入条件。禁止将输出放入数据、源码或主方法结果目录。

## 输出和恢复

默认基线任务根分别为：

```text
artifacts/audits/pythia_mimir_baselines7_v1/tasks
artifacts/audits/qwen_temporal_clean_baselines7_v1/tasks
```

每个条件保存 `BASELINE_REQUEST.json`、`PARTITIONS.json`，以及七个方法各自的 `REPORT.json` / `scores.npz`。`_executions/` 保存队列、状态和逐条件日志。重复同一命令会跳过已完成方法；部分失败时保留此前完成的方法，其他条件继续排队。模型、代码、参数或数据版本变更需要换输出目录，不能混入旧结果。

`run` 结束后自动汇总，也可独立执行 `summarize`。`_summary/COMPARISON.{json,csv,md}` 包含每个 seed 的主方法及七个基线、AUC/95% CI、pAUC、ROC TPR@1%/10% FPR、独立校准 TPR/实际 FPR，以及成本。Markdown 按所有所选 seed 汇总均值 ± 样本标准差；不对缺失 seed 做隐式平均。

Pythia 默认读取原工作区的 `artifacts/audits/pythia_mimir_v1/tasks`；Qwen 默认读取独立工作树的 `artifacts/audits/qwen_temporal_clean_v2/tasks`。比较时验证相同 manifest、模型、seed、划分及实际 score IDs/labels/order，并检查结果校验和。历史主方法保留其原代码 provenance，无须当前源码逐字相同，不会被重新运行或改写。

尚无匹配主方法结果时仍可完成基线，表中主方法显示 missing，`comparison_complete=false`。Qwen 清理版本主方法可另用下列已有入口运行，然后再次 `summarize`：

```bash
bash standalone/qwen_temporal_clean/run.sh run --gpus 0 1
```

缺失基线/无效结果返回非零状态；仅主方法缺失不会令已完成的基线任务失败。切勿把原始未清理 Qwen 结果与清理后的基线直接比较，程序会拒绝这类划分/数据混用。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  /home/mxd/lib/SD_MIA/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  standalone/pretraining_baselines/test_baselines.py
```

验证使用本地生成的微型模型，不会调度真实模型 GPU 实验。
