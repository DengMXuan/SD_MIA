# Qwen3 epoch 1 / KD 草稿消融

三个入口均使用已有 **Qwen3-8B-Base 目标 + Qwen3-1.7B-Base
`draft_auxiliary_distilled` 草稿**。epoch 1 指目标模型的训练条件，KD 草稿使用
该条件已完成的辅助数据蒸馏 checkpoint。脚本运行固定候选主方法
`main_fixed_sparse_positive`，拟合非成员 TCN；不重新训练语言模型。

| 实验 | 配置 | 默认独立任务数 |
|---|---|---:|
| 辅助数据量 | 3 数据集 × 3 seed × 5 个唯一配置 | 45 |
| 辅助分布变化 | News→Wiki、News→Arxiv × 3 seed | 6 |
| 查询次数 | 3 数据集 × 3 seed × B=1/2/4/8/16 | 45 |

默认数据集为 `wikitection`、`newstection`、`arxivtection`，seed 为
**1919、1949、1978**。同一个条件的模型训练/data seed、冻结数据划分、辅助内部划分、
接受反馈随机流、TCN 初始化/打乱、指标 bootstrap、worker 的 `PYTHONHASHSEED`
均一一对应。每条记录的随机流由该 seed 与固定记录 ID 确定。
脚本会检查模型护照、shared split 及其 hash，拒绝错配或 DP checkpoint。

## 执行

```bash
cd /home/mxd/lib/SD_MIA

# 不带参数也默认 dry-run；只读检查输入并打印矩阵，不创建结果或加载模型/tokenizer。
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_auxiliary_size.sh dry-run
bash experiments/scripts/robustness/robustness_main_qwen3_epoch1_kd_auxiliary_domain.sh dry-run
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_query_budget.sh dry-run

# 显式 run 才启动正式实验。以下三条依次执行，共用同一组卡。
export CUDA_VISIBLE_DEVICES=0,1,2
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_auxiliary_size.sh run --gpus 0 1 2
bash experiments/scripts/robustness/robustness_main_qwen3_epoch1_kd_auxiliary_domain.sh run --gpus 0 1 2
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_query_budget.sh run --gpus 0 1 2
```

每个任务是一个完整的 **数据集 × seed × 辅助配置/B**。每张卡最多运行一个任务，
结束后立即领取下一个；不同 seed 不固定绑卡。`--workers` 默认等于所选卡数，
可用 `--workers 2` 只启用列表前两张卡。单卡用 `--gpu 0`。
GPU 编号相对于父进程 `CUDA_VISIBLE_DEVICES`：物理卡 `2,5` 应传 `--gpus 0 1`；
worker 单独暴露所分配的卡为 `cuda:0`，条件换卡后不因逻辑设备编号改变而失去恢复能力。
并发限制作用于本次启动的任务池；独立启动的多个任务池应分配不同 GPU。

脚本可从任意工作目录调用，默认仓库 `.venv/bin/python`，可通过
`SD_AUDIT_PYTHON` 覆盖；默认离线、OMP/MKL 各 2 线程。
TCN 拟合/评分沿用现有主方法的 CPU 执行口径，默认最多 30 epoch、验证集早停，
bootstrap 为 200 次。`--detector-epochs`、`--bootstrap` 可修改；正式对比应保持一致。

```bash
# 先做真实 tokenizer 数据准备与检查，完全不加载语言模型，不产生实验指标。
bash experiments/scripts/robustness/robustness_main_qwen3_epoch1_kd_auxiliary_domain.sh prepare --gpus 0 1

# 选定数据集/seed，适合先跑一组；之后完整 run 可复用其结果。
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_query_budget.sh run \
  --benchmarks wikitection --seeds 1919 --gpu 0

# 只读核验状态；不完整或无效时返回 2。
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_auxiliary_size.sh status
# 重新核验并导出逐 seed 和均值/标准差表。
bash experiments/scripts/ablation/ablation_main_qwen3_epoch1_kd_auxiliary_size.sh summarize
```

三个入口均支持 `dry-run / prepare / run / status / summarize`。
`prepare` 仍按所选 GPU/worker 数量提供 CPU 并发槽位，但不访问 CUDA；
`run` 会再次验证并组装数据，因此不要求提前 `prepare`。

## 辅助数据量

固定 B=2；拟合数包含训练和验证，按 80%/20% 划分。

| 曲线 | 拟合 | 训练 | 验证 | 校准 |
|---|---:|---:|---:|---:|
| 拟合量 | 400 | 320 | 80 | 200 |
| 拟合量 | 800 | 640 | 160 | 200 |
| 拟合量 | 1200 | 960 | 240 | 200 |
| 校准量 | 400 | 320 | 80 | 200 |
| 校准量 | 400 | 320 | 80 | 600 |
| 校准量 | 400 | 320 | 80 | 1200 |

两条曲线共六个点，400/200 共用同一个结果，因此每个数据集/seed 运行五个任务，
导出六行；全矩阵 **45 个任务、54 行曲线结果**。拟合扩展使用固定嵌套训练/验证集合，
旧训练样本不迁入验证集；校准扩展固定拟合与测试。校准量曲线共享同一个已拟合检测器，
并发到达时会等待相同 fit 的锁后复用。改变校准数本身不会改变原始排名 AUC，
应重点比较校准后的 TPR、实际 FPR 和 conformal 分辨率。

输入为已完成预检查的九组额外 1000 条非成员：
`/home/mxd/lib/SD_MIA-pretraining-data/resource_curves/<dataset>/seed<seed>/extension/EXTENSION.json`。
可用 `--extension-root` 替换此根目录。脚本检查扩展所属的 pool、split、seed、tokenizer、
去重约定，并重新验证文本/token 身份；扩展不得使用原模型/审计四角色中的任何记录。
若该预检查目录缺失，dry-run 明确失败，不重新采样或借用测试集。
扩展样本必须通过当前 tokenizer、精确重复和 13-gram 检查；历史预检查数量不能替代本次输入校验。

## 非同分布辅助数据

固定 B=2 和 320/80/200；News 同 seed 的原 `audit_auxiliary` 600 条全部用于
检测器训练、验证、校准，内部角色与该 seed 的 News 原始分区相同。
Wiki/Arxiv 各自的目标、KD 草稿与原 2000 member + 2000 nonmember 测试集保持原条件。
所有反馈（包括 News 辅助反馈）均由 **目标领域的模型对** 采集。
News 的原草稿训练 `auxiliary` 2000 条不用于这项替换。

数据准备核对原文、token 精确重复与 13-gram 重叠（阈值 0.5），检查 News 辅助样本
与目标领域所有 6600 条模型/审计分配之间是否重叠。发现重叠时失败并给出记录 ID，
不会悄悄替换 News 样本。研究结论限于受控 SFT 成员身份；分布变化后应报告校准阈值
在目标测试集上的**实际 FPR**，不能把标称 1%/10% 当作已实现的误报率。

## 查询次数

固定原同领域 320/80/200 和同一测试集。每个 B 同时作用于训练、验证、校准、测试，
分别实际采集和拟合对应 B 的检测器。不同 B 的固定候选与前 B 个随机判定保持对应，
每个 B 有独立的采集计时；不从 B=16 的缓存截取后声称测得小 B 的耗时。

B 的单位是 **每个有效候选 token 位置的接受/拒绝判定次数**，文档总判定数为
B × 有效候选数。它不等于目标 forward 次数或网络请求数。本地参考实现批量复用
模型概率行生成独立验证随机数，因此 forward 次数可能不随 B 增长。
报告同时保留真实判定数、forward 计数及同步测量的耗时。
跨 B 的耗时对比应使用同型号 GPU 和相同 warmup 策略；每份报告记录实际 GPU 型号。

## 输出与恢复

默认独立任务根目录：`artifacts/audits/resource_curves_v1/tasks/`。

```text
tasks/<auxiliary|domain|query>/<dataset>/epoch1/seed<seed>/fit<N>_cal<N>_b<B>/
  REQUEST.json     # 固定任务请求
  EVALUATION.json  # point、观测签名、fit key、seed
  REPORT.json     # 效果、判定数、耗时、硬件、模型/数据来源
  scores.npz      # 校准/测试分数与测试 p-value
intermediate/<experiment>/<dataset>/epoch1/seed<seed>/
  fit<N>_cal<N>_b<B>/study/STUDY.json
  fit<N>_cal<N>_b<B>/observations/  # 按文档恢复的反馈与测量成本
  detectors/<fit-key>/            # FIT.json、detector.pt
reports/<experiment>/
  SUMMARY.json
  per_seed.csv
  mean_std.csv                    # 跨 seed 算术平均、样本标准差（ddof=1）
executions/<experiment>/<run|prepare>/<attempt>/
  JOBS.json
  STATUS.json
  0000.log ...
```

`run` 结束自动汇总。汇总表标明实际完成和预期 seed 数；缺失 seed 不算零，
只有一个完成 seed 时标准差为空。辅助实验共同基准在两条曲线中分别列出，
但指向同一结果。所有实验保留逐条件结果，便于按 seed 与原主方法对照。

相同命令可恢复：已完成文档校验后复用，未完成文档重新采集；完整观测可直接跳过
模型加载。检测器只复用匹配训练/验证观测、B、seed 和训练设置的结果，未完成 fit 会重拟合。
运行时重新散列实际模型权重；状态/汇总检查源码、数据、结果 hash 与 checkpoint 清单。
采集/拟合缓存的耗时报告为其原始测量成本，不当作恢复本次的墙钟时间。
计时不含模型加载、warmup、存档 I/O 与指标 bootstrap。

单条件失败后继续其他条件，整批最终返回非零。Ctrl-C/SIGTERM 会停止派发并清理 worker
及其子进程。每次启动都有独立日志目录，seed、GPU、退出码可在 `STATUS.json` 查询。
使用 `--log-root` 可改变日志目录。

改参数/源码/模型/数据后需使用新输出目录，例如
`--output-root artifacts/audits/resource_curves_v1/tasks/repeat2`，或指定仓库 artifacts
目录之外的新目录；不覆盖原审计、训练或数据目录。模型位置可由 `--model-root` 指定，
仍强制检查 Qwen3、epoch 1、KD 和 seed 条件。正式运行前不要求修改已有矩阵或结果。


## Python 扩展接口

`AuxiliaryBudget(fitting, calibration, validation=None)` 配置辅助样本分配。
`fitting` 包含训练与验证，默认验证数为 `fitting // 5`；总辅助量限定为 200–1600。
查询预算、嵌套划分及指标解释沿用上面的实验协议。

## Compose an experiment through the library

1. Obtain the frozen `shared` split manifest, the original `base_prepared`
   records/tokenizer using the existing model-specific loader, the matching
   frozen `adapter`, and the model/data fingerprint dictionary `sources`.
   Recompute fingerprints before reuse; checkpoint *paths alone* do not prove
   weight identity. Loading/warmup are the future caller's responsibility.
2. `auxiliary.select_extension(pool_path, shared_manifest_path, tokenizers,
   count=1000)` verifies the frozen pool and selects unassigned nonmembers.
   Its seed defaults to the frozen shared split's seed; an explicit different
   seed is rejected. It no longer defaults to the historical seed 20260922.
   `tokenizers` maps **all exact tokenizer source names in the shared manifest**
   to locally loaded tokenizer objects. Selection excludes every original
   member/nonmember/draft-auxiliary/audit-auxiliary record, and checks raw text,
   truncated token IDs, token bands and 13-gram near-duplicates against the
   original assignments and selected extensions. Insufficient eligible data
   fails explicitly. This function does not download tokenizers or write to
   the original pool/split. `save_extension` stores a separate manifest.
3. Build the study using `build_study`; materialize extensions for the current
   tokenizer with `extension_records`; call `prepare_study` to select exactly
   the union of records needed for that study.
4. Collect observations, fit and evaluate a chosen point using the interfaces
   below. Future scripts can loop over the points; this package never starts
   those loops automatically.

```python
from experiments.resource_curves import calibration_curve, fitting_curve
from experiments.resource_curves.auxiliary import extension_records
from experiments.resource_curves.partitions import build_study, prepare_study, save_study
from experiments.resource_curves.observations import collect_observations
from experiments.resource_curves.detector import fit_detector
from experiments.resource_curves.evaluation import evaluate
from experiments.resource_curves.storage import CACHE_ROOT, DATA_ROOT, RUN_ROOT, digest

# shared, extension, base_prepared, tokenizer_source, adapter, sources and
# condition_key are supplied by the future caller. condition_key identifies
# dataset, training seed, epoch, model pair and draft role.
# Check that the model passport's config.seed and config.data_seed both equal
# shared["seed"], and that its shared-split checksum matches, before composing.
study = build_study(shared, extension, calibration_curve(), name="calibration")
save_study(DATA_ROOT / condition_key / study["name"], study)
extra = extension_records(extension, base_prepared.tokenizer, tokenizer_source)
prepared = prepare_study(base_prepared, extra, extension, study)

observations = collect_observations(
    prepared, adapter, CACHE_ROOT / condition_key / "calibration_b2",
    sources={"frozen": sources, "extension_digest": digest(extension)},
    multiplicity=2, seed=shared["seed"],
)
point = study["points"][0]
detector = fit_detector(observations, point, CACHE_ROOT / condition_key / "detectors")
report = evaluate(observations, point, detector,
                  RUN_ROOT / condition_key / "calibration_fit400_cal200")
```

For the fitting curve replace `calibration_curve()` with `fitting_curve()` and
use distinct study/collection/result directories. For the query curve build a
study containing only `AuxiliaryBudget(400, 200)` and collect each multiplicity
into its own directory. All APIs are explicit about the selected budget.

The extension, study, every point and prepared records carry the same condition
seed. `prepare_study` supplies `prepared.condition_seed`; observation collection
requires it and rejects a different seed. Detector fitting and metric bootstrap
inherit the observation seed and reject overrides. This preserves the mapping
1919→1919, 1949→1949, 1978→1978 across stages. Older extension/study artifacts
without this seed contract must be rebuilt in a new dedicated output folder.

New outputs default to dedicated roots:

- `artifacts/audits/resource_curves_v1/splits/`: extension and study manifests.
- `artifacts/audits/resource_curves_v1/intermediate/`: observation records and detectors.
- `artifacts/audits/resource_curves_v1/tasks/`: reports and scored records.

The library refuses existing unrelated output directories and old artifact
roots, including aliases through symlinks. Reuse of the same collection/result
directory with changed settings or source fingerprints fails. Completed
record observations resume with checksums; an interrupted unfinished record
is recollected. A completed detector is reused only for identical fitting
observations/roles, B and training settings. An interrupted detector fit starts
again; there are no per-epoch training checkpoints.

## Outputs and interpretation

- Observations contain draft features and individual binary judgments, not
  target probabilities, probability gaps, target hidden states or generated
  target tokens. Metadata records coverage, actual judgments, forward counters,
  measured synchronized collection times, hardware and model/data provenance.
- `FIT.json` and `detector.pt` store fitting identity, train/validation IDs,
  normalization, history, early stopping, checksums and measured duration.
  Calibration IDs, test IDs and their outcomes are excluded from fit identity.
- `REPORT.json` reports AUC, raw/normalized pAUC at 10% FPR, empirical ROC TPR
  at 1%/10% FPR, and independently calibrated deployment TPR/**actual FPR**.
  `scores.npz` retains scores, IDs, labels, partitions and test conformal p-values.
- A calibration-only curve should not change raw-score AUC/pAUC or empirical
  ROC TPR when fitting and test records are fixed. Its effect is assessed using
  calibrated TPR, actual FPR and conformal resolution. Calibration has no role
  in learning or choosing the raw-score orientation.
- Timings separate collection, fitting, calibration scoring and test scoring.
  Stored collection/fit time is the original measured cost, not a claim that
  cache replay did that work again. GPU synchronization is used when applicable.
  Loading, caller warmup, archive I/O and metric/bootstrap reporting are excluded.

`observations.view(B)` can use the first B **actually recorded** bits for
efficacy-only comparisons. It cannot increase B. A B=2 view of a B=16 archive
has no measured B=2 collection runtime: matching collection and total latency
are reported as `null`, with B=16 origin costs labeled separately. To measure
the runtime curve, collect each budget separately with the same hardware and
warmup policy. Forward calls may stay constant because full-context model
rows are batched and reused for verifier randomness; timing here is the local
protocol reference implementation, not a measurement of deployed network RTT.

The previous count-only B=2 archive cannot be imported as individual bits or
expanded into other budgets. It is left untouched. This first API uses pooled
independent calibration, matching the active fixed-main audit; it does not
silently introduce another scoring/calibration variant.
