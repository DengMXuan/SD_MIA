# Qwen3 epoch 1 KD 主方法重跑

本实验只执行 `main_fixed_sparse_positive`，复用已有训练检查点，不重新训练目标
或草稿模型。共 9 项任务：WikiTection / NewsTection / ArXivTection ×
1919 / 1949 / 1978；目标 epoch 固定为 1，草稿固定为
`draft_auxiliary_distilled`。不调度 baseline、member 草稿或其他模型对。

## Seed 与数据

| 训练/冻结划分 seed | 主方法采集与 TCN seed | 审计辅助集内部分配 seed | AUC bootstrap seed |
|---:|---:|---:|---:|
| 1919 | 1919 | 1919 | 1919 |
| 1949 | 1949 | 1949 | 1949 |
| 1978 | 1978 | 1978 | 1978 |

读取每个条件已有的冻结划分与对应 checkpoint：

```text
artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3/
  <dataset>/epoch1/seed<seed>/
    checkpoints/target/
    checkpoints/draft_auxiliary_distilled/
```

每个条件使用 2000 member、2000 nonmember、600 audit auxiliary。
600 条辅助记录按对应 seed 分为 320 训练、80 验证、200 独立校准。
泛化性检测抽样不影响这里的数据使用。采集仍为每个有效 token 两次验证，
逐记录随机流由条件 seed 与记录 ID 派生；TCN 初始化/训练与 AUC bootstrap
直接使用该条件 seed。AUC bootstrap 沿用主方法的 200 次重采样。

## 运行

从仓库根目录执行：

```bash
cd /home/mxd/lib/SD_MIA

# 只读预检：应该显示 9 个 ready，以及四列一一对应的 seed。
bash experiments/sd_membership_sft/scripts/run_qwen_kd_epoch1.sh dry-run

# 正式运行；选择实际空闲的 GPU，每张 GPU 同时一个任务。
bash experiments/sd_membership_sft/scripts/run_qwen_kd_epoch1.sh run --gpus 0 1 2 3

# 查看状态、重新生成汇总。
bash experiments/sd_membership_sft/scripts/run_qwen_kd_epoch1.sh status
bash experiments/sd_membership_sft/scripts/run_qwen_kd_epoch1.sh summarize
```

单 GPU 可以使用 `run --gpus 0`。重复相同 run 命令会续跑，复用通过来源校验的
逐条观测和已完成检测器，完整任务直接跳过；不必重新执行已完成的 9 项。
有未完成或无效结果时 run/summarize 返回 2，并保留完整结果和错误信息。

等价 Python 入口是：

```bash
.venv/bin/python -m experiments.sd_membership_sft.audit.qwen_kd_epoch1 dry-run
```

该入口没有独立的 audit seed 参数，也不提供 epoch、草稿角色或数据集选择，
防止这次固定矩阵误扩展。检测器最多训练 30 epoch，仍沿用原有验证集早停；
`--detector-epochs` 指的是检测器预算，不是目标模型的 SFT epoch。

## 输出

新默认批次为 `artifacts/audits/qwen_kd_epoch1_condition_seed_v1/`，
与历史审计和五对模型质量评估的产物分开。

```text
qwen_kd_epoch1_condition_seed_v1/
  tasks/qwen3/<dataset>/epoch1/seed<seed>/draft_auxiliary_distilled/fixed/
    main_fixed_sparse_positive/REPORT.json
    main_fixed_sparse_positive/scores.npz
  intermediate/qwen3/<dataset>/epoch1/seed<seed>/draft_auxiliary_distilled/fixed/
    trajectories/  observations.npz  observations.npz.json  detector.pt  FIT.json
  executions/<attempt>/TASK.json, STATUS.json, worker.log
  reports/RESULTS.csv, RESULTS.md, SEED_SUMMARY.csv, SUMMARY.json
```

汇总有 9 条条件结果、3 条按数据集聚合的 seed 结果，报告完成 seed 数及均值/
样本标准差；不会把草稿分支或 baseline 当成额外重复实验。
更改参数或代码后需使用新的 `--output-root artifacts/audits/<new_batch>/tasks`，
不重新签署历史报告或绕过缓存校验。
