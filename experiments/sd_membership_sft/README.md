# SD 成员推理实验

当前批量审计：冻结 Qwen3 8B/1.7B，运行**固定候选主方法和 11 个 baseline**，自然 SD 实现及入口已移除，主方法标识仍为 `main_fixed_sparse_positive`。主方法是非成员条件接受分布 TCN、草稿难度特征、正向稀疏评分和独立非成员校准。审计辅助集 600 条按 320/80/200 划分为训练、验证、校准；测试集含 2,000 成员和 2,000 非成员。算法、数据划分和指标定义没有因目录重构改变。

## 代码导航

| 目录 | 内容 |
|---|---|
| `audit/` | Qwen 条件、就绪检查和 CLI；调用共享调度与报告，日常入口为 `audit.cli` |
| `finetune/` | 现有受控训练矩阵预检 |
| `analysis/` | 诊断和分析报告脚本 |
| `scripts/` | 兼容链接和冻结 revision 配方 |
| `archive/` | 历史方法实现；保留复现入口 |
| `docs/` | 设计和历史说明 |
| `../shared/` | 通用数据、模型、训练、草稿、协议、检测方法和审计实现 |
| `../../tests/` | 按领域集中维护的回归测试 |

完整依赖方向与扩展方法见 [代码结构](../../docs/code_structure.md)。旧 shell 启动器均链接到 `experiments/scripts/` 中按目的命名的唯一实现。旧 Python 导入及 `python -m` 入口由 [`../MODULE_ALIASES.json`](../MODULE_ALIASES.json) 和统一兼容加载器映射到正式模块；内部代码使用正式路径，不再维护逐文件 wrapper 或修改包搜索路径。

## 保存目录

完整目录合同见 [实验全生命周期产物目录](../../docs/artifact_layout.md)，路径常量集中在 [`../paths.py`](../paths.py)。

```text
artifacts/
  data/pools/                              # 冻结数据池
  training/controlled_sft_v2/
    runs/                                  # 训练护照、日志、权重兼容入口
    models/                                # 实际模型与草稿头权重
    splits/                                # 共享划分与审核证明
  audits/<batch>/
    tasks/                                 # 每个条件/方法的报告与分数
    intermediate/                          # 观测、检测器、未完成轨迹
    executions/                            # 调度、状态与运行日志
    reports/                               # 汇总 JSON、CSV、Markdown
  reports/figures/                          # 展示图表
  maintenance/                             # 迁移证据与开发备份
```

旧 `experiments/results/` 和 `artifacts/runs/` 等入口保留为兼容链接。训练护照、历史审计报告中的原路径和 SHA256 不改写。旧审计仍绑定当时的代码及请求路径，作为历史快照读取；新代码不重新签署旧结果。

自定义 `--output-root` 的审计中间产物放在任务的 `intermediate/`，汇总在根目录的 `reports/`，日志在 `executions/`。

## 运行完整审计矩阵

从仓库根目录执行。先检查，再选择 GPU 开始：

```bash
bash experiments/scripts/effectiveness/effectiveness_comparison_qwen3_sft_matrix.sh status
bash experiments/scripts/effectiveness/effectiveness_comparison_qwen3_sft_matrix.sh run --gpus 0 1 2
```

单卡用 `--gpus 0`；改 GPU 列表不改变实验配置。默认模型根目录为 `artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3`，审计结果根目录为 `artifacts/audits/qwen_condition_seed_v1/tasks`。错误审计 seed 的旧 `qwen_fixed_v1` 批次已删除，该路径仍被保留为禁止写入的新实验路径。

- 完整方法报告通过配置、来源、分数及检测器校验后跳过。
- 主方法未采完的轨迹逐条校验后复用，从缺失部分继续。
- 已有完整观测或检测器时分别复用，避免重新采集或拟合。
- baseline 按已保存的方法恢复；中断且未保存的方法需要重跑该方法，已完成的其他方法不重跑。
- 来源校验继续严格生效。目录迁移不重新签署历史结果。源码或请求变化后使用新批次，不提供“忽略来源变化”的开关。

查看/生成当前范围汇总：

```bash
bash experiments/scripts/effectiveness/effectiveness_comparison_qwen3_sft_matrix.sh summarize
```

写入 `artifacts/audits/qwen_condition_seed_v1/reports/`，包括 `SUMMARY.json` 和 CSV 报表；矩阵尚未完成时返回码为 2，部分结果仍会写出。

### WS/RS/BT 共享原始生成

WS、RS、BT 在同一次条件审计中复用一次相同种子、相同输入的贪心原始续写，后续方法仍独立生成扰动/改写后的续写。SaMIA 的 10 路采样不复用。新审计默认写入 `artifacts/audits/qwen_condition_seed_v1/tasks/`；已删除的 `qwen_fixed_v1/` 路径不得用于新实验。示例：

```bash
SD_AUDIT_PYTHON=/path/to/existing/.venv/bin/python \
  bash experiments/scripts/effectiveness/effectiveness_comparison_qwen3_sft_matrix.sh status \
  --model-root /path/to/existing/artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3 \
  --output-root artifacts/audits/qwen_condition_seed_v1/tasks
```

成本口径分开：第一个运行的 WS/RS/BT 包含原始续写生成，保留完整独立方法成本；后续方法只记录 `physical_incremental_*` 物理增量字段，独立成本字段留空。汇总的实际计算时间使用 `execution_group_seconds`，不把共享原始生成重复计入。后续方法的增量耗时不可与独立方法耗时直接比较；正式使用前仍需在 GPU 上核对分数。

## 训练与验证

```bash
bash experiments/scripts/training/train_effectiveness_fivepairs_epoch1_3.sh --status
.venv/bin/python -m pytest -q
```

默认训练写入上述训练区，权重写入模型区。自定义训练根目录时保持原有独立目录行为；自定义数据划分位置可设置 `SPLIT_ROOT`。

详细设计：[批量审计](docs/QWEN_AUDIT_MATRIX_DESIGN.md)。历史探索协议与方法盘点保留在 [方法开发记录](../../docs/history/method_development.md)。迁移核验和恢复说明见 [维护文档](../maintenance/README.md)。

## Qwen3 epoch 1 KD 主方法重跑

本实验只执行 `main_fixed_sparse_positive`，复用已有训练检查点，不重新训练目标
或草稿模型。共 9 项任务：WikiTection / NewsTection / ArXivTection ×
1919 / 1949 / 1978；目标 epoch 固定为 1，草稿固定为
`draft_auxiliary_distilled`。不调度 baseline、member 草稿或其他模型对。

### Seed 与数据

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

### 运行

从仓库根目录执行：

```bash
cd /home/mxd/lib/SD_MIA

### 只读预检：应该显示 9 个 ready，以及四列一一对应的 seed。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh dry-run

### 正式运行；选择实际空闲的 GPU，每张 GPU 同时一个任务。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh run --gpus 0 1 2 3

### 查看状态、重新生成汇总。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh status
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh summarize
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

### 输出

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
