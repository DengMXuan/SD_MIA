# 实验全生命周期产物目录

所有生成产物放在仓库根目录的 `artifacts/`。先按生命周期阶段定位，再按批次、模型对、数据集、epoch、seed、草稿角色定位。路径计算集中在 `experiments/paths.py`；`experiments/` 保存代码，`docs/` 保存维护中的说明。

```text
artifacts/
├── data/
│   └── pools/<dataset>/                  # 冻结原始池、清单及采集缓存
├── training/<batch>/
│   ├── splits/<dataset>/                 # 共享划分及 tokenizer 审核证明
│   ├── runs/<track>/<pair>/<dataset>/epochN/seedN/
│   │   ├── results.json, RESULTS.md       # 训练护照、记录 ID、训练结果
│   │   ├── 日志与运行状态
│   │   └── checkpoints, heads, adapters  # 指向本批次 models 的兼容链接
│   └── models/<track>/<pair>/<dataset>/epochN/seedN/
│       └── checkpoints, heads, adapters  # 大模型权重/草稿头/适配器实体
├── evaluations/<batch>/
│   ├── tasks/<pair>/<dataset>/epochN/seedN/<evaluation>/ # 请求、抽样、分数、报告
│   ├── intermediate/<pair>/<dataset>/epochN/seedN/<evaluation>/ # 逐批/逐条恢复
│   ├── executions/<attempt>/             # 评估 worker 日志及状态
│   └── reports/                          # 条件指标与跨 seed 汇总
├── audits/<batch>/
│   ├── tasks/<condition>/<role>/<protocol>/
│   │   ├── <method>/REPORT.json, scores.npz # 最终方法报告及逐条分数
│   │   └── 观测、检测器入口                  # 指向 intermediate 的兼容链接
│   ├── intermediate/<condition>/<role>/<protocol>/
│   │   ├── trajectories/                 # 逐条采集进度，用于断点恢复
│   │   ├── observations.npz[.json]        # 完整观测与来源证明
│   │   └── detector.pt, FIT.json          # 拟合产物及来源证明
│   ├── executions/<attempt>/             # TASK、STATUS、worker.log
│   ├── reports/                          # SUMMARY、RESULTS、SEED_SUMMARY
│   └── splits/                           # 仅资源曲线等审计专用划分
├── reports/
│   └── figures/<topic>/<condition>/       # 展示用图像及绘图数据
└── maintenance/
    ├── migrations/                       # 本次及后续迁移清单、核验证据
    ├── legacy_migrations/                # 2026-09-22 历史迁移记录
    └── branch_backups/                   # 开发过程备份
```

`condition` 在 Qwen 审计中是 `<dataset>/epochN/seedN`，跨模型审计在前面加 `<pair>`，DP 扫描在前面加 `epsilonN`。baseline 位于条件下的 `baseline/<method>`，不按草稿角色重复保存。

## 当前入口与默认目录

| 用途 | 入口 | 默认目录（相对仓库根） |
|---|---|---|
| 冻结数据池 | `datasets.pools` | `artifacts/data/pools/` |
| 统一训练矩阵 | `scripts/retrain_unified_matrix.sh` | `artifacts/training/controlled_sft_v2/runs/` |
| 训练权重 | 各训练入口自动安排 | `artifacts/training/controlled_sft_v2/models/` |
| 共享划分 | 统一训练 preflight | `artifacts/training/controlled_sft_v2/splits/` |
| 模型资产评估 | `model_quality.cli` | `artifacts/evaluations/model_quality_v2/` |
| 当前 Qwen 审计 | `audit.cli` | `artifacts/audits/qwen_condition_seed_v1/tasks/` |
| Qwen epoch 1 KD 主方法重跑 | `audit.qwen_kd_epoch1` | `artifacts/audits/qwen_kd_epoch1_condition_seed_v1/tasks/` |
| 跨模型审计 | `cross_model_audit.cli` | `artifacts/audits/cross_model_condition_seed_v1/tasks/` |
| DP 训练/审计 | `dp_defense.sweep` | `artifacts/training/dp_defense_v1/runs/`、`artifacts/audits/dp_defense_v1/tasks/` |
| 资源曲线 | `resource_curves.storage` | `artifacts/audits/resource_curves_v1/{tasks,intermediate,splits}/` |

Qwen 与跨模型 CLI 的 `--output-root` 指定任务目录，汇总自动写入同批次 `reports/`，调度日志写入 `executions/`。不要把 `--output-root` 指向整棵 `artifacts/` 或某批次的父目录。

错误审计 seed 的历史 `qwen_fixed_v1` 批次及其兼容链接已删除；该路径仍为保留名称，不作为新实验输出目录。

自定义目录遵循相同的产物名称，但保持独立：审计写入 `<output>/intermediate/`、`<output>/reports/`、`<output>/executions/`；自定义训练仍把权重保存在运行目录内。底层 baseline、pretraining 和单方法 API 接受显式输出路径，其局部文件合同保持不变；推荐路径见各自 README。

## 保留与续跑

- `data`、`splits`、训练护照、权重、方法报告、逐条分数和汇总属于实验依据，需要保留。
- `intermediate` 包含来源验证需要的观测和检测器，**不能作为普通可删除缓存整体清空**。逐条轨迹只有在完整观测逐项核对通过后才能按专用维护流程清理。
- `executions` 保存调度、失败和重试证据，不与最终方法成本混算。
- 展示图表从已完成报告/分数派生，统一放入 `reports/figures`。研究笔记和手工撰写的文档留在 `docs/`。
- 每个独立配置/来源使用独立批次名。路径变更不允许绕过请求哈希、代码来源、checkpoint 或分数校验。

## 已有产物迁移

```bash
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout plan
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout apply
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout verify
```

执行 `apply` 时必须停止实验写入；脚本还会检查已有 worker/coordinator 文件锁。它只做同盘 rename，不复制 2 TB 级权重，不删除实验文件，不改写已有 JSON、指标、分数和来源哈希。跨盘移动或目标冲突会失败。先保存清单，再移动；中断后重复 apply 原本复用原核验清单。

上述 `verify` 是 2026-09-25 迁移时的完整历史快照检查。旧 `qwen_fixed_v1` 批次删除后，原清单中的文件不再存在，重新运行该检查会失败；当前实验请使用对应入口的 `dry-run`/`status`。

旧 `artifacts/runs/`、`artifacts/models/`、`artifacts/cache/`、`artifacts/data/splits/`、`artifacts/figures/` 中的已迁移入口保留为链接；更早的 `experiments/results/` 链接继续可用。内部相对链接会随目录迁移修正。它们只供旧护照和脚本读取，新实验使用上面的正式目录。

核验覆盖所有普通文件的设备号、inode、大小和纳秒修改时间；除大型 `.safetensors` / `.bin` 权重外，额外计算 SHA-256，并核对旧路径和新路径指向同一个文件。迁移证据保存在 `artifacts/maintenance/migrations/20260925_lifecycle/`。

**旧路径可读不等于新代码可续跑旧审计。** 审计请求记录绝对路径，并且严格绑定运行时代码哈希。这次修改保留历史来源证明，不重新签署结果。因此旧审计作为历史快照读取；需要新审计时使用新的批次目录。训练护照和模型读取仍通过兼容链接工作，训练完成状态检查保持原规则。
