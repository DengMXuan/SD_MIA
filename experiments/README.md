# 实验入口

所有生成产物统一保存在仓库根目录 `artifacts/`。完整路径规则、各阶段保留策略和历史路径迁移见 [实验全生命周期产物目录](../docs/artifact_layout.md)。`experiments/` 维护实现和启动器，回归测试统一在根目录 `tests/`。共享框架、依赖方向和新增模型/草稿/防御的接入方法见 [代码结构与扩展指南](../docs/code_structure.md)。

| 实验 | 说明与入口 | 产物批次 |
|---|---|---|
| 共享实验框架 | [共享代码](shared/README.md)：模型、数据、训练、协议、方法、调度和报告 | 路径由实验入口指定 |
| 受控 SFT 训练与 Qwen 审计 | [SD 实验](sd_membership_sft/README.md) | `training/controlled_sft_v2/`、`audits/qwen_shared_reference_v1/` |
| 跨模型固定候选审计 | [跨模型实验](cross_model_audit/README.md) | `audits/cross_model_fixed_v1/` |
| 全参数 DP 防御 | [DP 实验](dp_defense/README.md) | `training/dp_defense_v1/`、`audits/dp_defense_v1/` |
| 查询次数、辅助数据量与非同分布消融 | [Qwen3 多 GPU 脚本](resource_curves/QWEN_ABLATIONS.md)、[资源曲线 API](resource_curves/README.md) | `audits/resource_curves_v1/` |
| Target-only baseline | [Baseline](baseline/README.md) | 矩阵内 `tasks/<condition>/baseline/`，或显式输出目录 |
| Pythia / MIMIR 预训练审计 | [预训练实验](pretraining/README.md) | `data/pretraining/`、`audits/pretraining_v1/` |
| 目录维护 | [维护说明](maintenance/README.md) | `maintenance/` |

以上产物批次均相对 `artifacts/`。训练批次包含 `runs / models / splits`；审计批次包含 `tasks / intermediate / executions / reports`。论文和展示图表保存到 `artifacts/reports/figures/`。

从仓库根目录运行当前主线：

```bash
bash experiments/sd_membership_sft/scripts/retrain_unified_matrix.sh --status
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh status
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh run --gpus 0 1 2
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh summarize
```

`status` 和 `dry-run` 不创建结果目录。审计汇总写入对应批次的 `reports/`；未完成矩阵的汇总仍会生成，但退出码为 2。

2026-09-22 以前的算法与命令说明见 [历史文档](sd_membership_sft/docs/history/README_before_20260922.md) 和 [历史方法目录](sd_membership_sft/archive/README.md)。历史文档保留当时路径。旧入口链接供既有护照读取，新实验使用正式目录；代码来源和请求校验不会因目录迁移而放宽。
