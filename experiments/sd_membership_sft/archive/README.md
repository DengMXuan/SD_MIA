# 历史实验归档

这些实现于 2026-09-17 移出活跃方法目录。归档保留负结果、不同信息权限和高成本支线的复现能力，不表示所有方法均无效。

| 范围 | 模块 |
|---|---|
| 强信息/成员监督检测器与汇总 | `full_delta_mia`, `stat_delta_mia`, `accept_only_mia`, `aggregate_permissions` |
| M1 监督拟合与报告 | `m1_fit`, `m1_evaluate` |
| M2 概率证据引导激活聚合 | `m2_features`, `m2_models`, `m2_fit`, `m2_evaluate` |
| 学习窗口和 EVT | `adaptive_window_accept_only` |
| 合成正例神经评分 | `neural_adaptive_accept_only`, `aggregate_neural_adaptive_accept_only` |
| 尺度门控与影子模型 | `interpretable_scale_gate`, `build_local_shadow_cache`, `shadow_scale_gate` |
| 高预算/闭环主动查询及组合 | `full_ai_query_allocation`, `analyze_full_ai_allocation`, `dynamic_marginal_query`, `analyze_dynamic_marginal_query`, `combine_shadow_active` |
| 截断前缀和 JS 主动策略 | `collect_counterfactual_accept_only`, `active_protocol_design` |
| 含无收益分支的历史矩阵 | `priority_accept_only`, `run_priority_matrix`, `run_direction_matrix` |

`MANIFEST.json` 列出26个模块、迁移前源码 SHA-256、原始行数以及兼容入口。迁移调整了相对导入和仓库路径，将公共函数移到父目录；它不是把原文件逐字复制到新位置。所有历史统计口径和方法参数保持不变。

可以使用 `python -m experiments.sd_membership_sft.archive.<module>` 运行具备命令行入口的模块。顶层旧命令继续转发，输出目录和完成标志不变；只有明确要求复现历史实验时才运行这些入口。有些影子模型命令会训练语言模型，不属于当前冻结模型流程。

当前方法请使用父目录的 `conditional_accept_only`、`difficulty_accept_only`、`combined_accept_only`。原结果、原始分数、检查点和数据池没有移动或删除。
