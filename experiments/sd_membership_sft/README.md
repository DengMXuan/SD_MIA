# SD 成员推理实验

当前批量审计：冻结 Qwen3 8B/1.7B，运行**固定候选主方法和 11 个 baseline**，自然 SD 实现及入口已移除，主方法标识仍为 `main_fixed_sparse_positive`。主方法是非成员条件接受分布 TCN、草稿难度特征、正向稀疏评分和独立非成员校准。审计辅助集 600 条按 320/80/200 划分为训练、验证、校准；测试集含 2,000 成员和 2,000 非成员。算法、数据划分和指标定义没有因目录重构改变。

## 代码导航

| 目录 | 内容 |
|---|---|
| `audit/` | Qwen 条件、就绪检查和 CLI；调用共享调度与报告，日常入口为 `audit.cli` |
| `finetune/` | 现有受控训练矩阵预检 |
| `analysis/` | 诊断和分析报告脚本 |
| `scripts/` | shell 启动器和冻结配方 |
| `archive/` | 历史方法实现；保留复现入口 |
| `docs/` | 设计和历史说明 |
| `../shared/` | 通用数据、模型、训练、草稿、协议、检测方法和审计实现 |
| `../../tests/` | 按领域集中维护的回归测试 |

完整依赖方向与扩展方法见 [代码结构](../../docs/code_structure.md)。根目录旧 shell 启动器继续转发至 `scripts/`。旧 Python 导入及 `python -m` 入口由 [`../MODULE_ALIASES.json`](../MODULE_ALIASES.json) 和统一兼容加载器映射到正式模块；内部代码使用正式路径，不再维护逐文件 wrapper 或修改包搜索路径。

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

## 运行当前审计

从仓库根目录执行。先检查，再选择 GPU 开始：

```bash
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh status
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh run --gpus 0 1 2
```

单卡用 `--gpus 0`；改 GPU 列表不改变实验配置。默认模型根目录为 `artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3`，审计结果根目录为 `artifacts/audits/qwen_shared_reference_v1/tasks`。旧 `qwen_fixed_v1` 为历史快照，当前入口禁止写入。

- 完整方法报告通过配置、来源、分数及检测器校验后跳过。
- 主方法未采完的轨迹逐条校验后复用，从缺失部分继续。
- 已有完整观测或检测器时分别复用，避免重新采集或拟合。
- baseline 按已保存的方法恢复；中断且未保存的方法需要重跑该方法，已完成的其他方法不重跑。
- 来源校验继续严格生效。目录迁移不重新签署历史结果。源码或请求变化后使用新批次，不提供“忽略来源变化”的开关。

查看/生成当前范围汇总：

```bash
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh summarize
```

写入 `artifacts/audits/qwen_shared_reference_v1/reports/`，包括 `SUMMARY.json` 和 CSV 报表；矩阵尚未完成时返回码为 2，部分结果仍会写出。旧 `qwen_fixed_v1` 结果保留为历史快照。

### WS/RS/BT 共享原始生成

WS、RS、BT 在同一次条件审计中复用一次相同种子、相同输入的贪心原始续写，后续方法仍独立生成扰动/改写后的续写。SaMIA 的 10 路采样不复用。新审计默认写入 `artifacts/audits/qwen_shared_reference_v1/tasks/`；原 `qwen_fixed_v1/` 结果保留为历史快照，不能用新代码继续写入。示例：

```bash
SD_AUDIT_PYTHON=/path/to/existing/.venv/bin/python \
  bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh status \
  --model-root /path/to/existing/artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3 \
  --output-root artifacts/audits/qwen_shared_reference_v1/tasks
```

成本口径分开：第一个运行的 WS/RS/BT 包含原始续写生成，保留完整独立方法成本；后续方法只记录 `physical_incremental_*` 物理增量字段，独立成本字段留空。汇总的实际计算时间使用 `execution_group_seconds`，不把共享原始生成重复计入。后续方法的增量耗时不可与独立方法耗时直接比较；正式使用前仍需在 GPU 上核对分数。

## 训练与验证

```bash
bash experiments/sd_membership_sft/scripts/retrain_unified_matrix.sh --status
.venv/bin/python -m pytest -q
```

默认训练写入上述训练区，权重写入模型区。自定义训练根目录时保持原有独立目录行为；自定义数据划分位置可设置 `SPLIT_ROOT`。

详细设计：[批量审计](docs/QWEN_AUDIT_MATRIX_DESIGN.md)。旧方法、旧命令和历史结果说明完整保存在 [重构前文档](docs/history/README_before_20260922.md)，其中路径和入口按当时状态记录。迁移核验和恢复说明见 [维护文档](../maintenance/README.md)。
