# SD 成员推理实验

当前批量审计：冻结 Qwen3 8B/1.7B，运行**固定候选主方法和 11 个 baseline**，不再调度自然 SD。主方法是非成员条件接受分布 TCN、草稿难度特征、正向稀疏评分和独立非成员校准。审计辅助集 600 条按 320/80/200 划分为训练、验证、校准；测试集含 2,000 成员和 2,000 非成员。算法、数据划分和指标定义没有因目录重构改变。

## 代码导航

| 目录 | 内容 |
|---|---|
| `audit/` | 当前矩阵调度、固定/自然主方法、baseline 调用、结果和效率汇总；日常入口为 `audit.cli` |
| `methods/` | TCN、难度特征、稀疏评分、校准和保留的组合消融 |
| `protocols/` | SD 协议、独立草稿/EAGLE-3/MTP 适配器、观测采集 |
| `finetune/`、`drafts/` | 训练配置与实现、模型预检、独立草稿及 EAGLE/MTP 训练 |
| `datasets/` | 数据池、共享划分、记录重建 |
| `core/` | 指标、四角色合同、缓存与通用运行支持 |
| `analysis/` | 诊断和分析报告脚本 |
| `scripts/` | 实际运行的 shell 启动器和固定模型版本 |
| `archive/` | 历史方法实现；保留复现能力，不进入当前实验调度 |
| `compat/` | 旧 Python 模块名的薄兼容入口；实际实现仅在对应新目录维护 |
| `tests/`、`docs/` | 回归测试、设计和历史说明 |

根目录的旧 shell 启动器继续转发至 `scripts/`。旧 `python -m experiments.sd_membership_sft.<module>` 导入/入口仍兼容；新代码使用分层模块名。`MODULES.json` 记录映射。`experiments/baseline/` 的实现保持原样。

## 保存目录

路径常量集中在 [`../paths.py`](../paths.py)。全部从仓库根目录查看：

```text
artifacts/
  models/
    controlled_sft_v2/          # 本轮训练权重：checkpoints、heads
    legacy/                    # 旧轮次权重，仍保留
  runs/
    training/controlled_sft_v2/ # 训练护照、日志，权重目录为链接
    audits/qwen_fixed_v1/      # 每个方法的 REPORT.json、scores.npz、执行日志
  cache/audits/qwen_fixed_v1/  # 主方法观测、检测器、未完成轨迹
  data/
    pools/                     # 冻结数据池
    splits/controlled_sft_v2/   # 共享数据划分与审核证明
  archive/                     # 历史实验和已停用流程的产物
  migrations/20260922_layout/  # 迁移前元数据备份、文件指纹、迁移清单与核验结果
```

`experiments/results/` 与 `experiments/data` 下的旧入口是兼容链接，不是另一份数据。训练护照内的历史路径不改写，以保留训练时的原始记录及 SHA256；它们通过这些链接继续有效。不要删除这些链接。当前审计元数据使用新路径，并保留迁移前备份。

自然 SD 的已有产物保留在对应审计条件的 `natural/` 子目录；当前入口不会继续调度它。自定义 `--output-root` 的审计缓存放在该任务自己的 `cache/`，避免多个独立实验共用缓存。

## 继续已暂停的审计

从仓库根目录执行。先检查，再选择 GPU 开始：

```bash
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh status
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh run --gpus 0 1 2
```

单卡用 `--gpus 0`；改 GPU 列表不改变实验配置。默认模型根目录为 `artifacts/runs/training/controlled_sft_v2/model_pairs/qwen3`，审计结果根目录为 `artifacts/runs/audits/qwen_fixed_v1`。显式使用旧的两个根目录也会解析到同一位置。不要更换实验输出根目录或已固定的数据/评分参数，否则不再是同一次实验。

- 完整方法报告通过配置、来源、分数及检测器校验后跳过。
- 主方法未采完的轨迹逐条校验后复用，从缺失部分继续。
- 已有完整观测或检测器时分别复用，避免重新采集或拟合。
- baseline 按已保存的方法恢复；中断且未保存的方法需要重跑该方法，已完成的其他方法不重跑。
- 来源校验继续严格生效。迁移脚本只对这次已核验的目录重构更新指纹，不提供“忽略来源变化”的开关。

查看/生成当前范围汇总：

```bash
bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh summarize
```

写入 `artifacts/runs/audits/qwen_fixed_v1/fixed_only_summary/`，包括 `SUMMARY.json` 和 CSV 报表；矩阵尚未完成时返回码为 2，部分结果仍会写出。原始全矩阵汇总保留作为历史快照，以当前 `fixed_only_summary/` 为准。

### 可选：复用 WS/RS/BT 的原始生成

`--reuse-robustness-reference` 让同一条件的 WS、RS、BT 复用一次相同种子、相同输入的贪心原始续写，后续方法仍独立生成扰动/改写后的续写。SaMIA 的 10 路采样不复用，方法定义及其耗时不变。此模式要求独立的 `--output-root`，不能写入上面的在跑审计目录；它也会把这一模式写进任务配置和每个方法的报告。示例：

```bash
SD_AUDIT_PYTHON=/path/to/existing/.venv/bin/python \
  bash experiments/sd_membership_sft/scripts/run_qwen_audit_matrix.sh status \
  --model-root /path/to/existing/artifacts/runs/training/controlled_sft_v2/model_pairs/qwen3 \
  --output-root artifacts/runs/audits/qwen_shared_reference_v1 \
  --reuse-robustness-reference
```

该模式将成本口径分开：第一个运行的 WS/RS/BT 包含原始续写生成，保留完整独立方法成本；后续方法只记录 `physical_incremental_*` 物理增量字段，旧的独立成本字段留空。汇总的实际计算时间使用 `execution_group_seconds`，不把共享原始生成重复计入。因此不能用后续方法的增量耗时与上面的独立方法耗时直接比较；分数应保持一致，但正式使用前仍需在 GPU 上核对一次。默认不启用此模式。

## 训练与验证

```bash
bash experiments/sd_membership_sft/scripts/retrain_unified_matrix.sh --status
.venv/bin/python -m pytest -q tests experiments/sd_membership_sft/tests
```

默认训练写入上述训练区，权重写入模型区。自定义训练根目录时保持原有独立目录行为；自定义数据划分位置可设置 `SPLIT_ROOT`。

详细设计：[批量审计](docs/QWEN_AUDIT_MATRIX_DESIGN.md)、[自然 SD](docs/NATURAL_SD_DESIGN.md)。旧方法、旧命令和历史结果说明完整保存在 [重构前文档](docs/history/README_before_20260922.md)，其中路径和入口按当时状态记录。迁移核验和恢复说明见 [维护文档](../maintenance/README.md)。
