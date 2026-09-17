# SD 成员推理实现

2026-09-17 清理后，当前方法与历史实验分开维护。清理仅涉及本目录；`experiments/baseline/`、`experiments/pretraining/`、已有 `tests/` 和实验结果/检查点均未修改。

## 当前方法

主线为 **非成员条件接受分布 TCN + 可选草稿难度特征 + global/sparse 评分 + 独立非成员校准**。只用可信非成员训练和选择小型检测器，冻结目标与草稿语言模型。数据划分、随机数、参数、候选词元范围和评分数值沿用已验证实验。

| 文件 | 职责 |
|---|---|
| `conditional_accept_only.py` | 接受位观测合同、非成员划分、基础 TCN、全局评分与基础对照；保留历史诊断所需的 span/paired 接口 |
| `difficulty_accept_only.py` | 草稿难度特征的读取与小型检测器拟合、预测、稀疏评分及扩充/分组校准；不加载历史因果/集成/分配实验 |
| `collect_draft_difficulty.py` | 已有冻结草稿的难度特征采集与恢复 |
| `combined_accept_only.py` | 复用冻结检测器的 2×2×3 组合验证；保留完整消融结果 |
| `lowq_baseline.py` | 固定 low-q 多尺度基线，不再计算不需要的 learned-window 分支 |
| `serial_accept_only.py` | 独立的自然串行 SD 采集/评估，不能与固定候选指标混排 |
| `analyze_conditional_accept_only.py`, `summarize_*_validation.py` | 已有实验的完整汇总，保留负结果及协议边界 |

四个常用配置为 q/位置或难度特征 × global 或 sparse。当前探索性候选为难度特征 + sparse + 1200 条统一校准；难度分组作为可选消融，尚未成为默认推荐。组合超过单项的 AUC 增量仍需确认；平均误报率不保证各条件误报率。

命令从仓库根目录执行：

```bash
# 基础非成员 TCN；仅训练小型检测器
.venv/bin/python -m experiments.sd_membership_sft.conditional_accept_only \
  --benchmark wikitection --epoch 1 --budget 2 --device cpu

# 当前难度特征实验入口：匹配的 q-only 与难度增强检测器
# 沿用 priority_validation/features 下的缓存和完成标志
.venv/bin/python -m experiments.sd_membership_sft.difficulty_accept_only \
  --benchmark wikitection --epoch 1 --device cpu

# 使用已有冻结检测器验证组合；完成的运行会跳过
.venv/bin/python -m experiments.sd_membership_sft.combined_accept_only matrix --jobs 3
.venv/bin/python -m experiments.sd_membership_sft.summarize_combination_validation
```

以上默认 CPU；语言模型特征采集仍可按原命令使用可用 GPU。不需要重新微调语言模型。

## 公共实现

| 文件 | 从旧实验中提取的功能 |
|---|---|
| `audit_runtime.py` | 仓库/缓存路径、固定随机种子、数据划分、非成员抽样、回放随机数、原子 JSON 写入 |
| `audit_metrics.py` | AUC、pAUC、保守尾部 conformal p 值和成员推理指标 |
| `replay_cache.py` | 固定候选回放的数据结构、缓存一致性检查和原有 EOS 去除规则；精确 p/delta 仅供模拟器使用 |
| `lowq_baseline.py` | low-q 位置统计与参考非成员标准化 |

当前检测器、难度特征采集、组合评估和汇总模块导入时不加载 `archive`。历史模块也复用这些公共函数，不再各维护一份划分/指标/回放定义。

数据构建、模型加载、模型适配、p/q 采集、M1 原始特征提取与机制诊断继续保留。与当前算法不同的信息权限不等于无效实现。

## 历史实验

26 个模块的完整实现移入 [`archive/`](archive/README.md)。顶层同名文件仅保留短的兼容入口，保障已有测试、外部导入和历史命令继续工作。例如：

```bash
# 两条命令指向同一份历史实现；仅在复现负结果时使用
.venv/bin/python -m experiments.sd_membership_sft.archive.priority_accept_only --help
.venv/bin/python -m experiments.sd_membership_sft.priority_accept_only --help
```

`priority_accept_only`、`run_priority_matrix`、`run_direction_matrix` 现在属于历史完整矩阵入口，不作为当前方法入口。因果接受历史、简单集成、不确定性折扣、熵查询分配等仍可复现，但不会随当前难度模型导入或训练。

保留兼容入口是为了渐进清理：旧文件名存在不表示算法仍有两份实现。不要删除这些入口，否则已有 `tests/` 和预训练路径中的旧导入会失效。

## 验证

```bash
.venv/bin/python -m pytest tests/ experiments/sd_membership_sft/tests/ -q
```

新增的目录内测试检查主线与归档的导入隔离、兼容入口、公共函数复用、low-q 数值一致性和静态检测器训练一致性。清理时另外做了已有检查点、划分和组合校准结果的前后数值复核；没有改写原实验报告。
