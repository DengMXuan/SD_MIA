# SD 成员推理实现

> 历史说明：2026-09-25 已移除自然 SD 代码及其旧命令示例。当前入口和目录以 [现行 README](../../README.md) 为准。

2026-09-17 清理后，当前方法与历史实验分开维护。清理仅涉及本目录；`experiments/baseline/`、`experiments/pretraining/`、已有 `tests/` 和实验结果/检查点均未修改。

## 当前方法

主线为 **非成员条件接受分布 TCN + 草稿难度特征 + sparse 评分 + 200条独立非成员校准**。只用可信非成员训练和选择小型检测器，冻结目标与草稿语言模型。当前数据合同从同一冻结池按 seed 生成四个互斥角色：2,000 条 member、2,000 条 nonmember、2,000 条 draft auxiliary、600 条 audit auxiliary。最后一类再固定分成 320 条检测器训练、80 条验证和 200 条校准；完整的 2,000+2,000 条审计对象只用于测试。

| 文件 | 职责 |
|---|---|
| `conditional_accept_only.py` | 接受位观测合同、非成员划分、基础 TCN、全局评分与基础对照；保留历史诊断所需的 span/paired 接口 |
| `difficulty_accept_only.py` | 草稿难度特征的读取与小型检测器拟合、预测、稀疏评分及扩充/分组校准；不加载历史因果/集成/分配实验 |
| `collect_draft_difficulty.py` | 已有冻结草稿的难度特征采集与恢复 |
| `combined_accept_only.py` | 复用冻结检测器的 2×2×3 组合验证；保留完整消融结果 |
| `collect_deployment_observations.py` | 从四角色训练护照重建600+2000+2000条记录；冻结草稿和目标模型，持久化草稿难度特征与B=2接受位，目标概率只在验证器内存中短暂存在 |
| `deployment_accept_only.py` | 当前独立600条非成员资源合同：重新拟合小型难度TCN、使用稀疏评分＋200统一校准，在2000＋2000测试集上评估 |
| `lowq_baseline.py` | 固定 low-q 多尺度基线，不再计算不需要的 learned-window 分支 |
| `sd_protocol.py`, `protocol_models.py` | 新的独立草稿/EAGLE-3/MTP 冻结适配器；固定候选验证 |
| `collect_protocol_observations.py`, `protocol_archive.py` | 固定候选观测、四角色来源校验和断点恢复 |
| `protocol_accept_only.py` | 固定候选 TCN、稀疏评分与文档级独立校准 |
| `analyze_conditional_accept_only.py`, `summarize_*_validation.py` | 已有实验的完整汇总，保留负结果及协议边界 |

历史组合实验仍保留 q/位置或难度特征 × global或sparse × 三种校准方式。当前部署候选固定为难度特征＋sparse＋200条统一校准，不再要求1200条校准或默认难度分组。测试集扩大只改变评估精度，不参与检测器拟合、评分选择或阈值设定。

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

# 重新微调完成后，为每个验证器随机种子采集四角色观测。
# 该命令依次加载冻结草稿和冻结目标，二者不会同时训练或更新；archive
# 只包含 logq、三个已登记的草稿难度特征和两次二元接受结果。
.venv/bin/python -m experiments.sd_membership_sft.collect_deployment_observations \
  --run-dir experiments/results/sft_runs/four_role_v1/wikitection_qwen3_8b_epoch1 \
  --acceptance-seed 20260914 --gpu 0 \
  --output experiments/results/sft_runs/deployment_observations/wikitection_epoch1/seed20260914.npz

# 当前部署资源合同；校验 archive 及来源 sidecar，只重新拟合小型 TCN。
.venv/bin/python -m experiments.sd_membership_sft.deployment_accept_only matrix --jobs 3
.venv/bin/python -m experiments.sd_membership_sft.deployment_accept_only summarize
```

以上默认 CPU；语言模型特征采集仍可按原命令使用可用 GPU。

## 四角色数据池

三个原始池各含 8,000 条记录。`build_controlled_split` 对完整池按选定 seed 洗牌，然后依次执行模型对应的 token 长度门槛、截断后 token 精确去重和 13-gram 近重复门槛，最后冻结 2,000/2,000/2,000/600 四个角色。`audit_auxiliary` 不会从测试 nonmember 中切分。

候选数据必须先写到单独文件，再用 `merge` 校验并安全追加。合并器保留原池顺序，检查候选 manifest 和 SHA-256，按 record ID、文本 SHA-256、词级 13-gram 去重，并将顶层采集时间更新为所有来源区间的并集。数据和 manifest 通过同一把文件锁及事务日志发布；若进程在两次原子 rename 之间中断，下一次仓库读取会先完成事务再校验哈希。原地覆盖前仍会写入 `backups/`：

```bash
# 示例：将单独采集的候选池安全合并到恰好 8,000 条
.venv/bin/python -m experiments.sd_membership_sft.pools merge \
  --benchmark wikitection \
  --candidate-paths experiments/data/pools/wikitection/extension_candidates_202609.jsonl \
  --target-records 8000
```

部署观测使用 `sd_mia_accept_only_deployment_v1` 固定 schema。每个 `.npz`
必须带同名 `.npz.json` sidecar；加载器会校验 archive SHA-256、生产模块、
特征名、run manifest、草稿 checkpoint 和目标 verifier checkpoint 指纹。
额外字段、缺少 sidecar 或来源哈希不一致都会失败，已有 `REPORT.json` 也
只有在 archive 与 sidecar 哈希均未变化时才会复用。

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

## Qwen3 36 配置统一审计矩阵

入口为 `run_qwen_audit_matrix.sh`，复用已经微调好的 Qwen3-8B / 1.7B。
默认组合为三个数据集 × epoch 1/3 × seed 1919/1949/1978 × KD/member 两种草稿。
每个配置报告以下 12 个变体：

- 固定候选主方法：难度条件 TCN + 正向稀疏评分。
- 11 个 baseline：loss、min_k_prob、min_k_pp、recall、icp_mia、petal、sead、ws、rs、bt、samia。

18 个目标条件展开为 **54 个 worker 任务、234 份独立方法输出、432 行配置/方法结果**。
baseline 只依赖目标模型，每个目标运行一次，两种草稿配置引用同一结果；汇总成本
不会重复计算，也不会把这些重复展示的行当作额外 seed。

在仓库根目录执行：

```bash
# 只读检查训练护照、共享划分、检查点分片和已有结果；不加载大模型或启动 GPU 实验
bash experiments/sd_membership_sft/run_qwen_audit_matrix.sh dry-run

# 正式运行：GPU 编号是物理编号，请填写空闲卡；默认仅使用 GPU 0
bash experiments/sd_membership_sft/run_qwen_audit_matrix.sh run --gpus 0

# 相同命令可恢复已校验的轨迹、检测器和方法结果
bash experiments/sd_membership_sft/run_qwen_audit_matrix.sh status
bash experiments/sd_membership_sft/run_qwen_audit_matrix.sh summarize
```

可通过 `--gpus 0 1 2` 并行调度，每张卡最多一个 worker。每次分派前检查 GPU
已用显存；默认允许不超过 1024 MiB 的后台占用，可用环境变量
`SD_AUDIT_GPU_MAX_USED_MIB` 调整。Ctrl-C 停止继续分派并终止、回收正在运行的
worker，已持久化的数据保留。方法失败不会丢弃其他独立任务的成功结果；重新
执行 `run` 可重试失败任务。训练尚未完成的条件显示为 pending，本次运行只
处理当前 ready 条件，不等待新训练完成。矩阵不完整时 `run` / `summarize`
返回非零状态（正常汇总的未完成状态为 2）。

默认路径：

- 模型：`experiments/results/sft_runs/unified_matrix_audit600_v2/model_pairs/qwen3`
- 结果：`experiments/results/sft_runs/qwen_audit_matrix_v1`

可用 `--model-root` / `--output-root` 指定其他目录，或限定一个子集：

```bash
bash experiments/sd_membership_sft/run_qwen_audit_matrix.sh run \
  --benchmarks wikitection --epochs 1 --seeds 1919 --gpus 0
```

主方法使用完整原回答，B=2；旧任务身份中的起点和轮数字段只作兼容占位。

`--audit-seed 20260914` 控制审计随机性，与三个模型训练/数据 seed 分开；
`--detector-epochs 30` 控制仅在非成员上拟合的小检测器。大模型始终冻结。
修改起点、预算或其他设置时使用新的结果目录；恢复、status、summarize 应使用
与原运行相同的模型路径、输出路径、筛选条件和审计参数。来源、参数或检查点
发生变化时拒绝复用旧报告，不会覆盖后继续混合汇总。

所有方法共享 600 条 audit auxiliary：主方法 320 拟合 + 80 验证 + 200 校准；
baseline 最多使用同一批前 400 条作为参考/拟合，后 200 条仅做阈值校准。
PETAL 回归拟合属于前 400 条参考数据；ReCaLL 默认使用其中 4 条。
草稿训练用的另外 2,000 条 auxiliary 不加入此审计预算。所有方法测试同一批
2,000 member + 2,000 nonmember，并排除人为追加的 EOS。baseline 的实现文件
保持原样，通过本目录适配器接入数据与报告；其概率/生成文本访问权限在报告中
单独标明，不等同于主方法的 accept-only 权限。

结果包括（四种汇总文件现在位于 `fixed_only_summary/` 下，其余原始产出路径不变）：

- `RESULTS.csv` / `RESULTS.md`：逐配置、逐方法结果，未完成行明确标记。
- `SEED_SUMMARY.csv`：每个数据集/epoch/草稿/方法跨 seed 的均值、样本标准差
  和完成 seed 数。单 seed 不给标准差，不把均值置信区间当作合并样本置信区间。
- `SUMMARY.json`：完整结构化结果、错误及去重后的成功方法工作量。
- 各方法 `REPORT.json` / `scores.npz`：指标、来源哈希、4,200 条校准/测试记录的
  ID、标签、原始分数及分区。主方法另存观测归档、逐轨迹缓存和 `detector.pt`。
- `executions/*/`：每次 worker 尝试的 TASK、STATUS 和日志，保留重试记录。

统一指标口径：AUC；FPR ∈ [0, 0.10] 的 pAUC 原始面积及面积/0.10；
ROC TPR@10%/1%FPR（保持并列分数完整，取不超过 FPR 上限的可实现点）；
独立非成员校准后的 TPR、实际测试 FPR。AUC 给出 200 次分层 bootstrap 区间，
校准 TPR/FPR 给出 Wilson 区间。ROC 是描述性评估，不用于挑方向或部署阈值。

效率记录准备/拟合、校准、测试三个阶段的时间、按 4,000 测试文档摊销的
ms/record、吞吐、目标/草稿调用、输入/输出 token 工作量及峰值 allocated 显存。
GPU 计时边界同步，数值为墙钟时间。主指标排除模型/数据加载、预热、归档 I/O
和 ROC/bootstrap 报告开销；worker 总耗时另外记录。逻辑生成序列数与真实
forward 调用数分开，token 工作量不称为 FLOPs。汇总按 execution group 去重。
当前 SD 验证器重建完整前缀，所测速度反映此实验实现，不代表带 KV cache 的
生产 SD 引擎。检测器在 CPU 上拟合；GPU 主要用于冻结模型的观测采集。

本轮只实现、测试和 dry-run，**没有启动完整矩阵，没有新增成员推理效果结论**。
完整测试命令 `.venv/bin/python -m pytest -q tests experiments/sd_membership_sft/tests`
通过 **245 项测试**（2026-09-19）。
只读检查时 18 个 Qwen 目标条件、90 个任务均 ready；真实 Wiki/epoch1/seed1919
的 CPU 数据重建验证了 4,600 个唯一 ID 及 320/80/200/4,000 分区。该检查不验证
运行时显存容量。EAGLE/MTP 不在本 Qwen 独立草稿矩阵中，原有适配接口保留。
设计记录见 [QWEN_AUDIT_MATRIX_DESIGN.md](../QWEN_AUDIT_MATRIX_DESIGN.md)。

## Qwen3 / Gemma 4 完整微调矩阵

`retrain_model_pairs.sh` 固定生成 36 个 experiment condition：两个模型对、三个数据集、两个 target epoch 设置和三个 condition seed。每个 condition 保存 target SFT、auxiliary-data KD draft 和 member-data SFT draft 三个完整 checkpoint，共 108 个。

模型 revision 固定在 `model_pair_revisions.env`。正式启动前可以只做本地缓存、tokenizer、数据池与 GPU 预检，或查看不会启动训练的完整调度计划：

```bash
# 不训练：检查模型快照、配对 tokenizer、数据池和 GPU 3–6
experiments/sd_membership_sft/retrain_model_pairs.sh --preflight-only

# 不训练：打印 Gemma smoke gate 和其余 35 个 condition
experiments/sd_membership_sft/retrain_model_pairs.sh --dry-run

# 正式执行；必须先完成 Gemma newstection/epoch1/seed1919 smoke condition
experiments/sd_membership_sft/retrain_model_pairs.sh
```

脚本只复用已完整保存的 checkpoint。当前 smoke condition 如果已经有完整 target checkpoint，会从两个缺失的 draft 阶段继续；任一 condition 失败都会令最终命令返回非零状态，且不会把不完整矩阵标为完成。
