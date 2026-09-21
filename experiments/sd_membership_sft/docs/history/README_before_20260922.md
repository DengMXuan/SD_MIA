# SD 成员推理实现

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
| `serial_accept_only.py` | 独立的自然串行 SD 采集/评估，不能与固定候选指标混排 |
| `sd_protocol.py`, `protocol_models.py` | 新的独立草稿/EAGLE-3/MTP 冻结适配器；单步自然 SD 与固定候选验证 |
| `collect_protocol_observations.py`, `protocol_archive.py` | 多起点自然轨迹、固定候选观测、四角色来源校验和断点恢复 |
| `protocol_accept_only.py` | 自然 SD 因果检测器、固定候选 TCN、稀疏评分与文档级独立校准 |
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

## 多起点自然 SD 与预测头适配（2026-09-19）

新增入口支持独立草稿、EAGLE-3、原生单层 MTP。自然 SD 从正文 token 的
指定比例处启动，例如 `--starts 0.5 0.75`；也可使用 `--starts suffix64`。
每个起点使用独立轨迹和随机流，拒绝后的修正影响后续前缀，不恢复原文。
同一文档的所有轨迹始终属于同一训练/验证/校准/测试角色。

首版每轮提议一个 token，接受后可补充目标 token，拒绝后采样修正 token，
遇到 EOS 提前停止。`--rounds-per-start 32` 是**每个起点**的上限，两个
起点最多合计 64 轮；相同总预算的比较可以使用单起点 32 轮与双起点各
16 轮。此实现不是完整 EAGLE-3 树搜索，也不宣称推理加速：采用完整前缀
重算以保证正确性，记录实际模型前向次数、输入 token 工作量和生成长度。
隐藏状态字节数是逻辑张量传输量，并非实测网络流量。

先指定一个**已完成且具有四角色数据合同**的训练目录，然后采集：

```bash
SD_AUDIT_RUN_DIR=/absolute/path/to/completed/condition

.venv/bin/python -m experiments.sd_membership_sft.collect_protocol_observations collect \
  --run-dir "$SD_AUDIT_RUN_DIR" --adapter plain --protocol natural \
  --starts 0.5 0.75 --rounds-per-start 16 --device cuda:0 \
  --output-dir experiments/results/sft_runs/protocol_audit/example_natural

.venv/bin/python -m experiments.sd_membership_sft.protocol_accept_only \
  --observations experiments/results/sft_runs/protocol_audit/example_natural/observations.npz \
  --output-dir experiments/results/sft_runs/protocol_audit/example_natural/evaluation
```

EAGLE-3/MTP 分别使用 `--adapter eagle3` / `--adapter mtp`，并将 run-dir
指向对应新矩阵的条件目录；要求 `checkpoints/target` 和
`heads/auxiliary_head` 都有完成标记，训练护照和共享划分审计一致。端侧
预测头使用云端隐藏状态，但检测器观测归档仅保存草稿特征和接受计数。
MTP 使用与冻结目标匹配的 embedding/output 权重，并检查原生一步预测
的位置偏移。两个头适配器目前只经过接口/协议测试，**尚未经过真实模型验证**。

固定候选对照使用同一采集命令的 `--protocol fixed`，另设输出目录；
它对真实正文候选进行 B=2 验证，不使用自然轨迹起点。EAGLE 草稿词表外
的候选不伪造有限 q，不产生接受样本；每篇文档记录总候选数和支持数，
没有任何支持候选时明确失败。不同协议和不同候选覆盖率不能混报。

自然 SD 使用因果 GRU 预测当前接受分布，只输入已发生的接受历史；固定
候选复用难度 TCN。两个协议都报告预先固定的正向、负向和双侧 global /
sparse 评分，以及 q-only 和接受率对照。所有起点的固定备择证据先按
文档累积再混合，用同预算的独立非成员文档校准；另报各起点的结果。
测试成员不参与方向、起点、评分或阈值选择。600 条审计辅助数据仍固定
划分为 320/80/200，完整 2,000+2,000 条审计对象只用于测试。

每个轨迹持久化后可恢复；参数、输入、程序或模型来源变化会拒绝复用。
每个 `.npz` 都带哈希校验 sidecar。正式评估拒绝不满足四角色合同的旧
检查点数据，也拒绝运行冒烟归档。旧模型可用以下独立入口验证流程：

```bash
.venv/bin/python -m experiments.sd_membership_sft.collect_protocol_observations smoke \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --adapter plain --starts 0.5 0.75 --rounds-per-start 2 --device cuda:0 \
  --output-dir experiments/results/sft_runs/protocol_audit/old_qwen_smoke
```

`smoke` 默认使用合成文本，也可通过 `--smoke-text-file` 指定文本；不拟合
检测器、不报告成员推理指标。旧 Qwen 的实测归档见
`experiments/results/sft_runs/natural_sd_extension_smoke_20260919/`：50% 起点
在第一轮接受 EOS 后停止，75% 起点完成两轮；只证明运行和提前终止流程。
原有 `serial_accept_only.py` 保留为旧划分下的历史多 token 原型，不作为
本次正式四角色评估入口。设计与验证说明见 `NATURAL_SD_DESIGN.md`。

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
2026-09-20 起，按用户要求停止后续自然 SD 审计。shell 入口默认每个配置只报告以下 12 个变体：

- 固定候选主方法：难度条件 TCN + 正向稀疏评分。
- 11 个 baseline：loss、min_k_prob、min_k_pp、recall、icp_mia、petal、sead、ws、rs、bt、samia。

18 个目标条件展开为 **54 个 worker 任务、234 份独立方法输出、432 行配置/方法结果**。
baseline 只依赖目标模型，每个目标运行一次，两种草稿配置引用同一结果；汇总成本
不会重复计算，也不会把这些重复展示的行当作额外 seed。

已经保存的自然 SD 报告、轨迹和检测器全部保留，但不再调度或纳入新汇总。
本次范围调整仅在 shell 启动入口完成：实验 Python 源文件、保留任务的参数、
标识、输出路径与哈希均不改变，已完成固定候选/baseline 结果继续复用。
新汇总写入原结果目录下的 `fixed_only_summary/`，不覆盖旧的全方法汇总。
正在运行的旧进程仍使用旧计划；需 Ctrl-C 并等其退出后，用下面的 shell 命令
重新启动。直接 `python -m ...qwen_audit_matrix` 仍为原来的含自然 SD 全矩阵入口。

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

历史自然 SD 默认 `--starts suffix64 --rounds-per-start 32`，从原回答尾部 64 token
之前开始，每轮提议一个 token，接受/修正后沿实际生成前缀继续，EOS 提前停止。
这表示最多 32 个候选判定，不是保证生成 64 token。保留 `--starts 0.5 0.75`
等多起点设置，文档内各起点证据合并后再校准。固定候选始终使用完整原回答，
B=2；两种协议不是等查询预算，比较时应同时查看成本。
现在的 shell 入口保留上述参数仅为兼容已有任务签名，不会因此运行自然 SD。
恢复时继续使用原值，不要为取消自然 SD 而更改这些参数。

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
forward 调用数分开，token 工作量不称为 FLOPs。自然 SD 两种评分共享采集与
拟合，每行保留独立使用成本，汇总按 execution group 去重。
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
