# 实验指南

所有正式 shell 入口位于 `experiments/scripts/`。实验配置和调度不再以临时项目名创建新 `standalone/` 目录。

## 命名与目录

审计脚本使用 `<目的>_<方法角色>_<模型和主要配置>.sh`：

- `effectiveness`：有效性；`robustness`：分布变化、低重叠或 DP 防御；`ablation`：查询/辅助预算或评分消融。
- `main`：主方法；`baseline`：target-only 基线；`comparison`：已有完整对比矩阵。
- 模型、epoch、草稿类型、B、数据划分等决定实验含义的配置写入名称；seed、GPU、输出位置由参数控制。
- `training/train_<目的>_...`、`data/prepare_...`、`diagnostics/diagnostic_...` 分别表示训练、CPU 数据准备和机制/模型质量诊断。

旧 shell 路径是兼容链接，不维护第二份实现。主方法和 baseline 的数值计算源码仍保留在原位置，因为已保存结果记录了其路径和哈希；目录整理不放宽来源校验，也不搬动已有模型、冻结数据或结果。`standalone/` 仅保留这类实现，正式入口以本页为准。

## GPU 队列与共同用法

在任意工作目录均可用脚本的绝对路径启动。默认使用仓库 `.venv/bin/python`、离线模型访问。审计和训练入口不带参数时只打印计划；数据采集使用各自显式子命令。

```bash
# 先检查配置、数据和输出目录。
bash experiments/scripts/robustness/robustness_main_pythia_mimir7gram02_b2.sh dry-run
# 一张卡一个条件，空闲 worker 立即领取下一个条件。
bash experiments/scripts/robustness/robustness_main_pythia_mimir7gram02_b2.sh run --gpus 3 4 5
# 单独运行相同测试划分上的七个 baseline。
bash experiments/scripts/robustness/robustness_baseline_pythia_mimir7gram02_seven.sh run --gpus 3 4 5
bash experiments/scripts/robustness/robustness_baseline_pythia_mimir7gram02_seven.sh summarize
```

矩阵任务按领域/数据集 × seed × 配置拆分；单条件内部有依赖的阶段顺序执行。单条件失败会记录并继续其他条件，最终返回非零。相同请求可按原实验恢复协议复用完成结果；改变模型、数据、源码或参数需新输出目录。队列保存每次启动的 `JOBS.json`、`STATUS.json` 和 worker 日志；Ctrl-C/SIGTERM 停止派发并清理其进程组。

使用 `gpu_pool` 的入口（预训练、资源曲线、DP、训练）支持 `--gpu N` / `--gpus N ...`、`--workers N`；少于卡数时只使用前 N 张卡。训练支持 `--gpus` 和 `--workers`，并兼容 `MATRIX_GPUS` 环境变量。旧 SFT/跨模型/模型质量审计使用其已有动态调度器，通过 `--gpus` 指定并发卡数，不接受 `--workers`。

预训练、资源曲线、DP、训练队列的 GPU 编号是父进程 `CUDA_VISIBLE_DEVICES` 中的逻辑位置，子进程只看到自己的 `cuda:0`。例如 `CUDA_VISIBLE_DEVICES=3,5` 时使用 `--gpus 0 1`。旧 SFT/跨模型/模型质量审计按物理卡编号调度，运行这些入口时不要额外设置设备掩码。多个独立启动的队列应分配不同空闲 GPU；进程锁不能阻止外部程序抢占显存。

## 有效性

以下脚本相对 `experiments/scripts/effectiveness/`。默认 seed 为 1919/1949/1978；状态以当前 `dry-run` / `status` / `summarize` 输出为准，不把文档中的历史测量当作当前完成状态。

| 脚本 | 范围 |
|---|---|
| `effectiveness_main_qwen3_epoch1_kd_b2.sh` | Qwen3，SFT epoch 1，KD 草稿，Wiki/News/Arxiv |
| `effectiveness_baseline_qwen3_epoch1_seven.sh` | 同条件七个 target-only baseline |
| `effectiveness_main_fourmodels_epoch1_b2.sh` | 四模型主方法矩阵 |
| `effectiveness_comparison_qwen3_sft_matrix.sh` | Qwen SFT 固定候选完整对比矩阵 |
| `effectiveness_comparison_multimodel_fixed.sh` | 跨模型固定候选完整对比矩阵 |
| `effectiveness_main_pythia_mimir13gram08_fullpile_b2.sh` | Pythia，7 个 13_gram_0.8 领域及单独 full-Pile none 划分，24 条件 |
| `effectiveness_baseline_pythia_mimir13gram08_seven.sh` | 只对 7 个领域补充 baseline，21 条件，历史主方法只读 |
| `effectiveness_baseline_pythia_mimir13gram08_fullpile_seven.sh` | 原预训练 baseline 矩阵，含 full-Pile，共 24 条件 |
| `effectiveness_main_pythia_wikimia64_128_b2.sh` | WikiMIA 64/128 词，6 条件，只运行主方法 |
| `effectiveness_baseline_pythia_wikimia64_128_seven.sh` | WikiMIA 同条件，只运行 baseline |
| `effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh` | Qwen3/Gemma4，同一时间代理数据，默认长度匹配 |
| `effectiveness_main_qwen3_temporal_clean_matched_b2.sh` | 原 Qwen 文本清洗/长度匹配实验 |
| `effectiveness_baseline_qwen3_temporal_matched_seven.sh` | 对应 Qwen 清洗实验 baseline |
| `effectiveness_main_qwen3_temporal_shared_b2.sh` | 原 Qwen 历史成员/共享 SFT 非成员时间代理协议 |

不要将不同时间代理协议的结果直接当作同一测试集。详见 [时间代理协议](temporal_pretraining.md)、[baseline 协议](pretraining_baselines.md)、[SFT 协议](../experiments/sd_membership_sft/README.md)。

## 鲁棒性与消融

| 目录/脚本 | 固定配置与变量 |
|---|---|
| `robustness/robustness_main_pythia_mimir7gram02_b2.sh` | Pythia 主方法，低重叠 GitHub/arXiv × 3 seed |
| `robustness/robustness_baseline_pythia_mimir7gram02_seven.sh` | 同划分 baseline，独立运行 |
| `robustness/robustness_main_qwen3_epoch1_kd_dp_epsilon1_4_8.sh` | Qwen3 epoch1 KD；epsilon=1/4/8，要求训练已完成 |
| `robustness/robustness_main_qwen3_epoch1_kd_auxiliary_domain.sh` | News 辅助 → Wiki/Arxiv；B=2；6 条件 |
| `ablation/ablation_main_qwen3_epoch1_kd_auxiliary_size.sh` | 拟合量/校准量，5 个唯一配置 × 3 数据集 × 3 seed |
| `ablation/ablation_main_qwen3_epoch1_kd_query_budget.sh` | B=1/2/4/8/16 × 3 数据集 × 3 seed |
| `ablation/ablation_main_pythia_preserved_b2_cpu.sh` | Pythia 缓存评分改进；默认 seed1919、B=2；用 `--benchmark mimir13/mimir7/wikimia` 独立执行，12 个固定评分，见[协议](reports/pythia_preserved_evidence.md) |
| `ablation/ablation_main_pythia_tail_b2_cpu.sh` | Pythia 接受率尾部与有界正向补充；默认 seed1919、B=2；三个 benchmark 独立执行，13 个固定评分，见[协议](reports/pythia_tail_evidence.md) |
| `ablation/ablation_main_pythia_q_features_b2_cpu.sh` | q 特征与非成员位置参照的三轮开发；按文档分组隔离确认部分，见[协议](reports/pythia_q_exploration.md) |
| `ablation/ablation_main_pythia_q_reference_confirm_b2_cpu.sh` | 冻结开发选择后，7 条件 × 3 seed 的确认与七基线对照；复用 B=2 反馈，CPU 非成员预测器 |

资源曲线支持 `dry-run / prepare / run / status / summarize`。B 是每个有效候选 token 的判定次数，不是 forward 次数。DP 包装入口支持 `dry-run / run`，完整 DP 汇总、非 DP 参考与对比命令见 [DP 协议](../experiments/dp_defense/README.md)；资源配置和接口见 [资源曲线协议](../experiments/resource_curves/README.md)。

DP 审计默认读取 `artifacts/training/dp_defense_v1/runs` 并选择三个数据集的 27 个条件，
不会自动寻找其他训练批次。若使用已训练的 Wiki v2，传入
`--benchmarks wikitection --model-root artifacts/training/dp_defense_v2/runs --audit-root artifacts/audits/dp_defense_v2/tasks`；
完整命令见 [Wiki v2 用法](../experiments/dp_defense/README.md#使用已训练的-wiki-v2-模型)。
DP 的 `dry-run` 只打印计划，不代表所选模型已完成训练。

## 三个独立 Pythia benchmark

原 `paper_positive_controls` 已拆除。对应配置分别在 `experiments/pretraining/benchmarks/mimir13.py`、`mimir7.py`、`wikimia.py`；每次只读所选 benchmark 的输入，没有 `--experiment all`。主方法与 baseline 各有入口、队列、日志和汇总；缺少另一实验数据不会阻止当前实验。

目标为冻结 Pythia-6.9B，revision `c0e3eee36dc47af0c49f361c74cfe459c09f7f23`；草稿为 Pythia-1.4B，revision `fedc38a16eea3bd36a96b906d78d11d2ce18ed79`。最大输入 512 token，主方法 B=2。600 个独立非成员分成训练 320、验证 80、校准 200。七个 baseline 为 Loss、Min-K%、Min-K%++、SEAD、PETAL、ReCaLL、ICP-MIA。

- **MIMIR 13_gram_0.8**：7 领域，每 seed 400 member + 400 nonmember，历史主方法读取 `artifacts/audits/pythia_mimir_v1/tasks/`，缺少历史主方法时拒绝重新生成。
- **MIMIR 7_gram_0.2**：官方 revision `02500d3b7cece0cb7628e939ba9fc93fdb6362ae`。GitHub 268+268，arXiv 400+400。600 辅助非成员来自同领域 13_gram_0.8，排除与全部 7-gram nonmember 的精确 token 重复和 13-word 近重叠。辅助与测试分布不同，应作为低重叠诊断单独报告。
- **WikiMIA**：官方 revision `a89ab76d88f704e9bc5870ac39cc9d458a2a70ac`。64 词 284+258，128 词 139+111；全部官方记录用于测试。辅助来自创建日期不早于 2024-01-01 的独立 Wikipedia 事件页，不使用官方测试负例。两个辅助来源的文本提取方式不同，来源记录保留此限制。64 词为主诊断，128 词为长度补充；旧/新事件标签是时间代理，不能称为已验证 Pile 成员。

```bash
# CPU 数据准备。MIMIR 仅显式 --allow-download 时联网。
bash experiments/scripts/data/prepare_pythia_mimir7gram02.sh --source github
bash experiments/scripts/data/prepare_pythia_mimir7gram02.sh --source arxiv
bash experiments/scripts/data/prepare_pythia_wikimia64_128.sh build-aux
bash experiments/scripts/data/prepare_pythia_wikimia64_128.sh prepare --length 64
bash experiments/scripts/data/prepare_pythia_wikimia64_128.sh prepare --length 128
```

扩展 Wiki 辅助需 `collect-aux --output <新路径> --contact <email或URL>`，随后 `build-aux --checkpoint <新checkpoint> --output <新辅助文件>`。已有 checkpoint 的冻结检查继续保留。`prepare --aux-file` 可指定独立辅助数据。所有审计默认离线；`dry-run` 不创建输出或加载模型。

为继续复用已经冻结的数据和结果，旧产物批次名保留：数据在 `artifacts/data/paper_positive_controls_v1/`；新主方法/baseline 在 `artifacts/audits/paper_positive_controls_v1/tasks/<mimir08|mimir02|wikimia>/` 下各自的 `main/`、`baselines7/`。`mimir08` 主方法只读历史批次。

支持 `--sources`、`--seeds`、`--data-root`、`--output-root`、`--log-root`；三个入口的领域选择彼此独立。汇总写入当前任务根的 `reports/<main|baseline>/`；默认日志写入相邻 `<任务根名>_executions/<main|baseline>/`。主方法汇总只要求主方法完成；baseline 汇总报告自身完成情况，并明确列出对比主方法的缺失/无效行。任一所选方法缺失或无效返回 2，不把部分 seed 误算为完整组。

## 训练与诊断

`training/` 的正式训练需要显式 `run`。普通 SFT 入口支持 `dry-run / preflight / status / run`，兼容 `--dry-run / --preflight-only / --status`；保留原冻结 revision、训练参数、split 校验和恢复条件。

| 脚本 | 范围 |
|---|---|
| `train_effectiveness_qwen3_gemma4_epoch1_3.sh` | 36 条件；Gemma smoke 条件成功后才派发其余条件 |
| `train_effectiveness_eagle3_mtp_epoch1_3.sh` | 54 条件；共享 MTP source 转换一次，各条件独立训练 target/aux/member |
| `train_effectiveness_fivepairs_epoch1_3.sh` | 共享 preflight 后顺序执行上述两矩阵，共 90 条件 |
| `train_effectiveness_qwen3_fourrole_epoch1_3.sh` | 兼容原四角色 recipe；6 条件，历史 seed=20260824；仅 dry-run/status/run |
| `train_robustness_qwen3_epoch1_kd_dp_epsilon1_4_8.sh` | DP 训练，独立于审计；dry-run/run |

```bash
bash experiments/scripts/training/train_effectiveness_fivepairs_epoch1_3.sh dry-run --gpus 3 4
bash experiments/scripts/training/train_effectiveness_fivepairs_epoch1_3.sh run --gpus 3 4
```

`diagnostics/diagnostic_models_fivepairs_epoch1_quality.sh` 支持多 GPU 模型质量评估。`diagnostics/diagnostic_main_pythia_delta_fp32.sh` 将精确 p/q 采集按领域 × seed 分配到多 GPU，完成后联合分析所有条件并保留原 Holm 多重检验校正；使用独立 `pythia_delta_gpu_v1` 批次。`ablation/ablation_main_pythia_evidence_b2_cpu.sh` 只重算已保存证据分数。数据下载、语料清洗、已保存分数重算、图表和汇总属于 CPU 工作，不为这些操作加载 GPU。历史概率差机制分析和证据评分协议见 [delta 诊断](reports/pythia_delta.md)、[证据评分](reports/pythia_evidence.md)。

新增实验应复用共享队列，将配置放到对应模块，在本页登记目的、主要参数、输入、输出和解释限制。避免复制评估器、创建含多个无关实验的 `all` 模式，或修改现有结果以绕过来源检查。
