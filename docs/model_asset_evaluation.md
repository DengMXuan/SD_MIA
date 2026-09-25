# 已有模型资产评估方案

状态：评估口径已确认并实现。代码测试和只读资产检查不代表正式 GPU 实验已运行。

## 范围与随机性

- 覆盖已登记的五对模型、三个数据集、1919 / 1949 / 1978 三个条件 seed。
- 泛化性仅评估 epoch 1 的目标模型，并以对应的未微调基座为参照。
- 接受率仅评估 epoch 1 目标模型与对应的辅助数据 KD 草稿。
- 主方法随机性、审计辅助集内部划分、AUC bootstrap 与对应条件 seed
  一一对应，不增加阶段偏移。泛化性抽样与 bootstrap 也复用对应条件 seed。
- 保留历史实验的原始记录；新规则用于后续重新运行的实验。

## 已确认的目标模型泛化性评估

沿用文档续写的 BLEU-4、ROUGE-1、ROUGE-L 指标，报告微调目标的
member–nonmember 差距，以及基座与微调目标在相同样本上的质量变化。
该评估衡量当前数据分布上的续写质量，不等同于独立任务上的通用能力评估。

读取对应条件 seed 的已冻结划分，抽取 500 条 member 和 500 条 nonmember。
保留全部抽中记录，不因长度不足丢弃或替换。使用目标模型的 tokenizer；
草稿属于普通模型、EAGLE 或 MTP，不改变目标模型泛化性评估的定义。

对每条记录的响应 token 序列（长度为 L）：

- L >= 384：前 256 个 token 为上下文，接下来的 128 个为参考续写。
- L < 384：前约 2/3 为上下文，其余约 1/3 为参考续写；整数切分采用
  `floor(2 * L / 3)` 个上下文 token。
- 生成采用贪心解码，生成上限等于该记录的参考续写长度；允许正常提前 EOS。
- 基座与微调目标使用相同记录、提示、上下文和参考续写。

例如，300 个 token 分为 200 + 100，128 个 token 分为 85 + 43。

泛化性评估与成员审计是两个实验。抽样不修改原始划分，也不排除后续审计
使用这些记录；主方法仍使用对应条件下完整的 2000 member、2000 nonmember、
600 audit auxiliary。泛化性评估不改变后续审计的数据成员身份。

## 接受率评估

沿用历史脚本的原文前缀条件，测量 `sum_v min(p(v), q(v))` 和 top-1 一致率。
每个条件从 member、nonmember、草稿训练 auxiliary 各抽取 256 条；这第三类
不是 600 条审计辅助数据。按角色分别汇总，同时报告三类等量混合的 overall。
auxiliary 参与过 KD，不能把其结果或混合结果标为纯留出表现；nonmember 结果
可单独用于比较留出数据上的匹配质量。

只加载对应 epoch 1 目标与 KD 草稿：普通模型使用 `draft_auxiliary_distilled`，
EAGLE/MTP 使用 `auxiliary_head`。不运行 member 草稿、不比较未适配草稿。
复用当前统一协议适配器的 EAGLE 词表映射及 MTP step-0 对齐。

沿用历史接受率评估的响应位置口径：包含末尾附加的 EOS，排除提示位置。
每条文档先对位置取均值，再对文档等权平均，bootstrap 重采样单位也是文档，
默认 1000 次，seed 直接取条件 seed。EAGLE 词表外概率为零，不删除原文 token
落在词表外的位置；另外报告原文 token 词表覆盖率。MTP 若有响应位置没有预测，
明确报错而不是悄悄缩小分母。

此项衡量原文上下文下的下一 token 分布匹配，不是完整生成轨迹的实测接受率，
也不推导部署加速比。模型运行前检查归一化、因果性及前缀数值漂移。

## 数值校验修复决策（2026-09-25）

已确认正式实验保留 BF16 推理，不将正式结果统一改为 FP32。
因果性检查与前缀数值漂移分开：保持序列长度与被测前缀不变，
扰动未来 token 检查因果性；另外记录完整序列与截断前缀的数值差异，
异常项使用 FP32 复核，复核仍不一致时阻断。此修复同时适用于主方法与
接受率共用的适配器检查，不以直接放宽现有阈值替代诊断。

FP32 复核是校验步骤，不改变正式 BF16 指标的精度口径，也不表示所有
BF16 数值漂移均可忽略。校验使用对应 seed 的审计辅助训练子集中的一条
记录，不用评测 member/nonmember 调整校验。接受率每次续跑都会重新校验，
即使数值缓存已经齐全；结果写入 `VALIDATION.json` 并绑定最终报告校验和。

在最多 64 个 token 的两个位置，分别使用两组保持长度不变的未来 token
扰动；因果性比较的绝对/相对容差均为 `1e-5`，总变差距离上限为 `1e-5`。
前缀比较保留原有 `atol=0.15, rtol=0.01`，另以总变差距离超过 `0.01`
触发 FP32 复核。FP32 前缀复核要求 `atol=1e-3, rtol=1e-4` 且总变差
距离不超过 `1e-4`。报告保存漂移数值与是否复核；这些阈值用于校验分流，
不把 BF16 与 FP32 的正式指标宣布为等价。复核不支持、显存不足或不通过
都阻断该条件，不能静默跳过；复核结束（含异常路径）恢复原参数精度与缓冲区。

此检查是选定样本和位置上的运行时检查，不是对所有输入的因果性证明。

真实模型验证还发现两个后端调用问题，已纳入本次修复：Gemma4 在当前软件栈的
memory-efficient SDPA 内核上出现严重的长度相关差异；评测中的 Gemma4
固定使用 math SDPA（仍为 BF16），FP32 复核也使用 math SDPA。该限制同时覆盖
目标模型生成与协议适配器前向，不改变模型训练设置。EAGLE-3 的远程实现
需要显式二维 attention mask 才构造因果掩码，协议适配器现已按训练时的调用
约定传入全有效位置掩码，禁止头部读取未来隐藏状态。未修改 checkpoint 文件。

## 冻结数据与 tokenizer

所有评估使用目标模型 tokenizer。两对普通模型的历史划分只记录了共享草稿
tokenizer 的审计证明，因此额外核对目标 tokenizer 重建的全部四类记录 ID、
prompt hash、response hash 与训练护照完全相同后才接受。EAGLE/MTP 直接使用
目标 tokenizer 的审计证明。两种路径均不改写划分、训练护照或原始数据。

随机抽样在每个角色内重新初始化为条件 seed；同一冻结清单下不同模型对抽到
相同原始记录 ID。不同 tokenizer 可以产生不同长度与参考文本，跨模型指标
应结合这一点解释；基座与微调目标始终严格使用相同切分。

member/nonmember 的 bootstrap 独立重采样；基座/微调模型的 bootstrap 使用
相同记录索引配对。0.03 仅为原有均值差提示阈值，置信区间单独报告，不把
阈值通过解释为统计等价或“没有过拟合”的证明。

## 运行

从仓库根目录执行以下命令。默认五对模型 × 三个数据集 × 三个 seed，固定
epoch 1；共 45 个条件、45 项泛化性任务和 45 项接受率任务。

```bash
cd /home/mxd/lib/SD_MIA

# 只读：检查训练护照、划分证明、所需权重分片及本地基座快照，不运行推理。
.venv/bin/python -m experiments.model_quality.cli dry-run

# 正式运行；每张指定 GPU 同时只有一个 worker。按实际空闲 GPU 修改列表。
.venv/bin/python -m experiments.model_quality.cli run --gpus 0 1 2 3

# 相同命令可以续跑：逐批恢复生成结果、逐条恢复接受率，完整任务直接跳过。
.venv/bin/python -m experiments.model_quality.cli status
.venv/bin/python -m experiments.model_quality.cli summarize
```

只运行其中一项或一个条件：

```bash
.venv/bin/python -m experiments.model_quality.cli run --gpus 0 \
  --evaluations generalization
.venv/bin/python -m experiments.model_quality.cli run --gpus 0 \
  --evaluations acceptance
.venv/bin/python -m experiments.model_quality.cli run --gpus 0 \
  --pairs qwen3 --benchmarks wikitection --seeds 1919
```

生成默认 batch size 为 4；接受率按单条文档执行。`--batch-size` 只影响泛化性。
更改抽样量、batch size、bootstrap 次数或评测依赖源码时，应通过 `--output-root`
指定新批次，以免与旧请求混用。正式 worker 使用本地缓存和离线模式，不自动
下载模型。泛化性依次加载基座、微调目标，避免同时占用两份大模型显存。
首次运行会对权重计算完整哈希；status/dry-run 使用清单检查，不读取整套权重。
评测源码按入口的本地导入依赖递归收集，包含函数内部导入、包初始化及
兼容/模型注册表；模型目录自带的远程实现随 checkpoint 指纹绑定。
该范围是保守的模块依赖范围，主方法独立的 `audit/reporting.py` 不在其中。
修改计算、数据处理、适配器或校验依赖仍会阻止复用，不关闭来源校验。

旧单次泛化性入口也已接到同一实现：

```bash
.venv/bin/python -m experiments.shared.training.generalization \
  --run-dir artifacts/training/controlled_sft_v2/runs/model_pairs/qwen3/wikitection/epoch1/seed1919 \
  --gpu 0 --batch-size 4 --bootstrap-repeats 1000
```

## 产物与恢复

```text
artifacts/evaluations/model_quality_v2/
  tasks/<pair>/<dataset>/epoch1/seedN/<generalization|acceptance>/
    REQUEST.json        # 参数、代码来源、权重哈希、运行环境
    SAMPLES.json        # 抽中记录；泛化性同时保存上下文/参考 token 与长度
    VALIDATION.json     # 接受率的因果性、数值漂移和 FP32 复核记录
    scores.npz          # 逐条指标及记录 ID
    REPORT.json         # 指标、置信区间、划分与来源证明
    REPORT.md           # 可读报告
  intermediate/<pair>/<dataset>/epoch1/seedN/<evaluation>/
    REQUEST.json        # 中间结果绑定同一请求
    *.json              # 分批生成或逐条接受率结果，附校验摘要
  executions/<attempt>/ # 任务描述、日志与退出状态
  reports/
    SUMMARY.json
    conditions.csv      # 每条件、角色、指标及置信区间
    seeds.csv           # 同模型/数据集/评估下跨 seed 均值、样本标准差与完成数
```

报告最后写入，只有请求、来源与产物哈希全部匹配才视为完成。失败任务保留已
完成的批次，其他独立任务继续执行；run/summarize 若仍有未完成任务返回 2。
汇总只针对命令所选范围，并显式报告完成 seed 数，不把部分结果伪装为完整三 seed。

## 后续主方法 seed 策略

Qwen、跨模型、单条件 API 与 DP 审计入口均默认使用对应训练条件 seed。
采集根 seed、检测器初始化与训练、600 条辅助记录的 320/80/200 内部分配、
AUC bootstrap 使用同一数值。逐记录采集仍按原有规则从根 seed 与记录 ID
派生随机流；这里的一一对应不意味着所有文档共用同一串随机数。
baseline 复用同一辅助划分和条件 seed。显式 `--audit-seed` 只允许等于条件
seed；运行多 seed 矩阵时应省略该参数。

新默认审计批次为 `artifacts/audits/qwen_condition_seed_v1/tasks` 和
`artifacts/audits/cross_model_condition_seed_v1/tasks`，历史批次保持原样。
资源曲线内部划分使用冻结清单的 seed，采集显式传入该 seed，检测器和指标
默认继承采集 seed。历史 legacy 分区接口保留原默认值用于解释旧实验。

## 历史接受率实现核查

Git 历史中存在 `experiments/sd_membership_sft/acceptance_comparison.py`
（普通草稿）和 `experiments/sd_membership_sft/head_acceptance.py`
（EAGLE-3 / MTP）。两者在提交 `2122905`（2026-09-08）中删除，
可从该提交的父版本读取。

两者均在原文前缀下计算下一 token 分布的重叠
`sum_v min(p(v), q(v)) = 1 - TV(p, q)`，同时报告 top-1 一致率；
先对每条记录的响应位置取均值，再对记录取均值并进行 bootstrap。
该量是给定原文前缀时草稿采样 token 的期望接受概率，不是完整生成轨迹的
实测接受率，也不能仅凭原文前缀条件就认定其为部署接受率的严格上界。

旧脚本默认从 member、nonmember、草稿训练 auxiliary 各抽取 256 条；
这里的 auxiliary 不是 600 条 detector audit auxiliary。
旧脚本会重建划分，并使用 seed 偏移及硬编码 bootstrap seed，不能直接用于
本次已经确认的冻结划分和条件 seed 规则。沿用口径时还需适配当前模型路径、
EAGLE 词表映射和 MTP 位置对齐。

## 实现验证

- 全量 CPU 回归：`402 passed`。
- 质量矩阵只读资产预检：45 个 epoch 1 条件、90 项任务全部 ready。
- 后续主方法矩阵只读预检：270 个 worker 全部 ready，每个条件 seed 对应
  90 个 worker，任务中的 audit seed 与条件 seed 全部一致。
- 在五对模型的 WikiTection / epoch 1 / seed 1919 真实冻结数据上完成
  tokenizer、原划分、500+500 抽样与短文档处理检查，每对均保留完整样本；
  接受率抽样均为 256 member + 256 nonmember + 256 KD auxiliary。
- 未启动正式 GPU 推理。实际模型生成、接受率数值与峰值显存需要运行后报告，
  预检或 CPU 测试不作为模型质量结论。

## 旧批次归档与从零重跑

用户于 2026-09-25 确认保留 BF16，旧结果归档、全部重新计算。原
`artifacts/evaluations/model_quality_v1` 已移至
`artifacts/archive/evaluations/model_quality_v1_20260925T084604`，旧路径以符号链接
保留。归档保存 1,931 个原文件的校验清单、原请求以及 68 份与原请求哈希
相符的源码快照，原结果文件未改写。`ARCHIVED.json` 明确禁止复用；评测入口
拒绝写入该目录，包括通过旧路径或显式 `--output-root` 写入。

默认输出改为 `model_quality_v2`。上面的普通 `run` 命令将从零执行全部
90 项任务，不导入任何旧缓存或旧报告；后续仅在新批次内部按相同请求续跑。
本次修复只进行回归测试、少量样本校验及只读预检，未启动正式重跑。

## 本次修复验证

- 全量代码测试：424 项通过。
- 四个原失败的 Qwen3 接受率样本：修复后因果性检查、FP32 前缀复核均通过。
- Gemma4、Qwen3 EAGLE-3、Llama EAGLE-3、Qwen3.5 MTP：各用一条 Wiki /
  epoch 1 / seed 1919 审计辅助记录，短序列检查及显式 FP32 复核均通过。
- 五对模型 × 三数据集 × 三 seed × 两项评测的 90 项任务通过只读资产预检。
- 归档的 1,931 个原文件重新校验一致；新批次没有写入任何正式结果。

上述短样本验证不等同于所有 45 个条件已完成正式实验。正式运行仍会逐条件
执行校验，失败条件保留日志并阻断，不通过降低阈值绕过。

## 本批次 EAGLE 字节码误报修复

`model_quality_v2` 首轮结束时，74 项报告完成，16 项 EAGLE 接受率任务因
首次导入产生 `__pycache__/eagle3.cpython-312.pyc` 而在末尾清单校验失败。
这些任务已保存全部 768 条指标和通过的模型校验记录。

修复后，模型目录清单和完整指纹使用相同资产文件集合，仅排除导入生成的
`__pycache__/*.pyc`、`*.pyo`，不排除远程 Python 源码或任何权重/配置。
本次通过固定版本维护脚本核对原始完整指纹并保留原元数据，恢复的是
**同一 v2 批次**内已验证的计算；不导入已归档 v1 的结果。恢复失败任务时
仍重新执行适配器校验，再读取本批次指标缓存生成报告。

修复轨迹见 `artifacts/evaluations/model_quality_v2/repairs/bytecode_inventory_v1/`。
原始失败执行日志保留，恢复尝试写入新的 `executions/` 目录。

维护核验分两类并逐 checkpoint 记录：失败项目引用的权重，以及原指纹已包含
字节码、需要改变摘要的 checkpoint，重读完整内容验证原 SHA256；其余已有
成功报告且原指纹不含字节码的 checkpoint，在原报告输出、来源与当前资产
清单全部一致后保留原摘要。两种核验依据在修复计划中分别标记，不将轻量
清单检查表述为重新计算完整权重哈希。

运行时先完成维护核验，再恢复失败任务（GPU 编号按可用设备调整）：

```bash
cd /home/mxd/lib/SD_MIA
.venv/bin/python -m experiments.maintenance.repair_quality_bytecode
# 上一步成功后执行：
.venv/bin/python -m experiments.model_quality.cli run --gpus 1 2 3 4
.venv/bin/python -m experiments.model_quality.cli status
```

维护和恢复命令均可中断后继续。恢复运行跳过已有的 74 项报告；全部完成后，
状态应为 90 项完成、0 项失败。
