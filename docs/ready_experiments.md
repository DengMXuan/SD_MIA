# 当前可运行实验与调用方法

更新日期：2026-09-26。本轮提供四个批量脚本；只做过 dry-run、tokenizer 数据核验和
CPU 功能测试，没有启动正式 Pythia/Qwen 审计或 DP 训练。
四个脚本的 shell 语法及实际 dry-run 均通过；多 GPU 调度另用真实 CPU 子进程测试。
最新全仓回归：**525 项测试通过**，覆盖动态派发、seed/设备映射、失败继续和中断清理。
模型与依赖已缓存；无需重新下载。GPU 显存占用和正式耗时尚需运行后观察。

| 实验 | 默认矩阵 | 脚本 |
|---|---|---|
| Pythia/MIMIR 主方法 | 7 领域 + full_pile × 3 seed，共 24 条件 | `experiments/pretraining/scripts/run_pythia_mimir.sh` |
| Qwen 时间代理主方法 | 3 seed，共 3 条件 | `experiments/pretraining/scripts/run_qwen_temporal.sh` |
| DP 目标训练 + KD 蒸馏 | Qwen、epoch 1；3 数据集 × 3 seed × ε=1/4/8，共 27 条件 | `experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh` |
| DP 主方法审计 | 使用上一步对应的 27 组 DP 目标/KD 模型 | `experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh` |

所有脚本默认 **1919、1949、1978**。预训练扩展不微调语言模型，但会拟合非成员 TCN；
DP 的 epoch 1 是目标模型的训练预算，KD 沿用参考配方的 384 步。四个脚本的主方法均为 B=2、
检测器默认 30 epoch、辅助数据 320 拟合 / 80 验证 / 200 校准。

## 共同用法

```bash
cd /home/mxd/lib/SD_MIA
# 如需选择物理卡，可先设置；以下 --gpu 0 指可见设备中的第一张卡
export CUDA_VISIBLE_DEVICES=0
```

脚本可从任意工作目录调用，默认使用仓库 `.venv/bin/python`（可由 `SD_AUDIT_PYTHON` 覆盖），
并启用离线模式和每个 CPU 库 2 线程。**不带参数等同 dry-run，显式 run 才执行实验。**
默认前台、单卡、逐条件顺序执行。不同条件使用独立目录；相同命令可恢复已完成阶段，
数据、模型、源码或参数改变后应使用新结果目录。DP 中断阶段会重训，完整阶段可复用。

### 多 GPU 与 worker

四个脚本均支持 `--gpus 0 1 2 --workers 3`。每个 worker 执行一个完整条件，每张卡最多
一个 worker；条件完成后自动领取下一个任务，不按 seed 固定绑卡，seed 对应关系保持不变。
省略 `--workers` 则每张所选 GPU 一个 worker；显式数量须在 1 到 GPU 数量之间，数量较少时
只使用列表前 N 张卡。可并发条件数少于 worker 数时按条件数启动。

例如使用物理卡 0、1、2，分别执行所需实验命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
bash experiments/pretraining/scripts/run_pythia_mimir.sh run --gpus 0 1 2 --workers 3
bash experiments/pretraining/scripts/run_qwen_temporal.sh run --gpus 0 1 2 --workers 3
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh run --gpus 0 1 2 --workers 3
# 对应训练完成后执行
bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh run --gpus 0 1 2 --workers 3
```

`--gpus` 与 `--gpu` 互斥，编号相对于父进程的 `CUDA_VISIBLE_DEVICES`。
例如 `CUDA_VISIBLE_DEVICES=2,5` 时应传 `--gpus 0 1`。worker 将分配到的卡单独暴露为
`cuda:0`，避免条件换卡后仅因设备编号变化而无法恢复。单卡 `--gpu N` 用法继续可用。
预览只检查参数并规划，不探测硬件；运行前请选用本次实验可用的卡，并发上限约束本次启动的任务池。
DP 训练每个 worker 仅目标模型的 CPU 梯度累加器约需 32 GB 主机内存，选择并发数时需考虑这一开销。

每次执行有独立的日志目录，终端打印其位置。`JOBS.json` 保存条件和命令，`STATUS.json`
保存状态、seed、分配的 GPU 和每条件 `.log` 的位置。默认目录如下，可用 `--log-root` 覆盖：

- Pythia：`artifacts/audits/pythia_mimir_v1/executions/run/<attempt>/`
- Qwen：`artifacts/audits/qwen3_temporal_shared_split_v1/executions/run/<attempt>/`
- DP 训练：`artifacts/training/dp_defense_v1/executions/train/<attempt>/`
- DP 审计：`artifacts/audits/dp_defense_v1/executions/audit/<attempt>/`

普通条件失败会继续处理其他条件，最终返回非零；Ctrl-C 或 SIGTERM 停止调度并清理全部 worker
及其子进程。相同命令可重新启动并复用实验已完成阶段。`prepare` 也可多 worker，但仅执行 CPU
数据准备；将 `run` 换为 `dry-run` 始终不会启动 worker 或创建日志/实验结果。

如需后台运行，在下列任一 `bash ... run` 命令前加 `nohup`、结尾加 `> /tmp/自定日志名.log 2>&1 &`。
使用不同日志文件，并避免多个进程同时占用同一个条件的输出目录。

## 1. Pythia + MIMIR 主方法有效性

目标为固定版本的 Pythia 6.9B，草稿为 Pythia 1.4B。7 个领域是 `arxiv`、`dm_mathematics`、
`github`、`hackernews`、`pile_cc`、`pubmed_central`、`wikipedia_(en)`，另有混合 `full_pile`。
每领域为 400 成员测试 / 400 非成员测试 / 600 独立非成员辅助；full_pile 为 2000 / 2000 / 600。

```bash
# 只读核验 24 个已冻结 manifest 并打印计划
bash experiments/pretraining/scripts/run_pythia_mimir.sh dry-run
# 执行全部 24 条件
bash experiments/pretraining/scripts/run_pythia_mimir.sh run --gpu 0
# 可选：先跑一个条件；之后执行完整命令会复用已完成结果
bash experiments/pretraining/scripts/run_pythia_mimir.sh run --sources 'wikipedia_(en)' --seeds 1919 --gpu 0
# 检查报告完整性，缺失或无效时返回非零
bash experiments/pretraining/scripts/run_pythia_mimir.sh summarize
```

输入：`/home/mxd/lib/SD_MIA-pretraining-data/mimir/prepared/<source>/seed<seed>/manifest.json`。
结果：`artifacts/audits/pythia_mimir_v1/tasks/<source>/seed<seed>/main_fixed_sparse_positive/REPORT.json`。
`--output-root` 可替换 `tasks/` 根目录。

官方标签用于预训练成员推断验证。报告应说明 1.4B 草稿也训练过 The Pile；512 是 token
截断上限，文本并非全部等长。full_pile 单独报告，不能作为 7 个领域宏平均。

## 2. Qwen 时间划分主方法有效性

目标 Qwen3-8B-Base、草稿 Qwen3-1.7B-Base，直接使用固定版本的公开基础模型。
每个 seed 的 2,000 条历史成员**完整保留此前同 seed 选出的样本**，历史来源仍为
WikiText-103 raw train 重建的旧 Wikipedia 文章。非成员和辅助改为复用原受控 SFT
WikiTection 划分中的 `nonmember` 2,000 条与 `audit_auxiliary` 600 条，ID 和顺序均不变。
`auxiliary` 是原草稿训练用的角色，此处不使用。

原时间 manifest 本身也用了这三个 seed；它与原 SFT 的近期样本不同，是因为抽样方法不同。
新脚本只组装上述已冻结角色，不重新抽取历史样本，不改原 manifest、数据池、SFT 划分或模型。
公开 condition seed 同时控制主方法辅助分区、接受反馈、TCN 和指标估计。

```bash
bash experiments/pretraining/scripts/run_qwen_temporal.sh dry-run
# 可选：只准备数据和加载 tokenizer，不加载语言模型、不执行审计
bash experiments/pretraining/scripts/run_qwen_temporal.sh prepare
# 默认全部 3 seed；run 会自动完成数据准备，无需先手动 prepare
bash experiments/pretraining/scripts/run_qwen_temporal.sh run --gpu 0
# 可选子集
bash experiments/pretraining/scripts/run_qwen_temporal.sh run --seeds 1919 --gpu 0
bash experiments/pretraining/scripts/run_qwen_temporal.sh summarize
```

新数据首次运行时生成到
`/home/mxd/lib/SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1/seed<seed>/manifest.json`。
结果：`artifacts/audits/qwen3_temporal_shared_split_v1/tasks/wikitection/seed<seed>/main_fixed_sparse_positive/REPORT.json`。
`--data-root` 可指定包含原时间/MIMIR 数据的完整根目录；`--reference-root` 可修改读取 Qwen
SFT 数据护照的根目录；`--output-root` 可替换结果 `tasks/` 根目录。

这是**时间代理标签**实验，历史文章不等于已知训练成员。原历史样本仍全部 512 token；
近期测试实际为 150–512 token，辅助约 169–512 token（不同 seed 的最短长度不同）。
复用原划分意味着保留这部分长度差异，分析结果时应考虑来源和长度偏差。
完整数据统计见 [DATA_INVENTORY.md](../experiments/pretraining/DATA_INVENTORY.md)。

## 3. DP 防御：epoch 1 + KD

三个数据集为 `wikitection`、`newstection`、`arxivtection`，默认 ε=1/4/8、
max_grad_norm=1.0、目标阶段 δ=5e-6；每条件从公开基础模型训练 DP 目标，再向其蒸馏 KD 草稿。
共 27 个条件、54 份模型产物；不训练 member 草稿，不运行基线。

### 3.1 单独训练

```bash
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh dry-run
bash experiments/dp_defense/scripts/train_qwen_epoch1_kd.sh run --gpu 0
```

训练目录：`artifacts/training/dp_defense_v1/runs/qwen3/epsilon<ε>/<dataset>/epoch1/seed<seed>`。
该脚本完成后不会自动审计。完整 DP 训练尚未测量耗时；实现采用逐文档梯度和 CPU 累加器。
参考护照提供数据与配方，初始化使用公开基础模型，不能从已普通 SFT 的权重开始声称 DP。

### 3.2 训练完成后，单独运行主方法

```bash
bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh dry-run
bash experiments/dp_defense/scripts/run_qwen_epoch1_kd_main.sh run --gpu 0
```

每条件使用匹配的 DP 目标和 KD 草稿，重新采集、拟合和校准。缺失或来源不匹配的检查点
会报错，不会自动触发训练。结果为：
`artifacts/audits/dp_defense_v1/tasks/qwen3/epsilon<ε>/<dataset>/epoch1/seed<seed>/draft_auxiliary_distilled/fixed/main_fixed_sparse_positive/REPORT.json`。

两个 DP 脚本都可追加 `--benchmarks wikitection --seeds 1919 --epsilons 4` 只跑一个条件。
改 `--model-root` 时，两脚本必须传同一个目录；`--audit-root` 改审计结果目录。
DP dry-run 会显示每条件的 train 和 audit 两条命令，但 run 严格只执行脚本名称对应的阶段。

同一条件的参考模型 seed、数据 seed、shared split seed 和公开训练/审计 seed 一一对应；
DP Poisson 采样及噪声采用独立未公开随机流，不由公开 seed 决定。

### 3.3 生成同设置的非 DP 主方法参考，再比较

比较器会核对模型条件、seed、方法、角色、参数，以及校准/测试记录的 ID、标签和划分。
使用下面的独立批次生成 9 个非 DP 参考；这些参考可供三个 ε 共用。
这只重新审计既有 epoch 1 目标/KD 检查点，不训练语言模型，也不运行 11 个基线。

```bash
.venv/bin/python -B -u - <<'PY'
from pathlib import Path
from experiments.shared.audit.evaluation import evaluate_main
from experiments.shared.models.registry import MODEL_PAIRS

model_root = MODEL_PAIRS['qwen3'].run_root
output = Path('artifacts/audits/dp_reference_v1/tasks/qwen3')
role = 'draft_auxiliary_distilled'
for benchmark in ('wikitection', 'newstection', 'arxivtection'):
    for seed in (1919, 1949, 1978):
        key = Path(benchmark) / 'epoch1' / f'seed{seed}'
        report = evaluate_main(
            model_root / key, output / key / role / 'fixed',
            draft_role=role, device='cuda:0', audit_seed=seed, detector_epochs=30,
        )
        print(benchmark, seed, report['metrics'], flush=True)
PY

.venv/bin/python -B -m experiments.dp_defense.compare \
  --dp-root artifacts/audits/dp_defense_v1/tasks \
  --reference-root artifacts/audits/dp_reference_v1/tasks \
  --output-dir artifacts/audits/dp_defense_v1/reports
```

比较输出为 `COMPARISON.csv`、`SEED_SUMMARY.csv`、`COMPARISON.json`。
完整矩阵应有 27 个匹配结果；比较器汇总的是已有报告，退出成功本身不证明矩阵完整。
上述是攻击有效性比较，不能单凭它得出模型效用未损失的结论；效用结果需另行评估。

## 4. 辅助数据量消融 / 5. 查询次数消融

这两类现已提供独立 CLI，另有 News→Wiki/Arxiv 非同分布辅助数据入口。
默认均为 **Qwen3 epoch 1 的 KD 草稿、1919/1949/1978 三个对应 seed**，
支持一卡一个条件的多 GPU 动态队列、恢复和汇总。完整配置与输出见
[Qwen 消融脚本指南](../experiments/resource_curves/QWEN_ABLATIONS.md)。

```bash
bash experiments/resource_curves/scripts/run_qwen_auxiliary_size.sh run --gpus 0 1 2
bash experiments/resource_curves/scripts/run_qwen_auxiliary_domain.sh run --gpus 0 1 2
bash experiments/resource_curves/scripts/run_qwen_query_budget.sh run --gpus 0 1 2
```

三条命令依次执行，分别为 45、6、45 个独立条件。辅助量的共同 400/200 基准复用，
因此导出六个曲线点、每数据集/seed 实际运行五个配置。不带参数默认 dry-run。

以下保留已有 API 组合调用示例，使用同一模型条件，复用已选好的 1000 条扩展非成员，
不重新下载或扩充原池。

`RESOURCE_STUDY=auxiliary` 执行两条辅助量曲线：

- 拟合总数 400/800/1200（含 20% 验证），校准固定 200。
- 校准数 200/600/1200，拟合总数固定 400。
- B 固定 2，每个条件测试集固定 2000 成员 + 2000 非成员。

将命令首行改为 `RESOURCE_STUDY=query`，则执行 B=1/2/4/8/16 的查询曲线，拟合/校准固定
400/200。B 是**每个支持的候选 token 位置的独立接受判断次数**，不是每篇文档的总查询数。
每个 B 都重新采集，才能报告该 B 的实测效率；不从 B=16 缓存裁切后声称测得了 B=2 耗时。

以下默认覆盖三个数据集、三个 seed。要先跑一个条件，缩小末尾两个循环的列表即可。

```bash
RESOURCE_STUDY=auxiliary .venv/bin/python -B -u - <<'PY'
import gc
import json
import os
from pathlib import Path
import torch

from experiments.shared.models.registry import MODEL_PAIRS
from experiments.shared.models.loading import prepare_records, load_adapter
from experiments.shared.audit.provenance import sources_for
from experiments.shared.audit.artifacts import check_sources_light
from experiments.shared.protocols.collect_protocol_observations import protocol_prompt_ids
from experiments.resource_curves import AuxiliaryBudget, calibration_curve, fitting_curve
from experiments.resource_curves.auxiliary import extension_records
from experiments.resource_curves.partitions import build_study, prepare_study, save_study
from experiments.resource_curves.observations import collect_observations, fixed_trace
from experiments.resource_curves.detector import fit_detector
from experiments.resource_curves.evaluation import evaluate
from experiments.resource_curves.storage import CACHE_ROOT, DATA_ROOT, RUN_ROOT, digest

mode = os.environ['RESOURCE_STUDY']
if mode == 'auxiliary':
    studies = [('calibration', calibration_curve(), (2,)),
               ('fitting', fitting_curve(), (2,))]
elif mode == 'query':
    studies = [('query', (AuxiliaryBudget(400, 200),), (1, 2, 4, 8, 16))]
else:
    raise ValueError('RESOURCE_STUDY must be auxiliary or query')

data_root = Path('/home/mxd/lib/SD_MIA-pretraining-data/resource_curves')
role = 'draft_auxiliary_distilled'

def run_condition(benchmark, seed):
    key = Path('qwen3') / benchmark / 'epoch1' / f'seed{seed}' / role
    run = MODEL_PAIRS['qwen3'].run_root / benchmark / 'epoch1' / f'seed{seed}'
    folder = data_root / benchmark / f'seed{seed}'
    extension = json.loads((folder / 'extension/EXTENSION.json').read_text())
    shared = json.loads(Path(extension['shared_split_path']).read_text())
    cfg, base = prepare_records(run, 'plain', role)
    if not cfg.seed == cfg.data_seed == shared['seed'] == extension['seed'] == seed:
        raise ValueError('model/data/extension seed mismatch')
    tokenizer_source = f'{cfg.draft_model}@{cfg.draft_revision}'
    extra = extension_records(extension, base.tokenizer, tokenizer_source)
    sources = {'frozen': sources_for(run, ['target', role], adapter='plain'),
               'extension_digest': digest(extension)}
    adapter = load_adapter(run, 'plain', 'cuda:0', role)
    check_sources_light(sources['frozen'])
    try:
        for name, budgets, multiplicities in studies:
            study = build_study(shared, extension, budgets, name=name)
            save_study(DATA_ROOT / key / name, study)
            prepared = prepare_study(base, extra, extension, study)
            # Warm up on a fitting auxiliary only; exclude this from collection timing.
            first_id = study['points'][0]['partitions']['train'][0]
            record = prepared.records[prepared.record_ids.tolist().index(first_id)]
            for b in multiplicities:
                fixed_trace(adapter, protocol_prompt_ids(record, prepared.tokenizer),
                            list(record.response_ids), seed=seed, multiplicity=b)
                observations = collect_observations(
                    prepared, adapter, CACHE_ROOT / key / name / f'b{b}',
                    sources=sources, seed=seed, multiplicity=b,
                )
                for point in study['points']:
                    detector = fit_detector(
                        observations, point, CACHE_ROOT / key / 'detectors',
                        seed=seed, device='cpu', epochs=30,
                    )
                    budget = point['budget']
                    label = f"b{b}_fit{budget['fitting']}_cal{budget['calibration']}"
                    report = evaluate(observations, point, detector,
                                      RUN_ROOT / key / name / label, metric_seed=seed)
                    print(benchmark, seed, name, label, report['metrics'], flush=True)
    finally:
        del adapter
        gc.collect()
        torch.cuda.empty_cache()

for benchmark in ('wikitection', 'newstection', 'arxivtection'):
    for seed in (1919, 1949, 1978):
        run_condition(benchmark, seed)
PY
```

报告位于 `artifacts/audits/resource_curves_v1/tasks/qwen3/<dataset>/epoch1/seed<seed>/draft_auxiliary_distilled/<curve>/<point>/REPORT.json`。
中间观测和检测器在同批次 `intermediate/`，划分在 `splits/`。

校准量曲线会复用相同的拟合检测器，因此固定测试的原始 AUC/pAUC 应保持不变；
该曲线主要看独立校准后的 TPR、实际 FPR 和校准分辨率。拟合量曲线会重新拟合检测器。
查询曲线报告的时间是本地固定候选协议实现成本，不等于部署系统的网络延迟或投机解码加速。

## 建议执行次序

可以先运行 Pythia 的一个领域/seed 确认 GPU 端到端可用，再完成 MIMIR 和 Qwen 时间代理矩阵。
随后运行辅助量与查询曲线；DP 单独安排完整训练时间，再进行审计与同条件比较。
所有参考与审计都使用当前代码和独立输出目录，避免混用旧来源指纹的结果。
