# 从全新检出复现

本文从源码和公开外部资产开始。仓库只包含代码、固定配置、测试和协议文档；数据池、模型快照、训练检查点及报告写入被 Git 忽略的 `artifacts/`。历史实验报告及其来源哈希不会在全新检出后自动出现。

## 环境与 CPU 检查

要求 Python 3.12、[uv](https://docs.astral.sh/uv/) 和用于正式推理的 CUDA GPU。从仓库根目录执行：

```bash
uv sync --locked --extra pretraining --extra dp
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q
```

只运行预训练实验时可省略 `--extra dp`。正式 shell 入口默认使用仓库 `.venv/bin/python`；已有兼容环境可通过 `SD_AUDIT_PYTHON` 指定。CPU 测试使用小模型或替身，不证明大模型 GPU 推理、速度或数值结果已复现。

## Pythia / MIMIR：一条独立的主方法与基线路径

该路径不需要先训练语言模型。目标为固定 revision 的 `EleutherAI/pythia-6.9b`，草稿为 `EleutherAI/pythia-1.4b`；版本号定义在 [`pretraining/data.py`](../experiments/pretraining/data.py)。MIMIR 使用官方 `iamgroot42/mimir` 数据集的固定 revision；其访问可能需要在 Hugging Face 页面同意条款，并在本机 SDK 登录。不要把访问令牌写进仓库。

先获取两个模型的固定快照。以下命令只下载模型，不运行推理：

```bash
HF_HUB_OFFLINE=0 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from experiments.pretraining.data import TARGET, DRAFT
for model in (TARGET, DRAFT):
    snapshot_download(repo_id=model['repo_id'], revision=model['revision'])
PY
```

以下用 `github`、seed 1919 演示从官方 13-gram 缓存生成冻结 manifest。扩大矩阵时，对其他领域及 seed 1949、1978 重复同一准备过程；各领域的测试/辅助规模见 [预训练协议](../experiments/pretraining/README.md)。下载和冻结代码会记录官方来源、数据文件哈希、模型版本和 tokenizer 指纹。

```bash
HF_HUB_OFFLINE=0 .venv/bin/python - <<'PY'
from pathlib import Path
from experiments.pretraining.datasets import download_mimir, prepare_mimir

root = Path('artifacts/pretraining_inputs/mimir')
cache = download_mimir(
    source='github', split='ngram_13_0.8', cache_size=1000,
    local_dir=root / 'official', local_files_only=False,
)
prepare_mimir(
    **cache, output_dir=root / 'prepared/github/seed1919',
    seed=1919, n_per_class=400, n_aux=600,
)
PY
```

`dry-run` 会核对 manifest，不会加载模型；只有显式 `run` 才调度 GPU。将 `0` 替换为可用的逻辑 GPU 编号：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_pythia_mimir13gram08_fullpile_b2.sh dry-run \
  --data-root artifacts/pretraining_inputs --sources github --seeds 1919
bash experiments/scripts/effectiveness/effectiveness_main_pythia_mimir13gram08_fullpile_b2.sh run \
  --data-root artifacts/pretraining_inputs --sources github --seeds 1919 --gpu 0
bash experiments/scripts/effectiveness/effectiveness_baseline_pythia_mimir13gram08_fullpile_seven.sh run \
  --data-root artifacts/pretraining_inputs/mimir/prepared \
  --main-root artifacts/audits/pythia_mimir_v1/tasks \
  --sources github --seeds 1919 --gpu 0
```

主方法输出位于 `artifacts/audits/pythia_mimir_v1/tasks/github/seed1919/`，基线输出位于 `artifacts/audits/pythia_mimir_baselines7_v1/tasks/github/seed1919/`。二者分别保存请求、来源校验、分数和报告。重新运行相同命令会校验并复用完整条件；改变数据、模型或源码时使用新的输出根目录。更多领域、full-Pile 和汇总命令见[实验指南](experiments.md)与[基线协议](pretraining_baselines.md)。

MIMIR 7-gram 低重叠实验要先准备同领域 13-gram 的官方缓存和 seed1919 manifest；上面的 `artifacts/pretraining_inputs/mimir` 即是 `--auxiliary-root` 所需结构。然后运行：

```bash
bash experiments/scripts/data/prepare_pythia_mimir7gram02.sh \
  --source github --auxiliary-root artifacts/pretraining_inputs/mimir --allow-download
bash experiments/scripts/robustness/robustness_main_pythia_mimir7gram02_b2.sh dry-run \
  --sources github --seeds 1919
```

准备入口还支持 `--data-root` 和 `--official-root`；它们分别控制冻结 7-gram 输出和官方缓存位置。`--allow-download` 只下载缺少的 7-gram 文件，不会替代必需的 13-gram 辅助来源。完整条件执行命令见[实验指南](experiments.md#三个独立-pythia-benchmark)。

## Qwen 受控微调与 DP

Qwen 审计须先取得 WikiTection、NewsTection、ArxivTection 数据池，按固定 seed 生成共享划分，并训练目标及对应草稿。数据采集入口位于 `experiments/scripts/data/`，固定模型版本在 `experiments/sd_membership_sft/scripts/model_pair_revisions.env`，训练入口位于 `experiments/scripts/training/`。从仓库根目录依次查看[实验指南](experiments.md)、[SFT 协议](../experiments/sd_membership_sft/README.md)和[DP 协议](../experiments/dp_defense/README.md)，先运行各入口的 `dry-run` / `preflight`，完成训练后再运行审计。

README 中的 Qwen 示例在全新检出时显示 `pending: training passport is not ready` 并退出 2；这是缺少先决数据和检查点的状态。审计入口不会自动下载大型模型，也不会从历史报告重建权重。训练和评估默认离线加载固定模型版本；第一次运行前须在本机缓存对应版本。

## 复核边界

报告中的完成数和指标只能由相应 `artifacts/` 重新生成并校验。测试通过、`dry-run` 可规划，以及从官方数据准备 manifest，分别验证不同阶段；任何一步都不能代替真实 GPU 推理。外部数据集的访问权限、模型发布状态与硬件环境可能影响从零运行，程序会对缺失资产或来源不一致明确报错。
