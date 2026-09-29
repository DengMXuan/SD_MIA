# SD-MIA

基于投机解码接受反馈的训练数据成员审计。主方法固定候选、默认 B=2，使用独立非成员数据训练、验证和校准检测器。

- [从全新检出复现](docs/reproduction.md)：安装、获取固定版本模型与数据、准备 MIMIR、运行主方法和基线。
- [公开源码快照](docs/publication.md)：排除旧 Git 历史中的非源码材料。
- [实验指南与脚本目录](docs/experiments.md)：有效性、鲁棒性、消融、训练和数据准备。
- [主方法接口](docs/main_method_api.md)与[方法说明](docs/methodology_outline_zh.md)。
- [代码结构](docs/code_structure.md)与[产物目录](docs/artifact_layout.md)。

要求 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。从仓库根目录安装并运行不需要模型权重的检查：

```bash
uv sync --locked --extra pretraining --extra dp
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q
```

正式入口位于 `experiments/scripts/`。审计和训练默认只打印计划，显式 `run` 才运行计算。以 Qwen3 epoch 1 KD 审计为例，运行前必须先准备数据池、训练护照和对应检查点；全新检出时 `dry-run` 会报告条件未就绪并退出 2：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh dry-run
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh run --gpus 0
```

模型、数据与实验产物不随源码发布。默认的模型加载是离线的；首次运行须按[复现指南](docs/reproduction.md)获取固定版本外部资产。可设置 `SD_AUDIT_PYTHON` 指向现有 Python 环境。
