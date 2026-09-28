# SD-MIA

基于投机解码接受反馈的训练数据成员审计。主方法固定候选、默认 B=2，仅使用独立非成员训练、验证和校准检测器。

- [实验指南与脚本目录](docs/experiments.md)：有效性、鲁棒性、消融、训练和数据准备。
- [主方法接口](docs/main_method_api.md)与[方法说明](docs/methodology_outline_zh.md)。
- [代码结构](docs/code_structure.md)与[产物目录](docs/artifact_layout.md)。
- [论文展示方案](docs/experiment_presentation_plan_zh.md)、[历史方法开发](docs/history/method_development.md)。

正式启动脚本统一位于 `experiments/scripts/`，默认 dry-run；只有显式 `run` 启动实验。先查看计划，再选择空闲 GPU：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh dry-run
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_epoch1_kd_b2.sh run --gpus 3 4
```

使用项目 `.venv`（Python 3.12）；可设置 `SD_AUDIT_PYTHON` 指向已有环境。测试：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q
```
