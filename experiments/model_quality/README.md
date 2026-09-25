# 模型资产评估

执行五对已有模型的 epoch 1 目标泛化性和 KD 草稿接受率评估。
完整协议、短文档处理、seed 规则、产物结构与恢复行为见
[评估方案](../../docs/model_asset_evaluation.md)。

```bash
.venv/bin/python -m experiments.model_quality.cli dry-run
.venv/bin/python -m experiments.model_quality.cli run --gpus 0 1 2 3
.venv/bin/python -m experiments.model_quality.cli status
.venv/bin/python -m experiments.model_quality.cli summarize
```

默认 45 个条件、90 项任务；按实际空闲 GPU 修改列表。dry-run/status 只读，
不会启动推理。仅运行一项可加 `--evaluations generalization` 或
`--evaluations acceptance`；相同命令可续跑。

默认使用新批次 `artifacts/evaluations/model_quality_v2/`。旧 `model_quality_v1`
已归档，只保留查阅兼容路径；新批次从零计算，不复用旧缓存。正式推理保持
BF16；校验会记录数值漂移，并在需要时执行 FP32 复核。
