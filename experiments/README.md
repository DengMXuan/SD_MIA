# 实验代码

[实验指南](../docs/experiments.md)是当前启动命令、分类、配置和多 GPU 用法的统一入口。

- `scripts/`：按 effectiveness / robustness / ablation / training / data / diagnostics 分类的正式 shell 入口。
- `launchers/`：编排、训练队列与独立 benchmark 调度；复用 `shared/core/gpu_pool.py`。
- `pretraining/benchmarks/`：MIMIR 13-gram、MIMIR 7-gram、WikiMIA 的独立配置与数据准备。
- `shared/`：模型、数据、训练、协议、方法、结果校验及调度实现。
- `sd_membership_sft/`、`cross_model_audit/`、`dp_defense/`、`pretraining/`、`resource_curves/`、`model_quality/`：各类实验实现及协议。
- `baseline/`：target-only 算法；`figures/`：CPU 图表；`maintenance/`：产物维护。

旧 shell 入口是兼容链接。回归测试集中在根 `tests/`，生成产物集中在根 `artifacts/`。
详见 [代码结构](../docs/code_structure.md)、[产物目录](../docs/artifact_layout.md)。
