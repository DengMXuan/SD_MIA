# 冻结实现兼容区

正式启动脚本已统一到 [`experiments/scripts/`](../experiments/scripts/)，使用方法见 [实验指南](../docs/experiments.md)。这里保留已被数据/结果来源哈希引用的数值计算和采集实现；不再新增实验启动目录。旧 shell 入口指向正式脚本，测试已移到 `tests/pretraining/`。

- `pretraining_baselines`：[baseline 协议](../docs/pretraining_baselines.md)。
- `qwen_temporal_clean`、`qwen_gemma_temporal`：[时间代理协议](../docs/temporal_pretraining.md)。
- `qwen_temporal_mediawiki`：[语料构建](../docs/temporal_data.md)。
- `pythia_delta`、`pythia_evidence`：[delta 诊断](../docs/reports/pythia_delta.md)、[评分消融](../docs/reports/pythia_evidence.md)。

原 `paper_positive_controls` 的配置与数据准备已移到 `experiments/pretraining/benchmarks/`，拆成三个独立 benchmark，主方法与 baseline 分别调度。
