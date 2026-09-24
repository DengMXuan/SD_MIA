# 共享实验框架

此包维护跨基座、草稿类型和防御实验复用的实现。实验入口只负责选择条件、设备及结果批次；数据合同、模型加载、主方法、调度和报告在这里维护。

- `models/model_pairs.json`：固定模型身份；`models/adapters.py`：草稿家族加载与协议接口。
- `data/`、`training/`、`drafts/`：数据准备、普通训练与草稿实现。
- `protocols/`、`methods/`：观测协议、检测特征、拟合与评分。
- `audit/evaluation.py`：单条件 `inspect_run` / `evaluate_main` 与训练校验接口。
- `audit/scheduler.py`、`audit/reporting.py`：所有普通矩阵共享的调度和聚合。
- `audit/config.py`：统一审计默认参数及 baseline 执行设置。
- `core/`：底层合同、缓存、指标与运行支持。

此包不导入 `sd_membership_sft`、`cross_model_audit` 或 `dp_defense` 入口。需要训练机制特有验证时，由调用者注入。完整扩展步骤、兼容边界和测试入口见 [代码结构与扩展指南](../../docs/code_structure.md)。
