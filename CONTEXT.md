# SD-MIA 项目约定

运行入口与配置以 [实验指南](docs/experiments.md) 为准；不要在此复制实验进度、命令和结果数字。

- 当前主方法为固定候选、原始前缀上的 accept-only 审计，默认 B=2；用非成员训练 320、验证 80、独立校准 200。成员标签只用于最终评估，不选模型、方向、阈值或超参数。
- 目标与草稿在审计期间冻结。受控 SFT、预训练成员基准和时间代理标签分别解释；时间标签不表示已验证的训练成员身份。
- seed 与模型、数据划分、随机反馈、检测器和 bootstrap 一一对应。重复 seed 不等于独立数据集。
- 正式 shell 入口统一在 `experiments/scripts/`，名称包含目的、方法角色和主要配置。共享实现保持单份；新实验不另建临时 `standalone/` 项目。
- GPU 实验按独立条件动态排队。数据准备、状态和 dry-run 不启动 GPU；正式计算使用显式 `run`。
- 模型、数据、分数和源码均保留来源校验。整理启动脚本不改变数值实现；改变实验参数或源码后使用新输出批次，不修改旧产物绕过校验。
- 代码在 `experiments/`；冻结兼容实现见 `standalone/README.md`；回归测试统一在 `tests/`；生成数据和结果在 `artifacts/`。

详细说明：[复现指南](docs/reproduction.md)、[方法接口](docs/main_method_api.md)、[方法论](docs/methodology_outline_zh.md)、[代码结构](docs/code_structure.md)、[产物目录](docs/artifact_layout.md)、[架构决策](docs/adr/)。
