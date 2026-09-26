# 代码结构与扩展指南

代码放在 `experiments/`，测试统一在 `tests/`，生成数据与结果放在 `artifacts/`。产物路径约定见 [全生命周期目录](artifact_layout.md)，路径函数集中在 `experiments/paths.py`。

## 目录职责

```text
experiments/
  paths.py                   # 产物根目录与阶段路径规则
  shared/
    core/                    # 数据合同、指标、分区、缓存与通用运行支持
    data/                    # 数据池、冻结划分、tokenizer 验证
    evaluation/              # 目标泛化性、KD 接受率、冻结数据与断点恢复
    models/                  # 模型身份、草稿家族、加载与就绪检查
    training/                # 普通受控 SFT 配置、训练和记录重建
    drafts/                  # plain / EAGLE-3 / MTP 训练与底层实现
    protocols/               # 协议、观测采集与归档
    methods/                 # 检测器、特征、评分与校准
    audit/                   # 单条件评估、调度、来源验证、成本与报告
  baseline/                  # target-only 算法及独立 CLI
  sd_membership_sft/         # Qwen 入口、冻结训练脚本、预检、分析与历史方法
  cross_model_audit/         # 跨模型任务选择、CLI 与 worker 入口
  dp_defense/                # DP 训练、会计、校验、矩阵与防御前后比较
  resource_curves/           # 查询和辅助数据规模实验
  pretraining/               # 预训练审计的数据合同与入口
  maintenance/               # 产物迁移及核验工具
  MODULE_ALIASES.json         # 显式旧模块名映射
  _compat.py                 # 唯一兼容加载器
tests/
  architecture/ baseline/ cross_model/ data/ dp_defense/
  methods/ pretraining/ resource_curves/ sft/ storage/ training/
```

依赖方向是实验入口调用共享实现。`shared/` 不导入 SFT、跨模型或 DP 入口；防御专属证明通过 `RunVerification` 注入。共享 baseline 审计适配器调用 `baseline.engine.score_methods`，不调用命令行模块的私有函数。

Qwen 和跨模型的任务选择、就绪策略可以不同，但 worker 调度只在 `shared/audit/scheduler.py` 实现，矩阵聚合只在 `shared/audit/reporting.py` 实现。DP 报告附加隐私信息并保留专属比较 schema，检测方法仍复用同一实现。WS/RS/BT 共享原始生成的物理增量成本不填入独立方法成本列；展示到多个草稿角色的 target-only 结果按执行组去重计费。

## 新增模型基座

1. 在 `shared/models/model_pairs.json` 添加唯一组合名，填写 `name`、`adapter`、`target`、`draft`、`target_revision` 和 `draft_revision`。使用明确的固定版本；草稿家族须已注册。
2. 沿用数据划分与训练护照合同准备训练结果。普通训练实现位于 `shared/training/`，草稿训练位于 `shared/drafts/`。现有 shell 脚本是冻结配方，注册新组合不会自动创建其训练脚本；应显式定义训练参数及检查模型支持情况。
3. 用 `shared.audit.evaluation.inspect_run` 验证护照和草稿角色，再用跨模型 CLI 的 `--model-pairs <name>` 做只读规划。训练根目录由注册表统一解析，审计任务路径含组合名。
4. 先验证模型加载、tokenizer、协议和小规模推理，再执行完整矩阵。仅添加注册项不代表新的模型架构已经通过真实推理验证。

单条件调用见 [主方法 API](main_method_api.md)。现有普通训练的基础模型与草稿 revision 仍与冻结 shell 配方一致，架构测试检查该约束。

## 新增草稿模型类型

在 `shared/models/adapters.py` 注册一个 `DraftFamily`：定义角色名、检查点目录类别（`checkpoints` 或 `heads`）、草稿加载函数和协议工厂，并说明是否需要共享 tokenizer。新增实现放在 `shared/drafts/` 或 `shared/protocols/`；注册必须在模型目录表加载前完成，所有 worker 进程均需执行注册。

协议工厂返回满足采集器接口的对象，负责该草稿与目标之间的协议，不把模型名称分支放进调度器。角色必须准确对应检查点，不用辅助草稿替代缺失的 member 草稿。已有 plain、EAGLE-3、MTP 是三个参考实现。

家族注册覆盖加载/评估边界。全新草稿训练方式仍需实现训练配方、就绪验证及来源校验；超出现有两类检查点合同的模型还需扩展相应合同。DP 不会因草稿注册而自动成立：当前 DP head 训练只支持 EAGLE-3 和 MTP，其他类型明确拒绝，应先实现并验证其可训练参数、逐文档梯度、教师依赖与隐私组合规则。

## DP 与其他防御实验

`dp_defense` 保留 DP 训练和会计实现。`evaluation_verification()` 提供共享评估所需的 `RunVerification(verify, source_files)`：前者核验完整训练与隐私护照，后者补充来源指纹。带 DP 标记的检查点在没有验证器时被拒绝，不会被当作普通检查点处理。

新防御应在自己的实验包中实现训练与证明，复用模型、划分、主方法和报告。额外元数据或新防御护照需要明确扩展验证合同；`RunVerification` 不是绕过 DP 校验的开关。防御前后重新采集、拟合和校准，并用一致的模型、角色、审计设置及记录 ID 比较，避免把换数据或换检测设置造成的变化解释为鲁棒性。

DP sweep 支持 `--model-pairs`。矩阵目录包含 `<model_pair>/epsilon<预算>/<benchmark>/epoch<N>/seed<N>`，跨模型和跨隐私预算分别保存。见 [DP 操作说明](../experiments/dp_defense/README.md)。

## 预训练成员与时间代理评估

`pretraining/datasets.py` 冻结 MIMIR 官方标签或历史/近期文本代理标签，
`pretraining/evaluation.py` 校验预训练模型快照与独立非成员分区。
它直接调用 `shared.audit.fixed.run_prepared_main`；受控 SFT / DP 的 `run_main`
也调用同一实现。预训练场景无需伪造 SFT 护照或草稿训练分区，且不会重新实现 TCN 或评分。
准备、单条件评估和标签解释见 [预训练接口](../experiments/pretraining/README.md)。

## 兼容与验证

已移除的自然 SD 串行采集器及其旧别名不再提供入口。主方法名称 `main_fixed_sparse_positive`、输出路径及 B=2/TCN/校准规则保持不变。`starts` 和 `rounds_per_start` 仅作为旧请求身份字段保留，不再对应任何生成分支。

其余旧 Python 名称通过 `MODULE_ALIASES.json` 延迟解析到同一个正式模块对象；旧 `python -m` 入口继续转发到其 `main()`。内部代码只导入正式模块，不再维护 80 个 wrapper 或修改 `__path__`。旧 shell 转发入口和产物软链接继续保留。

来源哈希覆盖共享实现、模型目录表、baseline 实现和兼容清单。历史报告保持原始请求与来源指纹；重构不会重新签署历史结果。新代码运行应使用新批次，已有来源不匹配会明确拒绝恢复。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/architecture tests/dp_defense
```

架构测试约束依赖方向、内部正式导入、测试位置、兼容模块身份、扩展接入和来源覆盖；DP 集成测试覆盖现有五种组合的角色、预算附加与矩阵路径唯一性。CPU 和替身模型测试不代替真实 GPU 训练、推理及效率测量。

模型资产质量评估入口为 `experiments/model_quality/cli.py`；协议、矩阵范围和运行命令见 [模型资产评估方案](model_asset_evaluation.md)。
