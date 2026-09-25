# 主方法与差分隐私实验调用接口

这些函数位于 `experiments.shared.audit.evaluation`，供实验脚本调用，不负责调度矩阵。DP 调用显式注入训练校验器；旧 `experiments.cross_model_audit.api` 仍作为自动注入该校验器的兼容入口。新增模型与草稿类型见 [代码结构与扩展指南](code_structure.md)。普通微调检查点与 DP 检查点使用同一个固定候选主方法：B=2、非成员 TCN、正向稀疏评分；检测器训练/验证/校准/测试规模仍为 320/80/200/4000。

## 支持范围

| 模型注册名 | 模型组合 | `draft_role` |
|---|---|---|
| `qwen3` | Qwen3 8B-Base / 1.7B-Base | `draft_auxiliary_distilled`、`draft_member_sft` |
| `gemma4` | Gemma 4 12B / E2B | 同上 |
| `qwen3_8b_eagle3` | Qwen3 8B / EAGLE-3 | `auxiliary_head`、`member_head` |
| `llama31_8b_eagle3` | Llama 3.1 8B / EAGLE-3 | 同上 |
| `qwen35_9b_mtp` | Qwen3.5 9B / 原生 depth=1 MTP | 同上 |

入口从 `results.json` 的模型名称、固定 revision 和协议记录识别组合，要求现有共享划分及完成标记。它不是任意 Hugging Face 检查点的通用导入器。

## 只读检查与普通主方法评估

从仓库根目录的 Python 环境调用：

```python
from pathlib import Path
from experiments.shared.audit.evaluation import inspect_run, evaluate_main
from experiments.shared.models.registry import MODEL_PAIRS

reference = MODEL_PAIRS["qwen3_8b_eagle3"].run_root / "wikitection/epoch1/seed1919"
info = inspect_run(reference)  # 只读检查两种草稿；不加载模型做推理
role = info["draft_roles"][0]
outputs = Path("artifacts/audits/main_api_example/tasks")
ordinary_output = outputs / info["model_pair"] / "ordinary" / role

# 以下调用实际进行推理与检测器训练；不会训练语言模型或启动 baseline。
ordinary = evaluate_main(reference, ordinary_output, draft_role=role, device="cuda:0")
```

`evaluate_main` 返回校验后的报告字典。方法报告在
`<output_dir>/main_fixed_sparse_positive/REPORT.json`，并保存分数、观测和检测器。
同参数、同来源重复调用会复用已完成阶段。每个模型、数据条件、草稿角色、隐私预算应使用独立目录；修改审计参数或源码后不得混用旧缓存。默认 `audit_seed=None` 表示继承对应训练/数据条件的 seed（1919/1949/1978），检测器、辅助集内部分配和 AUC bootstrap 共用该 seed；显式传入的 audit_seed 必须与条件一致。`detector_epochs` 默认 30。

EAGLE/MTP 首次采集前检查前缀一致性、概率归一化和固定候选协议。检测器仍只接收草稿特征与接受反馈；草稿头自身依赖目标隐藏状态，报告会明确这一访问条件。该参考实现的成本不能直接解释为生产推测解码加速。

## DP 规划、训练与同方法比较

DP 需要可选依赖：`uv sync --locked --extra dp`。

```python
from experiments.dp_defense.api import plan_private_training, train_private
from experiments.dp_defense.compare import compare_reports
from experiments.dp_defense.artifacts import evaluation_verification

private_run = Path("artifacts/training/dp_api_example/runs") / info["model_pair"] / "epsilon4"
request = plan_private_training(reference, private_run, epsilon=4, gpu=0)
# 只读规划：核对配方、来源与会计；不加载模型权重、不创建输出目录。

# 未来显式调用才会训练。可复用校验通过的完整阶段。
passport = train_private(reference, private_run, epsilon=4, gpu=0)
private_output = outputs / info["model_pair"] / "epsilon4" / role
private = evaluate_main(
    private_run, private_output, draft_role=role, device="cuda:0",
    verification=evaluation_verification(),
)

method = "main_fixed_sparse_positive"
comparison = compare_reports(
    private_output / method / "REPORT.json",
    ordinary_output / method / "REPORT.json",
)
```

DP 训练复用普通实验的冻结数据划分和配方，**从固定公开基础模型/原生头初始化**，不会把已非 DP 微调的权重继续训练后称作 DP 模型。目标做全参数 DP；独立 member 草稿做全参数 DP，member 草稿头只对其可训练参数做逐文档裁剪与加噪。辅助草稿/头用独立辅助集合向 DP 目标重新蒸馏，属于后处理。

`epsilon` 分别约束目标和 member 适配阶段；默认每阶段 δ=5×10⁻⁶。因此 ε=4 的辅助部署上限是 (4, 5×10⁻⁶)，member 部署基本组合上限是 (8, 10⁻⁵)。报告附实际会计结果。多版本联合发布需另行组合；成员 ID 等实验元数据不是 DP 发布物。完整边界见 [DP 文档](../experiments/dp_defense/README.md)。

普通与 DP 结果各自重新采集观测、拟合 TCN、校准和测试，不复用普通模型的检测器。`compare_reports` 要求模型、方法、角色、审计设置以及校准/测试 ID、标签、分区一致，返回各指标的 DP 减普通结果差值。旧 Qwen 报告缺省模型名可兼容，但不允许跨模型比较。

训练恢复以完整阶段为单位：目标或辅助阶段已完成则复用；中断的阶段从公开初始化重新训练。训练与评估均验证来源和输出归属，不覆盖参考模型或当前 Qwen 审计目录。

## 验证边界

已对五类现有 WikiTection/epoch1/seed1919 检查点执行只读检查与 DP 规划，并用 CPU 测试验证会计、头训练机制、各模型/角色的主方法评分、隐私报告和恢复。评分集成测试使用合成观测及真实 TCN 拟合，不等于真实大模型推理测试。

尚未运行完整 GPU DP 训练或新增跨模型审计；显存需求、完整模型推理兼容性及防御效果需未来实测。批量规划使用 `python -m experiments.dp_defense.sweep dry-run --model-pairs ...`，支持现有五种模型组合；本次未启动 GPU 实验。
