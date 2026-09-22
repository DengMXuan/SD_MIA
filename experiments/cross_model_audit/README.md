# 独立跨模型验证入口

本目录为后续跨模型验证提供可选入口。**现有 `sd_membership_sft/run_qwen_audit_matrix.sh`、共享模块、baseline 实现及当前 Qwen 结果均未修改。** 本目录不在旧 Qwen 的 `runtime_files()` 指纹范围内，因此新增这里的代码不会使当前结果或断点失效。新进程也不替换、修改旧模块的全局函数。

只运行固定候选主方法和 11 个 baseline，不调度自然 SD，不触发语言模型训练或 DP 训练。检测器仍仅用非成员数据拟合。算法复用现有的 TCN 拟合、正向稀疏评分、分区、指标、baseline 打分和 SD 协议实现；独立维护模型注册、草稿头路由和批量调度。

## 支持的模型

| `--model-pairs` | 目标与草稿 | 两种草稿分支 |
|---|---|---|
| `gemma4`（默认） | Gemma 4 12B / E2B | `draft_auxiliary_distilled`、`draft_member_sft` |
| `qwen3_8b_eagle3` | Qwen3 8B / EAGLE-3 | `auxiliary_head`、`member_head` |
| `llama31_8b_eagle3` | Llama 3.1 8B / EAGLE-3 | `auxiliary_head`、`member_head` |
| `qwen35_9b_mtp` | Qwen3.5 9B / native MTP | `auxiliary_head`、`member_head` |
| `qwen3`（显式选择） | Qwen3 8B-Base / 1.7B-Base | 两种独立草稿；在新目录独立评估 |

注册表固定模型名称和训练时 revision。预检核对训练配置、共享划分、tokenizer 审核、权重分片，以及 EAGLE/MTP 所选 head 的完成标记、来源版本、成员/辅助分支和冻结目标绑定。baseline 只依赖目标阶段，不要求草稿头完成。

MTP 沿用现有 depth=1 协议适配器；EAGLE 沿用现有层融合与词表映射。它们是完整上下文重建的参考实现，耗时不代表生产 SD 引擎的 KV-cache 优化性能。额外记录隐藏状态字节数及固定候选覆盖数量，便于解释不同协议的成本和可观察位置差异。

## 使用

自行编写实验脚本时，使用 `inspect_run` / `evaluate_main`，普通与 DP 检查点共用同一入口；见[主方法与 DP 调用接口](../../docs/main_method_api.md)。单次调用只评估一个条件的一种草稿，不调度矩阵或 baseline。

从仓库根目录运行。以下 `status`/`dry-run` 只读，不加载模型权重、不做推理、不写实验输出：

```bash
bash experiments/cross_model_audit/run.sh status --model-pairs gemma4

bash experiments/cross_model_audit/run.sh dry-run \
  --model-pairs gemma4 qwen3_8b_eagle3 llama31_8b_eagle3 qwen35_9b_mtp
```

未来准备正式验证时，先选择一个配置：

```bash
bash experiments/cross_model_audit/run.sh run \
  --model-pairs gemma4 \
  --benchmarks newstection --epochs 1 --seeds 1919 --gpus 0
```

或在指定 GPU 上调度四类其他模型的完整矩阵：

```bash
bash experiments/cross_model_audit/run.sh run \
  --model-pairs gemma4 qwen3_8b_eagle3 llama31_8b_eagle3 qwen35_9b_mtp \
  --gpus 0 1 2
```

默认三个数据集、epoch 1/3、seed 1919/1949/1978。每类模型有 18 个目标条件、36 个草稿配置、54 个 worker 任务、234 个独立方法结果；baseline 在两个草稿分支展示时复用，形成 432 行汇总。更换 GPU 列表不改变实验签名。

`--model-root` 只用于单类模型的训练目录覆盖，多类模型使用注册表中的各自目录。模型仍从 `artifacts/runs/training/controlled_sft_v2/{model_pairs,speculator_matrix}/` 的训练记录和模型链接读取。

**不要用新入口接续当前 Qwen 审计。** 当前实验继续使用原命令。新入口显式选择 `qwen3` 时会创建独立评估结果，不读取或覆盖原 Qwen 结果；输出参数如果指向原 Qwen 审计目录或其父/子目录、兼容链接，会被拒绝。

## 保存与恢复

```text
artifacts/runs/audits/cross_model_fixed_v1/
  <model_pair>/<dataset>/epochN/seedN/
    baseline/<method>/{REPORT.json,scores.npz}
    <draft_role>/fixed/<method>/{REPORT.json,scores.npz}
  executions/<attempt>/{TASK.json,STATUS.json,worker.log}
  fixed_only_summary/{RESULTS.csv,RESULTS.md,SEED_SUMMARY.csv,SUMMARY.json}

artifacts/cache/audits/cross_model_fixed_v1/
  <model_pair>/<dataset>/epochN/seedN/<draft_role>/fixed/
    trajectories/  observations.npz  observations.npz.json  detector.pt  FIT.json
```

输出按模型隔离；跨 seed 汇总也按模型分组，不把不同模型当作额外 seed。包含 AUC、pAUC@10%FPR、ROC TPR@1%/10%FPR、独立校准下实际 FPR/TPR，以及时间、吞吐、显存和查询/Token 数。效率口径继承现有矩阵，baseline 展示复用不会重复计入物理成本。

重复相同命令会校验并跳过完整方法；主方法复用逐条轨迹、完整观测和已拟合检测器。baseline 未保存的单个方法需要重跑。新结果的来源指纹同时包含本目录和所调用的共享算法，不绕过来源校验。自定义输出根目录的缓存保存在任务自身的 `cache/`。

```bash
bash experiments/cross_model_audit/run.sh summarize --model-pairs gemma4
```

未完成全部任务时，`summarize` 仍输出部分结果，返回码为 2；摘要会列出未完成项及错误。不同 `--model-pairs`/数据集选择的汇总反映本次选择范围，原始方法结果不变。

## 当前验证范围

EAGLE/MTP 主方法在首次采集前，用检测器拟合集合中的一条辅助记录检查完整上下文与前缀预测的一致性、概率归一化和固定候选轨迹。未通过检查则停止；复用缓存也要求已有通过记录。

已验证五类模型全部 270 个任务的只读预检、双 head 路由、模型/权重来源校验、独立输出保护、跨模型汇总及缓存/检测器恢复；使用 CPU 合成观测测试方法完整评分路径。未启动新模型的正式 GPU 审计，实际推理兼容性和显存峰值需在未来首个真实配置运行时确认。

```bash
.venv/bin/python -m pytest -q experiments/cross_model_audit/tests
```
