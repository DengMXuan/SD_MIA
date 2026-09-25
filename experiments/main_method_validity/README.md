# 四模型 epoch 1 主方法实验

对 Gemma4、Qwen3 EAGLE-3、Llama 3.1 EAGLE-3、Qwen3.5 MTP 的已训练
epoch 1 目标和辅助数据草稿运行固定候选主方法。每对模型覆盖 WikiTection、
NewsTection、ArxivTection 与条件 seed 1919、1949、1978，共 36 项。
Gemma4 使用 `draft_auxiliary_distilled`，其余三组使用 `auxiliary_head`。
不调度 baseline、成员草稿、epoch 3 或模型训练。

从仓库根目录执行：

```bash
bash experiments/main_method_validity/run_four_models.sh dry-run
bash experiments/main_method_validity/run_four_models.sh run --gpus 0 1 2 3
bash experiments/main_method_validity/run_four_models.sh status
bash experiments/main_method_validity/run_four_models.sh summarize
```

运行时根据实际空闲 GPU 修改 `--gpus`。每张 GPU 同时一个 worker；调度器复用
现有 GPU 文件锁，因此会拒绝与正在运行的 Qwen 审计共用同一 GPU。相同 `run`
命令会从通过来源校验的结果续跑。`dry-run` 和 `status` 不创建实验产物，
也不加载模型推理。`run` 完成后会自动汇总；仅需刷新报告时再用 `summarize`。

默认输出是独立批次 `artifacts/audits/four_model_epoch1_main_condition_seed_v1/`：
`tasks/` 存放逐方法报告与分数，`intermediate/` 存放轨迹、观测和检测器，
`executions/` 存放日志，`reports/` 存放逐条件和跨 seed 汇总。
新入口的源码摘要绑定到每个任务请求；更改入口、参数或计算代码后使用新的
`--output-root artifacts/audits/<new_batch>/tasks`，不复用旧结果。

采集、TCN、辅助集 320/80/200 内部分配与 AUC bootstrap 均使用对应条件
seed。每条件使用 2000 member、2000 nonmember、600 audit auxiliary；
主方法的剩余协议与 [Qwen 专用实验](../sd_membership_sft/docs/QWEN_KD_EPOCH1_RERUN.md)
一致。新脚本不修改或读取 Qwen 当前批次的产物。
