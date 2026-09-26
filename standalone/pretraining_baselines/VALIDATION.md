# 验证记录

2026-09-27，在独立工作树完成；没有启动真实模型 GPU 实验。

- CPU 集成验证：`test_baselines.py`，**10 passed**。覆盖真实微型 Pythia/Qwen target 的七种方法、权重冻结、target-only 加载、无 EOS/有效词表评分、参考/校准/测试隔离、三个条件 seed、完整及部分恢复、校验和及错配拒绝、CLI 数据集选择。
- MIMIR 全量 dry-run：**8 领域 × 3 seed = 24 条件、168 个基线结果**，全部 manifest 与已有主方法划分一致。
- Qwen 全量 dry-run：**2 清理版本 × 3 seed = 6 条件、42 个基线结果**；默认仅 `length_matched` 为 3 条件、21 个结果。
- 使用只写 `/tmp` 的汇总演练：**24 个 Pythia 主方法结果全部通过验证**；Qwen 两个清理版本的 6 个主方法结果均明确显示 missing。尚未运行的基线同样显示 missing，无 invalid 结果。缺少基线时命令按设计返回 2。
- `bash -n run.sh` 通过。

旧实验源码、旧数据和旧结果未改写。运行说明见 README.md；请指定空闲 GPU 后执行正式实验。
