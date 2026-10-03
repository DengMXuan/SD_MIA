# WikiTection ε=8 DP 模型质量与 KD 草稿接受率

固定范围：Qwen3-8B-Base DP 目标、Qwen3-1.7B-Base 辅助数据 KD 草稿，epoch 1，条件 seed 1919、1949、1978。每个 seed 有泛化性与接受率两项任务，共六项。只读取训练检查点；结果写入独立目录 `artifacts/evaluations/dp_quality_v1/qwen3/epsilon8/wikitection/epoch1`。

## 运行

```bash
bash experiments/scripts/diagnostics/dp_qwen3_epsilon8_wiki_quality.sh dry-run
bash experiments/scripts/diagnostics/dp_qwen3_epsilon8_wiki_quality.sh run --gpus 0
bash experiments/scripts/diagnostics/dp_qwen3_epsilon8_wiki_quality.sh status
bash experiments/scripts/diagnostics/dp_qwen3_epsilon8_wiki_quality.sh summarize
```

有多张空闲 GPU 时可传 `--gpus 0 1 2`；每张 GPU 同时运行一个任务。`run` 可续跑，先核对 DP 训练护照及完整权重哈希。也可用 `--seeds 1919` 或 `--evaluations acceptance` 选子集。若修改抽样量、batch size 或 bootstrap 次数，使用新的 `--output-root`。

结果包含每项任务的 `REPORT.json`、`REPORT.md`、`scores.npz`，以及 `reports/SUMMARY.json`、`conditions.csv`、`seeds.csv`。只有三 seed 的六项任务全部完成，默认 `summarize` 才返回成功。

## 指标口径

- 泛化性：在冻结的 member 和 nonmember 中各抽 500 条，以原始 Qwen3-8B 基座为同样本对照，贪心续写后计算 BLEU-4、ROUGE-1、ROUGE-L。报告微调目标的 member−nonmember 差值及 95% bootstrap 区间，还报告基座−微调目标的同样本质量差。它衡量 WikiTection 分布上的续写质量，不能直接代表其他任务的通用能力。
- 接受率：从 member、nonmember、KD auxiliary 各抽 256 条，报告每条文档上 `Σ min(p(v), q(v))` 的平均值、95% bootstrap 区间、top-1 一致率和原文 token 词表覆盖率。这里用原文前缀的教师强制条件计算**单 token 期望接受率**；它不是真实推测生成轨迹的实测接受率，也不能直接推出加速比。泛化讨论应主要看未参与训练的 nonmember，auxiliary 曾用于 KD。
- 隐私预算：报告绑定 target+KD 草稿的 ε、δ 和训练请求哈希。此配置草稿为目标模型的后处理，组合预算上限 ε=8。

抽样与 bootstrap 固定为各条件 seed。源文件、训练请求、检查点清单和结果校验和随报告保存；续跑会核对来源。质量评估不会改写训练护照、冻结划分或检查点。
