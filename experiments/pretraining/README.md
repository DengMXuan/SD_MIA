# 预训练场景：Pythia / MIMIR 与 Qwen 时间划分

当前主方法入口为 `experiments.pretraining.evaluation.evaluate_main`。它复用
`shared.audit.fixed.run_prepared_main`，采用固定候选 B=2、非成员 TCN、正向稀疏评分，
与受控 SFT / DP 主方法共用采集、拟合、评分、指标和成本实现。目标和草稿语言模型始终冻结，
不进行微调；检测器仍需在独立非成员样本上拟合。没有新增实验脚本或调度器。

此前的 p/q、Q/H 提取和 M1 工作流保留在 [LEGACY.md](LEGACY.md)，它们不是这里的当前主方法。
其分区、检测器和缓存不能用于本入口。

## 支持的模型与标签

| 场景 | 目标 / 独立草稿 | 数据标签 |
|---|---|---|
| MIMIR | Pythia 6.9B / 1.4B，standard 非 deduped 版本 | 官方 train=member，test=nonmember |
| Qwen 时间划分 | Qwen3-8B-Base / Qwen3-1.7B-Base | 历史文本为 presumed member，发布后新文本为 nonmember proxy |

Pythia 默认 revision 固定为 `c0e3eee36dc47af0c49f361c74cfe459c09f7f23`（6.9B）和
`fedc38a16eea3bd36a96b906d78d11d2ce18ed79`（1.4B），对应已缓存的 standard 最终模型；
官方模型卡说明 main 对应 step143000。已有 manifest 仍使用其中记录的版本，不自动迁移。
两者都训练过 The Pile，因此草稿不是受控 SFT 中的 member-blind 辅助草稿。

Qwen 模型 revision 取自共享模型注册表，使用原始 Base 模型和独立 1.7B 草稿，
不依赖 SFT 结果文件，也不使用 EAGLE 头。两个模型必须有相同的完整 tokenizer。
目标训练成员关系未知；报告明确写入 `membership_verified=False`，不能把时间代理 AUC
直接写成有真实成员标签的 MIA AUC。文章发布时间也不能排除旧内容被复制到新页面。

## 数据准备接口

`experiments.pretraining.datasets` 提供以下函数；它们仅加载 tokenizer，不加载语言模型权重。

```python
prepare_mimir(
    member_file, nonmember_file, output_dir,
    source=..., split=..., seed=1919, n_per_class=...,
    n_aux=600, max_tokens=512, models=None, source_provenance=None,
)

prepare_temporal(
    historical_files, recent_pool, output_dir,
    seed=1919, historical_format="wikitext103",
    historical_provenance={
        "published_before": "2016-09-26",
        "reference": "https://arxiv.org/abs/1609.07843",
    },
    n_per_class=2000, n_aux=600, min_tokens=512, max_tokens=512,
)
```

MIMIR 输入为官方缓存 JSONL（JSON 字符串，或含 `text` 的对象），保留原标签和来源行号。
辅助集从官方非成员中独立预留，不与测试集重叠。`source_provenance` 可记录下载版本；
未提供时明确标为用户提供的官方缓存，不伪造下载证明。原有下载器
`experiments.pretraining.prepare` 仍可获取固定版本
`iamgroot42/mimir@02500d3b7cece0cb7628e939ba9fc93fdb6362ae`，不新增下载脚本。
本次该固定版本端点返回 HTTP 401，尚无真实 MIMIR 数据冻结产物；需要先取得官方缓存。

规模须显式选择：每类 1000 条的缓存无法支持 2000+2000 测试和 600 辅助样本。
例如可选每类 300 条测试、600 条辅助，或取得更大的官方缓存；过滤后不足会报错，不会缩减
或借用测试样本。`full_pile` 的混合语料结果不能当作某个单一领域结果。

时间数据支持两种历史来源：

- `wikitext103`：按顺序读取完整 raw train parquet 分片，重建跨分片的整篇文章，
  去掉标题/章节标题，保留每篇前 12,000 字符，再按 tokenizer 截断。必须提供早于模型发布日
  的出版证明。需要可选依赖 `pretraining` 中的 pyarrow。
- `historical_wiki`：已有采集器生成的 JSONL 和相邻 `.manifest.json`，校验文件哈希、
  `presumed_member_temporal_proxy` 标签语义、正 revision ID，以及创建和快照日期。

新文本复用 WikiTection pool 及其 `.manifest.json`，逐条检查创建日期严格晚于
Qwen3 发布日 **2025-04-29**。历史和新文本之间、测试和辅助之间统一做模型可见 token
精确去重与 13-gram 近似去重。跨角色文档 ID 不得重复。

`min_tokens` 默认 128；已准备的正式候选数据显式设为 512，使所有角色都用 512 个文本 token，
即首 token 作上下文、511 个评分 token。WikiText raw 的标点/空格处理和精选文章来源仍不同于
新 Wikipedia，长度统一不能消除全部来源偏差。见 [数据清单](DATA_INVENTORY.md)。

两种准备函数都原子发布 `records.jsonl` / `manifest.json`，拒绝覆盖已有目录。
记录来源、哈希、原标签、选样 seed、模型版本、tokenizer 指纹和实际 token ID。
原文不套聊天模板，不插入 BOS/EOS；Pythia 补齐的 LM-head 行不属于真实词表，
在概率归一化前排除。此约定随数据与报告保存。

## 当前主方法调用

```python
from experiments.pretraining.evaluation import evaluate_main

report = evaluate_main(
    manifest_path, output_dir,
    seed=1919, device="cuda:0", detector_epochs=30,
    detector_train=320, detector_validation=80, calibration=200,
)
```

这是一个条件的功能调用；用户后续脚本负责选择 GPU、数据集和 seed。
`seed` 必须与 manifest 的 `selection_seed` 相等，可分别准备 1919、1949、1978。
辅助分区、逐文档接受反馈、TCN 和 AUC bootstrap 都从该条件 seed 确定。
本场景没有微调 seed；不借用某个 SFT 模型或其成员分配。

默认 600 个辅助非成员按 320/80/200 分配给检测器训练、验证与校准，所有成员及
测试非成员只用于最终测试。可显式调整三个辅助分区大小，但必须全部为正并恰好耗尽辅助集。
减少校准集会降低可达到的低 FPR 分辨率，测试应报告实际样本量。

`local_files_only=True` 为默认值，模型须已缓存；只有显式设置为 False 才允许下载。
Hub 模型须固定完整 commit revision；也可使用本地完整 checkpoint 目录。
实际加载的是校验过的快照目录。源代码、数据、tokenizer 和模型资产均绑定来源指纹。
同参数重入可恢复逐文档观测、检测器或已完成报告；参数、来源或数据改变必须换输出目录。
输出需与源码、模型、数据分离，已有无归属结果不会被接管；同一输出目录有独占锁。

产物包括：

- `MAIN_REQUEST.json`、`PARTITIONS.json`：请求与各分区文档 ID。
- `observations.npz`、`detector.pt`、`FIT.json`：草稿特征、接受反馈及非成员检测器；
  观测中不保存目标概率。
- `main_fixed_sparse_positive/REPORT.json`、`scores.npz`：校验后的报告和分数。

报告提供 AUC、原始及归一化 pAUC@10%、ROC TPR@1%/10% FPR；另列独立校准后的
TPR 和实际 FPR，不能混用这两种指标。成本沿用准备/校准/测试拆分，包含检测器拟合；
模型加载与预热不计入稳态评分成本。参考实现的时间不代表生产投机解码的加速。

## 验证范围

`tests/pretraining/test_current_main.py` 使用真实随机初始化的小型 GPT-NeoX 与 Qwen3 模型，
验证冻结权重、无目标概率泄漏、三种 seed、分区隔离、同 seed 冷启动确定性、缓存恢复、
篡改拒绝，以及历史文章跨分片重建和日期检查。它验证功能，不证明大模型效果。
本次未运行 Pythia/Qwen 正式审计、语言模型微调或效率实验。
