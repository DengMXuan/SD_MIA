# 预训练扩展数据清单（2026-09-26）

本次仅准备数据与功能，没有运行真实模型的成员推断实验。
原工作区 `/home/mxd/lib/SD_MIA` 的代码、数据池、模型与实验结果未改动。

## 已准备的 Qwen 时间代理数据

持久目录：`/home/mxd/lib/SD_MIA-pretraining-data/`。该目录独立于原工作区及 `/tmp` 开发 worktree，
不由 Git 管理；同步代码分支后可直接使用这些绝对路径。
总清单为 `INVENTORY.json`，包含各 manifest 及 records 的 SHA-256。

| seed | 子目录 | 历史测试 / 近期测试 / 独立近期辅助 | 文本 token 数 |
|---|---|---|---|
| 1919 | `qwen3_wikitext_temporal_512_seed1919` | 2000 / 2000 / 600 | 全部 512 |
| 1949 | `qwen3_wikitext_temporal_512_seed1949` | 2000 / 2000 / 600 | 全部 512 |
| 1978 | `qwen3_wikitext_temporal_512_seed1978` | 2000 / 2000 / 600 | 全部 512 |

各子目录包含 `manifest.json` 和 `records.jsonl`。辅助集在后续主方法调用时按同 seed 分成
320 训练、80 验证、200 校准；所有测试记录独立于这三部分。不同 seed 是同一个候选池的
重复随机划分，不能视为三个互不重叠的数据集。首 token 为上下文，余下 511 token 评分。

记录文件 SHA-256：

```text
1919 15558caf336538c45edadeac294b1cfe77b24383ebc940d18bdc254bf8b26270
1949 86564e659e2f09d394a2b988d2c09d02dfc0e835122c379862aa99677821b15b
1978 c618356b27d012172ca0fef38b44af0aefdd79d0985f01d434969218b614b7b1
```

历史来源是本地已有的 `Salesforce/wikitext`，固定 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`，`wikitext-103-raw-v1/train` 的两个完整 parquet 分片。
缓存路径为 `/home/mxd/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/`
加上述 revision 和配置名。按分片顺序重建出 **29,443 篇文章**，不会把不同段落当独立文档。
原始论文 [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)
于 **2016-09-26** 公开，并介绍 WikiText 语料，满足早于模型发布的历史条件。

两个源分片的 SHA-256 分别为：

```text
train-00000 74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212
train-00001 ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720
```

近期来源是原有 `/home/mxd/lib/SD_MIA/artifacts/data/pools/wikitection/pool.jsonl`，
共 **8,000 篇**，创建日期在 **2026-04-01—2026-09-17**；读取后未修改。
其 SHA-256 为 `1bc14f8a153d90b90fc12132e6d813e06b8dcdb44eb9d9df85ed12eac1ef35da`。
依据 [Qwen3 官方发布页](https://qwenlm.github.io/blog/qwen3/)，采用 **2025-04-29**
作为保守边界，而非假定未知的预训练截止日。

所选公开模型：

- `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`

同一冻结 tokenizer 指纹为
`8ec78a5318c6883c3b8670208399b24585dd9e4870dd99296bf4cc550f6be4cf`。
长度不足和跨角色重复文本均在选样时剔除；每个 manifest 保存过滤统计。

**解释边界：**历史 Wikipedia 文本不等于已知训练成员；新页面也可能复用旧文字。
WikiText 是精选文章，保留旧式标点和空格处理，与新页面存在来源分布差异。
固定长度降低长度混淆，但不能消除这些差异。因此这组数据用于时间代理场景扩展，
不能替代 MIMIR 的官方标签验证。

## 其他已发现的历史来源

- WikiMIA 已有本地缓存，但 length256 只有 82 条（51 个 label1、31 个 label0）；
  原始标签针对更早的模型，不应把其 label0 直接解释为 2025 年 Qwen 的非成员。
  未把它混入本次 Qwen 数据。
- `artifacts/data/pools/wiki2023` 只有旧采集缓存和日志，没有可用的完整 pool。
  新接口支持 `historical_wiki` 格式，后续若完成采集可独立冻结，无需 SFT 数据护照。

## Pythia / MIMIR 就绪程度

本地 Pythia standard 6.9B、1.4B 模型已存在，配置及 tokenizer 已核对。
真实 tokenizer 均为 **50,277** 个 ID；两种模型的 LM-head 补齐大小分别为 50,432 和 50,304，
新入口沿用只对真实词表归一化的协议。两个 tokenizer 指纹均为
`3d8bca06cee31d7bc55ae90c100136f98f8cc05e254fb247c948728198f9ca47`。
固定 revision 及模型使用条件见 [README.md](README.md)。

此前未授权的 401/403 已解除。2026-09-26 使用本机 Hugging Face SDK 的有效登录重新访问，
官方数据文件授权成功，下载了固定 revision
`iamgroot42/mimir@02500d3b7cece0cb7628e939ba9fc93fdb6362ae` 的 **16 个文件、34,000 条记录**。
只下载 7 个领域的 `cache_100_200_1000_512/{train,test}/*_ngram_13_0.8.jsonl`，
以及 `cache_100_200_10000_512/{train,test}/full_pile.jsonl`，没有下载邻居缓存。

持久目录：`/home/mxd/lib/SD_MIA-pretraining-data/mimir/`：

- `official/`：官方原始 JSONL；每行是 JSON 字符串。train 保留为 member，test 保留为 nonmember。
- `DOWNLOAD.json`：每个官方文件的字节数、行数和 SHA-256，以及固定仓库 revision。
- `INVENTORY.json`：24 个冻结划分的位置、SHA-256、数量、长度与容量检查结果。
- `prepared/<source>/seed<seed>/{manifest.json,records.jsonl}`：可直接交给当前主方法入口。

已分别为 **1919、1949、1978** 准备：

| 数据 | 成员测试 | 非成员测试 | 独立非成员辅助 | 条件数 |
|---|---:|---:|---:|---:|
| 7 个领域，每领域 | 400 | 400 | 600 | 7 × 3 |
| full_pile 混合语料 | 2000 | 2000 | 600 | 1 × 3 |

辅助集的 600 条可按当前主方法默认分为 320 训练 / 80 验证 / 200 校准。
领域缓存的非成员只有 1000 条，保留这 600 条辅助后，平衡测试集的每类上限是 **400**。
不能同时声称使用完整 1000 条非成员测试且有独立的 600 条同源辅助。
full_pile 去重后可容纳每类至多 9400 条测试，本次仅冻结 2000/2000/600。
不同 seed 可重叠；full_pile 与领域缓存也不应作为互不相关的独立数据来源合并统计。

使用实际 Pythia tokenizer、`add_special_tokens=False`、最多 512 token 检查，
所有 34,000 条均可解析。每个 train/test 对内没有同标签精确 token 重复、跨标签精确 token
重复或不足 2 token 的记录。每个冻结划分还经实际 `load_evaluation(..., verify_draft=True)`
回读，验证哈希、真实词表、标签和角色互斥。以下为原始缓存截断后的长度：

| 来源 | 成员 / 非成员条数 | 平均 token：成员 / 非成员 | token 范围 |
|---|---:|---:|---:|
| `full_pile` | 10000 / 10000 | 280.1 / 280.7 | 109–512 |
| `arxiv` | 1000 / 1000 | 323.0 / 324.1 | 155–512 |
| `dm_mathematics` | 1000 / 1000 | 383.7 / 385.5 | 262–512 |
| `github` | 1000 / 1000 | 379.3 / 393.4 | 145–512 |
| `hackernews` | 1000 / 1000 | 318.8 / 318.2 | 141–512 |
| `pile_cc` | 1000 / 1000 | 259.7 / 259.6 | 126–512 |
| `pubmed_central` | 1000 / 1000 | 307.2 / 308.3 | 155–512 |
| `wikipedia_(en)` | 1000 / 1000 | 284.0 / 282.2 | 126–512 |

**实验解释：**这批官方标签可用于 Pythia 的预训练成员推断扩展，不需要微调。
当前主方法会冻结 6.9B 目标和 1.4B 草稿，使用独立辅助非成员拟合检测器。
1.4B 草稿本身也训练过 The Pile，须说明这是“共享预训练语料的公开草稿”，其暴露条件不同于
受控 SFT 的 member-blind 草稿。`full_pile` 是混合语料，不能当作 7 个领域宏平均。
`512` 是截断上限而非固定长度；例如 GitHub 两类的平均长度有差异，分析效果时应关注长度。
本次仅额外检查精确 token 重复；领域采用官方 ngram_13_0.8，full_pile 则无此过滤后缀，
不能将两者声称为相同近似去重协议。

本次只做下载、数据冻结和 tokenizer/CPU 功能测试；没有执行真实 Pythia 有效性或效率实验，
不能据此推断 AUC、低 FPR 表现或 GPU 运行时间。
