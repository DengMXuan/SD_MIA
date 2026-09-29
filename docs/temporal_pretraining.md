# 时间代理预训练审计

统一启动脚本见 [实验指南](experiments.md)。下文明确区分协议、冻结输入与历史测量；时间标签不等于已确认的训练集成员身份。

## Qwen 时间切分：清理与长度匹配 v2

预处理实现保留在 `standalone/qwen_temporal_clean/prepare.py`；统一启动入口调用冻结预训练主方法。清理产生独立数据与审计批次。

## 数据和规则

默认读取仓库同级的 `SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1/seed<seed>/`；也可通过 `--source-root` 指定其他位置。输入须先按 [预训练数据协议](../experiments/pretraining/README.md) 准备，每个 seed 保留 2000 历史样本、2000 近期测试样本和 600 近期辅助样本。保留 ID、分组、标签、组内顺序、模型版本和同 seed 对应关系。

所有角色使用同一规则，先处理完整已有文本，再重新分词、截取前缀：

- 把 `@-@`、`@,@`、`@.@` 还原为连字符、逗号、小数点，例如 `well @-@ known` → `well-known`。普通 `@`、邮件地址保留。
- 统一 Unicode NFC、HTML 实体、不可见字符和重复空格；修正标点及括号前后的分词空格、明确的英文缩写和成对双引号内空格。保留无法可靠判定的引号，不强行重配对。
- 移除 Wikipedia 数字引文标记、`[edit]`、引用条目和明确的 References 等末尾章节。
- 移除固定的翻译操作说明、Wikipedia:Translation 指引、stub 提示等网页模板。原近期样本确实带有这些内容，可能占据可见前缀的大半。
- 正文中的数字、实体、词汇和题材不按类别重写；不按模型概率或攻击分数筛选。

生成两个版本：

1. `clean_only`：只统一文本格式，最多 512 token。
2. `length_matched`（**默认运行**）：在清理基础上，让历史测试样本的 token 长度直方图与同 seed 的近期测试样本完全一致。按历史可用长度排序、seed 随机打散并列项后分配目标长度；不填充虚构文本、不更换文档。近期测试和辅助样本与 `clean_only` 保持一致。

清理掉模板、参考文献后，有些近期文本比原 128-token 选样下限更短。此版本保留原身份、角色和数量，实际最短长度写入派生 manifest，历史样本按同一长度分布匹配。没有用新文章回填，也没有沿用失真的旧 `min_tokens`。

## 准备和运行

以下命令从仓库根目录执行；在其他目录使用脚本绝对路径。

```bash
## CPU 准备两个版本；已准备的数据会检查哈希后复用。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh prepare

## 查看默认三个 seed 的“清理 + 长度匹配”任务。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh dry-run --gpus 2 3

## 在选定 GPU 上运行；每卡一个条件，结束后领取下一个。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh run --gpus 2 3

## 同时比较“只清理”和“清理 + 长度匹配”，共六个条件。
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh run \
  --variants clean_only length_matched --seeds 1919 1949 1978 --gpus 2 3

bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh summarize
```

`--gpus` 是当前 `CUDA_VISIBLE_DEVICES` 内的逻辑序号；调度器每个子进程只暴露指定 GPU，并使用 `cuda:0`。调度器管理本次任务队列，不会等待或干预其他实验的 GPU 进程；运行时应指定可用的卡。可用 `--gpu` 单卡、`--workers` 限制并发、`--seeds 1949` 选择单个 seed。

主方法条件保持：固定 Qwen3-8B-Base / Qwen3-1.7B-Base、原模型 revision、B=2、TCN 30 epoch、辅助数据 320/80/200 训练/验证/校准。同一 seed 用于数据长度分配和主方法实验。旧缓存不能复用到清理后的 token 序列；新结果目录已经隔离。

## 输出

相对仓库根目录：

- 数据：`artifacts/data/qwen_temporal_clean_v2/seed<seed>/<variant>/manifest.json` 和 `records.jsonl`。
- 审计：每个 seed 的 `REQUEST.json`、`AUDIT.json`、`_COMPLETE.json`，以及根目录 `SUMMARY.md/json`。
- 实验：`artifacts/audits/qwen_temporal_clean_v2/tasks/<variant>/seed<seed>/`。
- 每卡任务队列、状态和日志：`artifacts/audits/qwen_temporal_clean_v2/executions/`。

`REQUEST.json` 固定源 manifest/records 哈希和清理代码哈希；已完成目录拒绝不同源或规则。每条新记录保存原文本哈希、原 token 哈希和新可见长度，以便追溯。

## 审计边界

检查模型可见前缀的伪标记、标点空格、引文/UI 模板、字符/token、数字比例、标点比例、段落密度和 token 长度，并给出历史/近期两类的描述性 KS 距离与标准化均值差。

所有角色的完全相同 token 序列均拒绝；成员与非成员间的近重复也拒绝。非成员测试和辅助集之间可能仍有不同文章共享较长正文段落，保留冻结文章 ID 并在 `cross_role_near_pairs` 中完整列出；没有把这些重叠悄悄删去或宣称不存在。

清理与长度匹配控制的是表面格式及长度。历史 WikiText 与近期网页抽取仍可能有题材、信息框和写作风格差异，**不能声称两类完整文本同分布**。历史样本仍只是时间代理成员标签，日期不能证明真实预训练成员身份。

`prepare` 会在所选数据根目录写入 `SUMMARY.md` 与逐 seed `AUDIT.json`，供数据质量检查。

## Qwen3 / Gemma 4 的 Wiki 时间划分实验

本入口比较仓库已登记的 `qwen3`（Qwen3-8B-Base + Qwen3-1.7B-Base）和
`gemma4`（Gemma 4 12B + E2B）两组冻结预训练模型与独立草稿。两组模型在同一
seed 下使用**相同的历史页面 ID**，并沿用已有 WikiTection 划分中相同的
2000 个近期非成员 ID 与 600 个近期辅助 ID。历史池从 2023 年创建的页面提取
截至 2023-12-31 的正文；近期池来自 2026 年创建的页面。两个角色都经过同一
`qwen_temporal_clean_v2.clean_text` 规则，再分别使用各模型自己的 tokenizer。

默认的 `length_matched` 条件将 2000 篇历史文章的可见 token 前缀截短，使
**成员组与非成员组在该模型 tokenizer 下的长度直方图完全相同**。另有
`clean_only` 条件供对照。两个条件都保留原有页面 ID 和辅助集；辅助集不参与
长度匹配。历史页面是否真的进入预训练语料未知，因此结果仅是**时间代理标签**
下的审计结果，不能解释为已验证的真实训练成员检测。Gemma 4 官方模型卡写明
2025-01 数据截止；Qwen3 使用 2025-04-29 发布日期来界定固定权重之后的
2026 文章。证据与局限见
[日期核查记录](temporal_model_cutoff_sources_2026-09-27.md)。

## 运行

从仓库根目录执行。准备步骤仅读取已完成的历史/近期池和冻结划分，使用本机缓存
的两个 tokenizer；不进行网络采集或 GPU 推理。一次生成三个 seed、两组模型、
两种长度条件：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh prepare
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh dry-run
```

在确认所选 GPU 空闲后运行审计。默认每个模型 × 三个 seed 各一个
`length_matched` 条件，共 6 个任务；单 GPU 顺序执行。`--gpu` 是当前
`CUDA_VISIBLE_DEVICES` 下的逻辑序号，单个 worker 内使用 `cuda:0`。

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh run --gpu 0
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh summarize
```

只运行其中一组或一个 seed：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh dry-run --models gemma4 --seeds 1919
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh run --models gemma4 --seeds 1919 --gpu 0
```

同时审计未匹配长度的对照条件：

```bash
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_gemma4_temporal_matched_b2.sh run --variants clean_only length_matched --gpu 0
```

可以用 `--gpus 0 1 --workers 2` 将任务分配到两张空闲 GPU；`--detector-epochs`
默认 30。每个任务调用现有的 `experiments.pretraining.evaluation.evaluate_main`，
目标和草稿权重均冻结，辅助集划分为检测器拟合 320、验证 80、校准 200 条。
脚本默认离线加载固定 revision 的模型；首次运行前这些权重必须在本机缓存。

## 文件与后台任务隔离

准备结果写入 `artifacts/data/qwen_gemma_temporal_v1/seed<seed>/`，每个模型和
条件各有 `manifest.json`、`records.jsonl`。`SELECTION.json` 记录共用页面 ID，
`COMPLETE.json` 与 `REQUEST.json` 锁定产物和来源哈希。审计结果写入
`artifacts/audits/qwen_gemma_temporal_v1/tasks/<model>/<variant>/seed<seed>/`，
worker 日志写入相邻的 `executions/`。准备和运行均不写入原历史池、近期池、
共享实验源码或旧审计目录。`dry-run` 会检查数据哈希、来源、模型配置、页面 ID、
分组数量和长度直方图；不会占用 GPU。中断后的审计可用同一命令恢复；如果
冻结来源发生变化，脚本会拒绝复用旧数据或结果。
