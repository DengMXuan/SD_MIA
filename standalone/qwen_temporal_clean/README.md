# Qwen 时间切分：清理与长度匹配 v2

独立工作树中的新实验入口。旧历史/近期数据、原实验脚本、观测缓存、TCN 与结果均保留。预处理代码在本目录；运行器调用现有固定模型的预训练主方法，不改评分实现。

## 数据和规则

读取 `/home/mxd/lib/SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1/seed<seed>/`，保留 1919、1949、1978 每个 seed 的 2000 历史样本、2000 近期测试样本和 600 近期辅助样本。保留 ID、分组、标签、组内顺序、模型版本和同 seed 对应关系。

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

在任意目录执行；以下路径对应当前独立工作树。

```bash
# CPU 准备两个版本；已准备的数据会检查哈希后复用。
bash /home/mxd/.codex/worktrees/dp-throughput/SD_MIA/standalone/qwen_temporal_clean/run.sh prepare

# 查看默认三个 seed 的“清理 + 长度匹配”任务。
bash /home/mxd/.codex/worktrees/dp-throughput/SD_MIA/standalone/qwen_temporal_clean/run.sh dry-run --gpus 2 3

# 在选定 GPU 上运行；每卡一个条件，结束后领取下一个。
bash /home/mxd/.codex/worktrees/dp-throughput/SD_MIA/standalone/qwen_temporal_clean/run.sh run --gpus 2 3

# 同时比较“只清理”和“清理 + 长度匹配”，共六个条件。
bash /home/mxd/.codex/worktrees/dp-throughput/SD_MIA/standalone/qwen_temporal_clean/run.sh run \
  --variants clean_only length_matched --seeds 1919 1949 1978 --gpus 2 3

bash /home/mxd/.codex/worktrees/dp-throughput/SD_MIA/standalone/qwen_temporal_clean/run.sh summarize
```

`--gpus` 是当前 `CUDA_VISIBLE_DEVICES` 内的逻辑序号；调度器每个子进程只暴露指定 GPU，并使用 `cuda:0`。调度器管理本次任务队列，不会等待或干预其他实验的 GPU 进程；运行时应指定可用的卡。可用 `--gpu` 单卡、`--workers` 限制并发、`--seeds 1949` 选择单个 seed。

主方法条件保持：固定 Qwen3-8B-Base / Qwen3-1.7B-Base、原模型 revision、B=2、TCN 30 epoch、辅助数据 320/80/200 训练/验证/校准。同一 seed 用于数据长度分配和主方法实验。旧缓存不能复用到清理后的 token 序列；新结果目录已经隔离。

## 输出

相对工作树根目录：

- 数据：`artifacts/data/qwen_temporal_clean_v2/seed<seed>/<variant>/manifest.json` 和 `records.jsonl`。
- 审计：每个 seed 的 `REQUEST.json`、`AUDIT.json`、`_COMPLETE.json`，以及根目录 `SUMMARY.md/json`。
- 实验：`artifacts/audits/qwen_temporal_clean_v2/tasks/<variant>/seed<seed>/`。
- 每卡任务队列、状态和日志：`artifacts/audits/qwen_temporal_clean_v2/executions/`。

`REQUEST.json` 固定源 manifest/records 哈希和清理代码哈希；已完成目录拒绝不同源或规则。每条新记录保存原文本哈希、原 token 哈希和新可见长度，以便追溯。

## 审计边界

检查模型可见前缀的伪标记、标点空格、引文/UI 模板、字符/token、数字比例、标点比例、段落密度和 token 长度，并给出历史/近期两类的描述性 KS 距离与标准化均值差。

所有角色的完全相同 token 序列均拒绝；成员与非成员间的近重复也拒绝。非成员测试和辅助集之间可能仍有不同文章共享较长正文段落，保留冻结文章 ID 并在 `cross_role_near_pairs` 中完整列出；没有把这些重叠悄悄删去或宣称不存在。

清理与长度匹配控制的是表面格式及长度。历史 WikiText 与近期网页抽取仍可能有题材、信息框和写作风格差异，**不能声称两类完整文本同分布**。历史样本仍只是时间代理成员标签，日期不能证明真实预训练成员身份。

本次完成数据准备、13 项清理/匹配回归检查、真实 manifest 兼容性验证和队列 dry-run；没有启动占用后台 GPU 的正式重跑。具体实测分布见数据根目录的 `SUMMARY.md` 与 `AUDIT.json`。
