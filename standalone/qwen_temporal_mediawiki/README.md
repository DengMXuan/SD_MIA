# Qwen 时间代理：两组共用 MediaWiki 正文抽取

历史组复用近期 WikiTection 的 MediaWiki 创建日志、元数据预筛规则、
`_wiki_page_record` 修订版查询、`action=parse` 全文渲染、正文解析器、
700–9000 字符过滤与近重复检查。历史组由独立的断点续跑调度器管理，
固定模型发布前的修订版。近期组继续使用
既有 `wikitection/pool.jsonl` 和每个 seed 已冻结的 `nonmember`、
`audit_auxiliary` ID、顺序与文本。**此入口创建新数据批次；不会覆盖 WikiText
实验、已清理数据或现有审计结果。**

## 后台实验影响

当前资源消融实验的来源指纹覆盖 `experiments/shared/`、
`experiments/sd_membership_sft/`、`experiments/baseline/` 和
`experiments/resource_curves/`。本入口只新增 `standalone/qwen_temporal_mediawiki/`
文件，调用既有底层抽取函数，不修改上述目录。修改共享采集器会改变消融实验来源
指纹，故不在后台实验运行时修改。

## 收集并准备数据

从仓库根目录执行。收集步骤访问 Wikipedia API，默认选择 2023 年首次创建的
英文主名字空间页面，并固定截至 2023-12-31 的最后一份修订版。默认收集
3000 篇合格文章，最多读取 **50000 条**创建事件，并在其中最多保留
**10000 个**通过元数据预筛的页面。MediaWiki 请求间隔默认 **1 秒**，使用项目仓库网址作为
User-Agent 联系地址。每篇历史文章通常至少需要修订版查询和正文渲染两次
请求，加上候选预筛。请求耗时超过 1 秒后额外暂停 5 秒；HTTP 429/503
也会触发退避。实际耗时取决于页面筛选通过率和 API 响应，尚未实测。
若某个历史修订版的 `action=parse` 返回 `permissiondenied`、`nosuchrevid`
或 `missingtitle`，该页会记录 `skip unavailable historical revision` 并跳过。
其他 API 错误仍会停止，
但中断或出错前已记录的创建事件、预筛结果、文章尝试与成功正文都保留在
`historical_pool/checkpoint/`。同一命令重跑会从断点继续，成功的 API
响应也会从 `api_cache` 复用；正式池文件在达到目标条数后发布。
`run.sh status` 可查看已落盘的创建事件、预筛、正文尝试、可用文章和缓存数。
旧版本中断时已有的 API 缓存会复用，但因旧版本没有文章断点文件，第一次
使用新版入口仍要从缓存重建文章日志。
日期、条数和两个上限可用 `--window-start`、`--window-end`、`--snapshot-at`、
`--records`、`--candidate-limit`、`--survivor-limit` 调整；所有日期必须早于
Qwen3 发布日 2025-04-29。日期或正文规则改变时需使用新输出目录。

```bash
bash standalone/qwen_temporal_mediawiki/run.sh collect
bash standalone/qwen_temporal_mediawiki/run.sh prepare
bash standalone/qwen_temporal_mediawiki/run.sh status
```

请求由采集器统一限速；本入口尚未实现账号认证，按 Wikimedia 机器人
政策将匿名 Action API 并发固定为 1。Wikimedia 2026 年公布的 API 限额中，
只有 IP 标识的请求是 10 次/分钟，带合规 User-Agent 的未登录脚本与
新注册账号均为 200 次/分钟。1 秒是本实验选取的最小请求间隔，
并非 Wikimedia 指定的固定值；实际分档以服务端识别为准。
核查记录见 [WIKIMEDIA_RATE_LIMITS.md](WIKIMEDIA_RATE_LIMITS.md)。
按默认设置采集：

```bash
bash standalone/qwen_temporal_mediawiki/run.sh collect \
  --records 3000
```

如果有真实的公开项目联系邮箱或网址，可用 `--contact` 标识请求来源；
它不等于账号认证，也不会单独提高 API 配额。注册新账号同样没有更高分档。
调小 `--request-interval` 要观察是否出现 429/503，否则反复退避可能更慢。
若默认上限内不足 3000 篇，可在同一池目录提高
`--candidate-limit` 或 `--survivor-limit` 后继续，断点与 API 缓存都会保留。
3000 是采集池总数，并非每个 seed 的
输出数；三个 seed 各自需要 2000 篇通过 128–512 token 和跨组去重筛选的
历史文章。若准备阶段报告候选不足，需增大 `--records` 并采集到新的池目录。

第一步保存 `historical_pool/pool.jsonl`、相邻 hash manifest 和
`COLLECTION.json`，记录全文抽取路径、快照日期和源码哈希；第二步按
1919、1949、1978 三个 seed 选历史文章、核对同 seed 冻结的近期划分，
分别写出 `shared_split/seed<seed>/manifest.json` 和 `records.jsonl`。
Qwen3 目标与草稿使用同一 tokenizer，测试每组 2000 篇、近期辅助 600 篇，
token 长度为 128–512。历史与近期文章跨组去重；历史候选不足时停止，
不会缩减样本数或借用旧 WikiText 文章。

## 使用现有清理、长度匹配和主方法

历史组现在也来自渲染后的 Wikipedia 页面；仍用原 v2 清理器对两组施加
同一文本规则，并在默认 `length_matched` 版本中匹配长度。数据和审计结果
使用新的目录，旧缓存不会复用：

```bash
bash standalone/qwen_temporal_clean/run.sh prepare \
  --source-root artifacts/data/qwen_temporal_mediawiki_v1/shared_split \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1

bash standalone/qwen_temporal_clean/run.sh dry-run \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1 \
  --output-root artifacts/audits/qwen_temporal_mediawiki_clean_v1/tasks

bash standalone/qwen_temporal_clean/run.sh run --gpu 0 \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1 \
  --output-root artifacts/audits/qwen_temporal_mediawiki_clean_v1/tasks
```

`run` 默认执行三个 seed 的 `length_matched`；对比两个清理版本可加
`--variants clean_only length_matched`。三个模型实验仍仅使用时间代理标签，
页面创建早于模型发布并不能证明模型确实训练过该文本。历史修订版渲染还
可能展开当前模板，匹配抽取方式也不能保证题材和写作风格一致。
