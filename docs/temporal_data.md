# 时间代理语料构建与采集限制

## Qwen 时间代理：两组共用 MediaWiki 正文抽取

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
bash experiments/scripts/data/prepare_qwen3_temporal_mediawiki.sh collect
bash experiments/scripts/data/prepare_qwen3_temporal_mediawiki.sh prepare
bash experiments/scripts/data/prepare_qwen3_temporal_mediawiki.sh status
```

请求由采集器统一限速；本入口尚未实现账号认证，按 Wikimedia 机器人
政策将匿名 Action API 并发固定为 1。Wikimedia 2026 年公布的 API 限额中，
只有 IP 标识的请求是 10 次/分钟，带合规 User-Agent 的未登录脚本与
新注册账号均为 200 次/分钟。1 秒是本实验选取的最小请求间隔，
并非 Wikimedia 指定的固定值；实际分档以服务端识别为准。
核查记录见 [WIKIMEDIA_RATE_LIMITS.md](temporal_data.md)。
按默认设置采集：

```bash
bash experiments/scripts/data/prepare_qwen3_temporal_mediawiki.sh collect \
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
bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh prepare \
  --source-root artifacts/data/qwen_temporal_mediawiki_v1/shared_split \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1

bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh dry-run \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1 \
  --output-root artifacts/audits/qwen_temporal_mediawiki_clean_v1/tasks

bash experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_clean_matched_b2.sh run --gpu 0 \
  --data-root artifacts/data/qwen_temporal_mediawiki_clean_v1 \
  --output-root artifacts/audits/qwen_temporal_mediawiki_clean_v1/tasks
```

`run` 默认执行三个 seed 的 `length_matched`；对比两个清理版本可加
`--variants clean_only length_matched`。三个模型实验仍仅使用时间代理标签，
页面创建早于模型发布并不能证明模型确实训练过该文本。历史修订版渲染还
可能展开当前模板，匹配抽取方式也不能保证题材和写作风格一致。

## Wikimedia API 限速与认证核查（2026-09-27）

## 结论

本项目从 `https://en.wikipedia.org/w/api.php` 调用 Action API。Wikimedia 在 2026 年启用的跨项目 API 限额按**客户端身份**区分：只按 IP 识别的请求为 10 次/分钟；带合规 `User-Agent` 的匿名机器人为 200 次/分钟；新注册或编辑较少的已认证账号也是 200 次/分钟；成熟编辑者账号为 2000 次/分钟。获得社区批准的 bot flag、在 Wikimedia Cloud Services (WMCS) 运行且有合规 `User-Agent`、或获基金会豁免的客户端，可免于这层 API 限额，但仍受运营层限速和机器人政策约束。这些数字尚处 2026 年试验阶段，可能变化。**单纯注册账号或注册 OAuth 应用，不会自动提高本采集器可用的请求额度。**[1][2]

认证可以帮助正确识别请求，并让客户端按该账号的现有权限获得对应限额。官方接受 bot password、OAuth 和 owner-only consumer 等方式；OAuth 1 和部分 owner-only 流程需要会话 cookie 才能被新限速网关正确识别。若账号是新账号，认证后仍是 200 次/分钟。官方 FAQ 明确说明新账号的限额较低，是为了防止抓取者通过批量注册账号绕过限额。[1][3]

## 为什么仍要控制请求速率

全局限额不是唯一约束。Wikimedia 的机器人政策对 Action API 还要求：匿名请求的总并发保持为 1、总速率低于 5 次/秒；已认证请求的并发可到 3、速率可到 10 次/秒；若某个 API 请求花费超过 1 秒，下一次请求前等待 5 秒。遇到 HTTP 429 应遵守 `Retry-After`。政策还指出，Action API 不适合批量取页面 HTML，鼓励在可行时使用数据转储、批量请求和更易缓存的接口。[2] 2026 年限额文档补充：429 或 503 一般会带 `Retry-After`；没有时至少等待 5 秒或指数退避。[1]

MediaWiki 的一般 API 礼仪页仍写着“读取请求没有硬性的速度上限”，但同页也明确说 Wikimedia 项目的 API 请求另受专门的 API 限额约束。因此，不能用一般礼仪页那句话否定 2026 年 Wikimedia 专项限额。[1][4]

共享采集器原始默认值 `_WIKI_INTERVAL = 7.5` 秒，并注释“低于未识别客户端 10 次/分钟”；独立历史采集入口已把最小请求间隔设为 **1 秒**，匿名并发限制为 1，耗时超过 1 秒的请求后再暂停 5 秒。`_request` 对 429/503 仍会退避。上述间隔都是项目配置，并非 Wikimedia 的固定要求。代码已发出 `SD-MIA-research/0.1 (https://github.com/DengMXuan/SD_MIA)`，若 Wikimedia 判定这个标识符合“有意义且可联系”的 `User-Agent` 要求，请求可能落入 200 次/分钟档；光设置 `--contact` 并不会提高配额，实际身份分类以服务端为准。[1][5] 当前采集还使用 `action=parse` 渲染历史修订版，单次请求可能较重，提速时也需考虑上述耗时请求规则。[2]

## 与本实验有关的可行方式

1. 先确保 `User-Agent` 包含项目名、版本、真实可联系的邮箱或网页，并检查 429/503 及 `Retry-After`。Python 脚本应设置标准 `User-Agent`；`Api-User-Agent` 主要用于浏览器 JavaScript 无法修改浏览器自带 `User-Agent` 的场景。[5]
2. 若仍需更高吞吐，可用个人 Wikimedia 账号认证，并确认账号是否已有成熟编辑者身份。新账号的 200 次/分钟与合规匿名机器人相同；认证本身不解除 Action API 的并发和耗时请求规范。[1][2][3]
3. 社区机器人可申请 bot flag，适合 Wikimedia 社区认可的机器人；研究用途可以参考官方 FAQ 关于数据转储、WMCS 和研究访问的指引。不能为了绕过限额分散到多个账号或 `User-Agent`。[2][3][6]

## 官方来源

[1] [Wikimedia APIs/Rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)：2026 年限额表、认证注意事项、429/503 处理。

[2] [Wikimedia Robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy)：Action API 并发与速率、重请求等待、批量获取建议、豁免路径。

[3] [Wikimedia APIs/Rate limits/FAQ](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits/FAQ)：新账号、认证方式、bot flag、研究项目说明。

[4] [API:Etiquette](https://www.mediawiki.org/wiki/API:Etiquette)：一般读取礼仪与 Wikimedia 专项限额的关系。

[5] [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)：标识和联系方式格式、`Api-User-Agent` 适用场景。

[6] [Wikimedia Foundation API Usage Guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines)：遵守退避指令、不得通过多个标识隐藏单个操作者的过量请求。
