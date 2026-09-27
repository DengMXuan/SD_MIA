# Wikimedia API 限速与认证核查（2026-09-27）

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
