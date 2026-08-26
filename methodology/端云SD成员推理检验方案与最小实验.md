# 端云协同推测解码的成员推理检验方案与最小实验

> 版本：2026-08-24  
> 研究对象：端侧部署白盒草稿模型 $q_\phi$，云端部署完整验证模型 $p_\theta$  
> 术语约定：本文统一使用“检验、审计、评估者、适应性客户端”等中性表述。论文原文中的 MIA 在本文中称为“成员推理检验”。

## Material Passport

- 研究阶段：研究问题收敛 + 方法设计 + 单卡可行性 pilot
- 证据状态：`PILOT_SUPPORTED`，尚非生产部署结论
- 实验代码：[sd_membership_pilot.py](../experiments/sd_membership_pilot.py)
- 弱记忆结果：[pilot_epoch1/RESULTS.md](../experiments/results/pilot_epoch1/RESULTS.md)
- 高记忆上界：[pilot/RESULTS.md](../experiments/results/pilot/RESULTS.md)
- Qwen3 真正 SFT 代码：[sd_membership_sft/](../experiments/sd_membership_sft/)
- Qwen3 真正 SFT 结果：[qwen3_1p7b_to_8b_epoch1/RESULTS.md](../experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch1/RESULTS.md)
- Qwen3 强记忆对照：[qwen3_1p7b_to_8b_epoch4/RESULTS.md](../experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch4/RESULTS.md)
- 信号边界：只使用端云协议本来返回的验证结果；不使用时间、包长、功耗或其他侧信道

## 1. 结论先行

最值得做的不是“把普通 LLM 成员推理方法直接跑在端侧小模型上”，而是研究一种端云 SD 特有的**草稿—验证联合成员审计**：

1. 端侧拥有 $q_\phi$ 的参数、logits、隐藏状态和梯度，因而天然获得一个与云端 $p_\theta$ 高度对齐的白盒参考模型；
2. 随机采样型 SD 的正常验证结果满足

   $$
   A_{i,r}\sim \operatorname{Bernoulli}\!\left(
   \min\left\{1,\frac{p_\theta(x_i\mid x_{<i})}{q_\phi(x_i\mid x_{<i})}\right\}\right),
   $$

   所以“接受/拒绝”不是普通二值反馈，而是云端概率与白盒草稿概率之比的随机观测；
3. 贪心型 PipeSD 也会把“与云端 argmax 连续匹配了多长”直接返回端侧；SpecEdge 明确返回 verified tokens 和额外 token，SLED 明确返回被拒绝位置及修正 token。它们都暴露了**协议语义反馈**，并非侧信道；
4. 客户端控制草稿模型时，还可以对候选 token 的 $q$ 做分级调整，以验证反馈反推被截断的 $p/q$ 或近似 $p$。这一能力在普通文本生成 API 中不存在；
5. 草稿常由验证模型蒸馏、剪枝或同源训练得到。现有蒸馏隐私研究表明，学生模型并不必然降低成员信号，某些样本上反而会放大。因此，白盒草稿本身和 draft–verifier 的耦合都应纳入检验。

最小实验已经给出初步支持：在随机化、同来源的 1-epoch 弱记忆设置中，基础草稿的 Min-K% AUC 仅为 **0.536**，而未绑定草稿采样过程的 verifier 接受率达到 **0.612**，白盒草稿与协议反馈联合达到 **0.644**，接近直接读取云端 logits 的上界 **0.654**。pilot 固定待检记录的候选 token，但使用部署草稿给出的原始 $q$，没有调整 logits；因此它验证的是 L3“客户端可选择候选、服务端不证明候选确由 $q$ 采样”的接口，而不是 L2 被动自然生成。这回答了最关键的创新性质疑：**有效信号主要来自端云 draft–verify 关系，而不只是一般的小模型成员分数。**

## 2. 研究问题与可证伪假设

### RQ1：协议反馈是否增加可检出的成员信息？

在成员、非成员严格同分布且草稿单独接近随机时，验证接受率、接受前缀长度和修正 token 是否仍能显著提高云端验证模型的成员判别能力？

- H1：`draft + verifier transcript` 的 AUC 比 `draft only` 至少高 0.05；
- H1-null：加入协议反馈后，配对 AUC 差异不显著或低于 0.05。

### RQ2：白盒草稿的价值具体来自哪里？

比较四类信息：

- 输出层：loss、Min-K%、entropy、margin、rank；
- 内部状态：逐层 pooled activation、线性探针；
- 梯度几何：梯度范数、单步扰动后的表示漂移；
- 查询规划：用白盒 (q) 选择少量高信息 token，而非均匀查询整段文本。

- H2：在相同云端验证次数下，白盒 token 选择优于随机 token 选择；
- H2-null：白盒选择只反映文本难度，不能改善低 FPR 或查询效率。

### RQ3：草稿来源如何改变风险？

比较：独立预训练草稿、仅在无交集辅助集上蒸馏的草稿、与目标共享数据的草稿、从目标剪枝/截层的草稿。

- H3：草稿来源对风险的影响是非单调的。共享数据可能增强草稿自身成员信号；高质量蒸馏也可能让 $q$ 与 $p$ 同步移动，从而减弱单纯 $p/q$ 残差；
- H3-null：草稿来源只影响接受率，不影响成员信息。

### RQ4：哪些协议设计最敏感？

比较贪心匹配、无损随机采样、只返回最终 verified sequence、返回逐 token 接受位、拒绝时返回完整目标分布等接口。

- H4：未绑定草稿权重/分布的随机采样接口风险最高；服务端绑定 $q$ 可抑制适应性概率测量，但不能消除正常 draft–target agreement 信号。

## 3. 威胁模型与访问层级

### 3.1 审计目标

给定候选记录 $x=(x_1,\ldots,x_L)$，判断它是否属于云端验证模型的某一明确训练集合：

- 预训练数据成员；
- SFT/领域适配数据成员；
- 文档级成员（书籍、论文、病历或代码库）；
- 数据集级成员（某个集合是否整体参与训练）。

首篇工作建议以 **SFT/领域适配成员 + 文档级成员** 为主，因为成员定义清晰、可随机化，并且端云部署更可能把领域专用模型压缩为本地草稿。预训练成员可作为更难的后续验证。

### 3.2 访问层级

| 层级 | 端侧能力 | 典型协议 | 可用信号 |
|---|---|---|---|
| L0 普通生成 | 只看到最终文本 | 常规云端 API | 文本重合、稳定性、n-gram coverage |
| L1 贪心验证 | 白盒草稿 + verified tokens | PipeSD 类 | 每个位置是否与 target argmax 匹配、接受前缀长度、修正 token |
| L2 随机验证 | 白盒草稿 + 接受结果 | SLED/经典无损 SD 类 | $\min(1,p/q)$ 的随机观测、修正 token、bonus token |
| L3 适应性草稿 | L2 + 可调草稿 logits/窗口/随机种子 | 未绑定本地草稿的 verifier-only 接口 | 分级 (q) 探测、单 token 验证、重复前缀统计 |
| L4 分布回传 | L3 + 拒绝时返回目标分布 | DSSD 类变体 | 被拒绝位置的直接目标概率或修正分布 |

主结果必须分别报告 L1、L2、L3，不能把 L3 的结果包装成所有端云 SD 都自动具备的风险。

### 3.3 合规与非合规分支

- **被动协议审计**：草稿按部署模型正常生成，只汇总自然出现的接受轨迹；
- **接口内适应性审计**：候选 token、(q) 和窗口均满足接口格式，但本地 logits 可调整；
- **绑定草稿审计**：服务端通过签名、远程证明或本地副本固定 (q)，客户端只能改变输入和随机种子。

三者应独立报告。若论文只展示可调 (q) 的最强结果，评审很容易质疑现实性。

## 4. 为什么这不是普通 LLM 成员推理

| 普通成员推理 | 端云 SD 特有信息 |
|---|---|
| 只评估一个目标模型 $p$ | 客户端持有与 $p$ 为加速目的而协同训练的白盒 $q$ |
| 参考模型通常要自行寻找或训练 | 部署协议直接把草稿作为天然参考模型交给客户端 |
| 黑盒 API 通常不返回每个 token 的概率 | 接受位本身就是 (p/q) 的随机函数 |
| 无法控制目标模型用于比较的参考分布 | 客户端可改变本地 (q)、草稿窗口和候选路径 |
| 最终输出掩盖内部纠错过程 | 客户端必须接收 accepted prefix、修正 token 或 verified sequence 才能更新 KV cache |
| 单模型训练来源 | 蒸馏、剪枝、共享 tokenizer/embedding 会转移或重塑成员信号 |

论文的核心贡献应表述为：**端云 SD 将目标模型概率与客户端已知概率通过验证规则显式耦合，形成一个协议内的、可校准的成员审计接口。** 草稿白盒内部状态是增强项，接受比值才是最难被一般场景替代的主线。

## 5. 方法：DraVer-MemAudit

### 5.1 候选记录切窗

将文档切成 (W) 个等长、不重叠窗口。每个窗口保留一段 prefix，并对后续 token 做检验。所有成员/非成员必须长度匹配、来源匹配，并清除跨集合 n-gram 重叠。

### 5.2 模块 A：白盒草稿筛选

对每个 token 计算：

$$
\ell_i^q=\log q_\phi(x_i\mid x_{<i}),\quad
H_i^q,\quad \mathrm{margin}_i^q,\quad \mathrm{rank}_i^q.
$$

第一版使用 Min-K% 选取 (q) 下最不自然的 10%–20% token。随后加入：

$$
G_i=\lVert h_i\rVert_2
\sqrt{\sum_v q_i(v)^2-2q_i(x_i)+1},
$$

它等于该 token 对 LM-head 权重梯度范数，不需要显式保存巨大梯度矩阵。还可加入逐层隐藏状态线性探针，以及单步梯度扰动后的表示漂移；后者应被视为可选增强，因为 2026 年相关证据仍较新，需要独立复现。

输出 token 集合 (I(x))，目标是在相同云端验证预算下最大化成员信息。

### 5.3 模块 B：验证比值审计（主方法）

在未绑定草稿采样过程的接口上，对 $i\in I(x)$ 固定候选记录中的 token，并采用 $\gamma=1$ 的验证，避免“一处拒绝导致后续 token 不可见”的前缀删失。提交的 $q_i(x_i)$ 仍来自原始部署草稿；若服务端要求可验证的采样随机性或可信执行证明，此步骤应归入不适用。重复 $R$ 次，记录接受次数 $c_i$，用 Jeffreys 平滑：

$$
\widehat\alpha_i=\frac{c_i+1/2}{R+1},\qquad
\alpha_i=\min\left(1,\frac{p_i}{q_i}\right).
$$

当 $\alpha_i<1$ 时，

$$
\log p_i-\log q_i=\log\alpha_i.
$$

当 $\alpha_i=1$ 时只知道 $p_i\ge q_i$，因此它是右删失的似然比观测。第一版不必强行恢复 $p_i$，直接使用以下特征更稳健：

- $\operatorname{mean}_{i\in I}\widehat\alpha_i$；
- $\operatorname{min}/\operatorname{quantile}(\widehat\alpha_i)$；
- 接受前缀长度分布；
- 修正 token 与候选 token 的 rank/distance；
- 按 $q_i$、entropy、梯度范数分桶后的条件接受率。

这相当于由协议自动提供一个以草稿为参考模型的、难度校准后的成员分数。

### 5.4 模块 C：接受反馈层析（次要、强访问分支）

若客户端可调草稿 logits，设置

$$
q_i^{(l)}=\min(0.95,2^l q_i),\qquad l=0,1,\ldots,L_q.
$$

找到第一个非饱和层级后估计

$$
\widehat p_i=q_i^{(l)}\widehat\alpha_i^{(l)}.
$$

然后可在重建的 token log-probability 上运行 Min-K%、InfoRMIA 或文档级聚合。这个分支概念上最能体现“客户端控制 (q)”的独特性，但最小实验显示它在弱记忆设置下尚未稳定超过直接接受率，因此当前应作为**待优化假设**，不能作为已证实主结果。

下一步可用自适应二分和 sequential probability ratio test（SPRT）降低查询量，而不是固定 $R\times L_q$。

### 5.5 贪心协议分支

对于 PipeSD 类贪心 NAV，云端返回：

$$
m_i=\mathbf 1[x_i=\arg\max_v p_i(v)].
$$

特征为候选 suffix 的连续匹配长度、逐位置 match rate、首次不匹配位置以及修正 token。它不能恢复完整概率，但最小实验表明即使 1 个 epoch，也能从基础草稿选定 token 上得到 AUC 0.588。

### 5.6 融合与文档级聚合

记录级特征：

$$
z(x)=[S_q, S_{\mathrm{hidden}},S_{\mathrm{grad}},S_{\mathrm{accept}},
S_{\mathrm{greedy}},S_{\mathrm{correction}}].
$$

只在独立 audit-calibration 集上训练 L2 正则线性模型；阈值也只在该集合确定。文档级结果可使用：

- 窗口分数的 trimmed mean；
- top-k window pooling；
- Stouffer 聚合，并用 block bootstrap 处理相邻窗口相关性；
- “至少 (r) 个窗口超过阈值”的组级检验。

不得在测试集合调 k、层数、阈值或融合权重。

## 6. 必须包含的对照与消融

### 6.1 一般成员基线

- target LOSS / perplexity；
- target Min-K% Prob；
- neighborhood calibration；
- reference-model likelihood ratio；
- document-level aggregation；
- text-only n-gram coverage；
- 若有足够 reference models，加入 LiRA/InfoRMIA；
- target logits 直接访问作为不可部署上界。

### 6.2 端云 SD 分解

| 编号 | 草稿白盒 | 协议反馈 | 目的 |
|---|---:|---:|---|
| B0 | 否 | 否 | 文本/模型无关负对照 |
| B1 | 是 | 否 | 草稿单独风险 |
| B2 | 否或仅 token 概率 | 是 | 验证轨迹单独贡献 |
| B3 | 是 | 是 | 完整 DraVer-MemAudit |
| B4 | 是，可调 (q) | 是 | 接受反馈层析强访问分支 |
| B5 | 否 | 直接 target logits | 云端概率上界 |

核心论文结论必须建立在 `B3 > B1` 且 `B3 > B0` 上，而不是只证明 B1 有效。

### 6.3 草稿来源

- 同 tokenizer 的独立小模型；
- 在与候选记录无交集的辅助集上做 logit distillation；
- 在与 target 同源但不完全重合的数据上蒸馏；
- 共享 embedding/前若干层；
- 结构剪枝或 early-exit 草稿；
- 检索式草稿作为非参数对照。

### 6.4 协议因素

- $\gamma\in\{1,2,4,8\}$；
- greedy 与 stochastic；
- 每 token 反馈、只返回 accepted prefix、只返回最终 verified sequence；
- 是否返回 $q(x_i)$、完整 $q$、修正分布；
- 是否绑定草稿权重、temperature、随机种子；
- 重复前缀是否允许重置 KV 状态；
- 同步与流水线不会改变语义反馈，但会改变可查询频率，应单独统计。

## 7. 严格实验设计

### 7.1 数据构造

优先级如下：

1. **随机化 SFT benchmark**：从同一来源池随机抽样并注入 target 训练，成员身份由训练日志和 SHA-256 确认；
2. **Pythia/OLMo 可追溯预训练 benchmark**：利用公开数据顺序和 checkpoint，不使用时间切分猜测成员；
3. **文档级 benchmark**：以完整文档为分组单位划分，禁止同一文档的相邻窗口跨成员集合；
4. 时间切分只作外部有效性补充，不能作为主因果证据。

每个样本记录：来源、长度、token 数、压缩率、稀有 token 数、n-gram overlap、训练 step、重复次数。成员与非成员做分层匹配。

### 7.2 模型矩阵

建议第一阶段选择共享 tokenizer 且已在端云工作中使用的模型对：

- OPT-125M → OPT-6.7B/13B（与 DSSD 设置接近）；
- Qwen2/2.5-0.5B → 7B；
- Llama 1B/3B → 8B；
- Pythia-160M/410M → 2.8B/6.9B，用于可追溯预训练成员。

至少 3 个 draft–target pair、每个 target 5 个训练随机种子。固定成员集合，改变训练随机性，报告逐样本决策稳定性。

### 7.3 指标与样本量

- AUC 及配对 bootstrap 95% CI；
- TPR@0.1%、1%、5% FPR；
- 成员 posterior calibration / ECE；
- 每条记录的 verifier round、接受反馈 bit、上/下行字节与云端 FLOPs；
- 每个来源/长度/重复度子组的 worst-group AUC；
- 逐样本跨 seed 决策方差；
- 草稿加速质量：acceptance rate、平均接受长度、target output exactness。

低 FPR 结果需要足够非成员样本：0.1% FPR 建议至少 10,000 个非成员，1% FPR 至少 1,000 个。当前 pilot 每类测试样本仅 112 个，所以低 FPR 数字只用于调试，不能作为论文主结论。

### 7.4 完整性闸门

任一条件不满足时，不允许声称“成员信息来自端云 SD”：

- 模型无关 BoW/长度分类器 AUC 显著高于 0.55；
- 成员/非成员存在来源或时间差异；
- 相邻/重复窗口跨集合；
- 只报告 AUC、不报告低 FPR；
- threshold 在测试集调参；
- 只跑一个训练 seed；
- 只与草稿模型比较，不与 target Min-K、reference calibration 比较；
- 把 L3 可调 (q) 的结论外推到绑定草稿的 L1/L2 接口。

## 8. 已完成的最小实验

### 8.1 设置

- 硬件：只使用 GPU 1（A100 80GB）；GPU 4–6 原有任务完全未触碰；
- 峰值显存：12.45 GiB；
- 语料：当前目录 11 篇端云 SD 一手论文 PDF，仅在运行时抽取文本，不持久化原文；
- 构造：按来源分层随机划分 160 members、160 nonmembers、160 auxiliary records；各记录 64 GPT-2 tokens，窗口不重叠并做 SHA-256 去重；
- target：GPT-2 XL；
- drafts：8-layer 基础截层、在无交集辅助集上蒸馏、从微调 target 共享权重截层；
- audit calibration：每类 48 条；held-out test：每类 112 条；
- 两个记忆强度：target 训练 1 epoch 与 8 epochs；
- transcript：固定候选记录 token；原始-(q) 接受率每个 token 重复 24 次，层析分支最多 5 个 (q) 层级；
- 统计：500 次分层 bootstrap；低 FPR 阈值对离散并列分数采取保守处理，保证实际 FPR 不超预算。

### 8.2 弱记忆设置（1 epoch，主 pilot）

| 信号 | AUC | 95% CI | 解读 |
|---|---:|---:|---|
| 模型无关 hashed BoW | 0.444 | [0.370, 0.525] | 无证据表明标签可由文本分布直接区分 |
| 基础草稿 Min-K% | 0.536 | [0.462, 0.613] | 草稿单独接近随机 |
| 基础草稿 + 贪心匹配 | 0.588 | [0.528, 0.662] | PipeSD 类反馈已有增益 |
| 基础草稿 + 随机接受率 | 0.612 | [0.542, 0.686] | (p/q) 协议信号优于草稿单独分数 |
| 基础草稿 + 联合特征 | **0.644** | **[0.571, 0.711]** | 比草稿 Min-K% 高 0.108 |
| target logits Min-K% 上界 | 0.654 | [0.575, 0.727] | 联合方法接近直接概率访问上界 |
| 辅助蒸馏草稿 Min-K% | 0.536 | [0.465, 0.614] | 蒸馏后草稿自身仍接近随机 |
| 辅助蒸馏草稿 + 接受率 | **0.643** | **[0.565, 0.714]** | 比草稿自身高 0.107 |

主要证据是 `unattested verifier feedback > draft only`，而不是草稿白盒探针本身。隐藏状态线性探针在本 pilot 中只有 0.50–0.55 AUC，尚无支持；这也避免把所有白盒信号都包装成有效结果。真正的被动 L2 自然生成已在 P1 单列验证，但结果必须与此前固定候选 token 的 L3/模拟 verifier 结果分开报告。

### 8.3 高记忆上界（8 epochs）

- target logits Min-K%：AUC 1.000；
- 基础草稿 Min-K%：AUC 0.536；
- 基础草稿贪心匹配：AUC 0.999；
- 基础草稿随机接受率：AUC 0.968；
- 联合方法：AUC 1.000；
- 辅助蒸馏草稿 Min-K%：AUC 0.656；
- 辅助蒸馏草稿的 q-min 层析：AUC 0.715，优于随机 token 层析的 0.678。

这个设置只说明当 target 明显记忆时，协议反馈能高保真暴露该差异；不能据此估计生产预训练风险。

### 8.4 当前未被支持的假设

- q-min 层析在 1-epoch 设置只有 AUC 0.514–0.571，没有稳定超过直接接受率；
- 隐藏状态和梯度范数代理在当前样本量下接近随机；
- 低 FPR TPR 仍很低且方差大；
- P0 已在一个固定数据划分上跑完 5 个训练 seed；跨数据划分和文档级隔离仍未验证；
- 当前 verifier 是本地语义仿真，尚未接入 PipeSD/SpecEdge 的真实网络代码路径。

因此目前最稳妥的论文主线是**接受比值 + 白盒草稿校准**，而不是接受反馈层析或复杂内部表示。

### 8.7 P0：训练 seed/epoch 稳定性与低 FPR 审计集

为区分“单次 SFT 偶然现象”和训练强度效应，P0 固定 `data_seed=20260824` 与 `audit_seed=20260824`，在 Qwen3-8B-Base target、Qwen3-1.7B-Base base draft 上运行 5 个训练 seed（`20260824`–`20260828`）和 `0/1/2/4` epoch。每个主条件使用 320 members、320 nonmembers，其中 48/类只用于 calibration；另对 epoch1/epoch4 使用 1,048/类，使 held-out test 恰好为 1,000/类。P0 只保留 base draft，避免把 auxiliary/member draft 的训练条件混入稳定性矩阵。

320/类矩阵的联合白盒+协议 AUC 如下：

| target SFT epoch | mean AUC | SD | range |
|---:|---:|---:|---:|
| 0 | 0.512 | 0.000 | 0.512–0.512 |
| 1 | 0.601 | 0.016 | 0.589–0.620 |
| 2 | 0.874 | 0.012 | 0.859–0.889 |
| 4 | 0.997 | 0.001 | 0.996–0.998 |

在 1,048/类大审计条件中，epoch1 的联合 AUC 为 0.594、TPR@1%FPR 为 0.037；epoch4 为 0.990、TPR@1%FPR 为 0.768。P0 支持“当前 adapter SFT 的记忆强度是主要驱动因素，且在固定数据划分上跨训练 seed 稳定”的描述，但不支持将结果外推为生产 SFT 或预训练成员风险。完整结果见 [P0 SUMMARY](../experiments/results/qwen3_sft/p0_matrix/SUMMARY.md)。

### 8.8 P1：真正被动 L2 自然采样协议

P1 使用已有 epoch1/epoch4 target adapter，给定每条 response 的前 48 个 token 作为 context，由 Qwen3-1.7B-Base 自然采样 4-token block；没有固定候选 token、调节 q 或做 q-min 选择。Qwen3-8B-Base target 对每个候选 token 使用温度一致的
`min(1,p/q)` 接受概率，首次拒绝时从 `(p-q)+` residual 分布采样 correction；每条记录独立重复 8 轮。每个条件为 160/类，48/类用于 calibration，112/类用于 held-out test，因此该阶段是协议真实性验证，不承担低 FPR 结论。

| target SFT epoch | acceptance AUC | joint passive-L2 AUC (95% CI) | TPR@1%FPR |
|---:|---:|---:|---:|
| 1 | 0.515 | 0.505 [0.432, 0.573] | 0.009 |
| 4 | 0.436 | **0.631 [0.554, 0.708]** | 0.054 |

epoch1 的自然接受和联合统计均接近随机；epoch4 的联合统计出现描述性信号，但单独接受率方向反而低于 0.5，说明不能预设“接受率越高越像 member”。因此 P1 的当前结论是：强 adapter SFT 条件下，完整被动 L2 统计可能具有成员可识别性；epoch1 的弱记忆条件尚无支持。该结果仍受单一数据划分、本地 target 仿真和 160/类规模限制。完整结果见 [P1 SUMMARY](../experiments/results/qwen3_sft/p1_passive_l2/SUMMARY.md)。

### 8.5 Qwen3 真正 instruction SFT（单 seed）

本轮已将 target 训练改为真正的 instruction SFT：每条样本是“论文标题/固定技术写作指令 → 论文片段 assistant response”，prompt token 的 label 为 `-100`，只有 assistant response 参与 loss。target 为 `Qwen/Qwen3-8B-Base`，draft 为 `Qwen/Qwen3-1.7B-Base`；为适配单张 A100 80GB，target 和训练后的 draft 条件使用 LoRA，因此结果的准确表述是**adapter-based SFT membership**。

- 硬件：GPU 1，NVIDIA A100-SXM4 80GB；峰值显存 19.60 GiB；运行时间约 143 秒（模型已缓存）;
- 构造：160 members、160 nonmembers、160 auxiliary records；每类 48 条用于 audit calibration，112 条用于 held-out test；来源为 11 篇端云 SD 论文；
- target SFT loss：2.993；auxiliary draft distillation 最终 loss：1.198；member-data draft SFT loss：3.317；

| 信号 | AUC | 95% CI | 解读 |
|---|---:|---:|---|
| 模型无关 hashed BoW | 0.478 | [0.401, 0.553] | 未见明显标签分布偏移 |
| 基础 draft Min-K% | 0.457 | [0.385, 0.533] | draft-only 接近随机 |
| 基础 draft + 接受率 | 0.588 | [0.510, 0.659] | 协议反馈有描述性增益 |
| 基础 draft + 联合特征 | 0.566 | [0.491, 0.636] | 单 seed 证据较弱 |
| auxiliary-distilled draft + 接受率 | **0.631** | **[0.549, 0.699]** | 当前 Qwen3 条件的最强单一协议信号 |
| auxiliary-distilled draft + 联合特征 | **0.629** | **[0.556, 0.701]** | 白盒特征未明显超过接受率 |
| member-SFT draft + 联合特征 | 0.589 | [0.513, 0.661] | 同数据 draft 作为风险上界条件 |

该结果支持“auxiliary-distilled draft 的 verifier acceptance feedback 可能携带 SFT member 信息”的方向性判断，但尚不能证明跨 seed 稳定，也不能外推到 Qwen3 的预训练成员或真实网络 endpoint。q-min tomography 在该条件下没有超过直接接受率，应继续作为待验证分支。

### 8.6 Qwen3 4-epoch 强记忆对照

在完全相同的数据划分、seed、模型对和 LoRA 配置下，将 target 与 member-SFT draft 的训练轮数改为 4，作为强记忆 stress condition：

| 信号 | AUC | 95% CI |
|---|---:|---:|
| 基础 draft Min-K% | 0.457 | [0.385, 0.533] |
| 基础 draft + 接受率 | 0.995 | [0.985, 1.000] |
| 基础 draft + 联合特征 | 0.996 | [0.990, 1.000] |
| auxiliary-distilled draft + 接受率 | 0.999 | [0.997, 1.000] |
| auxiliary-distilled draft + 联合特征 | 0.998 | [0.994, 1.000] |
| member-SFT draft + 联合特征 | 0.999 | [0.997, 1.000] |

这说明当 adapter SFT 造成强记忆时，协议接受反馈可以稳定放大 target 与 nonmember 的差异；它验证的是方法在强记忆边界下的可检出性，不应被当作生产 SFT 或预训练风险的估计。完整表格见 [epoch4/RESULTS.md](../experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch4/RESULTS.md)。

## 9. 后续最小充分实验包

### Phase A：把 pilot 变成可投稿证据

1. 在 1、2、4、8 epoch 上各跑 5 seeds；
2. 增加到每类至少 2,000 records，并做文档级隔离；
3. 加入 target LOSS、Min-K%、neighborhood、reference likelihood ratio；
4. 报告配对 $\Delta$AUC、TPR@1%FPR 与逐样本稳定性；
5. 把接受率从仿真改为真实 verifier endpoint 返回值。

### Phase B：模型和草稿来源外推

1. OPT-125M→6.7B、Qwen-0.5B→7B、Pythia-160M→2.8B；
2. 独立、辅助蒸馏、同数据蒸馏、共享权重四类草稿；
3. greedy/stochastic 两类协议；
4. 白盒 token selection 与随机 selection 在固定 query budget 下比较。

### Phase C：协议缓解评估

评估以下设计对隐私、精确性和加速的三方权衡：

- 服务端绑定草稿权重、temperature 和 (q)；
- 会话 nonce + KV 状态绑定，限制同一 prefix 的重复独立验证；
- 只返回完成 KV 更新所必需的最小 verified sequence；
- 对重复 prefix 做隐私预算/频率限制；
- 隐私感知蒸馏：排除高风险点、bottleneck projection、DP-SGD；
- 对 target 与 draft 同时做成员审计，而不是只评估 target；
- 接受结果批量化只能减少观测精度，不能完全隐藏 client 已知 draft 与 verified output 的差异，应诚实量化残余风险。

## 10. 文献证据矩阵

| 证据 | 可借鉴内容 | 本方案如何使用 | 注意事项 |
|---|---|---|---|
| [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | 无损 SD 接受规则 | $A\sim\mathrm{Bernoulli}(\min(1,p/q))$ 的理论基础 | 原文不是隐私研究 |
| [PipeSD, ICML 2026](https://arxiv.org/abs/2605.13319) | 端侧 draft、云端 greedy NAV、返回 accepted/corrected tokens | L1 协议分支 | 其“privacy enhancement”不是成员隐私证明 |
| [SpecEdge, NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119940) | server 返回 verified tokens + bonus token | 证明客户端正常可见验证差异 | 重点是 serving，不评估训练成员 |
| [SLED, SEC 2025](https://doi.org/10.1145/3769102.3770608) | 随机接受公式、拒绝位置和修正 token 反馈 | L2 协议分支 | edge server 不等同远端公有云 |
| [DSSD, 2025](https://arxiv.org/abs/2507.12000) | 端侧发送 token/q，拒绝时下行目标分布 | L4 强反馈分支 | 论文状态按预印本处理 |
| [Mattern et al., 2023](https://arxiv.org/abs/2305.18462) | 邻域难度校准 | 一般基线 | 需要生成高质量邻域文本 |
| [Shi et al., 2023/2024](https://arxiv.org/abs/2310.16789) | Min-K% token 聚合 | 白盒 token 筛选与 target 上界 | 时间切分 benchmark 可能混入分布差异 |
| [Duan et al., COLM 2024](https://arxiv.org/abs/2402.07841) | 大规模预训练成员检验常接近随机；时间差异可造成虚高 | 负结果解释与严格同分布设计 | 不应期待所有预训练场景都强可检出 |
| [Meeus et al., USENIX Security 2024](https://www.usenix.org/conference/usenixsecurity24/presentation/meeus) | 文档级窗口聚合 | 文档级主任务 | 后验数据构造仍需严控 |
| [Meeus et al., SaTML 2025](https://arxiv.org/abs/2406.17975) | 后验 benchmark 的分布偏差审计 | BoW 闸门、随机注入、来源匹配 | 是本方案完整性要求的核心依据 |
| [Hayes et al., NeurIPS 2025](https://arxiv.org/abs/2505.18773) | 强检验在真实 LLM 上仍有限；逐样本决策不稳定 | 多训练 seed 与稳定性报告 | AUC 改善不等于单样本结论可靠 |
| [LUMIA, 2025](https://arxiv.org/abs/2411.19876) | 内部状态逐层线性探针 | 白盒草稿增强基线 | 需避免高维小样本过拟合 |
| [Cui et al., 2025/2026](https://arxiv.org/abs/2505.11837) | 蒸馏学生不一定更私密，部分样本信号可增强 | 草稿来源轴与缓解方案 | 当前为 arXiv 证据，需独立复现 |
| [Jagielski et al., 2023](https://arxiv.org/abs/2303.03446) | 学生可继承 teacher 训练成员影响 | 草稿模型风险动机 | 非 LLM 专属，不能替代端云实验 |
| [InfoRMIA, 2025](https://arxiv.org/abs/2510.05582) | token-level 信息聚合 | 层析恢复后可选评分器 | 较新预印本，作为扩展而非唯一基线 |

## 11. 预期论文贡献

如果 Phase A–B 复现成功，可把贡献收敛为四点：

1. 首次形式化端云 SD 的训练成员隐私面，区分草稿白盒、验证语义反馈和草稿可调性；
2. 揭示接受规则本身构成 (p/q) 的协议内统计接口，并提出 DraVer-MemAudit；
3. 建立无时间差异、无来源差异、含模型无关负对照和跨 seed 稳定性的端云 SD 成员基准；
4. 系统评估草稿绑定、反馈最小化和隐私感知蒸馏的 privacy–exactness–speed 三方权衡。

其中第 2 点回答“为什么必须在端云 SD 场景研究”，第 3 点回应当前 LLM 成员研究最常见的分布偏差质疑。
