# 五组模型与 Wiki 时间划分：官方日期证据

核查日期：2026-09-27。本笔记区分**训练数据截止时间**、**权重/模型发布日期**与**实验文章日期**。发布日期只能证明更晚才首次出现的内容不可能进入已发布的固定权重；它不能证明更早的文章实际被训练使用。仓库固定的五组目标/草稿标识和 revision 见 [`model_pairs.json`](../experiments/shared/models/model_pairs.json)。

当前 WikiTection 池的真实 manifest 将近期文章首次创建窗口记为 **2026-04-01 至 2026-09-17**，而非 `pools.py` 中已过时的默认窗口注释；见 [`pool.manifest.json`](../artifacts/data/pools/wikitection/pool.manifest.json)。历史采集器选择 **2023 年首次创建**、截至 **2023-12-31** 的修订版；见 [`README.md`](temporal_data.md)。

| 仓库模型组 | 目标模型的公开时间证据 | 草稿/预测头的公开时间证据 | 对 2023 历史组 / 2026 近期组的含义 |
|---|---|---|---|
| `qwen3`: Qwen3-8B-Base + Qwen3-1.7B-Base | [Qwen3 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B-Base)和[Qwen3 官方发布文](https://qwenlm.github.io/blog/qwen3/)；本仓库使用 **2025-04-29** 作为发布界线。模型卡没有可核实的精确预训练语料截止日。 | [1.7B 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B-Base)，同属 Qwen3；卡片没有可核实的精确截止日。 | 2026 文章晚于已发布权重；2023 文章只可称“可能在预训练窗口内”，不能确认成员。 |
| `gemma4`: Gemma 4 12B + E2B | [Google 12B 官方模型卡](https://huggingface.co/google/gemma-4-12B/blob/023679ed352de9bb66cc873c9009ce3482585c08/README.md#training-dataset)明确写明预训练数据 **cutoff date: January 2025**。 | [Google E2B 官方模型卡](https://huggingface.co/google/gemma-4-E2B/blob/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/README.md#training-dataset)也写明 **January 2025**。 | 2026 近期组晚于明确的预训练截止；2023 历史组早于截止，但未证实入训。 |
| `qwen3_8b_eagle3`: Qwen3-8B + RedHat EAGLE-3 | [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md)说明为预训练加后训练模型，但未给精确语料截止日期。它与 Base 是不同权重。 | [RedHat 官方预测头卡](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3/blob/08610ffa01dd9f16731fe8f627b85905b6aa51c4/README.md)记载 **2025-07-27 发布**；用 ShareGPT Vicuna 与 UltraChat 200k 训练，没有给出这些语料的逐文档截止日。 | 2026 近期组晚于目标和预测头发布；2023 历史组是否进过目标预训练或头训练均不能由日期断定。 |
| `llama31_8b_eagle3`: Llama 3.1 8B Instruct + RedHat EAGLE-3 | [Meta 官方 Llama 3.1 模型卡](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)给出知识截止 **December 2023**、发布于 **2024-07-23**。此仓库实际加载的是 [Unsloth 镜像](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)；其缓存 tokenizer 聊天模板也写有 `Cutting Knowledge Date: December 2023`。 | [RedHat 官方预测头卡](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3/blob/f4fa34a8f803a0ba75d048d6b3dbc1ad5149e9ac/README.md)记载 **2025-07-27 发布**，说明训练于 ShareGPT Vicuna 和 UltraChat 200k，但无精确文档截止日。 | 2026 近期组晚于目标和头发布。历史组的 **2023-12-31 修订版可能含有 Meta 截止后的文字**；整组不能被视作可靠的“截止前成员”代理。 |
| `qwen35_9b_mtp`: Qwen3.5-9B-Base 内置 MTP | [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)写明 Base 预训练权重、内置 MTP。[官方 Hugging Face 提交记录 API](https://huggingface.co/api/models/Qwen/Qwen3.5-9B-Base/commits/main?limit=100)显示仓库初始提交于 **2026-02-26**、权重于 **2026-02-27** 上传；[2026-02-27 提交的文件树](https://huggingface.co/api/models/Qwen/Qwen3.5-9B-Base/tree/6db4f547c43a5e3eb2a3784a3241e472b5d41bdb?recursive=false&expand=false)已包含四个 `model.safetensors` 分片。固定 revision `68c46c4b3498877f3ef123c856ecfde50c39f404` 是 **2026-04-23** 的 LICENSE 更新，未更新权重。模型卡**没有披露精确预训练数据截止日**。 | MTP 与目标同一仓库、同一 revision，并非单独发布的草稿模型。 | 固定权重先于 2026-04-01 的近期池起点上传，因此之后首次创建的文章不可能进入这些权重；2023 文章是否入训未知。 |

## 解释边界

1. **“模型截止早于非成员集”正是时间代理所需的方向。** 当前 2026 近期文章晚于 Gemma 4 明确公布的 2025-01 预训练语料截止，也晚于 Qwen3 和 Llama 3.1 原模型及 RedHat 预测头的发布、Qwen3.5-9B 固定权重的上传。Gemma 4 的精确发布日期在本机未核实；未公开精确截止日的 Qwen 模型只能依据固定权重公开时间排除其后的新文章，不能把发布日期冒充知识截止日。
2. **“发布前”不等于“被训练”。** 2023 历史页面即便早于模型发布，也不证明在训练语料中。Wiki 时间划分应报告为 `presumed member` / `post-release nonmember proxy`，不能给出真实成员推断的 AUC 解释。
3. **Llama 3.1 是最明显的历史组时间冲突。** 其 2023-12 知识截止与历史组 2023-12-31 快照接壤；需改用截至 2023-11 或更早的历史修订、并检查页面首次创建日，才有明确的“截止前”条件，仍无法证明真实入训。
4. RedHat 预测头各于 2025-07 发布，因而 2026 近期文章晚于头权重；它们使用的聊天训练集不能用目标模型的知识截止日期代替。将头替换为用本实验辅助数据蒸馏的版本时，还必须检查辅助数据与评估集是否重叠。

来源可核查性：Gemma 4、Qwen3.5、Qwen3-8B 和 RedHat 模型卡在本机 Hugging Face 缓存有对应 revision；Qwen3.5 的提交时间由宿主机只读查询官方 Hugging Face API 核验。对比 [2026-02-27 文件树](https://huggingface.co/api/models/Qwen/Qwen3.5-9B-Base/tree/6db4f547c43a5e3eb2a3784a3241e472b5d41bdb?recursive=false&expand=true) 与 [固定 revision 文件树](https://huggingface.co/api/models/Qwen/Qwen3.5-9B-Base/tree/68c46c4b3498877f3ef123c856ecfde50c39f404?recursive=false&expand=true)，四个模型分片的 LFS 哈希完全相同。Meta 官方卡及 Qwen3 Base 卡在当前缓存没有 README，其日期依据所链官方页面；未把未披露的精确 Qwen 数据截止日期写为事实。
