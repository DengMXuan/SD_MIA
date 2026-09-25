# 实验目录维护

## 2026-09-25 全生命周期整理

当前路径规则见 [产物目录合同](../../docs/artifact_layout.md)。维护入口为：

```bash
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout plan
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout apply
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout verify
```

执行迁移时停止实验写入；已有 worker/coordinator 锁被占用时拒绝执行。迁移使用同盘 rename，保留旧入口链接并修正内部相对链接；目标冲突或跨盘会失败。中断后重复 apply 使用原始清单恢复。没有权重复制、数据清理或来源哈希改写。

清单和核验结果位于 `artifacts/maintenance/migrations/20260925_lifecycle/`。旧审计来源和请求仍绑定旧代码/路径，不会自动变为新代码可续跑的结果。错误 seed 的 `qwen_fixed_v1` 批次随后经授权删除；原 `PLAN.json` 仍列有该批次的 940 个文件和 180 个内部链接。因此现在重新执行历史 `migrate_lifecycle_layout verify` 会因旧文件缺失而失败，不能将其作为当前实验的预检。新审计使用各自的 `dry-run`/`status` 校验。下方记录 2026-09-22 的独立历史迁移。


2026-09-22 迁移仅重构 SD SFT 实验及本地存储。baseline、pretraining、DP 的算法实现不变。日常使用见 [SD 实验入口](../sd_membership_sft/README.md)。维护脚本不参与正常运行，也不会自动执行。

## 本次迁移的证据链

本地 `artifacts/migrations/20260922_layout/` 保留以下轻量证据：

- `storage_moves.json`：物理路径迁移清单；同盘 rename 后留下旧路径链接，无权重复制或重写。
- `model_inventory_before.json`、`head_inventory_unchanged.json`：模型/草稿头文件 inode、大小和纳秒修改时间。
- `audit_binary_hashes_before.json`：分数、观测和检测器的迁移前 SHA256。
- `old_source_hashes.json`、`reviewed_new_source_hashes.json`：迁移前后固定的代码指纹。
- `VERIFIED.json`：迁移后核验结果。
- `RESUME_VERIFIED.json`：完整模型对哈希、历史指标/耗时不变、真实未完成任务复用 2,995 条轨迹的 CPU 验证。

2026-09-25 清理了已完成迁移的一次性大文件：`before_metadata.tar.gz`、
`metadata_plan.json`，以及 `20260922_cross_model/before_metadata.tar.gz`。
正常训练、审计和结果读取不使用它们；迁移前元数据回退已不可执行。

当时的一次性审计目录迁移严格核验旧来源、模型清单、所有方法产物以及每条轨迹。完整观测中的 features/counts 和 cost 必须与逐条缓存完全相同，才允许清理这些重复缓存。未完成集合的逐条缓存全部保留。该迁移脚本已随错误 seed 的旧审计批次删除。

随后 `apply` 应用保存的计划，逐文件原子更新 JSON、清理已证明重复的缓存、移动其余缓存并建立链接。该计划已清理，历史迁移命令不可重跑。

这是一项已固定版本的一次性迁移，不是通用的“重新签署结果”功能。新实验已使用新目录，不需要再迁移。修改算法后不得更新指纹以复用旧结果。

## 数据保留与历史回退限制

迁移时，权重、分数、指标、耗时与记录 ID 均保持不变。2026-09-25 后续清理删除了已停用的历史实验归档和自然 SD 结果；错误审计 seed 的 `qwen_fixed_v1` 固定候选主方法与 baseline 结果随后也已删除。已完成条件的逐条轨迹曾在与完整观测中的 features/counts/cost 逐条核对后清理。

`storage_moves.json` 仍记录原路径和新路径，但迁移前元数据与旧源代码的压缩备份已删除，无法再从本地完整回退到迁移前状态。已经清理的重复轨迹二进制也不在迁移备份中；对应的 features/counts/cost 仍保存在完整观测归档中。

本次训练护照中的旧路径刻意保留，避免改变数据/模型训练来源证明；保留的兼容链接承担这些记录到新存储的映射。迁移前后的源代码哈希仍保留用于核对。

## 历史复核真实断点（备份清理后不可重跑）

一次性历史复核曾针对迁移时暂停的 Wiki epoch1 seed1949 member 草稿任务重建记录，在第 2,996 条需要推理时立即截停。完成结论保留在 `RESUME_VERIFIED.json`；原元数据备份和错误 seed 旧批次均已清理，复核脚本也已删除，因此不能再次执行这个历史对照检查。

## 2026-09-25 模型资产评测字节码修复

`python -m experiments.maintenance.repair_quality_bytecode` 是固定版本的一次性
修复，只接受 `model_quality_v2` 及脚本中审核过的两份来源辅助代码版本。
它对失败任务及摘要需要变化的 checkpoint 核验原始完整指纹；其余成功任务
在资产清单、来源和报告产物一致后保留原摘要。同时核验冻结请求和失败项
全部缓存，随后保留原元数据备份，更新指纹规则并记录可续接的修复计划。它不训练模型、
不修改权重、不修改缓存指标，也不能用于授权算法变化后的结果复用。

修复记录位于该批次 `repairs/bytecode_inventory_v1/`：`before/` 保存旧请求、
缓存请求和旧报告，`source_before/`、`source_after/` 保存审核过的源码，
`VERIFIED_CHECKPOINTS.json` 保存逐字节核验结果，`PLAN.json`、`APPLIED.json`
记录元数据转换。应用完成后普通质量评测 `run` 仅恢复缺失报告的任务。

模型资产清单与完整指纹共用文件边界，只排除 `__pycache__` 下的 `.pyc/.pyo`。
远程 `.py` 实现、配置、tokenizer、权重和独立 `.pyc` 均继续受校验保护。

从仓库根目录依次执行，第一步成功后再启动第二步：

```bash
.venv/bin/python -m experiments.maintenance.repair_quality_bytecode
.venv/bin/python -m experiments.model_quality.cli run --gpus 1 2 3 4
.venv/bin/python -m experiments.model_quality.cli status
```

GPU 编号按实际可用设备调整。维护命令中断后可用同一命令继续，已保存的
完整指纹核验记录会在资产清单一致时复用。恢复运行跳过已有的 74 项报告，
对缺失的 16 项重新执行适配器校验并读取 v2 缓存生成报告。
