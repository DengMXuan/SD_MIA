# 实验目录维护

## 2026-09-25 全生命周期整理

当前路径规则见 [产物目录合同](../../docs/artifact_layout.md)。维护入口为：

```bash
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout plan
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout apply
.venv/bin/python -m experiments.maintenance.migrate_lifecycle_layout verify
```

执行迁移时停止实验写入；已有 worker/coordinator 锁被占用时拒绝执行。迁移使用同盘 rename，保留旧入口链接并修正内部相对链接；目标冲突或跨盘会失败。中断后重复 apply 使用原始清单恢复。没有权重复制、数据清理或来源哈希改写。

清单和核验结果位于 `artifacts/maintenance/migrations/20260925_lifecycle/`。旧审计来源和请求仍绑定旧代码/路径，保持历史快照，不会自动变为新代码可续跑的结果。下方记录 2026-09-22 的独立历史迁移。


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
正常训练、审计和结果读取不使用它们；迁移前元数据回退与下面的一次性复核脚本不再可执行。

迁移时，`python -m experiments.maintenance.migrate_audit_layout plan` 严格核验旧来源、模型清单、所有方法产物以及每条轨迹。完整观测中的 features/counts 和 cost 必须与逐条缓存完全相同，才允许清理这些重复缓存。未完成集合的逐条缓存全部保留。

随后 `apply` 应用保存的计划，逐文件原子更新 JSON、清理已证明重复的缓存、移动其余缓存并建立链接。该计划已清理，历史迁移命令不可重跑。

这是一项已固定版本的一次性迁移，不是通用的“重新签署结果”功能。新实验已使用新目录，不需要再迁移。修改算法后不得更新指纹以复用旧结果。

## 数据保留与历史回退限制

迁移时，权重、分数、指标、耗时与记录 ID 均保持不变。2026-09-25 后续清理删除了已停用的历史实验归档和自然 SD 结果；本次固定候选主方法与 baseline 的结果仍保留。已完成条件的逐条轨迹在与完整观测中的 features/counts/cost 逐条核对后清理。

`storage_moves.json` 仍记录原路径和新路径，但迁移前元数据与旧源代码的压缩备份已删除，无法再从本地完整回退到迁移前状态。已经清理的重复轨迹二进制也不在迁移备份中；对应的 features/counts/cost 仍保存在完整观测归档中。

本次训练护照中的旧路径刻意保留，避免改变数据/模型训练来源证明；保留的兼容链接承担这些记录到新存储的映射。迁移前后的源代码哈希仍保留用于核对。

## 历史复核真实断点（备份清理后不可重跑）

`verify_layout_resume.py` 曾针对迁移时暂停的 Wiki epoch1 seed1949 member 草稿任务重建记录，在第 2,996 条需要推理时立即截停。完成结论保留在 `RESUME_VERIFIED.json`；原元数据备份已清理，因此不能再次执行这个历史对照检查。
