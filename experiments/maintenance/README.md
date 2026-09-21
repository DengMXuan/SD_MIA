# 实验目录维护

2026-09-22 迁移仅重构 SD SFT 实验及本地存储。baseline、pretraining、DP 的算法实现不变。日常使用见 [SD 实验入口](../sd_membership_sft/README.md)。维护脚本不参与正常运行，也不会自动执行。

## 本次迁移的证据链

本地 `artifacts/migrations/20260922_layout/` 保存：

- `before_metadata.tar.gz`：迁移前代码和审计元数据；不包含模型、分数或观测二进制。
- `storage_moves.json`：物理路径迁移清单；同盘 rename 后留下旧路径链接，无权重复制或重写。
- `model_inventory_before.json`、`head_inventory_unchanged.json`：模型/草稿头文件 inode、大小和纳秒修改时间。
- `audit_binary_hashes_before.json`：分数、观测和检测器的迁移前 SHA256。
- `old_source_hashes.json`、`reviewed_new_source_hashes.json`：迁移前后固定的代码指纹。
- `metadata_plan.json`：每项元数据的旧哈希和新内容、已核验重复轨迹清单、恢复任务配置。
- `VERIFIED.json`：迁移后核验结果。
- `RESUME_VERIFIED.json`：完整模型对哈希、历史指标/耗时不变、真实未完成任务复用 2,995 条轨迹的 CPU 验证。

`python -m experiments.maintenance.migrate_audit_layout plan` 严格核验旧来源、模型清单、所有方法产物以及每条轨迹。完整观测中的 features/counts 和 cost 必须与逐条缓存完全相同，才允许清理这些重复缓存。未完成集合的逐条缓存全部保留。计划生成不覆盖已有计划。

`python -m experiments.maintenance.migrate_audit_layout apply` 应用已保存计划，逐文件原子更新 JSON、清理已证明重复的缓存、移动其余缓存并建立链接。若过程意外中断，可在代码、模型和实验均未改变时重跑同一个 apply。每步接受旧状态或计划中的新状态；不接受额外改动。最后逐份验证结果。

这是一项已固定版本的一次性迁移，不是通用的“重新签署结果”功能。新实验已使用新目录，不需要再迁移。修改算法后不得更新指纹以复用旧结果。

## 数据保留与回退

权重、分数、指标、耗时与记录 ID 均保持不变。历史结果归档保留；自然 SD 结果保留在原条件下。只清理完整观测已经精确覆盖的逐条重复缓存。

需要回退时，先停止全部实验并备份迁移后新增产物。对照 `storage_moves.json` 逆序恢复目录；仅移除相应的链接，再将物理目录移回，切勿递归删除链接目标。从元数据备份中选择恢复相关旧路径 JSON 和旧源代码。已经清理的完整轨迹二进制不在元数据备份中，其 features/counts/cost 均仍保存在完整观测归档中；恢复已完成实验不需要重建这些逐条副本。不要直接覆盖解包到运行中的目录。

历史训练护照中的旧路径刻意保留，避免改变数据/模型训练来源证明；兼容链接承担旧记录到新存储的映射。迁移前的源代码备份用于历史结果追溯。

## 复核真实断点（不启动 GPU 实验）

```bash
.venv/bin/python -m experiments.maintenance.verify_layout_resume
```

该核验针对迁移时暂停的 Wiki epoch1 seed1949 member 草稿任务：重建真实记录，使用临时目录读取已有轨迹，在第 2,996 条需要推理时立即截停。此检查固定了迁移时进度；恢复正式实验后进度变化，应以正常 status 和运行器校验为准。
