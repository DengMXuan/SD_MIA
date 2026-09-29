# 公开源码快照

当前开发仓库的旧提交仍包含已移出的专利材料和实验结果。只在现有仓库提交删除操作，再把整个仓库改为公开，仍会公开这些旧对象。

[`tools/prepare_public_release.py`](../tools/prepare_public_release.py) 从当前工作区的已跟踪源码和明确列出的新文档创建独立 Git 仓库。它不复制原 Git 历史、远程配置、本机 `artifacts/`、专利材料、实验 CSV 或报告；会核对 Markdown 本地链接，并要求先选定 `LICENSE`。输出必须是 `artifacts/` 下的新目录。

```bash
.venv/bin/python -B tools/prepare_public_release.py \
  --output artifacts/public_release_ready
git -C artifacts/public_release_ready rev-list --all --count
git -C artifacts/public_release_ready remote -v
git -C artifacts/public_release_ready ls-files
```

正常检查结果为一个根提交、没有远程，以及只含准备公开的文件。完成后从该独立目录创建新的公开仓库，并再次核对将要推送的文件和许可证。若必须沿用现有 GitHub 仓库地址，需要另行清理其所有公开 refs 与历史并协调强制更新；本工具不会改动原仓库或远程。

许可证会决定第三方能否复用代码及相关专利授权。选定许可证之前，仅可使用 `--draft` 生成供本地审阅的候选快照；该选项不会把候选快照标记为可发布。
