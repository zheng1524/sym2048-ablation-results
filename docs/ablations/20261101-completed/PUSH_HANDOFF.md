# 原仓库推送交接

**尚未上传；远端新分支和 PR 均未创建。** 实际推送返回 GitHub HTTP 403，账号
`zheng1524` 没有 `2986002971/sym2048` 的写权限，见 evidence/push-attempt-403.txt。
本机账号及 GitHub 连接都报告 pull=true、push=false。

独立本地分支：`experiment/completed-ablation-results-20261101`。
独立工作区：`/home/liu/other/sym2048-completed-20261101`。
自包含 bundle：`/home/liu/other/sym2048-completed-ablation-results-20261101.bundle`。

原仓库所有 38 个已有文件的 blob 和模式保持一致，main 仍为
`6eb6db44f0a3d70c15b92b5200404bb2b033d686`。正在运行的训练工作区未作调整。
本分支只增加独立消融实现及文档/结果路径，不包含教师、缓存或恢复包。

仓库所有者为 zheng1524 授予 Write，或在本机 gh 配置已有写权限的账号后，执行：

```bash
cd /home/liu/other/sym2048-completed-20261101
git push --set-upstream origin HEAD:refs/heads/experiment/completed-ablation-results-20261101
gh pr create --draft --base main \
  --head experiment/completed-ablation-results-20261101 \
  --title '新增独立消融实现与已完成训练快照' \
  --body-file docs/ablations/20261101-completed/PR_BODY.md
```

不使用 force push，不向 main 推送，不合并 PR。若远端后来出现不属于本快照的同名分支，
使用新的唯一分支名。bundle 含完整历史，可在另一台有写权限的机器克隆后推送。
