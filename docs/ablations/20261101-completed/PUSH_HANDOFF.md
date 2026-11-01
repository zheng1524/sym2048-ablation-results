# 已发布到独立私有仓库

仓库：https://github.com/zheng1524/sym2048-ablation-results

实验分支：`experiment/completed-ablation-results-20261101`。
新仓库的 main 保留原始基线 `6eb6db44f0a3d70c15b92b5200404bb2b033d686`；
实验实现及本快照在上述独立分支。未创建或合并 PR。

依用户明确要求，发布位置改为 zheng1524 自有仓库。原仓库
`2986002971/sym2048` 的所有分支和既有文件保持不变。先前对原仓库的 403
记录留在 evidence/push-attempt-403.txt，作为历史记录。

本地发布工作区仍为 `/home/liu/other/sym2048-completed-20261101`。
为保持原有配置，它的 origin 仍指向原仓库；向新仓库追加结果时应显式指定 URL：

```bash
git push https://github.com/zheng1524/sym2048-ablation-results.git \
  HEAD:refs/heads/experiment/completed-ablation-results-20261101
```

也可单独克隆新仓库的实验分支。后续是否合并回原仓库由用户另行决定。
本次上传为已经完成训练的阶段快照，全部成绩属于验证集；尚未完成的 A/C seed 2
以及正式测试结果未混入该快照。现有训练未受发布操作调整。
