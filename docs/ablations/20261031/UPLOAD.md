# Isolated upload handoff

The configured credentials currently have pull=true and push=false. Nothing has
been uploaded and no PR has been created. Main is unchanged. The local branch is
`experiment/distillation-d4-ablation-20261031`.

The local bundle at `/home/liu/other/sym2048-ablation-20261031.bundle` contains this
branch and its history. The background job refreshes it after completed or
interrupted execution. Current live state is in JOB_STATUS.json and REPORT.md;
downloaded bundles reflect their creation time, not subsequent live progress.

Once push permission is available, from this checkout:

```bash
git push origin experiment/distillation-d4-ablation-20261031
gh pr create --draft --base main \
  --head experiment/distillation-d4-ablation-20261031 \
  --title 'Add distillation and D4 architecture ablations' \
  --body-file docs/ablations/20261031/PR_BODY.md
```

Alternatively clone the bundle into another checkout before pushing. No merge,
main rewrite, alternate public repository, teacher upload or credential workaround
is part of this workflow. Read the report's actual completion scope before upload.
