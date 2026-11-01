新增独立的蒸馏/对称结构消融实现及已完成训练快照，便于审阅实际模型和验证曲线。

本次包含 A seed 0/1、B seed 0/1/2、C seed 0/1 的已完成 TD 模型，以及 A/C seed 0/1 的监督 best。A/C seed 2 未完成，未纳入快照；统一正式测试尚未开始，所有分数明确标为验证集。

相对原 main，所有既有文件的 blob 和模式均保持一致，差异仅为新增路径。main 不作修改，不请求合并。详见 docs/ablations/20261101-completed/README.md、audit.json 和 evidence/original-files-unchanged.json。

按原仓库规范提供 uv/Ruff/ty/pytest 检查记录，权重沿用 model.json＋params.msgpack；未包含教师、环境、缓存和优化器恢复包。
