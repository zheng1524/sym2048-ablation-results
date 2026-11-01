# 已完成消融训练快照（2026-11-01）

本快照交付已经完成的训练权重、配置、验证曲线和审计证据。
**本轮统一正式测试尚未开始；以下数值全部来自检查点选择使用的验证集，不作为最终消融结论。**

快照时间：`2026-11-01T12:42:06.208101+00:00`；冻结训练源码：`8258a9eda7b5fba0dda9d4032737c58091f8bcc4`。
发布分支：`experiment/completed-ablation-results-20261101`。

已完成：7 个 TD 模型、4 个监督阶段 best，共 11 份模型检查点及 157 条验证记录。
A、C 的 seed 2 尚未完成，本快照未复制其权重或恢复状态。B_practical 尚未开展。

| 组别 | 种子 | 实际 TD 更新 | best 步数 | 验证平均最大指数 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| d4_distill_td | 0 | 10000 | 9000 | 8.658203 | 98.05% | 67.97% |
| d4_distill_td | 1 | 10000 | 10000 | 8.662109 | 97.27% | 69.73% |
| d4_scratch_td | 0 | 10000 | 10000 | 6.996094 | 0.00% | 0.00% |
| d4_scratch_td | 1 | 10000 | 8000 | 6.996094 | 0.00% | 0.00% |
| d4_scratch_td | 2 | 10000 | 9000 | 7.000000 | 1.17% | 0.00% |
| plain_distill_td | 0 | 10000 | 10000 | 8.541016 | 95.90% | 58.40% |
| plain_distill_td | 1 | 10000 | 10000 | 8.591797 | 97.07% | 62.11% |

| 监督阶段诊断 | 种子 | 实际监督更新 | best 步数 | 验证平均最大指数 |
|---|---:|---:|---:|---:|
| d4_distill_td | 0 | 100000 | 75000 | 8.503906 |
| d4_distill_td | 1 | 100000 | 85000 | 8.490234 |
| plain_distill_td | 0 | 100000 | 95000 | 8.005859 |
| plain_distill_td | 1 | 100000 | 100000 | 8.000000 |

保护边界：
- 原仓库基线为 `6eb6db44f0a3d70c15b92b5200404bb2b033d686`。逐一核对其 38 个既有文件：Git blob 与文件模式全部保持一致。
- 原 README、RUN_STATUS、AGENTS、网络、训练、规则、评估、测试、pyproject.toml 和 uv.lock 均未修改。
- 新代码只位于 game2048/ablations/、新增脚本和新增测试；该源码与训练时冻结版本逐文件 SHA256 一致。
- 本次发布只在新的 docs/ablations/20261101-completed/ 命名空间增加快照；未切换或修改正在训练的工作区。

目录：
- checkpoints/：每个已完成阶段的 best/model.json 和 params.msgpack，沿用仓库模型两文件格式。
- curves/：原始 training.json、request.json、complete.json，含全部验证历史。
- completed-training.csv：已完成阶段、参数量、预算、指标、耗时及权重 hash。
- validation-curves.csv：157 条逐验证点记录；这是验证曲线 CSV，不是正式测试逐局 CSV。
- snapshot.json、audit.json、ARTIFACTS_SHA256.json：快照范围、正确性检查和全部文件校验和。
- evidence/：原文件保护清单、源码清单、环境、教师核验、GPU 检查及本次检查证据。
- logs/：仅已完成阶段的原始日志，扩展名 .txt；历史中断记录保留。

统一协议：
- 3×3 2048，优化终局最大块的期望指数；达到 256 后继续。推理使用合法动作上的网络贪心。
- A：D4（304833 参数）完整教师监督 100000 步，再 TD(λ) 10000 步。
- B：同一 D4 随机初始化直接 TD 10000 步，不读教师；A/B 的同 seed 初始 hash 已核对。
- C：普通 CNN（305403 参数，宽 83），同监督和 TD 预算；A/C 使用相同方向增强规则。
- 验证种子 400000–400511；计划正式测试种子 500000–501999。正式模型和超参冻结前不使用测试集。
- A 比 B 多出教师和监督成本；参数匹配不代表 FLOPs 或墙钟耗时相等。
- 训练使用 512 个 afterstate 的梯度分块累加，每个有效大 batch 只做一次优化器更新；预算未缩减。

资产与成本：
- 完整教师 41325017 状态，Bellman 残差 0，内容 SHA256 `6a66f3b52bd5bef3e270c777e4da063cae7e7038a0c5c858eb7bb7274466fc8e`。教师、虚拟环境、缓存、优化器恢复包均未上传。
- 本机 RTX 3060 Laptop 6GB；运行环境及显存 allocator 恢复记录见 RUNTIME.md。
- completed-training.csv 的 training_seconds 是保存的训练计时；all_process_attempt_seconds 包含该阶段历史失败/恢复尝试的进程墙钟时间。
- checkpoints/ 可独立用于推理；恢复训练还需要本地 runs/ 下未上传的完整恢复状态。

加载任意已发布 best 模型（不读取教师）：

```python
from pathlib import Path
from game2048.ablations.models import load_model

model, params, metadata, spec = load_model(
    Path("docs/ablations/20261101-completed/checkpoints/plain_distill_td/seed0/td")
)
print(spec, metadata["step"])
```

本分支继承冻结源码提交中的开发检查和 smoke 记录，它们在旧的 20261031 命名空间并明确标注 smoke。本目录没有将 smoke、部分训练或历史主线成绩混入本次已完成训练结果。

本次独立发布工作区的检查：Ruff、格式检查、ty 全部通过；CPU 全套 **38 项测试通过（82.59 秒）**。日志在 evidence/snapshot-*.txt。同一冻结源码的 GPU 全套检查记录也已随附。测试使用共享的已安装依赖、独立工作区源码及 CPU 后端，未改变正在运行的训练环境或源码。
