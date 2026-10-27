# 3×3 2048：完整教师 → 对称价值网络 → 采样规划

目标是最大化终局最大块的期望指数 `E[log2(max_tile)]`，不在达到 256 时结束。
同时报告 256/512/1024 达成率、最大块、棋盘数字之和及步数。
该目标会权衡不同结局的概率，并不等价于只追求偶尔出现的最高纪录。

完整教师已在提供的服务器构建并全量验证：**41,325,017** 个 D₄ 规范化非终局状态，
Bellman 残差 **0**，包括达到 1024 后继续游戏的状态。构建及中间文件约 4.3 GiB。
三个训练种子的正式实验已完成。每个种子、每种方法测试 2000 局：监督模型贪心的
256/512 达成率平均为 92.12%/51.00%，TD(λ) 微调后为 96.85%/66.28%，均未达到 1024。
**当前 shooting 和 MPPI 优先级规划显著弱于贪心，保留用于复现该轮对照，尚未重写。**
详细数据、模型位置和验证范围见 [RUN_STATUS.md](RUN_STATUS.md)。

## 安装与检查

```bash
uv sync                         # CPU 开发环境，Python 3.12
uv sync --extra cuda            # Linux NVIDIA GPU；本次在 4090D 上验证
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

远端项目：`/root/autodl-tmp/game2048-coursework`，uv：`.uv-bootstrap/bin/uv`。
远端原有 Conda Python 导入 readline 会崩溃，测试使用
`uv run python -m pytest -p no:capture -q` 绕过 pytest 的 readline 初始化。
这是解释器环境问题；GPU 网络/规划测试已实际运行通过。
下载不畅时可添加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 到 `uv sync`。

## 运行

```bash
# 完整枚举、反向动态规划、全量闭包/Bellman/索引核验；中断后同命令续算
NUMBA_NUM_THREADS=16 uv run game2048 teacher build artifacts/teacher3 --threads 16
uv run game2048 teacher inspect artifacts/teacher3

# 完整流水线的小规模运行，不能用它得出方法优劣结论
NUMBA_NUM_THREADS=16 uv run game2048 experiment artifacts/teacher3 runs/smoke --seeds 0 --smoke

# 复现已完成的正式对照；包括已知退化的 shooting/MPPI，耗时较长
NUMBA_NUM_THREADS=16 uv run game2048 experiment artifacts/teacher3 runs/full --seeds 0 1 2
```

实验入口分阶段使用独立进程，释放前一阶段 GPU 内存。已完成训练阶段跳过，评估阶段会核对
参数 hash、评估配置和逐局 CSV 后复用；未完成训练从最近保存的验证点恢复。
训练恢复需要原配置及优化器/状态文件，仅复制 `best/` 不包含恢复训练所需的全部文件。
改变算法做新实验时使用新的输出目录，避免混用历史结果。日志在实验目录的 `seed*-*.log`。
教师未完成或未核验时训练直接报错，
**不会边训练边补算教师**。

```bash
uv run game2048 train artifacts/teacher3 runs/seed0 --seed 0
uv run game2048 train artifacts/teacher3 runs/seed0 --seed 0 --resume
uv run game2048 evaluate runs/seed0/best runs/eval-greedy --mode greedy
uv run game2048 finetune runs/seed0/best runs/td0 --seed 0
uv run game2048 evaluate runs/td0/best runs/eval-td --mode greedy
uv run game2048 replay runs/td0/best runs/replay.json --mode greedy
```

`best/` 只有 `model.json` 与 `params.msgpack`，可复制到没有教师文件的机器推理。
推理只用 NN 与游戏规则，没有查表策略或隐藏求解器。
Gym 接口为 `gym.make("Game2048-v0", size=3, target=256)`，用于展示和参考验证。
Gym 观测是实际数字，JAX/Numba 内部棋盘则是指数，0 表示空格。
`render_mode="ansi"` 时 `env.render()` 返回字符串，`rgb_array` 返回 RGB 数组，
`human` 显示窗口。原始教师环境已保存为 Git 基线 `4a2a520`；工作区版本包含 3×3 默认值、
种子随机数、终局奖励和渲染接口修正。

规则实现分工：`kernels.py` 用 Numba 位棋盘和行查表执行教师枚举/求解；`jax_env.py`
执行批量训练/规划模拟；保留的 `envs/game2048.py` 是 Gym 参考环境。
`rules.py` 只提供编码、解码、D₄ 变换等工具，不再包含另一套 NumPy 移动/合并逻辑。
Numba 自行生成行规则表，测试将 Numba/JAX 与 Gym 的独立实现交叉核验。

## 方法

决策棋盘 `s` 经合法动作 `a` 确定性移动得到 afterstate `x=f(s,a)`。
网络输出标量 `Uθ(x)`，动作选择为 `argmax_a Uθ(f(s,a))`。
没有策略头、IRL 或动作偏好蒸馏。

```text
V*(s) = max_a U*(f(s,a))                              非终局
V*(s) = max_exponent(s)                               终局
U*(x) = Σ空格 e [0.9 V*(spawn(x,e,2)) + 0.1 V*(spawn(x,e,4))] / 空格数
```

每次合法移动后的落子使总和严格增加 2 或 4，因此按总和分层枚举、从大到小求值，
无需迭代收敛或保留巨大转移图。枚举从所有可能初始双块棋盘出发，覆盖全部动作和落子分支。
终局为解析边界；规范化非终局存入完整只读表。每个决策状态存四个精确 afterstate 标签。

教师使用 uint64 打包、D₄ 规范化、Numba 并行、float64 动态规划。训练标签转 float32。
标签数组与采样行号常驻 GPU；学生轨迹批量调用 CPU 哈希查表，缺失键报错。
教师使用只读 mmap，由系统缓存到内存，不进行逐样本文件读取或 Python 字典查询。

网络使用 D₄ 群卷积、残差块、群平均和中心/边/角汇聚，在数学结构上旋转镜像不变，
浮点实现允许舍入误差，不依赖数据增强；并列动作随机打破。
监督按“当前最大指数 × 空格数”分层采样，预热后半批使用学生访问状态，采集时有 5%
概率随机选择合法动作。按规范化 **afterstate** 的固定哈希留出约 5% 轨道，避免不同
父状态/旋转镜像造成标签泄漏。记录抽样留出标签的 RMSE，并用独立种子的整局验证成绩选模。

TD(λ) 是独立、无教师的在线微调。终局掩码截断跨局 bootstrap，λ-return 停止梯度。
相同验证种子选 best，允许保留微调前模型，不把退化结果强行当成提升。
当前 TD 使用网络贪心策略采集 32 步轨迹；MPPI 不参与 TD 的动作选择或参数更新。

## 历史采样规划基线（已知退化）

这里实现 **离散 MPPI-inspired 规划**，不声称满足连续控制 MPPI 的全部推导条件。
每个根动作单独分配候选；未来每步采样四方向的一个排列，共 24 种。
落子后按排列选第一个合法方向，不能通过非法动作原地停留骗过评价。
同一候选排列序列在多种落子情景复用，不能预知未来落子。
中间七步只按方向排列和合法性选择动作，不使用 NN 判断当前棋盘价值；NN 只在叶子评分。
这与网络引导的反馈规划存在明显差别，是当前退化需要诊断的重点。

候选价值为多情景平均；`softmax(value / temperature)` 加权更新各步的分类分布，
混入均匀探索。最佳候选与分布众数使用新落子样本复核，然后选择根动作。
中途终局用真实最大指数，否则叶子 afterstate 用 NN 评价。
它是有限视野、有限样本近似，**不保证优于贪心**，需要冻结模型对照。
shooting 每根动作一次采样 `K×I=192` 个计划，不迭代调整分布，但也会构造一个加权分布
众数候选；MPPI 每轮 64 个计划、共三轮。两者都保留最佳样本及分布众数，用 32 个新情景
复评。每次真实决策都从均匀分布重新开始，没有网络策略初始化或跨步热启动。

环境、随机数、规划迭代与叶子 NN 均在 JAX 编译路径，不逐节点调用 Python。
候选间共享情景随机数，目的是减小比较噪声；实际落子与规划随机数流分离。

| 参数 | 默认值 |
|---|---:|
| 群卷积通道 / 残差块 / head | 32 / 2 / 64 |
| AdamW 学习率 / weight decay / 梯度裁剪 | 3e-4 / 1e-5 / 1 |
| 决策棋盘 batch / 最多更新 / 预热 | 1024 / 100000 / 5000 |
| 验证间隔 / 局数 / patience | 5000 / 512 / 5 |
| TD 学习率 / λ / 最多更新 | 3e-5 / 0.8 / 10000 |
| TD 并行环境 / rollout | 128 / 32 |
| MPPI H / K 每根动作 / R / I | 8 / 64 / 4 / 3 |
| MPPI 温度 / 更新率 / 均匀混合 | 0.1 / 0.5 / 0.05 |
| 最终复核情景数 | 32 |
| 正式测试局数 / 起始种子 | 2000 / 100000 |

上表的“步”是参数更新步数，不是游戏移动次数。监督阶段每步 1024 个决策棋盘；
TD 每步采集 `128×32=4096` 个转移，再更新一次。规划参数默认值统一定义在 `config.py`，
CLI 与实验入口共用同一份配置；网络和监督训练默认值分别在 `ModelConfig`、`TrainConfig`。

热点规则、规范化、哈希、批采样、训练与规划均编译/批量化；验证 runner 缓存编译。
优先消除端到端瓶颈，目前没有引入自写 CUDA、Rust 或 Pallas 内核。

## 输出与展示

- `teacher3/manifest.json`：完整性门禁、状态数、全量残差和内容 hash。
- `supervised/training.json`：配置、验证曲线、留出 RMSE、恢复信息。
- CLI 输出的 `*/metrics.json`：终局指标、参数 hash、编译/执行时间、摊销每步时间；
  在逐局 CSV 写完后发布，作为本次评估完成标记。
- `*/games.csv`：逐局同种子结果；摊销时间不是单请求延迟。
- `results.json`：不同训练种子与方法的原始指标汇总。
- `replay.json`：单局动作/棋盘轨迹、终局棋盘，新生成的记录还包含规划配置；
  `replay` 不接受批量评估用的 `--games` 和 `--batch-size`。

展示可依次解释 afterstate、完整教师、D₄ 网络、采样规划，再展示真实对照结果。
报告正文不超过 5 页、宋体小四、单倍行距，分工填写真实成员贡献。
不要用 smoke 小样本成绩替代正式结果，也不要把教师最优值当作学生成绩。
