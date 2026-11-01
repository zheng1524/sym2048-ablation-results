原仓库要求的四项检查均已执行：

- uv run ruff check .：通过。
- uv run ruff format --check .：55 files already formatted。
- uv run ty check：通过。
- uv run pytest -q：38 passed in 82.59s，CPU 后端。

本次使用原训练工作区已安装的 uv 环境（--project /home/liu/other/sym2048-ablation --no-sync），并将 PYTHONPATH 设置为独立发布工作区。已确认 game2048.__file__ 指向独立发布工作区。未同步依赖、未改动锁文件、未使用正在训练的 GPU。

模型数据另行核验完整预算、20/11 次验证、512 个固定验证种子、best 选择、有限参数及 SHA256；结果见 audit.json。
