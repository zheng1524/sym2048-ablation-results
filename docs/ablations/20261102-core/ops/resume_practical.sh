#!/usr/bin/env bash
set -euo pipefail
cd /home/liu/other/sym2048-ablation
export PATH="/home/liu/.local/bin:/usr/local/cuda/bin:$PATH"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
unset JAX_DEFAULT_MATMUL_PRECISION
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/home/liu/other/sym2048-ablation/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
/home/liu/.local/bin/uv run --no-sync python - <<'PY'
import json
from pathlib import Path
from game2048.ablations.provenance import source_identity
r = Path('/home/liu/other/sym2048-ablation/runs/ablations-20261031-full')
request = json.loads((r / 'd4_scratch_practical/seed0/td/request.json').read_text())
assert source_identity() == request['source'], 'Frozen source/runtime mismatch'
assert (r / 'd4_scratch_practical/seed0/td/recovery.msgpack').is_file()
print('Frozen source/runtime verified; local recovery is ready.')
PY
if [ "${1:-}" = "--check" ]; then exit 0; fi
exec /home/liu/.local/bin/uv run --no-sync python -m game2048.ablations practical /home/liu/other/sym2048-ablation/runs/ablations-20261031-full --protocol-file /home/liu/other/sym2048-ablation/runs/ablations-20261031-full/protocol.json
