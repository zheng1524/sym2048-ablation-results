# Local runtime and recovery

The original 4090 endpoints were probed with the already configured key in batch
mode; both closed at SSH handshake. No password was requested or used. Both the
GitHub connector and existing local GitHub credentials report pull=true, push=false.
No push to main or an alternate repository was attempted.

Local hardware is an RTX 3060 Laptop 6GB with driver 570.133.07, system CUDA 12.8,
Ryzen 7 6800H, and 14GiB RAM. The clean uv environment uses managed CPython 3.12.16
and the original frozen project dependencies. The repository's CUDA 13 optional
extra requires a newer driver ([JAX installation documentation](https://docs.jax.dev/en/latest/installation.html)).
Without changing pyproject.toml or uv.lock, the following runtime-only additions
enable the **same JAX version** on this machine:

```bash
uv sync --frozen --managed-python
uv pip install --python .venv/bin/python --no-deps \
  jax-cuda12-plugin==0.12.0.1 jax-cuda12-pjrt==0.12.0.1 \
  nvidia-cudnn-cu12==9.10.2.21
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
uv run --no-sync python -c 'import jax; print(jax.devices())'
```

The complete environment is recorded in evidence/runtime-freeze.txt. All subsequent
GPU commands use these environment settings. Specifying JAX_PLATFORMS prevents silent
CPU fallback. These settings do not modify other environments or system drivers.
Deterministic GPU operations are requested so that gradient scatter reductions and
optimizer state reproduce exactly across interrupted and continuous runs. Runtime
versions and numerical flags are part of the resume identity.
The first CUDA probe loaded older shared cuDNN and failed convolution compilation;
the isolated cuDNN installation fixes that failure. Logs retain both attempts.

On 2026-10-31 at 22:54 Asia/Shanghai, A/seed0 reached update 5000 but failed
in its first full heldout-label validation: the default BFC GPU allocator could
not allocate a contiguous 2.54GiB buffer. The earlier single-update preflight had
not covered this validation path. The process exited and the complete atomic
recovery snapshot was still at step 0, so those 5000 updates must be replayed.
The original logs and interrupted marker are retained; no successful validation
or completed A checkpoint is claimed for this attempt.

Recovery on 2026-11-01 uses the `platform` allocator above, which releases unused
GPU allocations instead of retaining fragmented BFC pool chunks. This is an
operational memory-management change: source, protocol, model, update kernel,
learning rates, random streams and sample budgets are unchanged. The additional
allocator setting is explicitly audited in
`evidence/allocator-recovery-20261101.json`, alongside hashes of all three completed
B models and their unchanged validation histories. It does not bypass or relax
the existing source/protocol/checkpoint resume checks.

Before recovery, the same frozen source passed a full-size training update with
resident teacher arrays, 512-game validation, all eight 1024-decision heldout
validation batches, and 128×256 student collection plus exact teacher lookup.
Evidence is in `evidence/full-validation-memory-probe.{json,txt}`. The completed
B checkpoints are preserved; their 10000 updates are not rerun. The failed A
attempt remains in `stage-timings.json` for total wall-clock accounting.

The newly built teacher's verification passed before training. Its `layers/` and
`solved/` duplicate intermediates were archived to `intermediates.tar.gz` with gzip
level 1, then `tar --compare` verified every byte before the duplicate directories
were removed. Root keys/q/legal/index files and verified manifest remain intact.
The archive and its SHA256 are recorded in evidence/teacher-archive.json. To repeat
full verification later, extract the archive into a **new** teacher copy with the
root arrays, then run the original `teacher verify` command there.
During a second disk exhaustion, the byte-verified archive was moved temporarily
to `/dev/shm/sym2048-ablations-teacher-intermediates.tar.gz`. This is RAM-backed
storage; the active teacher does not depend on that archive, and its durable root
arrays and manifest remain in the experiment directory. The archive should be
copied back to durable storage once space is available.
After disk cleanup increased the available space, the archive was restored to the
durable teacher directory, its SHA256 checked again, and the RAM copy removed.

The disk was initially 97% full. A first full check attempt encountered disk
exhaustion, and smoke was subsequently resumed after another disk exhaustion;
both interruptions are preserved as failed evidence. Only this experiment's duplicate
teacher intermediates and incomplete download temporaries were removed. Other
projects, assets and processes were left untouched. Successful repeated checks
are recorded separately; failed logs are not represented as successful tests.

The formal background worker executes the CLI and refreshes JOB_STATUS.json and
REPORT.md every 30 seconds. It copies only validation snapshots, best checkpoints
and completed CSVs. Completion performs the strict summary verification, records
the actual results, creates a local delivery commit, and refreshes the git bundle.
Any process failure produces an explicitly interrupted report and preserves recovery
payloads. It never performs a network write. Use the exact same output and source
commit to resume. Code changes require a fresh namespace.

```bash
uv run --no-sync python scripts/execute_ablation_job.py \
  --teacher "$PWD/artifacts/ablations-20261031-teacher" \
  --output "$PWD/runs/ablations-20261031-full"
```

Best-only models in checkpoints/ support inference without teacher files. Full
optimizer/RNG/pool/board recovery remains under runs/ and is intentionally excluded
from Git, as is the teacher. Smoke deliverables use an explicit smoke namespace.
