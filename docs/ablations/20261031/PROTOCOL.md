# Distillation and D₄ ablation, 2026-10-31

Baseline: `6eb6db44f0a3d70c15b92b5200404bb2b033d686` (main at checkout;
no differences from the supplied review baseline). Implementation is additive on
`experiment/distillation-d4-ablation-20261031`; no historical source or artifact is rewritten.
The user's supplied execution document is the experimental specification.

The frozen machine-readable budgets are [configs/formal.json](configs/formal.json)
and [configs/smoke.json](configs/smoke.json). Smoke uses the **full** architectures
(D₄ 32/2/64 and plain 83/2/64), with small training/game counts. It cannot support
method comparisons. Formal seeds are 0, 1, 2; validation games 400000–400511;
test games 500000–501999, greedy only. Max exponent ≥10 returns the analytic 10
boundary in both models. Objective, moves, spawning and terminal rewards are unchanged.

| Group | Supervision | TD initialization | Architecture |
|---|---|---|---|
| d4_distill_td (A) | 100000 updates | supervised validation best | original D₄ |
| d4_scratch_td (B) | none, no teacher reads | seed's original random weights | original D₄ |
| plain_distill_td (C) | 100000 updates | supervised validation best | ordinary CNN |

TD is the same 10000-update λ=0.8 algorithm in every core group. Its step-zero
validation is eligible for best. Supervision always completes its fixed budget,
with exactly 20 scheduled validation events and no early stopping. Best is selected
strictly by validation mean_max_exponent, retaining the earlier model on ties.
A and B share exactly `split(PRNGKey(seed))[1]` initialization, recorded by SHA256.
A incurs additional teacher and supervised costs: A−B matches online TD budgets,
not total computation. A−C matches training algorithms/budgets and parameter counts,
not FLOPs, elapsed time, or the adaptively collected student states.

PlainValueNetwork has one-hot input, ordinary SAME convolutions (stem and two
two-convolution residual blocks), matching LayerNorm/SiLU locations, ordered
3×3 flatten, Dense(64)/SiLU/Dense(1). It has no group convolution, symmetry pooling,
canonicalized inference, or test-time averaging. Actual parameter counts are audited,
and width is fixed by structure, never test performance.

For A and C, teacher rows produce four scalar afterstates and their legal masks.
The canonical-afterstate hash holdout is determined **before** independent uniform
D₄ augmentation of each scalar input. Targets stay indexed to their afterstates;
illegal actions and holdout orbits are excluded from loss. The original stratified
sampling, 5% student exploration, collection cadence and half-student-pool rule
are retained. Heldout RMSE uses a separate random stream and augmented orientations.
Initialization, sampling, augmentation, collection, heldout diagnostics, TD tile
spawns, TD tie breaks and TD resets have independent named streams (`state.STREAMS`).
Game evaluation uses independent seed-addressed environment/tie streams, as before.

The common TD kernel adapts `training.finetune`, reusing `lambda_returns`, optimizer,
`greedy_action` and JAX rules. Supervision adapts `training.train`, reusing original
teacher arrays, decoding and student collector. New evaluators adapt
`evaluation.make_game_runner`/`evaluate` with an explicit architecture factory.
Original APIs, network classes and checkpoint formats remain untouched; the new
loader treats missing architecture as legacy D₄. Inference imports no teacher.

Stage requests bind the complete protocol, model dimensions/family, group, seed,
initial hash, upstream hash, source fingerprint and code commit; supervision also
binds teacher content hash. One atomic recovery payload contains the optimizer,
parameters, all RNGs, current TD boards, supervision pool/count/position, validation
history, best parameters/step/score and timing. JSON and best/latest directories
are reviewable views reconstructed from that payload. Unexpected exceptions roll
back to the last atomic snapshot; explicitly requested chunks save completed updates.
Nonempty unrecognized directories and mismatching resumes fail. Completion markers
are emitted only after the prescribed update budget; evaluations publish metrics
only after complete CSVs. Reuse verifies model identity, architecture, algorithm,
source, seeds, CSV hash, tile correspondence and recomputed metrics.

Each training stage is a separate process. All core TD models freeze before test
evaluation. If mean strict-B validation best is below mean A validation best, a
separately labeled diagnostic chooses among {3e-5, 1e-4, 3e-4} using only seed 3,
2000 updates per rate, and the same validation set; ties favor the lower rate.
It then restarts original seeds 0,1,2 from their original random weights for 100000
updates at the frozen rate. Core B remains intact. No algorithm, reward or exploration
changes are allowed. Incomplete diagnostics cannot establish scratch sufficiency.

The summary reports A−B and A−C per training seed, their mean and sample SD. Its
paired 95% normal intervals first average differences over fitted models within
each shared game seed, then take an interval over game seeds, using the existing
comparison helpers. These intervals describe game randomness conditional on the
fitted models, and do not account for training randomness. Three seeds are not
6000 independent training repetitions. Equal game seeds do not imply equal trajectories.

Run from this checkout in its uv environment:

```bash
uv run python -m game2048.ablations run --teacher artifacts/ablations-20261031-teacher \
  --output runs/ablations-20261031-smoke --seeds 0 1 2 --smoke
uv run python -m game2048.ablations run --teacher artifacts/ablations-20261031-teacher \
  --output runs/ablations-20261031-full --seeds 0 1 2
```

Re-running the exact command resumes matching stages. A B-only run uses
`--groups d4_scratch_td` and never opens its unused teacher argument. A standalone
`practical` command can complete an interrupted diagnostic using the formal protocol
file. Use `--help` for architecture-aware evaluation, audit and summary commands.
CPU and CUDA 12 runtime details and actual execution scope belong in REPORT.md.

Upload scope: new code, tests, frozen configs, compact curves/CSV/checkpoints/report
and check evidence only. Teacher, environment, caches and recovery payloads stay
local under ignored new namespaces. No edits to `.gitignore`, main, existing models,
teacher files or history; no PR merge. Push permission is read-checked before upload;
when unavailable, deliver a local commit and bundle, explicitly marked not uploaded.
