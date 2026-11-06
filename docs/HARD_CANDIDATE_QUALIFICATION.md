# Candidate reference qualification

`scripts/qualify_hard_candidates.py` replays private reference actions through the public `LayerEpisodeV16` and `public_remote_backend.configured_backend`. It does not import a model adapter or make paid requests. It never edits the catalog or promotes candidates into scored tasks.

Supply `public/index.json` (`tasks` list) and `private/tasks.json` (mapping by task ID). Each private task needs `mesh`, all source `pages`, a nonempty `reference_actions` list, and optional `initialization` / `preparation`. Actions must be forward paper operations; the runner appends `finish`. Relative source/goal paths resolve against `--asset-root` (default: public checkout). Pages must exist. Existing run directories are immutable.

Example from the public checkout, with the already configured public SSH transport:

```sh
ORIGAMI_BACKEND_MODE=ssh \
ORIGAMI_SSH_HOST=4090B \
ORIGAMI_REMOTE_ROOT=/data/ckpt/zhengzhuo/origami-public-v2-final-check \
ORIGAMI_REMOTE_PYTHON=/data/ckpt/zhengzhuo/miniconda3/envs/origami-codim/bin/python \
ORIGAMI_TOTAL_TIMEOUT=5400 ORIGAMI_IDLE_TIMEOUT=600 \
python scripts/qualify_hard_candidates.py \
  --dataset /absolute/path/to/candidate-dataset \
  --asset-root /absolute/path/to/source-asset-root \
  --task H102 --run reference-H102-01 \
  --output /absolute/path/to/candidate-study --group 4 \
  --physics-signature /Users/zane/Documents/ChatGPT/折纸benchmark/outputs/public-v3-pilot-20261107/physics-signature.json
```

Remote root above is an example: it must contain the exact matching current public physics sources and chosen mesh. SSH transport verifies their hashes before starting CPU work. Only CPU groups 4/5/6 are accepted; group is passed through `ORIGAMI_CPU_GROUP`. Local mode remains available through the same public backend but does not provide the SSH CPU-group lease. No automatic synchronization or source repair occurs.

Alternatively use `--public-json FILE --private-json FILE` instead of `--dataset`. A frozen `--physics-signature` (or `ORIGAMI_PHYSICS_SIGNATURE_FILE`) is mandatory and checked before any initialization/preparation motion. Physical settings and fixture mode are fixed to the current public protocol; the command has no switches for weakening paper thickness, strain, precision, contact or material. Default `--max-decisions 100` includes the terminal finish attempt.

Outputs are written to `OUTPUT/runs/RUN`:

- `qualification-plan.json`, private/public input snapshots, physics signature, code hashes and `qualification-source/`: provenance including exact scripts, source image hashes and attempt budget.
- `reference-attempts.json` and `steps/*-before/after.json`: every attempted action is persisted before execution, with raw result, rejection, rollback claim, timing and state. A killed run retains a `started` attempt, not a successful action.
- Original episode `events.json`, trajectory, material history, metrics and failure evidence remain. `report-episode.json` preserves the inherited report; incomplete finish output is backed up as `report-incomplete.json`.
- `candidate-initial.npz`, `annotated-initial.npz`; on complete paths `candidate-goal.npz`, `annotated-goal.npz`, `candidate-checkpoints.npz` (actual action endpoints and frame IDs). Incomplete runs use `*-prefix.npz` and never create a full candidate goal. These artifacts are evidence, not new formal task assets.
- `geometric-checks.json` / report checks use `IdealTrace` for a separate vertex-transform calculation. This uses recorded selected faces and axes: it does **not** independently establish that the annotation understood the source diagram correctly. Initial and preparation states are checked too. Limits remain RMS <1 mm and max <3 mm.
- Candidate reference pass additionally requires all planned actions, explicit finish, no environment failure, strict `structural_grading_v3.physical_evidence_pass(metrics)`, inherited material-memory requirements and all geometry checks. It does not infer physical validity from a screenshot or old boolean alone.
- `report.json` always has `score_eligible=false` and `task_success=false`; `candidate_reference_pass` reports the qualification outcome. This is neither an AI score nor automatic formal acceptance. Repetition, source annotation, checkpoint distinctness and task difficulty still need separate review.
- Preparation `SkillUnsupported` and `ActionRejected` / `PhysicsRejected` failures are separated from lost transport when identifiable, with `preparation-attempts.json`; rollback is left unknown where not explicitly verified. No continuation or retry occurs after a rejection.

Exit codes: 0 candidate reference checks passed; 2 incomplete/rejected/failed checks; 3 environment failure. `qualification-complete.json` means the qualification runner finalized a report, **not** that the candidate passed. A missing marker indicates interrupted reporting.

Offline checks (no native worker or network):

```sh
python -m unittest discover -s tests -p test_hard_candidate_qualification.py -v
python -m py_compile scripts/qualify_hard_candidates.py
python scripts/qualify_hard_candidates.py --help
```
