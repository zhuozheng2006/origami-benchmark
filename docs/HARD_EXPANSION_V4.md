# Real-source hard candidate expansion v4

This track adds **directed precrease-history tasks** from independent author diagrams. It does not claim completed animals/stars, strict reproduction of every printed action order, or demonstrated low frontier-model scores. The standard controller and `layer_policy_v15` prompt are unchanged. Sources are locally downloaded and hash-verified; source PDF/page images are not redistributed.

## Task scope

| ID | Original source and assigned section | Reference length after preparation | What is harder |
|---|---|---:|---|
| H102 | Gerwin Sturm, *Box Pleating Guide II*, page 1, 8×8 example | 28 actions + finish | 14 directed grid lines, alternating mountain/valley signs, turnover, real unfolding, 13 new lines after preparation |
| H103 | David Mitchell, *4Star*, Method 1 steps 8–15 | 28 actions + finish | 2:1 rectangular coordinates, two source pages, quarter/eighth landmarks, diagonal corner creases, turnover, 13 new lines after preparation |

The seven-candidate inventory is in `audit/hard-v4-candidates.json`. Only rows that pass complete physical references **and** scoring controls enter the runnable accepted catalog. Pending/unsupported candidates never enter the model-score denominator. H101 (Humphrey single-sheet component) failed the continuous 100µm clearance check at printed step 47; its failure remains in the evidence. Kite has annotation/partial-unfolding ambiguity; Pureland Cat/Penguin and NASA Dart require additional mesh, scope or operation qualification.

## Explicit scoring contract

Use `precrease-artifact-v4`. This is a separate contract from `structural-v3`; scores must not be pooled across them.

A pass requires all of the following:

- The same labelled material mesh and a valid prepared initial state.
- The final unfolded shape and paper-face orientation, with proper planar yaw and XYZ placement ignored. No reflection, scale or arbitrary 3D rotation fit.
- All required directed material hinges and crease bands formed on the **retained** physical branch. Preparation alone is insufficient; each task has nonempty new requirements.
- No prohibited opposite-direction coverage on single-sign target creases, and no obvious non-target full crease lines. Incidental bending is not claimed to be exact final CP equality.
- Explicit finish and complete finite contact/strain/precision evidence. Environment failures and malformed evidence cannot become successful model runs.

The contract deliberately accepts equivalent moving halves and action orders. It measures retained directed bending history, **not** the final plastic rest-angle field or strict printed action order. H103 step 13 has a printed order; this result-oriented track deliberately does not score that order. Undo cannot restore a spent attempt or make an undone crease count.

Hinge coverage/opposite-sign gates use 140 degrees; band coverage uses 130 degrees; obvious extra-band extraction uses 140 degrees. Physical settings remain 100µm thickness/continuous clearance, <1% stretch/compression, and the existing controlled endpoint and shape gates. All thresholds are versioned; no task-ID branches exist in the controller or scorer.

## Known interface boundary

The frozen controller limits a single motion to 600 integration frames. For a 2:1 rectangle, moving a large complementary half can exceed that cap even when the smaller-side reference is executable. `alternate-H103-02` preserves this `capability_limit` failure. A subsequent positive control reverses line endpoints and uses complementary halves only where the unchanged generic compiler supports them. This is not proof that every mathematically equivalent motion is supported. Keep capability rejections separate from model misinterpretation.

## Reproduce locally

From the repository root, using the tested benchmark environment:

```bash
python3.12 -m venv .venv-sources-v4
.venv-sources-v4/bin/python -m pip install -r requirements-sources-v4.txt
.venv-sources-v4/bin/python scripts/prepare_precrease_sources.py
python scripts/verify_release.py
python scripts/run_suite.py --suite precrease-hard --list --grading-contract precrease-artifact-v4
python scripts/run_suite.py --suite precrease-hard --reference --grading-contract precrease-artifact-v4 --output outputs/precrease-reference
```

For an agent, replace `--reference` with `--agent your_module:create --identity identity.json`. The agent receives the real diagram pages, current paper views, complete public attempt history and remaining budget. It does not receive reference actions, goal vertices, required hinge signs or score diagnostics during execution. The common `AccountedEpisode` enforces **40 total attempts**, including failed actions, undo, page inspection and finish. Use `summarize_public.py` with the generated experiment plan; preserve every failed/environment run.

`requirements-sources-v4.txt` pins only the source rendering dependencies. Native simulation installation and backend configuration remain in the main README. The new source preparation command enforces PDF bytes and RGB pixel hashes; do not silently substitute other pages/renderings.

## Evidence and limits

`data/precrease-hard-v4/qualification-summary.json` records independent positive references, real empty/sign negative controls, physical signatures and prepared-state correspondence. `examples/reference-evidence-v4/` retains their numeric evidence and failed candidates; `hard-v4/` provides source-page inspection, actual 3D trajectories and retained mountain/valley history overlays. Overlay samples are for display only; scoring uses full evidence.

Historical candidate1 and final candidate2 scoring files remain distinct. Synthetic score controls are explicitly labelled; they are not claimed as new physical replays. Reference checks establish scoped solvability and some scoring sensitivity, not frontier-model difficulty or real-finger robustness. No new paid model calls are required for this expansion's qualification.
