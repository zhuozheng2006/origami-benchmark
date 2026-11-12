# Packet loading experiment (PR #1)

The accepted change exposes a configurable, default-off loading experiment in
`LayerControllerV16(..., packet_angle_step_deg=0.388)` and
`compile_stack_fold_v15(..., packet_angle_step_deg=0.388)`.
The option is a researcher-side Python API; existing CLI/evaluation entrypoints
continue to use the legacy schedule. It is not part of the model action schema.
No special 0.388 constant is embedded in production code. Finite steps in
(0, 0.5] are accepted subject to the existing 600-frame action budget.

## Meaning and boundaries

The backend advances each requested frame by 1/60 second. Requesting 464 instead
of 360 frames lengthens a 180-degree motion from 6 to 7.733 seconds. This changes
loading speed, contact evolution and potentially plastic history; it is not a
fixed-duration integration convergence fix. The 0.8 mm arc-length bound and
600-frame rejection remain enforced. Budget rejection is not model failure.

The experiment applies when at least two selected planar material components
reach the free crease band. This is a geometry proxy, not a certified physical
layer count. It does not depend on whether a selector is `all` or an explicit
packet. Existing fixture selection rules are unchanged. A reverse motion uses
the original fold's effective step, including partial unfolds. Unrelated rigid
paper rotations and v14/v15 controllers retain their legacy behavior.

Every v16 fold/unfold records a versioned `loading_schedule`: configured and
effective angular increments, geometric component count (for new folds),
requested frames, nominal duration, actual successful integrator segments and,
when supplied by the backend, actual elapsed time. Actual segment counts can
exceed requested frames because the backend subdivides rejected attempts.
The v16 controller version is now `layer-boundary-v16-dev2`.

## Evidence correction

The published O03 sweep's seventh operation requested 360 frames with one hit
layer. Its 464 successful integration segments were produced by adaptive
subdivision over 6 seconds. PR #1 instead changed the following packet operation.
Equal segment counts do not demonstrate reproduction of that trajectory.
The original PR's reported O01/O02/O03 reruns have not been independently
reproduced here. Offline tests use archived geometry and a fake recording
backend; they do not claim physical accuracy or new task success.

## Scoring and frozen results

This change does not alter band-memory scoring, worker history sampling,
thresholds, or frozen H102/H103 results. Operation-boundary sampling can miss
internal peaks; finer loading alone does not repair that observation contract.
The previously reported old trajectory's per-frame 112.1-degree peak was also
below the 130-degree threshold. A new scoring contract must separately validate
finite input, rollback-aware per-substep history, and weighted simultaneous
band angles before temporal maxima (not sums of per-hinge maxima).

## Required before default adoption

Predeclare independent geometries, two/four/eight selected layers, both senses,
22.5/90/180-degree motions, scaled sheets, and equivalent selectors. Separate
fixed-duration time-step refinement from fixed-time-step loading-speed sweeps.
Report convergence of geometry, band peaks and threshold margins, strain,
continuous clearance, grip error and plastic dissipation, along with duration,
requested/actual steps, solver configuration and budget exhaustion. Use a
separate experimental evidence directory and bind its manifest/controller/backend
versions. Never overwrite archived trajectories or reinterpret frozen scores.

Validation for this revision: nine scheduling regressions, including O03 archived
selection equivalence and reverse scheduling; full offline suite. Native JGSL/IPCTK
physical convergence and paid model evaluations were not run.
