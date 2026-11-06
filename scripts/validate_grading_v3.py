"""Reproducible positive/negative controls and archived layer-evidence comparison.

No model inference, solver, secret configuration, or edits to original reports.
Synthetic geometry controls are not simulations or model benchmark scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rescore_structural_v3 import rescore
from origami_bench.diagram.structural_grading_v3 import VERSION, compare_placement, ordered_motion_checkpoints, combine_success, physical_evidence_pass
from origami_bench.diagram.layer_path_grading import ordered_checkpoints as old_path


def validate():
    base = np.array([[0, 0, 0], [.10, 0, .01], [0, .06, .015], [.04, .03, .04], [.08, .06, -.02]])
    first = base.copy(); first[1, 2] += .025
    second = first.copy(); second[3, 2] += .04
    targets = np.array([first, second])
    trajectory = np.array([base, (base+first)/2, first, second])
    controls = []

    def path_control(name, v, cps, expected):
        result = ordered_motion_checkpoints(v, cps, base)
        controls.append(dict(name=name, scope='synthetic_path_component', expected=expected,
                             actual=result['passed'], verified=result['passed'] == expected,
                             old_actual=old_path(v, cps)['passed'], details=result))

    path_control('reference_path', trajectory, targets, True)
    moved = trajectory.copy(); moved[:, :, :2] = moved[:, :, :2] @ np.array([[0, 1], [-1, 0]])
    moved += [.2, -.1, .006]; moved[:, :, 2] += np.arange(len(moved))[:, None]*.002
    path_control('global_yaw_xyz_and_height_drift', moved, targets, True)
    path_control('extra_samples_valid_decomposition', np.repeat(trajectory, 3, axis=0), targets, True)
    path_control('no_motion_static_padding', np.repeat(base[None], 100, axis=0), targets, False)
    path_control('skip_required_state', np.array([base, second]), targets, False)
    path_control('reverse_state_order', np.array([base, second, first]), targets, False)
    repeated = np.repeat(first[None], 4, axis=0)
    path_control('four_duplicate_checkpoints', trajectory, repeated, False)
    nearby = repeated.copy(); nearby[:, 1, 2] += np.arange(4)*.0001
    path_control('near_checkpoints_with_padded_frames', np.repeat(trajectory, 100, axis=0), nearby, False)
    rotated = first.copy(); rotated[:, :2] = rotated[:, :2] @ np.array([[0, 1], [-1, 0]])
    path_control('required_paper_rotation_present', np.array([base, first, rotated]), np.array([first, rotated]), True)
    path_control('required_paper_rotation_omitted', np.array([base, first, first]), np.array([first, rotated]), False)

    for name, matrix in [('mirror_material', np.diag([-1, 1, 1])), ('wrong_mountain_valley', np.diag([1, 1, -1])),
                         ('wrong_turnover', np.diag([-1, 1, -1])), ('scaled_paper', np.eye(3)*1.2)]:
        result = compare_placement(second @ matrix, second)
        controls.append(dict(name=name, scope='synthetic_shape_component', expected=False,
                             actual=result['passed'], verified=not result['passed'], details=result))

    for name, flags, expected in [('all_evidence_present', (True, True, True, False), True),
                                 ('missing_finish', (False, True, True, False), False),
                                 ('unknown_physics', (True, None, True, False), False),
                                 ('failed_physics', (True, False, True, False), False),
                                 ('missing_signed_crease_memory', (True, True, False, False), False),
                                 ('environment_fault', (True, True, True, True), False)]:
        actual = combine_success(dict(passed=True), dict(passed=True, contract_valid=True), *flags)
        controls.append(dict(name=name, scope='evidence_conjunction', expected=expected, actual=actual, verified=actual == expected))

    for name, args in [('empty_path', ([], targets, base)), ('empty_targets', (trajectory, [], base)),
                       ('nan_geometry', (trajectory*np.nan, targets, base))]:
        rejected = False
        try:
            ordered_motion_checkpoints(*args)
        except ValueError:
            rejected = True
        controls.append(dict(name=name, scope='input_validation', expected='ValueError',
                             actual='ValueError' if rejected else 'accepted', verified=rejected))

    for name, value in [('nan_compression', float('nan')), ('missing_compression', None), ('strain_at_limit', .01)]:
        actual = physical_evidence_pass([dict(clear=True, clearance=dict(clear=True), stretch=.001, compression=value, grip_error=.0001)])
        controls.append(dict(name=name, scope='physical_evidence_validation', expected=False, actual=actual, verified=not actual))

    cases = [('O01', 'examples/reference-evidence-v2/followup/final-O01'),
             ('O02', 'examples/reference-evidence-v2/sweep/O02'), ('O03', 'examples/reference-evidence-v2/sweep/O03'),
             ('L01', 'examples/reference-evidence-v2/followup/L01-longpath')]
    cases += [('O04', str(p.relative_to(ROOT))) for p in sorted((ROOT/'examples/grading-evidence-v3').iterdir()) if p.is_dir()]
    rows, evidence_hashes = [], {}
    for task, relative in cases:
        folder = ROOT/relative
        for name in ['trajectory.npz', 'final-state.json', 'metrics.json', 'material-history.json', 'events.json', 'run-config.json', 'report.json']:
            evidence_hashes[relative+'/'+name] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        result = rescore(folder, task)
        result.update(run=folder.name, evidence=relative)
        rows.append(result)
    assert all(x['verified'] for x in controls)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in evidence_hashes.items())
    source_files = ['origami_bench/diagram/structural_grading_v3.py', 'scripts/rescore_structural_v3.py', 'scripts/validate_grading_v3.py']
    return dict(grading_version=VERSION, controls=controls, control_count=len(controls),
                controls_verified=sum(x['verified'] for x in controls), archived_runs=rows,
                original_evidence_unchanged=True, evidence_sha256=evidence_hashes,
                grading_source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
                model_calls=0, physics_runs=0, official_scores_changed=False,
                caveat='Candidate contract evaluation of archived evidence; synthetic controls are not real physics or model runs. Height alignment changes only scoring coordinates, never contact/strain records.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep prior validation; choose a new output path')
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(controls=result['control_count'], verified=result['controls_verified'], archived_runs=len(result['archived_runs'])), ensure_ascii=False))
