"""Offline precrease-artifact-v4 extraction, rescoring and qualification controls.

No native simulator or model API is called. Original run inputs/reports are never
changed. Synthetic controls test isolated gates, never certify physical paths.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.document_grading import reference_creases
from origami_bench.diagram.layer_grading_v15 import extract_band_creases
from origami_bench.diagram.structural_grading_v3 import compare_placement, physical_evidence_pass
from origami_bench.diagram.precrease_artifact_grading import (
    CONTRACT, VERSION, signed_precrease_check, grade_precrease_artifact,
)
SCORING_FILE = 'precrease-grading-v4-final.json'
ENVIRONMENT_STATUSES = frozenset({'environment_error', 'reference_environment_failure'})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace evidence: {path}')
    else:
        path.write_text(text)


def asset(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _failed(task_id, error):
    return dict(task_id=task_id, grading_contract=CONTRACT, grading_version=VERSION,
                task_success=False, input_valid=False, physical_pass=False,
                memory_pass=False, finished=False, environment_error=True,
                failure_category='grading_evidence_error',
                shape=dict(passed=False), evidence_error=str(error),
                error_type=type(error).__name__, mode='offline_retained_evidence_rescore')


def rescore(folder, task_id, private_spec, asset_root):
    """Grade a run against the qualified SOURCE spec, never its own candidate goal.

private_spec is one task's mapping (or a path to a task mapping/registry). The
goal and initial_goal paths resolve against asset_root, and precrease_memory is
the source-reviewed, reference-derived fixed signed specification. Malformed or
missing evidence fails closed. No catalog lookup and no task-specific branch.
    """
    folder = Path(folder)
    try:
        if isinstance(private_spec, (str, Path)):
            private_spec = read(private_spec)
        if task_id in private_spec:
            private_spec = private_spec[task_id]
        if private_spec.get('grading_contract') != CONTRACT:
            raise ValueError('Explicit precrease-artifact-v4 task contract required')
        goal_data = np.load(asset(private_spec['goal'], asset_root))
        goal, rest = goal_data['vertices'], goal_data['rest']
        initial = np.load(asset(private_spec['initial_goal'], asset_root))['vertices']
        with np.load(folder / 'trajectory.npz') as data:
            frames = data['vertices']
            if not np.array_equal(data['faces'], goal_data['faces']):
                raise ValueError('Trajectory material face topology differs from source target')
        if frames.ndim != 3 or frames.shape[1:] != goal.shape or not len(frames) or not np.isfinite(frames).all():
            raise ValueError('Nonempty finite corresponding trajectory required')
        state, events = read(folder / 'final-state.json'), read(folder / 'events.json')
        if not np.array_equal(state['faces'], goal_data['faces']):
            raise ValueError('Final material face topology differs from source target')
        if not np.allclose(frames[-1], state['vertices'], rtol=0, atol=1e-12):
            raise ValueError('Final state disagrees with retained trajectory')
        config = read(folder / 'run-config.json')
        initial_frame = config['initial_frame']
        if type(initial_frame) is not int or not 0 <= initial_frame < len(frames):
            raise ValueError('Invalid prepared-state frame')
        initial_shape = compare_placement(frames[initial_frame], initial)
        history = read(folder / 'material-history.json')
        memory = signed_precrease_check(history, private_spec['precrease_memory'], rest)
        finished = bool(events) and events[-1]['action'] == {'op': 'finish'} and events[-1]['result'].get('status') == 'accepted'
        environment = any(e.get('result', {}).get('category') == 'environment_error'
                          or e.get('result', {}).get('status') in ENVIRONMENT_STATUSES
                          or e.get('status') in ENVIRONMENT_STATUSES for e in events)
        # A candidate run report may contain a supervisor/environment failure
        # after the last complete event; never turn that into a model failure.
        recorded = read(folder / 'report.json') if (folder / 'report.json').exists() else {}
        failure = recorded.get('failure')
        environment = bool(environment or recorded.get('environment_error')
                           or recorded.get('status') in ENVIRONMENT_STATUSES
                           or (isinstance(failure, dict) and failure.get('category') == 'environment_error')
                           or recorded.get('failure_stage') == 'supervisor_timeout')
        if memory.get('input_valid') is not True:
            raise ValueError('Invalid material-history grading evidence: ' + memory.get('reason', 'unknown'))
        metrics = read(folder / 'metrics.json')
        if not isinstance(metrics, list) or not metrics:
            raise ValueError('Missing retained physical metrics')
        for metric in metrics:
            if (not isinstance(metric, dict) or type(metric.get('clear')) is not bool
                    or not isinstance(metric.get('clearance'), dict)
                    or type(metric['clearance'].get('clear')) is not bool):
                raise ValueError('Missing known collision/clearance evidence')
            for key in ['stretch', 'compression', 'grip_error']:
                value = metric.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError('Invalid physical metric: ' + key)
        physical = physical_evidence_pass(metrics)
        result = grade_precrease_artifact(frames[-1], goal, physical_pass=physical,
            memory_pass=memory['passed'], finished=finished,
            initial_state_pass=initial_shape['passed'], environment_error=environment)
        result.update(task_id=task_id, mode='offline_retained_evidence_rescore',
            score_eligible=False, initial_shape=initial_shape, memory_diagnostics=memory,
            recorded_candidate_reference_pass=recorded.get('candidate_reference_pass'),
            recorded_status=recorded.get('status'),
            evidence_provenance={name: sha(folder / name) for name in [
                'trajectory.npz', 'material-history.json', 'metrics.json', 'events.json',
                'run-config.json', 'final-state.json']},
            target_provenance=private_spec.get('grading_provenance'),
            control_kind=(read(folder / 'candidate-private.json').get('control_kind')
                          if (folder / 'candidate-private.json').exists() else None))
        return result
    except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError, np.linalg.LinAlgError) as error:
        return _failed(task_id, error)


def extract_reference(folder, task_id, output_root):
    """Generate a NEW grading spec only from an independently checked reference."""
    folder, output_root = Path(folder), Path(output_root)
    report, complete = read(folder / 'report.json'), read(folder / 'qualification-complete.json')
    if complete['report_sha256'] != sha(folder / 'report.json') or complete['run_id'] != report['run_id']:
        raise ValueError('Completed reference report hash/run identity mismatch')
    if report['task_id'] != task_id or any(report.get(k) is not True for k in [
            'candidate_reference_pass', 'annotated_plan_complete', 'reference_actions_complete',
            'finished', 'physical_pass', 'geometry_pass', 'strict_physical_evidence_pass']):
        raise ValueError('A complete physically and independently geometrically accepted reference is required')
    private = read(folder / 'candidate-private.json')
    if private.get('control_kind'):
        raise ValueError('Negative controls may not define source targets')
    config, history = read(folder / 'run-config.json'), read(folder / 'material-history.json')
    cutoff = config['initial_frame']
    prepared = []
    for row in history:
        if row['steps'] > cutoff:
            break
        prepared.append(row)
    required = reference_creases(history)
    prepared_required = reference_creases(prepared)
    added = {key: [s for s in signs if s not in prepared_required.get(key, [])]
             for key, signs in required.items()}
    added = {key: signs for key, signs in added.items() if signs}
    if not added:
        raise ValueError('Reference adds no new signed crease beyond prepared state')
    initial = np.load(folder / 'candidate-initial.npz')
    rest = initial['rest']
    bands = extract_band_creases(history, rest)
    prepared_bands = extract_band_creases(prepared, rest)
    prepared_keys = {tuple(b['hinge_ids']): b['signs'] for b in prepared_bands}
    added_band_indices = [i for i,b in enumerate(bands)
                          if any(s not in prepared_keys.get(tuple(b['hinge_ids']), []) for s in b['signs'])]
    memory = dict(hinge_stencils=history[0]['hinges'], required_signed_hinges=required, prepared_signed_hinges=prepared_required,
                  task_added_signed_hinges=added, required_band_creases=bands,
                  prepared_band_creases=prepared_bands, task_added_band_indices=added_band_indices,
                  forbid_opposite_sign=True, forbid_obvious_extra_lines=True,
                  extraction_hinge_degrees=150.0, extraction_band_degrees=140.0,
                  process_order_scored=False, moving_half_scored=False)
    audit = signed_precrease_check(history, memory, rest)
    if audit['passed'] is not True:
        raise ValueError(f'Reference does not pass its nonempty strict signature: {audit}')
    output_root.mkdir(parents=True, exist_ok=True)
    for name, origin in [('goal', 'candidate-goal.npz'), ('initial', 'candidate-initial.npz')]:
        dest = output_root / f'{task_id}-{name}.npz'
        if dest.exists():
            if dest.read_bytes() != (folder / origin).read_bytes():
                raise FileExistsError(f'Different immutable target already exists: {dest}')
        else:
            dest.write_bytes((folder / origin).read_bytes())
    private.update(grading_contract=CONTRACT, grading_version=VERSION,
        goal=f'{task_id}-goal.npz', initial_goal=f'{task_id}-initial.npz',
        precrease_memory=memory, required_signed_hinges=required, required_band_creases=bands,
        checkpoint_plan_diagnostic_only=True,
        grading_provenance=dict(reference_folder=str(folder.resolve()), reference_run_id=report['run_id'],
            report_sha256=sha(folder / 'report.json'), private_input_sha256=sha(folder / 'candidate-private.json'),
            history_sha256=sha(folder / 'material-history.json'),
            goal_sha256=sha(folder / 'candidate-goal.npz'), initial_sha256=sha(folder / 'candidate-initial.npz')))
    write_new(output_root / f'{task_id}-private.json', private)
    write_new(output_root / f'{task_id}-reference-signature.json', audit)
    return private


def synthetic_controls(folder, task_id, private_spec, asset_root):
    """Gate unit-controls based on prepared/reference states, explicitly synthetic."""
    folder = Path(folder)
    history = read(folder / 'material-history.json')
    config = read(folder / 'run-config.json')
    prepared = []
    for row in history:
        if row['steps'] > config['initial_frame']:
            break
        prepared.append(row)
    goal_file = np.load(asset(private_spec['goal'], asset_root))
    goal, rest = goal_file['vertices'], goal_file['rest']
    initial = np.load(asset(private_spec['initial_goal'], asset_root))['vertices']
    memory_spec = private_spec['precrease_memory']
    controls = []
    def check(name, h, final, expected=False, physical=True, finished=True):
        m = signed_precrease_check(h, memory_spec, rest)
        g = grade_precrease_artifact(final, goal, physical_pass=physical, memory_pass=m['passed'],
                                    finished=finished, initial_state_pass=True, environment_error=False)
        controls.append(dict(name=name, expected_success=expected, observed_success=g['task_success'],
            expectation_met=g['task_success'] is expected, synthetic=True,
            physical_gate_assumed=physical, not_a_new_physical_replay=True,
            shape=g['shape'], memory_pass=m['passed'], finished=finished,
            missing_hinge_count=len(m.get('missing_hinges', [])),
            missing_band_count=len(m.get('missing_bands', [])),
            opposite_hinge_count=len(m.get('opposite_sign_hinges', [])),
            opposite_band_count=len(m.get('opposite_sign_bands', [])),
            extra_band_count=len(m.get('unexpected_bands', []))))
    check('retained_reference', history, goal, True)
    check('finish_only_from_prepared', prepared, initial)
    check('undo_all_task_actions_to_prepared', prepared, initial)
    check('flat_final_and_repeated_static_frames_cannot_supply_memory', prepared, goal)
    # A task-added band is removed, leaving all prepared evidence intact.
    bi = memory_spec['task_added_band_indices'][0]
    ids = memory_spec['required_band_creases'][bi]['hinge_ids']
    missing = copy.deepcopy(history)
    for row in missing:
        if row['steps'] > config['initial_frame']:
            for i in ids:
                row['angles'][i] = 0.0
    check('missing_one_new_source_band', missing, goal)
    wrong = copy.deepcopy(history)
    for row in wrong:
        if row['steps'] > config['initial_frame']:
            for i in ids:
                row['angles'][i] *= -1
    check('wrong_mountain_valley_one_new_source_band', wrong, goal)
    double = copy.deepcopy(history)
    for row in copy.deepcopy(history):
        if row['steps'] > config['initial_frame']:
            row['angles'] = [-float(x) for x in row['angles']]
            double.append(row)
    check('all_required_plus_opposite_sign_saturation', double, goal)
    extra = copy.deepcopy(history)
    row = copy.deepcopy(history[-1])
    for i in range(len(row['angles'])):
        if str(i) not in memory_spec['required_signed_hinges']:
            row['angles'][i] = float(np.pi)
    extra.append(row)
    check('obvious_non_target_crease_lines', extra, goal)
    # Material-angle occurrence is order independent; this tests that property,
    # not the physical executability of a reordered action sequence.
    check('synthetic_task_evidence_order_permutation', prepared + list(reversed(history[len(prepared):])), goal, True)
    check('missing_finish', history, goal, finished=False)
    check('unknown_physics', history, goal, physical=None)
    flipped = goal.copy(); flipped[:, 0] *= -1; flipped[:, 2] *= -1
    check('wrong_material_face_orientation', history, flipped)
    return dict(task_id=task_id, grading_contract=CONTRACT, synthetic_controls=controls,
                all_expectations_met=all(c['expectation_met'] for c in controls),
                note='Synthetic gate controls are not independent reference or physical negative runs.')


def qualify_task(task_id, private_spec, asset_root, run_folders, controls):
    runs = []
    reference_folder = Path(private_spec['grading_provenance']['reference_folder'])
    expected_physics = read(reference_folder / 'physics-signature.json')
    expected_initial = np.load(asset(private_spec['initial_goal'], asset_root))['vertices']
    expected_bands = private_spec['precrease_memory']['required_band_creases']
    for folder in run_folders:
        folder = Path(folder)
        if not (folder / 'qualification-complete.json').exists():
            continue
        q = read(folder / 'report.json')
        if q.get('task_id') != task_id:
            continue
        score = rescore(folder, task_id, private_spec, asset_root)
        write_new(folder / SCORING_FILE, score)
        p = read(folder / 'candidate-private.json')
        complete = read(folder / 'qualification-complete.json')
        complete_valid = (complete.get('report_sha256') == sha(folder / 'report.json')
                          and complete.get('run_id') == q.get('run_id'))
        kind = p.get('control_kind')
        physics = read(folder / 'physics-signature.json')
        backend = read(folder / 'run-config.json')['backend']
        physics_match = physics == expected_physics and all(backend.get(k) == v for k, v in expected_physics.items())
        actual_initial = np.load(folder / 'candidate-initial.npz')['vertices']
        initial_max_diff = (float(np.max(np.abs(actual_initial - expected_initial)))
                            if actual_initial.shape == expected_initial.shape else None)
        identical_initial = initial_max_diff is not None and initial_max_diff < 1e-12
        observed_bands = score.get('memory_diagnostics', {}).get('observed_bands')
        events = read(folder / 'events.json')
        rejected = [dict(attempt_index=e.get('index'), action=e.get('action'), result=e.get('result'))
                    for e in events if e.get('result', {}).get('status') != 'accepted']
        first_rejection = rejected[0] if rejected else None
        runs.append(dict(id=folder.name, run_id=q.get('run_id'), folder=str(folder.resolve()),
            control_kind=kind, alternate=bool(p.get('alternate_provenance')),
            complete=complete_valid, physical_sequence_pass=q.get('candidate_reference_pass'),
            task_success=score['task_success'], physical_pass=score['physical_pass'],
            memory_pass=score['memory_pass'], environment_error=score['environment_error'],
            initial_shape=score.get('initial_shape'), evidence_error=score.get('evidence_error'),
            physics_signature_match=physics_match, identical_prepared_vertices=identical_initial,
            initial_vertices_max_abs_difference_m=initial_max_diff,
            observed_bands_equal_source_reference=observed_bands == expected_bands,
            first_rejection=first_rejection,
            known_interface_boundary=bool(first_rejection and first_rejection['result'].get('category') == 'capability_limit'),
            model_evaluation=False,
            memory_signature_sha256=hashlib.sha256(json.dumps(
                observed_bands, sort_keys=True).encode()).hexdigest()))
    positive = [r for r in runs if not r['control_kind'] and r['complete'] and r['physical_sequence_pass'] is True and r['task_success'] is True and not r['environment_error'] and r['physics_signature_match'] and r['identical_prepared_vertices']]
    negative = [r for r in runs if r['control_kind'] and r['complete'] and r['physical_sequence_pass'] is True and r['physical_pass'] is True and r['task_success'] is False and not r['environment_error'] and not r['evidence_error'] and r['physics_signature_match'] and r['identical_prepared_vertices']]
    # Qualifier-negative names are metadata, not grading branches. A failed
    # environment/physical sequence cannot count as an informative task control.
    negative_kinds = [str(r['control_kind']) for r in negative]
    empty = any('empty' in k or 'finish' in k or 'missing_all_new_creases' in k for k in negative_kinds)
    sign = any('sign' in k or 'valley' in k for k in negative_kinds)
    accepted = (len({r['run_id'] for r in positive}) >= 2 and any(r['alternate'] for r in positive)
                and empty and sign and controls['all_expectations_met'])
    return dict(scoring_acceptance_pass=bool(accepted), positive_runs=[r['id'] for r in positive],
        negative_runs=[r['id'] for r in negative], runs=runs,
        gates=dict(two_independent_positive_runs=len({r['run_id'] for r in positive}) >= 2,
                   alternate_moving_half_positive=any(r['alternate'] for r in positive),
                   physical_empty_negative=empty, physical_wrong_sign_negative=sign,
                   all_observed_physics_signatures_match=bool(runs) and all(r['physics_signature_match'] for r in runs),
                   all_observed_prepared_states_identical=bool(runs) and all(r['identical_prepared_vertices'] for r in runs),
                   synthetic_controls=controls['all_expectations_met']),
        reason='All scoring qualification gates passed' if accepted else 'Scoring qualification incomplete; see missing gates',
        automatic_formal_promotion=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract')
    extract.add_argument('--task', required=True); extract.add_argument('--reference', type=Path, required=True)
    extract.add_argument('--output-root', type=Path, required=True)
    score = commands.add_parser('score')
    score.add_argument('--task', required=True); score.add_argument('--run', type=Path, required=True)
    score.add_argument('--private', type=Path, required=True); score.add_argument('--asset-root', type=Path, required=True)
    score.add_argument('--output', type=Path)
    controls_parser = commands.add_parser('controls')
    controls_parser.add_argument('--task', required=True); controls_parser.add_argument('--reference', type=Path, required=True)
    controls_parser.add_argument('--private', type=Path, required=True); controls_parser.add_argument('--asset-root', type=Path, required=True)
    controls_parser.add_argument('--output', type=Path, required=True)
    qualify = commands.add_parser('qualify')
    qualify.add_argument('--tasks', nargs='+', required=True); qualify.add_argument('--runs-root', type=Path, required=True)
    qualify.add_argument('--asset-root', type=Path, required=True); qualify.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        result = extract_reference(args.reference, args.task, args.output_root)
        print(json.dumps(dict(task_id=args.task, private=str(args.output_root / f'{args.task}-private.json'),
                             required_bands=len(result['precrease_memory']['required_band_creases']))))
    elif args.command == 'score':
        result = rescore(args.run, args.task, args.private, args.asset_root)
        if args.output: write_new(args.output, result)
        else: print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == 'controls':
        result = synthetic_controls(args.reference, args.task, read(args.private), args.asset_root)
        write_new(args.output, result)
        print(json.dumps(dict(task=args.task, all_expectations_met=result['all_expectations_met'], controls=len(result['synthetic_controls']))))
    else:
        tasks = {}
        for task in args.tasks:
            private = read(args.asset_root / f'{task}-private.json')
            controls = read(args.asset_root / f'{task}-synthetic-controls.json')
            tasks[task] = qualify_task(task, private, args.asset_root, sorted(args.runs_root.iterdir()), controls)
        result = dict(grading_contract=CONTRACT, grading_version=VERSION, tasks=tasks)
        # Summary is a derived live index; preserve prior snapshots before an
        # explicit refresh. Immutable per-run grading files remain unchanged.
        if args.output.exists() and read(args.output) != result:
            backup = args.output.with_name(args.output.stem + '-' + sha(args.output)[:12] + '.json')
            write_new(backup, read(args.output))
            args.output.unlink()
        write_new(args.output, result)
        print(json.dumps({t:v['gates'] for t,v in tasks.items()}, indent=2))


if __name__ == '__main__':
    main()
