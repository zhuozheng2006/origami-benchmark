"""Replay candidate reference actions, preserving incomplete physical evidence.

No model adapter or paid transport is imported. Outputs remain private candidate
qualification evidence; this command never promotes a task into a scored suite.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.attempt_ledger import atomic_json
from origami_bench.diagram.layer_episode_v16 import LayerEpisodeV16
from origami_bench.diagram.layer_protocol import signature
from origami_bench.diagram.layer_reference import IdealTrace
from origami_bench.diagram.public_remote_backend import configured_backend
from origami_bench.diagram.skills import SkillUnsupported
from origami_bench.diagram.structural_grading_v3 import physical_evidence_pass

VERSION = 'candidate-reference-qualification-v1'
SOURCE_FILES = ['scripts/qualify_hard_candidates.py'] + [
    'origami_bench/diagram/' + name + '.py' for name in (
        'layer_episode_v16', 'layer_controller_v16', 'layer_reference', 'layer_protocol',
        'public_remote_backend', 'local_backend', 'structural_grading_v3', 'hard_episode',
        'document_episode', 'layer_grading_v15')]
SETTINGS = dict(backend_variant='codim-ipc-directional', barrier_distance=.0002,
                compression_barrier=True, grip_solve_tolerance=.0002,
                grip_solve_metric='rms', newton_tolerance=1e-4)
FIXTURE_MODE = 'guided_directional_10000'
RMS_LIMIT_MM, MAX_LIMIT_MM = 1.0, 3.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate(public_path, private_path, task_id, asset_root=ROOT):
    public = json.loads(Path(public_path).read_text())
    rows = public.get('tasks', []) if isinstance(public, dict) else public
    matches = [r for r in rows if r.get('id') == task_id]
    if len(matches) != 1:
        raise ValueError('Exactly one matching public task is required')
    task = copy.deepcopy(matches[0])
    private = copy.deepcopy(json.loads(Path(private_path).read_text())[task_id])
    for key in ('id', 'title', 'instruction', 'page_count'):
        if key not in task:
            raise ValueError('Missing public task field: ' + key)
    if not re.fullmatch('[a-z0-9_]+', private.get('mesh', '')):
        raise ValueError('A public mesh name is required')
    actions = private.get('reference_actions')
    if not isinstance(actions, list) or not actions:
        raise ValueError('A nonempty private reference action list is required')
    if any(not isinstance(a, dict) or not isinstance(a.get('op'), str) for a in actions):
        raise ValueError('Reference actions must be objects with an operation')
    if any(a['op'] in ('finish', 'undo', 'view_page') for a in actions):
        raise ValueError('Reference must contain forward paper operations; finish is appended separately')
    for key in ('initialization', 'preparation'):
        if not isinstance(private.get(key, []), list):
            raise ValueError(key + ' must be a list')
    root = Path(asset_root).resolve()
    def resolve(value):
        p = Path(value)
        p = p.resolve() if p.is_absolute() else (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return str(p)
    private['pages'] = [resolve(p) for p in private.get('pages', [])]
    if type(task['page_count']) is not int or task['page_count'] < 1 or len(private['pages']) != task['page_count']:
        raise ValueError('Every declared source page must be present')
    for key in ('goal', 'initial_goal', 'checkpoints'):
        if private.get(key):
            private[key] = resolve(private[key])
    return task, private


def geometry_check(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 3 or not len(actual):
        raise ValueError('Geometry arrays must have the same nonempty (N,3) shape')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite reference geometry')
    delta = np.linalg.norm(actual - expected, axis=1) * 1000
    return dict(rms_mm=float(np.sqrt(np.mean(delta ** 2))), max_mm=float(delta.max()))


def geometry_pass(checks):
    return bool(checks) and all(
        np.isfinite(c['rms_mm']) and np.isfinite(c['max_mm']) and
        c['rms_mm'] < RMS_LIMIT_MM and c['max_mm'] < MAX_LIMIT_MM for c in checks)


def qualify(task, private, output, *, expected_physics, max_decisions=100,
            provenance=None, episode_factory=LayerEpisodeV16, backend_factory=configured_backend):
    """Injectable factories support offline tests; defaults always use public code."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing attempts are immutable; choose another run')
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError('max_decisions must be a positive integer including finish')
    if not isinstance(expected_physics, dict) or not expected_physics:
        raise ValueError('An explicit frozen physical signature is required')
    plan = dict(version=VERSION, run_id=str(uuid.uuid4()), task_id=task['id'],
                score_eligible=False, automatic_promotion=False, paid_model_calls=0,
                expected_reference_actions=len(private['reference_actions']),
                max_decisions_including_finish=max_decisions, settings=SETTINGS,
                fixture_mode=FIXTURE_MODE, expected_physics=expected_physics,
                geometry_thresholds=dict(rms_mm=RMS_LIMIT_MM, max_mm=MAX_LIMIT_MM),
                provenance=provenance or {}, code_sha256={name: sha(ROOT / name) for name in SOURCE_FILES}, source_pages=[dict(path=p, sha256=sha(p)) for p in private.get('pages', [])])
    episode = None
    backends, saved_folders, attempts, checkpoints, frames = [], [], [], [], []
    ideal, initial_check, preparation_checks = None, None, []
    actual_signature, base_report, failure = None, None, None
    metrics_ok = False
    all_actions = False
    stage = 'startup'

    def make_backend(mesh):
        nonlocal actual_signature
        b = backend_factory(mesh, **SETTINGS)
        backends.append(b)
        actual_signature = signature(b.configuration)
        # Must run before constructor initialization/preparation changes the paper.
        if actual_signature != expected_physics:
            raise RuntimeError('Frozen physical signature mismatch before preparation')
        original_save = b.save
        def capture_save():
            folder = Path(original_save())
            saved_folders.append(folder)
            return folder
        b.save = capture_save
        return b

    def persist():
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'qualification-plan.json', plan)
        atomic_json(output / 'reference-attempts.json', attempts)
        if actual_signature is not None:
            atomic_json(output / 'physics-signature.json', actual_signature)
        atomic_json(output / 'candidate-public.json', task)
        atomic_json(output / 'candidate-private.json', private)
        snapshot = output / 'qualification-source'
        if not snapshot.exists():
            snapshot.mkdir()
            for name in SOURCE_FILES:
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)

    def attempt(action, action_index):
        before = episode.backend.state()
        row = dict(attempt_id=len(attempts) + 1, reference_action_index=action_index,
                   action=copy.deepcopy(action), before_hash=before.get('hash'),
                   before_frame=before.get('steps'), status='started')
        attempts.append(row)
        stepdir = output / 'steps'
        stepdir.mkdir(exist_ok=True)
        atomic_json(stepdir / f'{row["attempt_id"]:03}-before.json', before)
        persist()  # A killed/timeout attempt cannot silently disappear.
        started = time.monotonic()
        try:
            result = episode.step(action)
            row.update(status='returned', result=copy.deepcopy(result))
            state = episode.backend.state()
            row.update(after_hash=state.get('hash'), after_frame=state.get('steps'))
            atomic_json(stepdir / f'{row["attempt_id"]:03}-after.json', state)
            return result, state
        except Exception as exc:
            row.update(status='environment_error', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            persist()
            print(json.dumps(dict(task=task['id'], attempt=row['attempt_id'], action=action.get('op'),
                                  result=row.get('result', row.get('error_type')), seconds=row['seconds'])), flush=True)

    try:
        episode = episode_factory(task, private, output, make_backend, max_decisions,
                                  fixture_mode=FIXTURE_MODE)
        persist()
        faces = np.asarray(episode.initial_state['faces'], int)
        rest = np.asarray(episode.backend.material, float)
        ideal = IdealTrace(rest, faces)
        for record in episode.initialization:
            ideal.apply(record, None)
        initial_check = geometry_check(episode.initial_state['vertices'], ideal.vertices)
        np.savez_compressed(output / 'candidate-initial.npz', vertices=episode.initial_state['vertices'],
                            rest=rest, faces=faces)
        np.savez_compressed(output / 'annotated-initial.npz', vertices=ideal.vertices, rest=rest, faces=faces)
        stage = 'reference_actions'
        for index, action in enumerate(private['reference_actions']):
            if len(attempts) >= max_decisions:
                break
            record_count = len(episode.controller.records)
            result, state = attempt(action, index)
            if result.get('status') != 'accepted':
                break
            if len(episode.controller.records) != record_count + 1:
                raise RuntimeError('Accepted forward operation lacks exactly one controller record')
            ideal.apply(episode.controller.records[-1], state['vertices'])
            checkpoints.append(np.asarray(state['vertices'], float).copy())
            frames.append(state['steps'])
            atomic_json(output / 'geometric-checks.json', ideal.checks)
        all_actions = len(checkpoints) == len(private['reference_actions'])
        if all_actions and len(attempts) < max_decisions:
            stage = 'finish'
            attempt({'op': 'finish'}, None)
        stage = 'save_and_verify'
        base_report = episode.finish('private_candidate_reference')
        # Keep the inherited old report intact, but never publish it as a formal score.
        atomic_json(output / 'report-episode.json', base_report)
        metrics_ok = physical_evidence_pass(json.loads((output / 'metrics.json').read_text()))
        with np.load(output / 'trajectory.npz', allow_pickle=False) as trace:
            prep = IdealTrace(rest, faces)
            for record in episode.initialization:
                prep.apply(record, trace['vertices'][record['frame']])
            preparation_checks = prep.checks
        if len(preparation_checks) != len(episode.initialization):
            raise RuntimeError('Preparation geometry evidence is incomplete')
    except Exception as exc:
        category = 'environment_error'
        if stage == 'startup' and (output / 'preparation-error.json').exists() and not any(getattr(b, 'poisoned', False) for b in backends):
            if isinstance(exc, SkillUnsupported):
                category = 'preparation_capability_rejection'
            elif 'ActionRejected' in str(exc) or type(exc).__name__ == 'PhysicsRejected':
                category = 'preparation_physical_rejection'
        failure = dict(stage=stage, category=category, error_type=type(exc).__name__, detail=str(exc))
        if (output / 'report.json').is_file() and base_report is None:
            shutil.copy2(output / 'report.json', output / 'report-incomplete.json')
        if category.startswith('preparation_'):
            records_path = output / 'preparation-records.json'
            records = json.loads(records_path.read_text()) if records_path.is_file() else []
            planned = private.get('initialization', []) + private.get('preparation', [])
            atomic_json(output / 'preparation-attempts.json', dict(accepted_records=records,
                        attempted_action_index=len(records), action=planned[len(records)] if len(records)<len(planned) else None,
                        result=failure, rollback_verified=None))
    finally:
        # Constructor failure may have allocated a backend without returning an episode.
        if episode is not None:
            try:
                episode.close()
            except Exception as exc:
                failure = failure or dict(stage='close', category='environment_error', error_type=type(exc).__name__, detail=str(exc))
        for backend in backends:
            try:
                backend.close()
            except Exception as exc:
                failure = failure or dict(stage='close', category='environment_error', error_type=type(exc).__name__, detail=str(exc))
        persist()
        for saved in saved_folders:
            if (saved / 'rejected-candidates').is_dir() and not (output / 'rejected-candidates').exists():
                shutil.copytree(saved / 'rejected-candidates', output / 'rejected-candidates')

    environment = bool((failure and failure.get('category') == 'environment_error') or any(
        a.get('status') == 'environment_error' or a.get('result', {}).get('category') == 'environment_error'
        for a in attempts))
    finished = bool(episode is not None and episode.finished)
    complete = bool(all_actions and finished and not environment)
    checks = ([initial_check] if initial_check is not None else []) + preparation_checks + (ideal.checks if ideal else [])
    geometry_ok = bool(complete and geometry_pass(checks))
    physical_ok = bool(metrics_ok and base_report and base_report.get('physical_pass') is True and not environment)
    memory_ok = bool(base_report and all(base_report.get(k, True) is True for k in
                                        ('memory_pass', 'unsigned_memory_pass', 'band_memory_pass')))
    qualified = complete and geometry_ok and physical_ok and memory_ok
    report = dict(version=VERSION, task_id=task['id'], run_id=plan['run_id'],
                  agent_kind='private_candidate_reference', score_eligible=False, task_success=False,
                  candidate_reference_pass=bool(qualified), annotated_plan_complete=complete,
                  reference_actions_complete=all_actions, finished=finished,
                  status=('candidate_reference_qualified' if qualified else 'reference_environment_failure'
                          if environment else 'candidate_preparation_rejected' if failure else 'candidate_reference_incomplete' if not complete else 'candidate_reference_failed_checks'),
                  attempts=len(attempts), accepted_reference_actions=len(checkpoints),
                  expected_reference_actions=len(private['reference_actions']),
                  remaining_decisions=max(0, max_decisions-len(attempts)),
                  physical_pass=physical_ok, retained_prefix_physical_pass=bool(base_report and base_report.get('physical_pass')),
                  geometry_pass=geometry_ok, memory_pass=memory_ok, initial_check=initial_check,
                  preparation_checks=preparation_checks, geometric_checks=ideal.checks if ideal else [],
                  environment_error=failure if environment else None, failure=failure,
                  strict_physical_evidence_pass=metrics_ok, fixture_mode=FIXTURE_MODE,
                  qualification_note='Candidate only. A complete replay and checks do not certify source annotation, difficulty, repeatability, or formal eligibility.',
                  ideal_trace_scope='Independent vertex transform calculation using recorded controller selections and axes; does not independently validate diagram interpretation or chosen layer selection.')
    if ideal is not None:
        suffix = '' if complete else '-prefix'
        np.savez_compressed(output / ('candidate-checkpoints' + suffix + '.npz'),
                            vertices=np.asarray(checkpoints, float).reshape((-1, len(ideal.vertices), 3)),
                            frames=np.asarray(frames, int))
        np.savez_compressed(output / ('annotated-goal' + suffix + '.npz'), vertices=ideal.vertices)
        if complete:
            np.savez_compressed(output / 'candidate-goal.npz', vertices=checkpoints[-1], rest=rest, faces=faces)
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'qualification-complete.json', dict(run_id=plan['run_id'], status=report['status'],
                                                           report_sha256=sha(output / 'report.json')))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, help='Directory containing public/index.json and private/tasks.json')
    parser.add_argument('--public-json', type=Path)
    parser.add_argument('--private-json', type=Path)
    parser.add_argument('--asset-root', type=Path, default=ROOT, help='Base for relative source/goal paths; default public checkout')
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True, help='Unique simple run name; old runs are never overwritten')
    parser.add_argument('--output', required=True, type=Path, help='Study directory; writes OUTPUT/runs/RUN')
    parser.add_argument('--group', type=int, choices=(4, 5, 6), default=4)
    parser.add_argument('--max-decisions', type=int, default=100, help='Includes final finish attempt')
    parser.add_argument('--physics-signature', type=Path, default=os.environ.get('ORIGAMI_PHYSICS_SIGNATURE_FILE'))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run):
        parser.error('run must be a simple directory name')
    if args.max_decisions < 1:
        parser.error('max-decisions must be positive')
    if args.physics_signature is None:
        parser.error('--physics-signature or ORIGAMI_PHYSICS_SIGNATURE_FILE is required')
    if args.dataset and (args.public_json or args.private_json):
        parser.error('Use either --dataset or both explicit JSON paths')
    public = args.dataset / 'public/index.json' if args.dataset else args.public_json
    private = args.dataset / 'private/tasks.json' if args.dataset else args.private_json
    if public is None or private is None:
        parser.error('Provide --dataset or both --public-json and --private-json')
    task, data = load_candidate(public, private, args.task, args.asset_root)
    expected_path = Path(args.physics_signature).resolve()
    expected = json.loads(expected_path.read_text())
    os.environ['ORIGAMI_CPU_GROUP'] = str(args.group)
    os.environ['ORIGAMI_PHYSICS_SIGNATURE_FILE'] = str(expected_path)
    provenance = dict(public_json=str(public.resolve()), public_sha256=sha(public),
                      private_json=str(private.resolve()), private_sha256=sha(private),
                      script_sha256=sha(__file__), physics_signature_sha256=sha(expected_path),
                      backend_mode=os.environ.get('ORIGAMI_BACKEND_MODE', 'local'), cpu_group=args.group)
    report = qualify(task, data, args.output / 'runs' / args.run, expected_physics=expected,
                     max_decisions=args.max_decisions, provenance=provenance)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['candidate_reference_pass'] else 3 if report['status'] == 'reference_environment_failure' else 2


if __name__ == '__main__':
    sys.exit(main())
