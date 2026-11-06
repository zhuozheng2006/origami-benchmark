"""Build a read-only old/new pilot review. Never performs inference or simulation.

The historical primary comparison always uses suffix -01. Later historical
attempts are shown separately, never substituted as best-of successes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rescore_structural_v3 import rescore
from origami_bench.diagram.model_identity import validate_identity

MODELS = {'gpt': 'gpt-6-astra', 'claude': 'claude-opus-5-5', 'gemini': 'gemini-4-argon'}
TASKS = ['O01', 'O02', 'O03', 'O04']


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def public_attempts(folder):
    ledger = read(folder / 'attempt-ledger.json', {})
    result, known_cost, unknown_cost = [], 0.0, 0
    for entry in ledger.get('attempts', []):
        response, outcome = entry.get('response', {}), entry.get('result', {})
        action_result = outcome.get('result', {})
        cost = response.get('cost_upper_cny')
        if isinstance(cost, (float, int)) and not isinstance(cost, bool):
            known_cost += cost
        else:
            unknown_cost += 1
        result.append(dict(
            attempt=entry.get('attempt_id'), action=response.get('action', outcome.get('action')),
            result={k: action_result.get(k) for k in ('status', 'category', 'detail', 'rollback_verified')},
            before=outcome.get('before'), after=outcome.get('after'),
            input_tokens=response.get('input_tokens'), output_tokens=response.get('output_tokens'),
            seconds=response.get('seconds'), wall_seconds=outcome.get('wall_seconds'),
            returned_model=response.get('returned_model'), api_call_id=response.get('api_call_id'),
            cost_upper_cny=cost,
        ))
    if not result:
        for i, event in enumerate(read(folder / 'events.json', [])):
            result.append(dict(attempt=i, action=event.get('action'), result=event.get('result', {}),
                               before=event.get('before'), after=event.get('after')))
        unknown_cost = len(result)
    return result, ledger.get('budget'), dict(known_upper_cny=known_cost,
        unknown_cost_attempts=unknown_cost, actual_provider_billing=False)


def run_record(folder, task, slug, phase, output, frames):
    recorded = read(folder / 'report.json')
    key = f'{phase}-{slug}-{task}-{folder.name}'
    value = dict(id=key, task=task, model_slug=slug, expected_model=MODELS[slug], phase=phase,
                 diagnostic=task == 'O04', available=recorded is not None,
                 source_directory=str(folder.relative_to(SOURCE)) if folder.is_relative_to(SOURCE) else str(folder),
                 source_report_sha256=sha(folder / 'report.json'),
                 original_status=recorded.get('status') if recorded else 'missing_result',
                 original_success=recorded.get('task_success') if recorded else None,
                 original_grading=recorded.get('grading_version') if recorded else None)
    if recorded is None:
        startup = read(folder / 'startup-or-runtime-error.json')
        if startup:
            value['original_status'] = startup.get('status', 'environment_error')
            value['startup_error'] = {k: startup.get(k) for k in ('error_type', 'detail')}
        value['identity_status'] = 'missing_report'
        return value
    experiment = read(folder / 'experiment-identity.json', {})
    identity = recorded.get('model_identity', experiment.get('model_identity'))
    valid = validate_identity(identity)
    value.update(requested_model=identity.get('requested_model') if valid else recorded.get('model'),
                 provider=identity.get('provider') if valid else recorded.get('provider'),
                 parameters=identity.get('parameters') if valid else recorded.get('model_settings'),
                 identity_status='declared_hashed' if valid else 'legacy_unverified_identity',
                 model_identity_sha256=identity.get('identity_sha256') if valid else None,
                 run_id=recorded.get('run_id'), repeat_id=recorded.get('repeat_id'),
                 policy_sha256=recorded.get('policy_sha256'),
                 release_manifest_sha256=recorded.get('release_manifest_sha256'),
                 grading_contract=recorded.get('grading_contract'),
                 score_eligible=recorded.get('score_eligible'), seconds=recorded.get('seconds'))
    attempts, budget, cost = public_attempts(folder)
    value.update(attempts=attempts, budget=budget, cost=cost)
    returned = set(recorded.get('returned_models', []))
    for path in sorted(folder.glob('model-turn-*.json')):
        turn = read(path, {})
        if isinstance(turn.get('returned_model'), str): returned.add(turn['returned_model'])
    value['returned_models'] = sorted(returned)
    value['returned_name_mismatch'] = bool(returned and any(x != value.get('requested_model') for x in returned))
    value['returned_name_missing'] = not returned
    value['rejections'] = sum(a.get('result', {}).get('status') == 'rejected' for a in attempts)
    value['undos'] = sum(isinstance(a.get('action'), dict) and a['action'].get('op') == 'undo' for a in attempts)
    value['environment_error'] = recorded.get('status') == 'environment_error'
    needed = ['trajectory.npz', 'final-state.json', 'metrics.json', 'material-history.json', 'run-config.json', 'events.json']
    missing = [name for name in needed if not (folder / name).is_file()]
    if missing:
        value['candidate_rescore'] = dict(status='unavailable', reason='missing_evidence', missing=missing)
    else:
        try:
            candidate = rescore(folder, task)
            value['candidate_rescore'] = dict(status='computed', result=candidate)
        except Exception as error:
            value['candidate_rescore'] = dict(status='unavailable', reason=type(error).__name__, detail=str(error))
    if (folder / 'trajectory.npz').is_file() and (folder / 'final-state.json').is_file():
        try:
            positions = np.load(folder / 'trajectory.npz')['vertices']
            state = read(folder / 'final-state.json')
            if positions.ndim != 3 or not np.isfinite(positions).all() or not len(positions):
                raise ValueError('Invalid trajectory')
            selected = np.unique(np.linspace(0, len(positions)-1, min(frames, len(positions)), dtype=int))
            payload = dict(units='m', vertices=np.round(positions[selected], 7).tolist(),
                           frame_numbers=selected.tolist(), total_frames=len(positions), faces=state['faces'])
            target = output / 'trajectories' / (key + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, separators=(',', ':')))
            value['trajectory'] = str(target.relative_to(output))
            value['trajectory_original_sha256'] = sha(folder / 'trajectory.npz')
            value['trajectory_sampled_frames'] = len(selected)
        except Exception as error:
            value['trajectory_error'] = type(error).__name__
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=SOURCE/'outputs/layer-paths-v17/runs')
    parser.add_argument('--new-root', type=Path, default=SOURCE/'outputs/public-v3-pilot-20261107')
    parser.add_argument('--output', type=Path, default=ROOT/'study-v3')
    parser.add_argument('--frames', type=int, default=72)
    args = parser.parse_args()
    if not 2 <= args.frames <= 240: parser.error('frames must be within 2..240')
    args.output.mkdir(parents=True, exist_ok=True)
    records, cells, extras = [], [], []
    for slug, model in MODELS.items():
        for task in TASKS:
            old = args.old_root / f'model-{model}-{task}-01'
            new = args.new_root / slug / task
            old_record = run_record(old, task, slug, 'old_primary', args.output, args.frames)
            new_record = run_record(new, task, slug, 'new', args.output, args.frames)
            records.extend([old_record, new_record])
            cells.append(dict(model_slug=slug, task=task, diagnostic=task=='O04', old=old_record['id'], new=new_record['id']))
            for folder in sorted(args.old_root.glob(f'model-{model}-{task}-*')):
                if folder == old or not (folder/'report.json').is_file(): continue
                record = run_record(folder, task, slug, 'old_additional', args.output, args.frames)
                records.append(record); extras.append(record['id'])
    summary = []
    for slug, model in MODELS.items():
        current = [r for r in records if r['phase']=='new' and r['model_slug']==slug and not r['diagnostic']]
        summary.append(dict(model_slug=slug, requested_model=model, planned_formal=3,
            reports_available=sum(r['available'] for r in current),
            passed=sum(r.get('original_success') is True for r in current),
            environment_errors=sum(r['original_status']=='environment_error' for r in current),
            missing=sum(not r['available'] for r in current),
            identity_complete=all(r.get('identity_status')=='declared_hashed' for r in current)))
    # Only expected safe aggregate keys are read, not credentials or call bodies.
    cost = {}
    for label in ['before','after']:
        source = read(args.new_root/f'cost-{label}.json')
        if source is not None:
            cost[label] = {k:v for k,v in source.items() if k in (
                'calls', 'configured_ceiling_cny', 'operating_ceiling_cny',
                'estimated_charges_cny', 'unresolved_reserves_cny', 'accounted_upper_cny')}
    if 'before' in cost and 'after' in cost:
        cost['delta'] = {
            k: cost['after'][k] - cost['before'][k]
            for k in ('calls', 'estimated_charges_cny', 'unresolved_reserves_cny', 'accounted_upper_cny')
            if isinstance(cost['after'].get(k), (float, int)) and isinstance(cost['before'].get(k), (float, int))
        }
    plans = {slug: dict(path=f'{slug}/experiment-plan.json',sha256=sha(args.new_root/slug/'experiment-plan.json'),
                       available=(args.new_root/slug/'experiment-plan.json').is_file()) for slug in MODELS}
    payload = dict(schema='pilot-comparison-v1', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        models=MODELS, tasks=TASKS, cells=cells, runs=records, additional_history=extras,
        new_formal_summary=summary, study_freeze_sha256=sha(args.new_root/'study-freeze.json'), plans=plans, costs=cost,
        historical_selection='Fixed first run -01 for all twelve cells; later runs preserved separately, never best-of.',
        comparability='Descriptive historical comparison only. Old weak identity, old retry/fallback transport and new no-retry transport differ; not a paired causal model-improvement estimate.',
        candidate_contract='structure-and-diagram-motion-v3-candidate1',
        evidence_boundary='Historical new-rule scores are offline evidence regrading, not new inference or simulation. New reported scores come from new physical episodes; missing results are never passes.')
    (args.output/'report.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(cells=len(cells),old_primary=sum(r['phase']=='old_primary' for r in records),
        old_additional=len(extras),new_reports=sum(r['phase']=='new' and r['available'] for r in records),
        rescore_available=sum(r.get('candidate_rescore',{}).get('status')=='computed' for r in records),
        output=str(args.output/'report.json')),ensure_ascii=False))


if __name__=='__main__':main()
