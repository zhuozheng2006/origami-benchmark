"""Identity-aware summaries with explicit missing/duplicate evidence accounting."""
import collections
import json
from pathlib import Path
from .model_identity import validate_identity


def summarize_runs(paths, plans=()):
    groups = collections.defaultdict(list)
    problems, excluded = [], 0
    expected = {}
    for path in plans:
        plan = json.loads(Path(path).read_text())
        for entry in plan['runs']:
            if entry['run_id'] in expected:
                problems.append(dict(reason='duplicate_planned_run', run_id=entry['run_id']))
            expected[entry['run_id']] = entry
    seen_paths, seen_ids, observed, valid = set(), set(), set(), set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_dir():
            continue  # shell globs include suite log files
        if path in seen_paths:
            problems.append(dict(reason='duplicate_directory', path=str(path)))
            continue
        seen_paths.add(path)
        report_path = path/'report.json'
        if not report_path.is_file():
            problems.append(dict(reason='missing_report', path=str(path)))
            continue
        try:
            r = json.loads(report_path.read_text())
        except (ValueError, OSError):
            problems.append(dict(reason='unreadable_report', path=str(path)))
            continue
        if not r.get('score_eligible'):
            excluded += 1
            continue
        run_id = r.get('run_id')
        if not run_id:
            problems.append(dict(reason='missing_run_identity', path=str(path)))
            continue
        if run_id in seen_ids:
            problems.append(dict(reason='duplicate_run_id', run_id=run_id))
            continue
        seen_ids.add(run_id)
        observed.add(run_id)
        identity = r.get('model_identity')
        if not validate_identity(identity):
            problems.append(dict(reason='missing_or_invalid_model_identity', run_id=run_id))
            continue
        required = ('task_id','suite','release_track','split','release_manifest_sha256','policy_sha256','repeat_id')
        if any(r.get(k) is None for k in required):
            problems.append(dict(reason='missing_protocol_identity', run_id=run_id))
            continue
        if expected:
            e = expected.get(run_id)
            if e is None:
                problems.append(dict(reason='unplanned_run', run_id=run_id))
                continue
            if any(r.get(k) != e.get(k) for k in ('task_id','repeat_id','release_manifest_sha256','policy_sha256','grading_contract')) or identity['identity_sha256'] != e.get('model_identity_sha256'):
                problems.append(dict(reason='planned_identity_mismatch', run_id=run_id))
                continue
        valid.add(run_id)
        key = (r['suite'], r['release_track'], r['split'], identity['identity_sha256'],
               r['release_manifest_sha256'], r['policy_sha256'], r.get('grading_contract','legacy-public-v2'))
        groups[key].append(r)
    result = []
    for (suite, track, split, identity_hash, freeze, policy, grading), rows in groups.items():
        by = collections.defaultdict(list)
        for r in rows:
            by[r['task_id']].append(r)
        returned = sorted({m for r in rows for m in r.get('returned_models', [])})
        if len(returned) > 1:
            problems.append(dict(reason='inconsistent_returned_models', identity_sha256=identity_hash, returned_models=returned))
        n = len(rows)
        env = sum(r.get('status') == 'environment_error' for r in rows)
        passed = sum(bool(r.get('task_success')) for r in rows)
        result.append(dict(suite=suite, track=track, split=split, model_identity=rows[0]['model_identity'],
            release_manifest_sha256=freeze, policy_sha256=policy, grading_contract=grading, returned_models=list(returned),
            returned_model_mismatch=any(m != rows[0]['model_identity']['requested_model'] for m in returned),
            returned_model_missing_calls=sum(r.get('returned_model_missing_calls', 0) for r in rows),
            runs=n, tasks=len(by), repeats={k:len(v) for k,v in by.items()}, passed=passed,
            environment_errors=env, environment_error_rate=env/n, end_to_end_success_rate=passed/n,
            conditional_success_rate=passed/(n-env) if n != env else None,
            macro_task_success_rate=sum(sum(bool(r.get('task_success')) for r in rs)/len(rs) for rs in by.values())/len(by)))
    missing = [expected[k] for k in expected.keys()-observed]
    unscorable = [expected[k] for k in expected.keys() & observed - valid]
    return dict(groups=result, excluded_oracle_diagnostic_or_replay=excluded, issues=problems,
                coverage=dict(planned=len(expected), observed_planned=len(expected.keys() & observed),
                              scorable_planned=len(expected.keys() & valid), missing=missing, unscorable=unscorable),
                comparison_ready=bool(expected) and not missing and not unscorable and not problems,
                warning='No plan means descriptive only. Identity is provider-reported, not authenticated weights. Never pool protocols or historical versions.')
