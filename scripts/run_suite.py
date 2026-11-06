"""Pre-register every task/repeat; preserve startup failures and frozen identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.model_identity import declared_identity, task_policy, policy_digest
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--suite', required=True)
p.add_argument('--split')
p.add_argument('--agent')
p.add_argument('--identity', type=Path)
p.add_argument('--repeats', type=int, default=1)
p.add_argument('--reference', action='store_true')
p.add_argument('--output', type=Path)
p.add_argument('--list', action='store_true')
p.add_argument('--allow-diagnostic', action='store_true')
p.add_argument('--grading-contract', choices=['legacy-public-v2','structural-v3'], default='legacy-public-v2')
a = p.parse_args()
registry = json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
ids = sorted(k for k,v in registry.items() if v['suite']==a.suite and (v['status']=='accepted' or a.allow_diagnostic) and (not a.split or v.get('split')==a.split))
if not ids: raise ValueError('Empty/unknown suite or split')
if a.grading_contract == 'structural-v3' and any(registry[x]['episode'] != 'layer' for x in ids):
    p.error('structural-v3 requires a layer-only suite')
if a.list:
    print('\n'.join(ids))
    sys.exit(0)
if not a.output or bool(a.agent)==bool(a.reference): p.error('Use --output and exactly one of --agent / --reference')
if a.repeats < 1: p.error('repeats must be positive')
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
freeze = hashlib.sha256((ROOT/'release-manifest.json').read_bytes()).hexdigest()
identity, declaration_error = None, None
if a.agent:
    try: identity = declared_identity(a.agent, a.identity)
    except Exception as error: declaration_error = type(error).__name__ + ': identity declaration unavailable'
if identity:
    declaration = a.output/'identity-declaration.json'
    declaration.write_text(json.dumps(identity, indent=2))
plan = dict(schema='experiment-plan-v1', experiment_id=str(uuid.uuid4()), model_identity=identity,
            declaration_error=declaration_error, suite=a.suite, release_manifest_sha256=freeze, runs=[])
for repeat in range(1, a.repeats+1):
    for tid in ids:
        name = tid if a.repeats == 1 else f'{tid}-r{repeat:02d}'
        specification = registry[tid]
        dataset = ROOT/'data'/specification['dataset']
        public_task = next(t for t in json.loads((dataset/'public/index.json').read_text())['tasks'] if t['id']==tid)
        plan['runs'].append(dict(run_id=str(uuid.uuid4()), task_id=tid, repeat_id=repeat,
            output=name, release_manifest_sha256=freeze,
            policy_sha256=policy_digest(task_policy(specification,public_task,dataset)), grading_contract=a.grading_contract,
            model_identity_sha256=identity['identity_sha256'] if identity else None))
(a.output/'experiment-plan.json').write_text(json.dumps(plan, indent=2))
for entry in plan['runs']:
    tid, name = entry['task_id'], entry['output']
    spec = registry[tid]
    cmd = [sys.executable, str(ROOT/'scripts/evaluate_public.py'), '--task', tid,
           '--output', str(a.output/name), '--run-id', entry['run_id'], '--repeat-id', str(entry['repeat_id'])]
    cmd += ['--reference'] if a.reference else ['--agent', a.agent]
    cmd += ['--grading-contract', a.grading_contract]
    if identity: cmd += ['--identity', str(declaration)]
    if a.allow_diagnostic: cmd += ['--allow-diagnostic']
    env = dict(os.environ)
    if env.get('ORIGAMI_API_LOGDIR'): env['ORIGAMI_API_LOGDIR'] = str(Path(env['ORIGAMI_API_LOGDIR'])/name)
    with (a.output/f'{name}.log').open('x') as log:
        result = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    report_path = a.output/name/'report.json'
    if not report_path.exists():
        report_path.parent.mkdir(exist_ok=True)
        data = ROOT/'data'/spec['dataset']
        task = next(t for t in json.loads((data/'public/index.json').read_text())['tasks'] if t['id']==tid)
        report_path.write_text(json.dumps(dict(task_id=tid, task_success=False, status='environment_error',
            failure_stage='subprocess_failed_without_completed_report', physical_pass=None,
            agent_kind='oracle_installation_check' if a.reference else 'external_agent:'+a.agent,
            score_eligible=bool(a.agent) and spec['status']=='accepted', suite=spec['suite'],
            split=spec.get('split', task.get('split','unspecified')), release_track=spec['track'],
            release_manifest_sha256=freeze, exit_code=result.returncode,
            run_id=entry['run_id'], repeat_id=entry['repeat_id'], model_identity=identity,
            policy_sha256=entry['policy_sha256'], grading_contract=entry['grading_contract'],
            returned_models=[], returned_model_missing_calls=0), indent=2))
    record = dict(task_id=tid, run_id=entry['run_id'], repeat_id=entry['repeat_id'],
                  exit_code=result.returncode, report=str(report_path), log=str(a.output/f'{name}.log'))
    with (a.output/'suite-attempts.jsonl').open('a') as log: log.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
