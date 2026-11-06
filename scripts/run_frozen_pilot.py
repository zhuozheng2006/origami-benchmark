"""Freeze a small catalog subset, then execute each preregistered run exactly once.

Credentials are loaded only at execution, never copied into the study. All models
use evaluate_public.py and one shared SQLite cost ledger. No request/run retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.model_identity import normalize_identity, task_policy, policy_digest
from verify_release import verify
from audit_cost_ledger import audit

ADAPTER = 'examples.openai_compatible:create'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def response_identity_evidence(api_dir):
    """Account for missing and malformed responses without hiding the failure."""
    api_dir = Path(api_dir)
    stems = {p.name.removesuffix('-request.json') for p in api_dir.glob('*-request.json')}
    stems.update(p.name.removesuffix('-response.json') for p in api_dir.glob('*-response.json'))
    names, missing, issues = set(), 0, []
    for stem in sorted(stems):
        path = api_dir / (stem + '-response.json')
        try:
            value = json.loads(path.read_text())
            name = value.get('response', {}).get('model')
            if not isinstance(name, str) or not name.strip():
                raise ValueError('missing returned model')
            names.add(name.strip())
        except (OSError, ValueError, TypeError, AttributeError):
            missing += 1
            issues.append(dict(call=stem, issue='missing_or_invalid_response_identity'))
    return dict(returned_models=sorted(names), returned_model_missing_calls=missing,
                response_identity_issues=issues)


def freeze(config_path, output, ledger):
    verify()
    cfg = json.loads(config_path.read_text())
    caps=[cfg[k] for k in ('stage_cap_cny','operating_cap_cny','cumulative_cap_cny')]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in caps) or caps!=sorted(caps):
        raise ValueError('Require positive finite stage <= operating <= cumulative caps')
    before=audit(ledger)  # read-only; never creates an empty replacement ledger
    if (before['configured_ceiling_cny'],before['operating_ceiling_cny']) != tuple(caps[2:]+[caps[1]]):
        raise ValueError('Existing shared ledger must match the authorized study ceilings')
    catalog = json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
    contract = cfg.get('grading_contract','structural-v3')
    if contract not in ('structural-v3','precrease-artifact-v4'):
        raise ValueError('Unsupported explicit pilot grading contract')
    ids = cfg['tasks']
    repeat_id = cfg.get('repeat_id', 1)
    if type(repeat_id) is not int or repeat_id < 1:
        raise ValueError('repeat_id must be a positive integer')
    if any(catalog[x].get('required_grading_contract', contract) != contract for x in ids):
        raise ValueError('Task and pilot grading contracts differ')
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Unique nonempty catalog task selection required')
    if any(catalog[x]['episode'] != 'layer' for x in ids):
        raise ValueError('This pilot requires already registered prepared-layer tasks')
    models = cfg['models']
    if len({x['slug'] for x in models}) != len(models):
        raise ValueError('Duplicate model slug')
    for model in models:
        if not model['slug'].isalnum() or model['cpu_group'] not in (4,5,6):
            raise ValueError('Invalid slug / approved CPU group')
        if model['endpoint'] != 'https://aicodelink.top/v1':
            raise ValueError('Pilot requires the explicitly authorized provider origin')
    if len({x['cpu_group'] for x in models}) != len(models):
        raise ValueError('Concurrent models must have separate CPU groups')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'cost-before.json',before)
    write_new(output/'physics-signature.json',cfg['physics_signature'])
    release = sha(ROOT/'release-manifest.json')
    plans = {}
    for model in models:
        folder = output/model['slug']; folder.mkdir()
        identity = normalize_identity(dict(requested_model=model['model'], endpoint=model['endpoint'],
            parameters=model.get('parameters',{}), output_cap=8192,
            output_parameter=model.get('output_parameter','max_tokens'), adapter_version='public-api-v3-pilot1'), ADAPTER)
        write_new(folder/'identity-declaration.json', identity)
        suites = sorted({catalog[tid]['suite'] for tid in ids})
        plan = dict(schema='experiment-plan-v1', experiment_id=str(uuid.uuid4()), model_identity=identity,
                    suite=suites[0] if len(suites) == 1 else 'mixed', release_manifest_sha256=release, runs=[])
        for tid in ids:
            spec = catalog[tid]; data = ROOT/'data'/spec['dataset']
            task = next(t for t in json.loads((data/'public/index.json').read_text())['tasks'] if t['id']==tid)
            plan['runs'].append(dict(run_id=str(uuid.uuid4()), task_id=tid, repeat_id=repeat_id, output=tid,
                score_eligible=spec['status']=='accepted',
                release_manifest_sha256=release, policy_sha256=policy_digest(task_policy(spec,task,data)),
                grading_contract=cfg.get('grading_contract','structural-v3'), model_identity_sha256=identity['identity_sha256']))
        write_new(folder/'experiment-plan.json', plan)
        plans[model['slug']] = sha(folder/'experiment-plan.json')
    value = dict(schema='frozen-pilot-v2-accounting', created=datetime.now(timezone.utc).isoformat(),
        config=cfg, release_manifest_sha256=release, plans=plans,
        ledger_path=str(ledger.resolve()), ledger_calls_sha256=before['calls_sha256'],
        ledger_accounting_sha256=before['accounting_sha256'],
        physics_signature_sha256=sha(output/'physics-signature.json'),
        notes=['One planned run per task and model; no automatic request or episode retries.',
               'Diagnostic tasks never enter formal scores.',
               'Provider model names are declarations, not independently attested identities.',
               'Historical comparisons are descriptive across versions, not a controlled causal ablation.'])
    write_new(output/'study-freeze.json', value)
    print(json.dumps(dict(frozen=str(output), runs=len(models)*len(ids), sha256=sha(output/'study-freeze.json'))))


def execute(study, credentials, ledger):
    verify()
    frozen = json.loads((study/'study-freeze.json').read_text()); cfg = frozen['config']
    if sha(ROOT/'release-manifest.json') != frozen['release_manifest_sha256']:
        raise ValueError('Frozen release changed')
    expected_accounting = frozen.get('ledger_accounting_sha256')
    if not isinstance(expected_accounting, str) or len(expected_accounting) != 64:
        raise ValueError('Legacy frozen study lacks accounting evidence binding; preserve it and create a new frozen study')
    current_budget = audit(ledger)
    if str(ledger.resolve()) != frozen['ledger_path'] or current_budget['calls_sha256'] != frozen['ledger_calls_sha256']:
        raise ValueError('Shared ledger differs from the audited frozen baseline; do not create or substitute a ledger')
    if current_budget['accounting_sha256'] != expected_accounting:
        raise ValueError('Shared accounting evidence changed after freeze; create a fresh reviewed study')
    if (current_budget['configured_ceiling_cny'], current_budget['operating_ceiling_cny']) != (cfg['cumulative_cap_cny'], cfg['operating_cap_cny']):
        raise ValueError('Shared budget ceilings changed after freeze')
    if sha(study/'physics-signature.json') != frozen['physics_signature_sha256']:
        raise ValueError('Frozen physical signature changed')
    for slug, digest in frozen['plans'].items():
        if sha(study/slug/'experiment-plan.json') != digest:
            raise ValueError('Frozen plan changed')
    # Exclusive marker prevents silent resumption/retries after interruption.
    write_new(study/'execution-start.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                                               study_sha256=sha(study/'study-freeze.json')))
    connections = json.loads(credentials.read_text())['connections']
    catalog = json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']

    def run_model(model):
        slug = model['slug']; folder = study/slug
        connection = connections[model['connection']]
        if connection['base_url'].rstrip('/') != model['endpoint']:
            raise ValueError('Credential origin differs from frozen declaration')
        env = dict(os.environ, ORIGAMI_API_BASE=model['endpoint'], ORIGAMI_API_KEY=connection['api_key'],
            ORIGAMI_MODEL=model['model'], ORIGAMI_API_PARAMETERS=json.dumps(model.get('parameters',{})),
            ORIGAMI_OUTPUT_PARAMETER=model.get('output_parameter','max_tokens'),
            ORIGAMI_INPUT_RATE=str(model['upper_cny_per_million'][0]),
            ORIGAMI_OUTPUT_RATE=str(model['upper_cny_per_million'][1]),
            ORIGAMI_COST_LEDGER=str(ledger), ORIGAMI_COST_LIMIT=str(cfg['cumulative_cap_cny']),
            ORIGAMI_OPERATING_LIMIT=str(cfg['operating_cap_cny']), ORIGAMI_COST_STAGE=cfg['cost_stage'],
            ORIGAMI_STAGE_LIMIT=str(cfg['stage_cap_cny']),
            ORIGAMI_BACKEND_MODE='ssh', ORIGAMI_SSH_HOST=cfg['remote_host'], ORIGAMI_REMOTE_ROOT=cfg['remote_root'],
            ORIGAMI_REMOTE_PYTHON=cfg['remote_root']+'/.venv-public/bin/python',
            ORIGAMI_PHYSICS_SIGNATURE_FILE=str(study/'physics-signature.json'),
            ORIGAMI_CPU_GROUP=str(model['cpu_group']))
        plan = json.loads((folder/'experiment-plan.json').read_text())
        for entry in plan['runs']:
            tid=entry['task_id']; spec=catalog[tid]; out=folder/tid
            if out.exists():
                raise FileExistsError('Existing run cannot be retried: '+tid)
            env['ORIGAMI_API_LOGDIR']=str(study/'api-calls'/slug/tid)
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_public.py'),'--task',tid,'--output',str(out),
                 '--agent',ADAPTER,'--identity',str(folder/'identity-declaration.json'),
                 '--run-id',entry['run_id'],'--repeat-id',str(entry['repeat_id']),'--grading-contract',entry['grading_contract']]
            if spec['status']!='accepted': cmd.append('--allow-diagnostic')
            timed_out=False
            with (folder/(tid+'.log')).open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=proc.wait(timeout=cfg['episode_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    timed_out=True
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: code=proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
            report_path=out/'report.json'
            complete=False
            if report_path.is_file() and (out/'evaluation-complete.json').is_file():
                try:
                    completed=json.loads((out/'evaluation-complete.json').read_text())
                    result=json.loads(report_path.read_text())
                    complete=(code==0 and completed.get('run_id')==entry['run_id']
                        and completed.get('grading_contract')==entry['grading_contract']
                        and completed.get('report_sha256')==sha(report_path)
                        and result.get('grading_contract')==entry['grading_contract']
                        and result.get('grading_candidate') is True and (out/'candidate-grading.json').is_file())
                except (ValueError,TypeError):pass
            if not complete:
                out.mkdir(exist_ok=True)
                if report_path.exists():report_path.rename(out/'report-incomplete.json')
                api_dir=Path(env['ORIGAMI_API_LOGDIR'])
                response_evidence = response_identity_evidence(api_dir)
                write_new(report_path,dict(task_id=tid,task_success=False,status='environment_error',
                    failure_stage='supervisor_timeout' if timed_out else 'subprocess_or_grading_incomplete',
                    physical_pass=None,agent_kind='external_agent:'+ADAPTER,score_eligible=spec['status']=='accepted',
                    suite=spec['suite'],split=spec.get('split','unspecified'),release_track=spec['track'],
                    release_manifest_sha256=frozen['release_manifest_sha256'],exit_code=code,
                    run_id=entry['run_id'],repeat_id=entry['repeat_id'],model_identity=plan['model_identity'],
                    policy_sha256=entry['policy_sha256'],grading_contract=entry['grading_contract'],
                    **response_evidence))
            record=dict(task_id=tid,model=model['model'],exit_code=code,run_id=entry['run_id'],
                        report=str(report_path),status=json.loads(report_path.read_text()).get('status'))
            with (folder/'suite-attempts.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    with ThreadPoolExecutor(max_workers=len(cfg['models'])) as pool:
        futures=[pool.submit(run_model,model) for model in cfg['models']]
        for future in futures: future.result()
    write_new(study/'execution-complete.json',dict(utc=datetime.now(timezone.utc).isoformat()))
    write_new(study/'cost-after.json',audit(ledger))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('--config',type=Path,required=True);f.add_argument('--output',type=Path,required=True);f.add_argument('--ledger',type=Path,required=True)
    e=commands.add_parser('execute');e.add_argument('--study',type=Path,required=True);e.add_argument('--credentials',type=Path,required=True);e.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze': freeze(a.config.resolve(),a.output.resolve(),a.ledger.resolve())
    else: execute(a.study.resolve(),a.credentials.resolve(),a.ledger.resolve())
