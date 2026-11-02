"""Run a frozen suite sequentially; every task has a separate immutable directory."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--split');p.add_argument('--agent');p.add_argument('--reference',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--list',action='store_true');p.add_argument('--allow-diagnostic',action='store_true');a=p.parse_args()
registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks'];ids=sorted(k for k,v in registry.items() if v['suite']==a.suite and (v['status']=='accepted' or a.allow_diagnostic) and (not a.split or v.get('split')==a.split))
if not ids:raise ValueError('Empty/unknown suite or split')
if a.list:print('\n'.join(ids));sys.exit(0)
if not a.output or bool(a.agent)==bool(a.reference):p.error('Use --output and exactly one of --agent / --reference')
if a.suite=='ultra' and not a.allow_diagnostic:p.error('Ultra uses legacy scoring and requires --allow-diagnostic')
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
freeze=hashlib.sha256((ROOT/'release-manifest.json').read_bytes()).hexdigest()
for tid in ids:
 cmd=[sys.executable,str(ROOT/'scripts/evaluate_public.py'),'--task',tid,'--output',str(a.output/tid)]
 cmd+=['--reference'] if a.reference else ['--agent',a.agent]
 if a.allow_diagnostic:cmd+=['--allow-diagnostic']
 import os
 env=dict(os.environ)
 if env.get('ORIGAMI_API_LOGDIR'):env['ORIGAMI_API_LOGDIR']=str(Path(env['ORIGAMI_API_LOGDIR'])/tid)
 with (a.output/f'{tid}.log').open('x') as log:
  result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 report_path=a.output/tid/'report.json'
 if not report_path.exists():
  report_path.parent.mkdir(exist_ok=True)
  spec=registry[tid]
  report_path.write_text(json.dumps(dict(task_id=tid,task_success=False,status='environment_error',failure_stage='subprocess_failed_without_completed_report',physical_pass=None,agent_kind='oracle_installation_check' if a.reference else 'external_agent:'+a.agent,score_eligible=bool(a.agent) and spec['status']=='accepted',suite=spec['suite'],split=spec.get('split','unspecified'),release_track=spec['track'],release_manifest_sha256=freeze,exit_code=result.returncode),indent=2))
 entry=dict(task_id=tid,exit_code=result.returncode,report=str(report_path),log=str(a.output/f'{tid}.log'))
 with (a.output/'suite-attempts.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
 print(json.dumps(entry),flush=True)
