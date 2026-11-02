"""Run a frozen suite sequentially; every task has a separate immutable directory."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--split');p.add_argument('--agent');p.add_argument('--reference',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--list',action='store_true');p.add_argument('--allow-diagnostic',action='store_true');a=p.parse_args()
registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks'];ids=sorted(k for k,v in registry.items() if v['suite']==a.suite and (v['status']=='accepted' or a.allow_diagnostic) and (not a.split or v.get('split')==a.split))
if not ids:raise ValueError('Empty/unknown suite or split')
if a.list:print('\n'.join(ids));sys.exit(0)
if not a.output or bool(a.agent)==bool(a.reference):p.error('Use --output and exactly one of --agent / --reference')
if a.suite=='ultra' and not a.allow_diagnostic:p.error('Ultra uses legacy scoring and requires --allow-diagnostic')
for tid in ids:
 cmd=[sys.executable,str(ROOT/'scripts/evaluate_public.py'),'--task',tid,'--output',str(a.output/tid)]
 cmd+=['--reference'] if a.reference else ['--agent',a.agent]
 if a.allow_diagnostic:cmd+=['--allow-diagnostic']
 import os
 env=dict(os.environ)
 if env.get('ORIGAMI_API_LOGDIR'):env['ORIGAMI_API_LOGDIR']=str(Path(env['ORIGAMI_API_LOGDIR'])/tid)
 result=subprocess.run(cmd,env=env,cwd=ROOT)
 print(json.dumps(dict(task_id=tid,exit_code=result.returncode)),flush=True)
