import json,subprocess,sys,pathlib,time
root=pathlib.Path.cwd();part=int(sys.argv[1]);tasks=json.loads((root/'data/evaluation-catalog.json').read_text())['tasks']
selected=sorted(k for k,v in tasks.items() if v['status']=='accepted')[part::3]
base=root/'outputs/v2-reference-sweep';base.mkdir(exist_ok=True,parents=True)
for tid in selected:
 start=time.monotonic();out=base/tid
 with (base/f'{tid}.log').open('x') as log:
  try:code=subprocess.run([sys.executable,'scripts/evaluate_public.py','--task',tid,'--reference','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
  except subprocess.TimeoutExpired:code=124
 report=json.loads((out/'report.json').read_text()) if (out/'report.json').is_file() else {}
 print(json.dumps(dict(task_id=tid,exit_code=code,success=report.get('success'),seconds=round(time.monotonic()-start,2))),flush=True)
