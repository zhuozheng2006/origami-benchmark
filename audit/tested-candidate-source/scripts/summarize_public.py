"""Report per-suite/per-agent scores; never pool different observation protocols."""
import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');a=p.parse_args();groups=collections.defaultdict(list);excluded=0
for path in a.runs:
 r=json.loads((path/'report.json').read_text())
 if not r.get('score_eligible'):excluded+=1;continue
 groups[(r.get('suite','layers-v1'),r.get('release_track'),r.get('agent_kind'),r.get('release_manifest_sha256'))].append(r)
result=[]
for (suite,track,agent,freeze),rs in groups.items():
 by=collections.defaultdict(list)
 for r in rs:by[r['task_id']].append(r)
 result.append(dict(suite=suite,track=track,agent=agent,release_manifest_sha256=freeze,runs=len(rs),tasks=len(by),repeats={k:len(v) for k,v in by.items()},passed=sum(bool(r.get('task_success')) for r in rs),environment_errors=sum(r.get('status')=='environment_error' for r in rs),macro_task_success_rate=sum(sum(bool(r.get('task_success')) for r in runs)/len(runs) for runs in by.values())/len(by)))
print(json.dumps(dict(groups=result,excluded_oracle_diagnostic_or_replay=excluded,warning='Incomplete task coverage must be disclosed; never pool suites or historical versions.'),indent=2))
