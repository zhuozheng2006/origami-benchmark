"""Summarize independent live episodes; diagnostic/oracle runs never inflate scores."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runs',type=Path,nargs='+')
a=p.parse_args()
reports=[json.loads((folder/'report.json').read_text()) for folder in a.runs]
eligible=[r for r in reports if r.get('score_eligible') is True]
success=sum(r.get('task_success') is True for r in eligible)
errors=sum(r.get('status')=='environment_error' for r in eligible)
print(json.dumps(dict(submitted_runs=len(reports),eligible_runs=len(eligible),passed=success,
    environment_errors=errors,success_rate_including_environment_errors=success/len(eligible) if eligible else None,
    warning='Report model, release, per-task coverage and repeats. Missing tasks are not automatically tested; oracle and action replays are excluded.'),indent=2))
