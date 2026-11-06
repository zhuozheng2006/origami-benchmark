"""Summarize frozen experiments without pooling models or hiding missing runs."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.public_summary import summarize_runs
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runs', type=Path, nargs='+')
p.add_argument('--plan', action='append', type=Path, default=[], help='Pre-registered experiment-plan.json; repeat for multiple models')
a = p.parse_args()
print(json.dumps(summarize_runs(a.runs, a.plan), indent=2))
