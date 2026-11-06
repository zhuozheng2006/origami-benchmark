"""Explicitly opt into a single generic fixture diagnostic, without paid agents.

No physical, geometric, contact or strain acceptance is loosened. The existing
qualifier archives the selected mode, controller source hash and complete path.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--fixture-mode', required=True, choices=[
    'guided_all_directional_10000', 'guided_pair_directional_10000'])
a,remaining=p.parse_known_args()
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('private_fixture_qualifier',root/'scripts/qualify_hard_candidates.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.FIXTURE_MODE=a.fixture_mode
base_qualify=m.qualify

def with_wrapper_provenance(*args,**kwargs):
    path=Path(__file__).resolve()
    kwargs['provenance']=dict(kwargs.get('provenance') or {},
        fixture_wrapper=str(path),
        fixture_wrapper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        experimental_fixture=a.fixture_mode,
        model_evaluation=False)
    return base_qualify(*args,**kwargs)

m.qualify=with_wrapper_provenance
sys.argv=[sys.argv[0],*remaining]
raise SystemExit(m.main())
