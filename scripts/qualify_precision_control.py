"""Isolated, stricter numerical-control experiment; never a model evaluation.

Only the hard-grip solve tolerance changes. Native/material/acceptance values
remain in the supplied explicit physical signature. All plans and raw failures
are written by the existing generic candidate qualifier.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--grip-solve-tolerance',required=True,type=float)
a,remaining=p.parse_known_args()
if not 0 < a.grip_solve_tolerance < .0002:
    raise ValueError('This control only permits stricter than the frozen 200um RMS tolerance')
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('private_precision_qualifier', root/'scripts/qualify_hard_candidates.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.SETTINGS=dict(m.SETTINGS,grip_solve_tolerance=a.grip_solve_tolerance)
base_qualify=m.qualify
def with_wrapper_provenance(*args, **kwargs):
    wrapper=Path(__file__).resolve()
    kwargs['provenance']=dict(kwargs.get('provenance') or {},
        precision_wrapper=str(wrapper),
        precision_wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        stricter_grip_solve_tolerance_m=a.grip_solve_tolerance)
    return base_qualify(*args, **kwargs)
m.qualify=with_wrapper_provenance
sys.argv=[sys.argv[0],*remaining]
raise SystemExit(m.main())
