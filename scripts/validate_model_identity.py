"""Offline synthetic identity/summary regression checks. No API or physics calls."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_model_identity import IdentityTests, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IdentityTests)
    names = [test.id().split('.')[-1] for test in suite]
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failures = {test.id().split('.')[-1]: detail for test,detail in result.failures+result.errors}
    helper = IdentityTests()
    different = helper.summarize([report('synthetic_A'), report('synthetic_B')])
    denominator = helper.summarize([report(returned_models=['A']),report(task_success=False,status='environment_error')])
    duplicate = report()
    duplicate_result = helper.summarize([duplicate,dict(duplicate)],duplicate_path=True)
    value = dict(kind='synthetic_unit_validation_not_model_benchmark', inference_calls=0, physics_runs=0,
        tests_run=result.testsRun, tests_passed=result.testsRun-len(failures),
        cases=[dict(name=name,passed=name not in failures,detail=failures.get(name)) for name in names],
        demonstrations=dict(different_models=dict(input_models=2,output_groups=len(different['groups'])),
            success_and_environment_failure=dict(input_runs=2,output_groups=len(denominator['groups']),
                success_rate=denominator['groups'][0]['end_to_end_success_rate'],
                environment_error_rate=denominator['groups'][0]['environment_error_rate']),
            duplicate_evidence=dict(input_reports=2,duplicate_directory_inputs=1,counted_runs=duplicate_result['groups'][0]['runs'])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
