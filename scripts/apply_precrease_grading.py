"""Apply the explicit precrease artifact contract to a new accounted episode."""
import json
from pathlib import Path
from scripts.score_precrease_candidates import rescore

ROOT = Path(__file__).resolve().parents[1]


def apply_candidate(folder, task_id, private_spec):
    folder = Path(folder)
    original = json.loads((folder / 'report.json').read_text())
    candidate = rescore(folder, task_id, private_spec, ROOT)
    for name, value in [('report-legacy.json', original), ('candidate-grading.json', candidate)]:
        with (folder / name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
    result = dict(original)
    result.update(candidate)
    # Rescoring must not change the run identity or turn oracle/diagnostic
    # evidence into an eligible model submission.
    for key in ('score_eligible', 'run_id', 'repeat_id', 'model_identity', 'agent_kind'):
        if key in original:
            result[key] = original[key]
    result.update(grading_contract='precrease-artifact-v4', grading_candidate=True,
                  legacy_task_success=original.get('task_success'))
    if candidate.get('environment_error') or candidate.get('input_valid') is False:
        result['status'] = 'environment_error'
    else:
        result['status'] = 'passed_scoped' if result['task_success'] else 'not_passed'
    (folder / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
