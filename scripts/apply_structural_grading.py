"""Attach opt-in scoring to a NEW execution; never run against archived folders."""
import json
from pathlib import Path
from scripts.rescore_structural_v3 import rescore


def apply_candidate(folder, task_id):
    folder = Path(folder)
    original = json.loads((folder/'report.json').read_text())
    candidate = rescore(folder, task_id)
    # Exclusive writes make reapplication impossible and keep original verdict.
    with (folder/'report-legacy.json').open('x') as f:
        json.dump(original, f, indent=2)
    with (folder/'candidate-grading.json').open('x') as f:
        json.dump(candidate, f, indent=2)
    result = dict(original)
    result.update(task_success=candidate['task_success'], structure_success=candidate['structure_success'],
                  shape=candidate['shape'], diagram_path=candidate['diagram_path'],
                  grading_contract='structural-v3', grading_version=candidate['grading_version'],
                  grading_candidate=True, legacy_task_success=original['task_success'],
                  goal_success=bool(candidate['structure_success'] and candidate['finished'] and
                                    candidate['physical_pass'] and candidate['memory_pass'] and
                                    not candidate['environment_error']))
    if original.get('status') == 'environment_error' or not candidate['diagram_path']['contract_valid']:
        result['status'] = 'environment_error'
        if not candidate['diagram_path']['contract_valid']:
            result['failure_stage'] = 'ambiguous_grading_contract'
    else:
        result['status'] = 'passed_scoped' if result['task_success'] else 'not_passed'
    # Diagnostics/oracle eligibility never becomes eligible through regrading.
    (folder/'report.json').write_text(json.dumps(result, indent=2))
    return result
