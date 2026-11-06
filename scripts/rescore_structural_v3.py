"""Opt-in candidate rescore. Original reports and the default scorer are untouched."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rescore_evidence import rescore as legacy_rescore
from origami_bench.diagram.structural_grading_v3 import VERSION, compare_placement, ordered_motion_checkpoints, combine_success, physical_evidence_pass


def rescore(folder, task_id, dataset=None):
    folder = Path(folder)
    registry = json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
    specification = registry[task_id]
    if specification['episode'] != 'layer':
        raise ValueError('This candidate contract is explicitly scoped to prepared layer tasks')
    dataset = dataset or specification['dataset']
    private = json.loads((ROOT/'data'/dataset/'private/tasks.json').read_text())[task_id]
    legacy = legacy_rescore(folder, task_id, dataset)
    trajectory = np.load(folder/'trajectory.npz')['vertices']
    initial_frame = json.loads((folder/'run-config.json').read_text())['initial_frame']
    goal = np.load(ROOT/private['goal'])['vertices']
    initial = np.load(ROOT/private['initial_goal'])['vertices']
    targets = np.load(ROOT/private['checkpoints'])['vertices']
    shape = compare_placement(trajectory[-1], goal)
    path = ordered_motion_checkpoints(trajectory, targets, initial, initial_frame)
    physical = bool(legacy['physical_pass'] and physical_evidence_pass(json.loads((folder/'metrics.json').read_text())))
    success = combine_success(shape, path, legacy['finished'], physical,
                              legacy['memory_pass'], legacy['environment_error'])
    recorded = json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else {}
    return dict(task_id=task_id, grading_version=VERSION, mode='candidate_evidence_rescore_not_new_simulation',
                score_eligible=False, task_success=success, structure_success=shape['passed'],
                shape=shape, diagram_path=path, finished=legacy['finished'], physical_pass=physical,
                memory_pass=legacy['memory_pass'], environment_error=legacy['environment_error'],
                recorded_task_success=recorded.get('task_success'), recorded_status=recorded.get('status'),
                legacy_recomputed_task_success=legacy['task_success'],
                legacy_shape=legacy['shape'], legacy_diagram_path=legacy['diagram_path'],
                scope='XYZ placement-invariant structural shape plus required relative board motions. Same labelled vertices, crease memory and recorded physical gates. No re-simulation or real-world certification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = rescore(args.run, args.task, args.dataset)
    data = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            f.write(data)
    else:
        print(data)
