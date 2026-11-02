"""Recompute scores from existing evidence. This is NOT trusted physical replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.document_pose_audit import compare_in_plane
from origami_bench.diagram.document_grading import crease_check
from origami_bench.diagram.hard_episode import unsigned_crease_check
from origami_bench.diagram.layer_grading_v15 import band_crease_memory_check, ordered_checkpoints


def rescore(folder, task_id, dataset='release-v1'):
    private = json.loads((ROOT/'data'/dataset/'private/tasks.json').read_text())[task_id]
    def read(name):
        return json.loads((folder/name).read_text())
    state, events, config = read('final-state.json'), read('events.json'), read('run-config.json')
    metrics, history = read('metrics.json'), read('material-history.json')
    vertices = np.load(folder/'trajectory.npz')['vertices']
    if not np.allclose(vertices[-1], state['vertices'], rtol=0, atol=1e-12):
        raise ValueError('Final state disagrees with trajectory')
    shape = compare_in_plane(state['vertices'], np.load(ROOT/private['goal'])['vertices'])
    path = ordered_checkpoints(vertices, np.load(ROOT/private['checkpoints'])['vertices'], config['initial_frame'])
    physical = bool(metrics) and all(
        m.get('clear') is True and m.get('clearance',{}).get('clear') is True
        and max(m['stretch'],m['compression']) < .01 and m['grip_error'] < .0005 for m in metrics)
    memory = (crease_check(history, private.get('required_signed_hinges',{}))
              and unsigned_crease_check(history, private.get('required_unsigned_hinges',[]))
              and band_crease_memory_check(history, private.get('required_band_creases',[])))
    finished = bool(events) and events[-1]['action']=={'op':'finish'} and events[-1]['result']['status']=='accepted'
    environment_error = any(e['result'].get('category')=='environment_error' for e in events)
    success = bool(finished and physical and memory and path['passed'] and shape['rms_mm']<1 and shape['max_mm']<3 and not environment_error)
    return dict(task_id=task_id,task_success=success,physical_pass=physical,memory_pass=bool(memory),
                shape=shape,diagram_path=path,finished=finished,environment_error=environment_error,
                mode='untrusted_evidence_rescore_not_new_simulation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',default='release-v1')
    a=p.parse_args()
    print(json.dumps(rescore(a.run,a.task,a.dataset),indent=2))
