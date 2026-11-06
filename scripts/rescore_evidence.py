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


def rescore(folder, task_id, dataset=None):
    registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
    specification=registry.get(task_id,{})
    dataset=dataset or specification.get('dataset','release-v1')
    private = json.loads((ROOT/'data'/dataset/'private/tasks.json').read_text())[task_id]
    if specification.get('required_grading_contract') == 'precrease-artifact-v4':
        from scripts.score_precrease_candidates import rescore as precrease_rescore
        return precrease_rescore(folder, task_id, private, ROOT)
    def read(name):
        return json.loads((folder/name).read_text())
    state, events = read('final-state.json'), read('events.json')
    finished = bool(events) and events[-1]['action']=={'op':'finish'} and events[-1]['result']['status']=='accepted'
    if specification.get('episode')=='step':
        from origami_bench.diagram.episode import score_episode
        result=score_episode(ROOT/'data'/dataset,task_id,folder,events,read('submitted-actions.json'),read('checkpoints.json'),finished)
        result['mode']='untrusted_evidence_rescore_not_new_simulation'
        return result
    metrics, history = read('metrics.json'), read('material-history.json')
    vertices = np.load(folder/'trajectory.npz')['vertices']
    if not np.allclose(vertices[-1], state['vertices'], rtol=0, atol=1e-12):
        raise ValueError('Final state disagrees with trajectory')
    goal=np.load(ROOT/private['goal'])['vertices']
    episode=specification.get('episode','layer')
    if episode=='document':
        error=np.linalg.norm(np.asarray(state['vertices'])-goal,axis=1)
        shape=dict(rms_mm=float(np.sqrt(np.mean(error**2))*1000),max_mm=float(error.max()*1000))
    else:shape = compare_in_plane(state['vertices'], goal)
    path=None
    if episode not in ('document','hard'):
        config=read('run-config.json')
        path = ordered_checkpoints(vertices, np.load(ROOT/private['checkpoints'])['vertices'], config['initial_frame'])
    physical = bool(metrics) and all(
        m.get('clear') is True and m.get('clearance',{}).get('clear') is True
        and max(m['stretch'],m['compression']) < .01 and m['grip_error'] < .0005 for m in metrics)
    memory = crease_check(history, private.get('required_signed_hinges',{}))
    if episode!='document':memory=memory and unsigned_crease_check(history, private.get('required_unsigned_hinges',[]))
    if episode not in ('document','hard'):memory=memory and band_crease_memory_check(history, private.get('required_band_creases',[]))
    environment_error = any(e['result'].get('category')=='environment_error' for e in events)
    success = bool(finished and physical and memory and (path is None or path['passed']) and shape['rms_mm']<1 and shape['max_mm']<3 and not environment_error)
    return dict(task_id=task_id,task_success=success,physical_pass=physical,memory_pass=bool(memory),
                shape=shape,diagram_path=path,finished=finished,environment_error=environment_error,
                mode='untrusted_evidence_rescore_not_new_simulation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',help='Default: resolve from the public task registry')
    a=p.parse_args()
    print(json.dumps(rescore(a.run,a.task,a.dataset),indent=2))
