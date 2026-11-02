"""Target-private diagram episodes, artifact logging, and scoped scoring."""
from pathlib import Path
import copy, hashlib, json, time
import numpy as np
from .skills import ACTION_GUIDE, SkillUnsupported, compile_fold, validate_fold
from ..reference_comparison import compare_reference

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'data/diagram2fold-v1.1'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def check_freeze(data=DEFAULT_DATA):
    data=Path(data);f=json.loads((data/'freeze.json').read_text())
    for name,sha in f['files'].items():
        path=(data/name).resolve()
        if not path.is_relative_to(data.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen data mismatch: '+name)
    return hashlib.sha256((data/'freeze.json').read_bytes()).hexdigest()


def semantic_match(actual, expected):
    try:
        line,d,n,angle=validate_fold(actual);eline,ed,en,eangle=validate_fold(expected)
        # Oriented normals are normalized toward the moving MATERIAL region.
        return bool(n@en>.9999 and np.max(np.abs((line-eline[0])@en))*.15<.001 and abs(angle-eangle)<=2)
    except (ValueError,TypeError,KeyError): return False


def render_observation(state, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v=np.asarray(state['vertices']);faces=np.asarray(state['faces'])
    fig=plt.figure(figsize=(8,4),facecolor='white')
    for k,(elev,azim) in enumerate([(90,-90),(25,-65)]):
        ax=fig.add_subplot(1,2,k+1,projection='3d')
        mesh=Poly3DCollection(v[faces]*1000,facecolors='#b3d8ed',edgecolors='#56758c',linewidths=.18,alpha=.95)
        ax.add_collection3d(mesh);ax.set(xlim=(-85,85),ylim=(-85,85),zlim=(-85,85));ax.set_box_aspect((1,1,1));ax.view_init(elev,azim)
        ax.set_xlabel('x / mm');ax.set_ylabel('y / mm');ax.set_zlabel('z / mm')
        ax.set_title('Top view' if k==0 else 'Oblique view')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


class Episode:
    def __init__(self, task_id, output, data=DEFAULT_DATA, backend_factory=None, observation_mode='image_only', max_decisions=12):
        self.data=Path(data).resolve();self.output=Path(output).resolve()
        if self.output.exists() and any(self.output.iterdir()): raise FileExistsError('Keep prior run; select a new output directory')
        self.output.mkdir(parents=True,exist_ok=True)
        self.freeze_hash=check_freeze(self.data)
        if observation_mode not in ('image_only','image_and_mesh'): raise ValueError('Unknown observation mode')
        self.observation_mode=observation_mode
        self.max_decisions=max_decisions
        pub=json.loads((self.data/'public/index.json').read_text())
        self.task=next(t for t in pub['tasks'] if t['id']==task_id)
        self.private=json.loads((self.data/'private/tasks.json').read_text())[task_id]
        self.events=[];self.actions=[];self.checkpoints=[];self.started=time.monotonic();self.closed=False;self.finished=False
        if backend_factory is None:
            from ..remote_codim import RemoteCodimClient
            backend_factory=RemoteCodimClient
        self.backend=backend_factory(self.private['mesh'])
        self.material=np.asarray(self.backend.material)
        state=self.backend.state();self.faces=np.asarray(state['faces'])
        frozen=np.load(self.data/'private'/self.private['targets'])
        if not np.array_equal(self.material,frozen['rest']) or not np.array_equal(self.faces,frozen['faces']):
            self.backend.close();raise RuntimeError('Backend mesh does not match frozen evaluation mesh')
        self.initial_hash=state['hash'];self.snapshots=[copy.deepcopy(state)]
        write_json(self.output/'run-config.json',dict(task_id=task_id,dataset_sha256=self.freeze_hash,observation_mode=observation_mode,backend=self.backend.configuration,skill_version='boundary-arc-v1.1-mesh-quantization',source_sha256=hashlib.sha256(Path(__file__).with_name('skills.py').read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def observation(self):
        state=self.backend.state()
        image=self.output/f'observation-{len(self.events):03d}.png'
        render_observation(state,image)
        # Deliberate allow-list: no reference state, metrics against goal,
        # expected action, crease schedule, mesh filename or audit result.
        observation=dict(task=copy.deepcopy(self.task),diagram=str(self.data/'public'/self.task['image']),current_image=str(image),action_guide=ACTION_GUIDE,current=dict(vertices=state['vertices'],faces=state['faces'],grips=state['grips'],material_coordinates=((self.material[:,:2]/.15)+.5).tolist(),state_hash=state['hash']),previous_action_result=self.events[-1]['result'] if self.events else None)
        if self.observation_mode=='image_only': observation['current']={'state_hash':state['hash']}
        observation.update(observation_mode=self.observation_mode,accepted_actions=copy.deepcopy(self.actions),decisions_used=len(self.events),max_decisions=self.max_decisions)
        write_json(self.output/f'observation-{len(self.events):03d}.json',observation)
        return observation

    def step(self, action):
        if self.finished: raise RuntimeError('Episode already finished')
        if len(self.events)>=self.max_decisions: raise ValueError('Action budget exhausted')
        before=self.backend.state()['hash'];instruction_index=len(self.actions);low=[];accepted=0;result={'status':'accepted'}
        try:
            if action=={'op':'finish'}:
                self.finished=True
            elif action=={'op':'undo'}:
                if not self.checkpoints: raise ValueError('Nothing to undo')
                previous=self.checkpoints.pop()
                for _ in range(previous['low_level_count']): self.backend.step({'op':'undo'})
                self.actions.pop();self.snapshots.pop()
            else:
                low=compile_fold(self.material,self.backend.state()['vertices'],self.faces,action)
                for cmd in low:
                    self.backend.step(cmd);accepted+=1
                self.actions.append(copy.deepcopy(action))
                self.checkpoints.append(dict(frame=int(self.backend.state()['steps']),low_level_count=len(low)))
                self.snapshots.append(copy.deepcopy(self.backend.state()))
                result['executed_axis']={k:low[-1][k] for k in ['axis_point','axis_direction','degrees']}
        except Exception as ex:
            # Low-level actions already roll back atomically. Roll back earlier
            # commands in this composite skill too, preserving the episode.
            rollback=True
            for _ in range(accepted):
                try: self.backend.step({'op':'undo'})
                except Exception: rollback=False;break
            if isinstance(ex,SkillUnsupported): category='capability_limit'
            elif isinstance(ex,(ValueError,TypeError,KeyError,IndexError)): category='invalid_action'
            elif 'ActionRejected' in str(ex): category='simulator_rejection'
            else: category='environment_error'
            if not rollback or self.backend.state()['hash']!=before: category='environment_error'
            result=dict(status='rejected',category=category,detail=str(ex),rollback_verified=rollback and self.backend.state()['hash']==before)
        event=dict(index=len(self.events),instruction_index=instruction_index,action=action,before=before,after=self.backend.state()['hash'],low_level_actions=low,result=result)
        self.events.append(event);write_json(self.output/'events.json',self.events)
        return result

    def finish(self, agent_kind='submitted_actions'):
        saved=self.backend.save()
        import shutil
        for name in ['trajectory.npz','metrics.json','actions.json','model-config.json']:
            shutil.copy2(saved/name,self.output/name)
        write_json(self.output/'submitted-actions.json',self.actions)
        write_json(self.output/'checkpoints.json',self.checkpoints)
        report=score_episode(self.data,self.task['id'],self.output,self.events,self.actions,self.checkpoints,self.finished)
        report.update(agent_kind=agent_kind,seconds=time.monotonic()-self.started,dataset_sha256=self.freeze_hash)
        write_json(self.output/'report.json',report)
        self.close();return report

    def close(self):
        if not self.closed:
            self.backend.close();self.closed=True


def score_episode(data,task_id,output,events,actions,checkpoints,finished):
    private=json.loads((Path(data)/'private/tasks.json').read_text())[task_id]
    goals=np.load(Path(data)/'private'/private['targets'])['targets']
    trajectory=np.load(Path(output)/'trajectory.npz')['vertices']
    metrics=json.loads((Path(output)/'metrics.json').read_text())
    expected=private['reference_actions']
    shapes=[]
    for i,c in enumerate(checkpoints):
        if i>=len(goals): break
        _,s=compare_reference(trajectory[c['frame']],goals[i]);shapes.append(s)
    semantics=[semantic_match(a,b) for a,b in zip(actions,expected)]
    attempted=[]
    for event in events:
        if isinstance(event['action'],dict) and event['action'].get('op')=='fold':
            idx=event.get('instruction_index',event['index'])
            attempted.append(semantic_match(event['action'],expected[idx]) if idx<len(expected) else False)
    complete=bool(finished and len(actions)==len(expected) and len(checkpoints)==len(goals))
    semantic_ok=complete and all(semantics)
    shape_ok=complete and all(s['rms_mm']<1 and s['max_mm']<3 for s in shapes)
    # Missing physical evidence never becomes success. This is the original,
    # fixed-mesh clearance policy; it is not general solid-thickness validity.
    constraints=bool(metrics) and all(m.get('clear') is True and m.get('clearance',{}).get('clear') is True and m['stretch']<.01 and m['compression']<.01 and m['grip_error']<.0005 for m in metrics)
    errors=[e['result'] for e in events if e['result']['status']=='rejected']
    environment_error=any(e['category']=='environment_error' for e in errors)
    success=bool(semantic_ok and shape_ok and constraints and not environment_error)
    return dict(task_id=task_id,task_success=success,completion=complete,diagram_semantic_success=semantic_ok,shape_success=shape_ok,fixed_mesh_controller_constraints_pass=constraints,physical_generalization_certified=None,
        scope='Finite-thickness Codim thin shell under ideal boundary grasps, fixed mesh + historical nonincident clearance rule. No general thickness, real tool access or calibrated material certification. Held checkpoints only, release stability NOT scored.',
        checkpoint_shape=shapes,semantic_per_step=semantics,attempted_diagram_steps_correct=attempted,perception_error_count=sum(not x for x in attempted),errors=errors,status='passed_scoped' if success else 'environment_error' if environment_error else 'not_passed',
        steps=len(metrics),peak_stretch=max((m['stretch'] for m in metrics),default=None),peak_compression=max((m['compression'] for m in metrics),default=None),max_grip_error_mm=max((1000*m['grip_error'] for m in metrics),default=None),
        limits=dict(shape_rms_mm=1,shape_max_mm=3,principal_strain=.01,grip_error_mm=.5,nominal_thickness_um=100,discrete_clearance_um=100),
        error_policy='Malformed/unsupported actions, simulator rejection, and environment faults are recorded separately. A correct diagram action rejected by the solver is not counted as a perception error.')
