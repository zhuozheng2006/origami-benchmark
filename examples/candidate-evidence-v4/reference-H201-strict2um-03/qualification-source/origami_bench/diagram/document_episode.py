from pathlib import Path
import copy,json,shutil,time,hashlib
import numpy as np
from .stack_controller import StackController,SkillUnsupported,VERSION
from .depth_renderer import render_public
from .attempt_ledger import atomic_json
class DocumentEpisode:
 def __init__(self,task,private,output,backend_factory,max_decisions=40):
  self.task=copy.deepcopy(task);self.private=copy.deepcopy(private);self.output=Path(output)
  if self.output.exists():raise FileExistsError('Existing attempts are immutable; use a new run ID')
  self.output.mkdir(parents=True);self.backend=backend_factory(private['mesh']);self.controller=StackController(self.backend);self.events=[];self.actions=[];self.finished=False;self.closed=False;self.max_decisions=max_decisions;self.started=time.monotonic();self.page=1;self.crop=[0,0,1,1];self.initialization=[]
  # A public initial orientation is reached by actual controller motion.
  for cmd in private.get('initialization',[]):self.initialization.append(self.controller.execute(cmd))
  self.controller.records=[];self.controller.snapshots=[];self.controller.fold_stack=[]
  self.initial_frame=self.backend.state()['steps'];self.initial_state=copy.deepcopy(self.backend.state())
  atomic_json(self.output/'run-config.json',dict(public_task=self.task,controller_version=VERSION,backend=self.backend.configuration,initialization=self.initialization,max_attempts=max_decisions,measurement_tool=False,automatic_panel=False,explicit_finish_phase=False))
 def observation(self):
  state=self.backend.state();image=self.output/f'observation-{len(self.events):03}.png';render_public(state,image)
  return dict(task={k:self.task[k] for k in ('id','instruction','title','page_count')},document=dict(page=self.page,page_count=self.task['page_count'],crop=self.crop),diagram=self.private['pages'][self.page-1],current_image=str(image),current=dict(state_hash=state['hash']),accepted_actions=copy.deepcopy(self.actions),previous_action_result=self.events[-1]['result'] if self.events else None,max_decisions=self.max_decisions)
 def step(self,action):
  if self.finished:raise RuntimeError('Episode finished')
  before=self.backend.state();low=[]
  try:
   op=action.get('op')
   if op=='finish':
    if action!={'op':'finish'}:raise ValueError('finish has no extra fields')
    self.finished=True
   elif op=='view_page':
    if not {'op','page'}<=set(action) or set(action)-{'op','page','crop'}:raise ValueError('view_page fields invalid')
    page=action['page'];crop=action.get('crop',[0,0,1,1]);c=np.asarray(crop,float)
    if type(page)is not int or not 1<=page<=self.task['page_count']:raise ValueError('Page index out of range')
    if c.shape!=(4,) or not np.isfinite(c).all() or (c<0).any() or (c>1).any() or min(c[2]-c[0],c[3]-c[1])<.08:raise ValueError('Crop must be nonempty normalized image rectangle')
    self.page=page;self.crop=c.tolist()
   else:
    record=self.controller.execute(action);low=record['low_level_actions'];self.actions=[r['action'] for r in self.controller.records]
   result=dict(status='accepted')
  except Exception as e:
   category='capability_limit' if isinstance(e,SkillUnsupported) else 'invalid_action' if isinstance(e,(ValueError,TypeError,KeyError,IndexError)) else 'simulator_rejection' if 'ActionRejected' in str(e) else 'environment_error'
   restored=self.backend.state()['hash']==before['hash']
   if not restored:category='environment_error'
   result=dict(status='rejected',category=category,detail=str(e),rollback_verified=restored)
  after=self.backend.state();event=dict(index=len(self.events),action=copy.deepcopy(action),before=before['hash'],after=after['hash'],low_level_actions=low,result=result,frame=after['steps'],document=dict(page=self.page,crop=self.crop),material_memory=after.get('material_memory'),time=after['time'])
  self.events.append(event);atomic_json(self.output/'events.json',self.events);return result
 def finish(self,agent_kind):
  saved=self.backend.save()
  for name in ['trajectory.npz','metrics.json','actions.json','model-config.json','material-history.json']:shutil.copy2(saved/name,self.output/name)
  state=self.backend.state();atomic_json(self.output/'final-state.json',state);render_public(state,self.output/'final.png');atomic_json(self.output/'controller-records.json',self.controller.records)
  metrics=json.loads((self.output/'metrics.json').read_text());physical=bool(metrics) and all(m.get('clear') is True and m.get('clearance',{}).get('clear') is True and max(m['stretch'],m['compression'])<.01 and m['grip_error']<.0005 for m in metrics)
  shape=None;goal=self.private.get('goal')
  if goal:
   expected=np.load(goal)['vertices'];actual=np.asarray(state['vertices']);error=np.linalg.norm(actual-expected,axis=1);shape=dict(rms_mm=float(np.sqrt(np.mean(error**2))*1000),max_mm=float(error.max()*1000))
  shape_pass=shape is not None and shape['rms_mm']<1 and shape['max_mm']<3
  # Material-memory tasks additionally require the requested creases to exist.
  from .document_grading import crease_check
  required=self.private.get('required_signed_hinges',{})
  history=json.loads((self.output/'material-history.json').read_text())
  memory_pass=crease_check(history,required)
  environment=any(e['result'].get('category')=='environment_error' for e in self.events)
  success=bool(self.finished and physical and shape_pass and memory_pass and not environment)
  report=dict(task_id=self.task['id'],agent_kind=agent_kind,task_success=success,finished=self.finished,physical_pass=physical,shape=shape,memory_pass=memory_pass,goal_available=bool(goal),status='passed_scoped' if success else 'environment_error' if environment else 'reference_unscored' if not goal else 'not_passed',attempts=len(self.events),retained_paper_operations=len(self.actions),peak_strain=max((max(m['stretch'],m['compression']) for m in metrics),default=None),material_memory=state.get('material_memory'),seconds=time.monotonic()-self.started,scope='Finite thickness, same ideal boundary grasps and historical contact rule; no real fingers or general collision certification.')
  atomic_json(self.output/'report.json',report);self.close();return report
 def close(self):
  if not self.closed:self.backend.close();self.closed=True
