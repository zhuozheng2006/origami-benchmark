"""Read-only layer experiments, including rejected candidates (never passed states)."""
from pathlib import Path
import json,re,sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/layer-paths-v14';DATA=ROOT/'data/document-layers-v14'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def folder(name,output=OUT):
 if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('Invalid run ID')
 p=output/'runs'/name
 if not p.is_dir():raise FileNotFoundError(name)
 return p
def summary(output=OUT,dataset=DATA):
 runs=[]
 for p in sorted((output/'runs').glob('*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  r=read(p);cfg=read(p.parent/'run-config.json',{});events=read(p.parent/'events.json',[]);audit=read(p.parent/'geometry-audit.json',{});checks=r.get('geometric_checks',[]) or audit.get('checks',[]);shape=r.get('shape') or (checks[-1] if checks else None)
  all_checks=checks+r.get('preparation_checks',[])+([r['initial_check']] if r.get('initial_check') else [])+([r['shape']] if r.get('shape') else [])
  prep_verified=r.get('agent_kind')!='private_layer_reference' or bool(r.get('initial_check') and r.get('preparation_checks'))
  good=bool(r.get('finished') and r.get('physical_pass') and prep_verified and checks and all(c.get('rms_mm',999)<1 and c.get('max_mm',999)<3 for c in all_checks))
  if r.get('model'):good=bool(r.get('task_success') and r.get('physical_pass') and r.get('finished'))
  worst=dict(rms_mm=max(c['rms_mm'] for c in all_checks),max_mm=max(c['max_mm'] for c in all_checks)) if all_checks else None
  runs.append(dict(id=p.parent.name,report=r,config=cfg,shape=shape,worst_endpoint=worst,reference_shape_and_physics=good,rejections=sum(e['result']['status']=='rejected' for e in events),last_error=next((e['result'].get('detail','') for e in reversed(events) if e['result']['status']=='rejected'),r.get('error',''))))
 for cfgfile in sorted((output/'runs').glob('*/run-config.json')):
  if (cfgfile.parent/'report.json').exists():continue
  cfg=read(cfgfile);events=read(cfgfile.parent/'events.json',[])
  runs.append(dict(id=cfgfile.parent.name,report=dict(finished=False,attempts=len(events),status='incomplete_record'),config=cfg,shape=None,reference_shape_and_physics=False,rejections=sum(e['result']['status']=='rejected' for e in events),last_error='记录尚未结束或缺少最终报告；不计为通过。'))
 db=sqlite3.connect('file:'+str(ROOT/'outputs/multimodel-small-v1/costs.sqlite3')+'?mode=ro',uri=True)
 try:cost=db.execute('select sum(coalesce(charge,reserve)) from calls').fetchone()[0]
 finally:db.close()
 return dict(runs=runs,tasks=read(dataset/'public/index.json',{}).get('tasks',[]),qualification=read(output/'qualification.json',{}),freeze=read(output/'policy-freeze.json'),cost_upper_cny=cost,budget_cny=100)
def strain(rest,faces,vertices):
 r=np.asarray(rest)[:,:2];f=np.asarray(faces);v=np.asarray(vertices);dm=np.stack([r[f[:,1]]-r[f[:,0]],r[f[:,2]]-r[f[:,0]]],axis=-1);ds=np.stack([v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]],axis=-1);sv=np.linalg.svd(ds@np.linalg.inv(dm),compute_uv=False)
 return np.maximum(sv[:,0]-1,1-sv[:,1])
def mark_retained_branches(events,initial_frame=0):
 # Frame numbers alone are ambiguous after undo. Track the surviving command branch.
 stack=[];paths=[];frame_stack=[initial_frame]
 for i,e in enumerate(events):
  action=e.get('action');op=action.get('op') if isinstance(action,dict) else None;accepted=e.get('result',{}).get('status')=='accepted'
  if accepted and op=='undo' and stack:stack.pop();frame_stack.pop()
  elif accepted and op in ('fold_line','unfold','turn_over','rotate_paper'):
   stack.append(i);frame_stack.append(e.get('frame',frame_stack[-1]))
  paths.append(tuple(stack));e['review_frame']=e.get('frame',frame_stack[-1])
 final=tuple(stack)
 for e,path in zip(events,paths):e['review_in_retained_branch']=final[:len(path)]==path
 return events

def replay(name,output=OUT,dataset=DATA,asset_prefix='/layer-assets/'):
 p=folder(name,output);events=read(p/'events.json',[]);cfg=read(p/'run-config.json',{});r=read(p/'report.json',{})
 events=mark_retained_branches(events,cfg.get('initial_frame',0))
 for i,e in enumerate(events):
  if not e['review_in_retained_branch']:
   for j in (i+1,i):
    obs=read(p/f'observation-{j:03d}.json',{})
    if obs.get('current',{}).get('state_hash')==e.get('after') and (p/f'observation-{j:03d}.png').exists():
     e['review_observation_image']=asset_prefix+name+f'/observation-{j:03d}.png';break
 answer=dict(report=r,config=cfg,events=events,preparation=read(p/'preparation-records.json',cfg.get('initialization',[])),ledger=read(p/'attempt-ledger.json',{}),vertices=[],faces=[],frame_ids=[],rejected=[],images=[dict(name=f.name,url=asset_prefix+name+'/'+f.name) for f in p.glob('*.png') if re.fullmatch('(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',f.name)])
 if (p/'trajectory.npz').exists():
  d=np.load(p/'trajectory.npz');v=d['vertices'];frames=[e.get('frame',0) for e in events];ids=np.unique(np.r_[np.arange(0,len(v),max(1,len(v)//280)),len(v)-1,frames]).astype(int);ids=ids[(ids>=0)&(ids<len(v))]
  answer.update(vertices=v[ids].round(8).tolist(),faces=d['faces'].tolist(),frame_ids=ids.tolist())
  if (p/'annotated-goal.npz').exists():
   target=np.load(p/'annotated-goal.npz')['vertices']
   if target.shape==v[-1].shape:
    error=np.linalg.norm(v[-1]-target,axis=1)*1000
    answer['review_reference']=dict(vertices=target.round(8).tolist(),vertex_error_mm=error.round(5).tolist(),rms_mm=float(np.sqrt(np.mean(error**2))),max_mm=float(error.max()),scope='Completed prefix only; review-only target, never included in model observations.')
  elif r.get('model'):
   task=read(dataset/'private/tasks.json',{}).get(r.get('task_id'),{})
   if task.get('goal'):
    target=np.load(task['goal'])['vertices'];actual=v[-1]
    if target.shape==actual.shape:
     ac=actual[:,:2].mean(0);bc=target[:,:2].mean(0);u,_,vt=np.linalg.svd((actual[:,:2]-ac).T@(target[:,:2]-bc));rot=u@vt
     if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
     aligned=np.column_stack(((target[:,:2]-bc)@rot.T+ac,target[:,2]));error=np.linalg.norm(actual-aligned,axis=1)*1000
     answer['review_reference']=dict(vertices=aligned.round(8).tolist(),vertex_error_mm=error.round(5).tolist(),rms_mm=float(np.sqrt(np.mean(error**2))),max_mm=float(error.max()),scope='Full task goal aligned by allowed XY placement; human review only, never model input.')
 elif (p/'before-action.json').exists():
  st=read(p/'before-action.json')['state']
  answer.update(vertices=[st['vertices']],faces=st['faces'],frame_ids=[st['steps']],trajectory_status='last_verified_state_only')
 else:
  snapshot=next((p/name for name in ['initial-state.json','preparation-state.json'] if (p/name).exists()),None)
  if snapshot:
   st=read(snapshot);answer.update(vertices=[st['vertices']],faces=st['faces'],frame_ids=[st['steps']],trajectory_status='last_verified_state_only')
 for file in sorted((p/'rejected-candidates').glob('*.npz')):
  d=np.load(file);values=strain(d['rest'],d['faces'],d['candidate']);answer['rejected'].append(dict(name=file.stem,reason=str(d['reason']),vertices=d['candidate'].round(8).tolist(),before=d['before'].round(8).tolist(),face_strain=values.round(8).tolist(),peak_strain=float(values.max())))
 for i,item in enumerate(read(p/'rejected-endpoints.json',[])):
  st=item['state'];answer['rejected'].append(dict(name='endpoint-'+str(i),reason=item['reason'],vertices=st['vertices'],face_strain=None,peak_strain=None))
 return answer
