"""Versioned bridge; same physical solver, adds read-only plastic-state evidence."""
from pathlib import Path
import os,sys,json,uuid,re,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));protocol=os.fdopen(os.dup(1),'w',buffering=1)
run=root/'outputs/interactive-codim'/uuid.uuid4().hex;run.mkdir(parents=True)
log=open(run/'solver.log','w');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
def send(o):protocol.write(json.dumps(o,allow_nan=False)+'\n');protocol.flush()
try:
 init=json.loads(sys.stdin.readline());mesh=init['mesh']
 if not re.fullmatch('[a-z0-9_]+',mesh):raise ValueError('invalid mesh')
 from origami_bench.layer_env import LayerEnvironment as CodimEnvironment
 import numpy as np
 e=CodimEnvironment(mesh,run,newton_tolerance=init.get('newton_tolerance',1e-4),max_newton_iterations=600,barrier_distance=init.get('barrier_distance',.001),backend_variant=init.get('backend_variant','codim-ipc'),compression_barrier=init.get('compression_barrier',False),grip_solve_tolerance=init.get('grip_solve_tolerance',.00002),grip_solve_metric=init.get('grip_solve_metric','l2'));original_state=e.state;history=[]
 def state():
  st=original_state();angles=np.array([e.sim.edgeInfo[i][0] for i in range(e.hinge_count)],dtype=np.float64)
  digest=hashlib.sha256(angles.tobytes()).hexdigest()
  st['material_memory']=dict(rest_angle_sha256=digest,rest_angle_l2=float(np.linalg.norm(angles)),dissipation_joules=sum(m['dissipation_joules'] for m in e.metrics))
  st['hash']=hashlib.sha256((st['hash']+digest).encode()).hexdigest();return st
 e.configuration['effective_activation_distance_m']=float(np.sqrt(e.sim.dHat2));e.configuration['layer_env_sha256']=hashlib.sha256((root/'origami_bench/layer_env.py').read_bytes()).hexdigest();e.configuration['soft_point_controls_supported']=hasattr(e.J,'Origami_SetSoftPointControls');e.configuration['document_worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();e.configuration['material_state_in_hash']=True
 e.configuration['worker_process_id']=os.getpid();e.configuration['worker_artifact_folder']=str(run)
 for module in ['control_geometry','contact_certification']:
  e.configuration[module+'_sha256']=hashlib.sha256((root/'origami_bench'/f'{module}.py').read_bytes()).hexdigest()
 e.configuration['hinge_stencils']=[list(st) for _,st,_ in e.hinge_data()]
 e.state=state;e.on_frame=lambda st:send({'frame':st})
 def save():
  e.save(run);(run/'material-history.json').write_text(json.dumps(history,indent=2))
 def evidence():
  return dict(hinges=[list(st) for _,st,_ in e.hinge_data()],angles=[float(a) for _,_,a in e.hinge_data()],rest_angles=[float(e.sim.edgeInfo[i][0]) for i in range(e.hinge_count)])
 history.append(dict(op='initial',state=state()['material_memory'],time=0,steps=0,**evidence()))
 send(dict(ok=True,state=state(),material=e.rest.tolist(),configuration=e.configuration))
 for line in sys.stdin:
  req=json.loads(line)
  try:
   if req.get('op')=='close':send({'ok':True});break
   if req.get('op')=='save':save();send(dict(ok=True,path=str(run)));continue
   st=e.step(req);history.append(dict(op=req['op'],state=st['material_memory'],time=st['time'],steps=st['steps'],**evidence()));send(dict(ok=True,state=st))
  except Exception as ex:send(dict(ok=False,error=type(ex).__name__+': '+str(ex),state=state()))
 save()
except Exception as ex:send(dict(ok=False,error=type(ex).__name__+': '+str(ex)))
