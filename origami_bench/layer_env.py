"""Transactional finite-thickness shell controller; Linux Codim runtime required.
Only selected material points receive target positions. No target mesh is exposed.
"""
from pathlib import Path
import os,sys,json,copy,hashlib
import numpy as np
from .control_geometry import rotate_vectors,midpoint_controls,rolling_offset
ROOT=Path(__file__).resolve().parents[1]
class ActionRejected(RuntimeError):pass
class LayerEnvironment:
 def __init__(self,mesh='horizontal',output=None,newton_tolerance=1e-4,barrier_distance=.001,max_newton_iterations=120,backend_variant='codim-ipc',crease_material=None,compression_barrier=False,grip_solve_tolerance=.00002,grip_solve_metric='l2'):
  if grip_solve_metric not in ('l2','rms'):raise ValueError('Unknown grip solver norm')
  self.grip_solve_tolerance=grip_solve_tolerance;self.grip_solve_metric=grip_solve_metric
  if type(compression_barrier)is not bool:raise ValueError('compression_barrier must be boolean')
  if not np.isfinite(grip_solve_tolerance) or not 0<grip_solve_tolerance<=.0005:raise ValueError('grip solver tolerance must stay within independent 0.5mm acceptance')
  if not np.isfinite(newton_tolerance) or not 0<newton_tolerance<=.01:raise ValueError('invalid Newton tolerance')
  if not np.isfinite(barrier_distance) or not 1e-5<=barrier_distance<=.01:raise ValueError('invalid contact barrier distance')
  if type(max_newton_iterations) is not int or not 20<=max_newton_iterations<=2000:raise ValueError('invalid Newton iteration limit')
  if backend_variant not in ('codim-ipc','codim-ipc-directional','codim-ipc-soft','codim-ipc-filtered','codim-ipc-filtered-density','codim-ipc-directional-density'):raise ValueError('unknown backend variant')
  backend=ROOT/'external'/backend_variant;sys.path[:0]=[str(backend/'build'),str(backend/'Python')]
  import Drivers,JGSL,ipctk
  if not Path(JGSL.__file__).resolve().is_relative_to((backend/'build').resolve()):raise RuntimeError('Use a fresh process when switching backend variants')
  if hasattr(JGSL,'Origami_DisableContactFilter'):JGSL.Origami_DisableContactFilter()
  if hasattr(JGSL,'Origami_SetYieldWeights'):JGSL.Origami_SetYieldWeights(JGSL.StdVectorXd([]),0)
  self.J=JGSL;self.ipc=ipctk;self.out=Path(output or ROOT/'outputs/codim-controller');self.out.mkdir(parents=True,exist_ok=True)
  path=ROOT/'tasks/crease_meshes'/f'{mesh}.json'
  if not path.exists():path=ROOT/'tasks/multifold_meshes'/f'{mesh}.json'
  self.meta=json.loads(path.read_text());uv=np.asarray(self.meta['vertices'],float);self.faces=np.asarray(self.meta['faces'],np.int32);self.rest=np.column_stack([uv,np.zeros(len(uv))]);self.grips={};self.trace=[];self.undo_states=[];self.time=0.
  obj=self.out/'initial.obj'
  with obj.open('w') as f:
   for v in self.rest:f.write('v '+' '.join(map(str,v))+'\n')
   for face in self.faces:f.write('f '+' '.join(str(i+1) for i in face)+'\n')
  (backend/'output').mkdir(exist_ok=True)
  os.chdir(backend)
  # Upstream concatenates every CLI argument into one log-directory component.
  # Keep that incidental path short; the experiment's real output is self.out.
  saved_argv=sys.argv
  sys.argv=['origami-controller','run-'+hashlib.sha256(str(self.out).encode()).hexdigest()[:16]]
  try:s=Drivers.FEMDiscreteShellBase('double',3)
  finally:sys.argv=saved_argv
  self.sim=s;s.output_folder=str(self.out)+'/'
  V=JGSL.Vector3d;s.add_shell_3D(str(obj),V(0,0,0),V(0,0,0),V(1,0,0),0);s.gravity=V(0,0,0);s.dt=1/60;s.frame_dt=s.dt;s.withCollision=True;s.PNTol=newton_tolerance
  s.initialize(800,3e9,.3,.0001,0);s.s=JGSL.Vector2d(1.009999 if compression_barrier else 1.01,.990001 if compression_barrier else 0);s.sHat=JGSL.Vector2d(1,1);s.kappa_s=JGSL.Vector2d(1e3,1e3 if compression_barrier else 0);s.initialize_OIPC(barrier_distance,.0001);JGSL.Origami_SetYieldMoment(1e-5);JGSL.Origami_SetMaxNewtonIterations(max_newton_iterations);JGSL.Origami_SetGripAbsoluteTolerance(grip_solve_tolerance)
  self.configuration=dict(mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),continuous_clearance_verifier='tight-inclusion+exact-plane-v3-piecewise',contact_barrier_parameter_m=barrier_distance,newton_velocity_RMS_tolerance_m_s=newton_tolerance,max_newton_iterations=max_newton_iterations,maximum_step_seconds=1/60,backend='Codim-IPC 9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a + local guarded extensions',mesh=mesh,youngs_modulus_Pa=3e9,poisson_ratio=.3,density_kg_m3=800,material_thickness_m=.0001,gravity=[0,0,0],friction=0,yield_moment_Nm_per_hinge=1e-5,contact_clearance_m=.0001,principal_strain_limit=.01,grip_absolute_L2_tolerance_m=grip_solve_tolerance if grip_solve_metric=='l2' else None,grip_acceptance_max_error_m=.0005,backend_binary_sha256=hashlib.sha256(Path(JGSL.__file__).read_bytes()).hexdigest())
  self.hinge_count=len(JGSL.Origami_GetHinges(s.X,s.edgeStencil));self.controls={};self.soft_controls={};JGSL.Origami_SetHingeControls([],self.hinge_count);self._set_soft_controls({});self.configuration['backend_variant']=backend_variant
  if backend_variant in ('codim-ipc-filtered-density','codim-ipc-directional-density'):
   reference_length=.005;weights=np.zeros(self.hinge_count)
   for i,st,_ in self.hinge_data():weights[i]=np.linalg.norm(self.rest[st[1]]-self.rest[st[2]])/reference_length
   JGSL.Origami_SetYieldWeights(JGSL.StdVectorXd(weights.tolist()),self.hinge_count)
   moment=self.configuration.pop('yield_moment_Nm_per_hinge')
   self.configuration['yield_density_experiment']=dict(reference_yield_moment_Nm=moment,reference_edge_length_m=reference_length,yield_moment_per_length_N=moment/reference_length,weights_min=float(weights.min()),weights_max=float(weights.max()),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),scope='yield moment proportional to rest material edge length; uncalibrated; rotational servo stiffness remains per hinge; not full mesh-objectivity certification')
  self.mesh=ipctk.CollisionMesh(self.rest,ipctk.edges(self.faces),self.faces);self.ccd=ipctk.TightInclusionCCD(tolerance=1e-8,max_iterations=1000000,conservative_rescaling=1.)
  self.contact_policy=None
  if backend_variant in ('codim-ipc-filtered','codim-ipc-filtered-density'):
   from .filtered_contact import ReferenceContactPolicy,certify_filtered_step
   self.contact_policy=ReferenceContactPolicy.from_flat_mesh(self.mesh,self.rest,offset=.0001,barrier_width=float(np.sqrt(s.dHat2)))
   keys=self.contact_policy.local_keys
   # Native shell contacts use complete PT/EE parents, not reduced closest features.
   pt=[list(k[1:]) for k in sorted(keys) if k[0]=='fv_candidates'];ee=[list(k[1:]) for k in sorted(keys) if k[0]=='ee_candidates']
   if any(k[0] not in ('fv_candidates','ee_candidates') for k in keys):raise ValueError('filtered native backend requires a pure triangle-shell collision mesh')
   JGSL.Origami_SetContactFilter(pt,ee,self.contact_policy.local_activation,len(self.rest))
   self.configuration['contact_policy']=self.contact_policy.metadata();self.configuration['continuous_clearance_verifier']='experimental-filtered-tight-inclusion+exact-plane-v3-piecewise';self.configuration['contact_clearance_m']=None
   initial=certify_filtered_step(self.mesh,self.rest,self.rest,self.contact_policy,ccd=self.ccd)
   if not initial['clear']:raise ActionRejected('Invalid filtered-contact initial state')
  else:
   from .contact_certification import certify_step
   initial=certify_step(self.mesh,self.rest,self.rest,min_distance=.0001,ccd=self.ccd)
   if not initial['clear']:raise ActionRejected('Initial mesh is incompatible with uniform100um contact; refine or select an explicitly different contact policy')
  self.configuration['initial_contact_certificate']=initial
  f=self.faces;self.inv=np.linalg.inv(np.stack([uv[f[:,1]]-uv[f[:,0]],uv[f[:,2]]-uv[f[:,0]]],axis=-1));self.frames=[self.rest.copy()];self.metrics=[];self.rejected_attempts=0;self.on_frame=None
  self.configuration.update(grip_solver_tolerance_m=grip_solve_tolerance,grip_solver_metric=grip_solve_metric,compression_barrier=compression_barrier,native_singular_value_limits=[s.s[i] for i in range(2)],native_strain_barrier_stiffness=[s.kappa_s[i] for i in range(2)])
  if crease_material is not None:
   from .crease_material import configure_line_material
   configure_line_material(self,crease_material)
 def positions(self):return np.array(self.J.Origami_GetPositions(self.sim.X),dtype=float)
 def _set_controls(self,controls):
  self.J.Origami_SetHingeControls([[i,*v] for i,v in controls.items()],self.hinge_count);self.controls=copy.deepcopy(controls)
 def _set_soft_controls(self,controls):
  if not hasattr(self.J,'Origami_SetSoftPointControls'):
   if controls:raise ActionRejected('This backend does not support compliant point controls')
  else:self.J.Origami_SetSoftPointControls([[i,*v] for i,v in controls.items()],len(self.rest))
  self.soft_controls=copy.deepcopy(controls)
 def hinge_data(self):return self.J.Origami_GetHinges(self.sim.X,self.sim.edgeStencil)
 def _snapshot(self):
  s=self.sim
  return dict(nodes=self.J.Origami_CloneNodes(s.X,s.nodeAttr),hinges=type(s.edgeInfo)(s.edgeInfo),ks=[s.kappa_s[i] for i in range(2)],kappa=[s.kappa[i] for i in range(3)],controls=copy.deepcopy(self.controls),soft_controls=copy.deepcopy(self.soft_controls),grips=copy.deepcopy(self.grips),time=self.time,frames=len(self.frames),metrics=len(self.metrics))
 def _restore(self,b):
  s=self.sim;s.X,s.nodeAttr=self.J.Origami_CloneNodes(*b['nodes']);s.edgeInfo=type(b['hinges'])(b['hinges']);s.kappa_s=self.J.Vector2d(*b['ks']);s.kappa=self.J.Vector3d(*b['kappa']);self.grips=copy.deepcopy(b['grips']);self.time=b['time'];s.t=self.time;self.frames=self.frames[:b['frames']];self.metrics=self.metrics[:b['metrics']]
  self._set_controls(b['controls']);self._set_soft_controls(b.get('soft_controls',{}))
  s.DBC=self.J.Origami_MakeGrips(self.grips,len(self.rest));s.DBCMotion=self.J.Storage.V2iV3dV3dV3dSdStorage()
 def _advance(self,targets,h,depth=0,controls=None,soft_controls=None):
  self.J.Origami_SetGripAbsoluteTolerance(self.grip_solve_tolerance*(np.sqrt(len(targets)) if self.grip_solve_metric=='rms' else 1))
  snap=self._snapshot();before=self.positions();s=self.sim;s.DBC=self.J.Origami_MakeGrips(targets,len(self.rest));s.DBCMotion=self.J.Storage.V2iV3dV3dV3dSdStorage();s.dt=h
  if controls is not None:self._set_controls(controls)
  if soft_controls is not None:self._set_soft_controls(soft_controls)
  try:
   s.advance_one_time_step(h);v=self.positions()
   if not np.isfinite(v).all():raise ActionRejected('nonfinite positions')
   f=self.faces;Ds=np.stack([v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]],axis=-1);sv=np.linalg.svd(Ds@self.inv,compute_uv=False);stretch=float(sv.max()-1);compression=float(1-sv.min())
   if max(stretch,compression)>=.01:raise ActionRejected('principal strain reached 1% limit')
   from .contact_certification import certify_step
   if self.contact_policy is None:clearance=certify_step(self.mesh,before,v,min_distance=.0001,ccd=self.ccd)
   else:
    from .filtered_contact import certify_filtered_step
    clearance=certify_filtered_step(self.mesh,before,v,self.contact_policy,ccd=self.ccd)
   if not clearance['clear']:raise ActionRejected('100um continuous clearance not certified' if self.contact_policy is None else 'filtered continuous contact not certified')
   griperror=max((np.linalg.norm(v[i]-p) for i,p in targets.items()),default=0.)
   if griperror>.0005:raise ActionRejected('grip did not reach its target')
   count,diss=self.J.Origami_CommitPlasticity(s.X,s.edgeStencil,s.edgeInfo,s.elemAttr,s.bendingStiffMult)
  except (RuntimeError,ValueError) as e:
   if depth>=6:
    try:
     diagnostic=self.out/'rejected-candidates';diagnostic.mkdir(exist_ok=True)
     np.savez_compressed(diagnostic/f'rejected-{self.rejected_attempts:06d}.npz',before=before,candidate=self.positions(),rest=self.rest,faces=self.faces,dt=h,grip_ids=np.array(list(targets),dtype=int),grip_targets=np.array(list(targets.values())),reason=str(e))
    except OSError:pass
   self._restore(snap);self.rejected_attempts+=1
   if depth>=6:raise ActionRejected(str(e)) from e
   half={i:((np.asarray(snap['grips'][i])+p)*.5).tolist() for i,p in targets.items()}
   half_controls=None if controls is None else {i:((np.array(snap['controls'][i])+v)*.5).tolist() for i,v in controls.items()}
   half_soft=None if soft_controls is None else midpoint_controls(snap['soft_controls'],soft_controls)
   self._advance(half,h/2,depth+1,half_controls,half_soft);self._advance(targets,h/2,depth+1,controls,soft_controls);return
  self.grips=copy.deepcopy(targets);self.time+=h;s.t=self.time;self.frames.append(v.copy());self.metrics.append(dict(time=self.time,dt=h,stretch=stretch,compression=compression,grip_error=griperror,plastic_hinges=count,dissipation_joules=diss,crease_controls=copy.deepcopy(self.controls),soft_controls=copy.deepcopy(self.soft_controls),soft_target_error=max((float(np.linalg.norm(v[i]-c[:3])) for i,c in self.soft_controls.items()),default=0.),clear=True,clearance=clearance))
  if self.on_frame is not None:self.on_frame(self.state())
 def state(self):
  v=self.positions();return dict(vertices=v.tolist(),faces=self.faces.tolist(),grips=sorted(self.grips),soft_controls=copy.deepcopy(self.soft_controls),crease_controls=copy.deepcopy(self.controls),time=self.time,steps=len(self.frames)-1,last_metrics=self.metrics[-1] if self.metrics else None,rejected_attempts=self.rejected_attempts,hash=hashlib.sha256(v.tobytes()+json.dumps([self.grips,self.controls]+([self.soft_controls] if self.soft_controls else []),sort_keys=True).encode()).hexdigest())
 def step(self,action):
  if not isinstance(action,dict):raise ActionRejected('action must be an object')
  op=action.get('op');schemas={'set_grips':{'op','vertices'},'release':{'op','vertices'},'move_grips':{'op','targets','frames'},'rotate_grips':{'op','vertices','axis_point','axis_direction','degrees','frames','roll_radii','roll_tangent','roll_start_angle_deg'},'attach_creases':{'op','hinges','stiffness','stiffness_per_length'},'drive_creases':{'op','targets','frames'},'release_creases':{'op','hinges'},'rotate_soft_grips':{'op','vertices','axis_point','axis_direction','degrees','frames'},'attach_soft_grips':{'op','vertices','stiffness','normal'},'move_soft_grips':{'op','targets','frames'},'release_soft_grips':{'op','vertices'},'wait':{'op','frames'},'undo':{'op'}}
  if op not in schemas or set(action)-schemas[op]:raise ActionRejected('unknown action or fields')
  if op=='undo':
   if not self.undo_states:raise ActionRejected('nothing to undo')
   self._restore(self.undo_states.pop());self.trace.pop();return self.state()
  snap=self._snapshot()
  try:
   if op in ('attach_soft_grips','release_soft_grips'):
    ids=action.get('vertices',list(self.soft_controls) if op=='release_soft_grips' else None)
    if not isinstance(ids,list) or any(type(i)is not int or i<0 or i>=len(self.rest) for i in ids) or len(set(ids))!=len(ids):raise ActionRejected('invalid compliant point ids')
    updated=copy.deepcopy(self.soft_controls)
    if op=='attach_soft_grips':
     stiffness=action.get('stiffness',10.)
     if type(stiffness) not in (int,float) or not np.isfinite(stiffness) or stiffness<=0:raise ActionRejected('invalid compliant stiffness')
     if any(i in self.grips for i in ids):raise ActionRejected('A material point cannot have both hard and compliant grips')
     normal=action.get('normal');direction=[]
     if normal is not None:
      n=np.asarray(normal,dtype=float)
      if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-6:raise ActionRejected('normal must be a finite unit vector')
      if not hasattr(self.J,'Origami_DirectionalPointResponse'):raise ActionRejected('This backend does not support directional point controls')
      direction=n.tolist()
     positions=self.positions()
     for i in ids:updated[i]=[*positions[i].tolist(),float(stiffness),*direction]
    else:
     for i in ids:updated.pop(i,None)
    self._set_soft_controls(updated)
   elif op in ('attach_creases','release_creases'):
    ids=action.get('hinges',list(self.controls) if op=='release_creases' else None)
    if not isinstance(ids,list) or any(type(i)is not int or i<0 or i>=self.hinge_count for i in ids) or len(set(ids))!=len(ids):raise ActionRejected('invalid hinge ids')
    updated=copy.deepcopy(self.controls)
    if op=='attach_creases':
     density='stiffness_per_length' in action
     if density and 'stiffness' in action:raise ActionRejected('Choose per-hinge stiffness or stiffness per material length, not both')
     k=action['stiffness_per_length'] if density else action.get('stiffness',.01)
     if type(k) not in (int,float) or not np.isfinite(k) or k<=0:raise ActionRejected('invalid rotational servo stiffness')
     geometry={i:(angle,float(np.linalg.norm(self.rest[st[1]]-self.rest[st[2]]))) for i,st,angle in self.hinge_data()}
     for i in ids:
      angle,length=geometry[i];effective=k*length if density else k
      if not np.isfinite(effective) or effective<=0:raise ActionRejected('invalid effective rotational servo stiffness')
      updated[i]=[angle,effective]
    else:
     for i in ids:updated.pop(i,None)
    self._set_controls(updated)
   elif op in ('set_grips','release'):
    ids=action.get('vertices',list(self.grips) if op=='release' else None)
    if not isinstance(ids,list) or any(type(i) is not int or i<0 or i>=len(self.rest) for i in ids) or len(set(ids))!=len(ids):raise ActionRejected('invalid material vertex ids')
    if op=='set_grips' and any(i in self.soft_controls for i in ids):raise ActionRejected('A material point cannot have both hard and compliant grips')
    v=self.positions()
    for i in ids:
     if op=='set_grips':self.grips[i]=v[i].tolist()
     else:self.grips.pop(i,None)
   else:
    frames=action.get('frames',30)
    if type(frames)is not int or not 1<=frames<=600:raise ActionRejected('frames must be 1..600')
    start=copy.deepcopy(self.grips);finish=copy.deepcopy(start)
    if op=='move_grips':
     raw=action.get('targets')
     if not isinstance(raw,dict) or not raw:raise ActionRejected('targets required')
     for key,p in raw.items():
      if not isinstance(key,(str,int)) or isinstance(key,bool):raise ActionRejected('invalid grip id')
      i=int(key)
      if str(i)!=str(key) or i not in start:raise ActionRejected('target must refer to an existing grip')
      point=np.asarray(p,dtype=float)
      if point.shape!=(3,) or not np.isfinite(point).all() or np.linalg.norm(point-start[i])/frames>.001:raise ActionRejected('invalid target or motion exceeds 1mm/frame')
      finish[i]=point.tolist()
    if op in ('rotate_grips','rotate_soft_grips'):
     rotation_start={i:c[:3] for i,c in self.soft_controls.items()} if op=='rotate_soft_grips' else start
     ids=action.get('vertices');center=np.asarray(action.get('axis_point'),dtype=float);axis=np.asarray(action.get('axis_direction'),dtype=float);degrees=action.get('degrees')
     if not isinstance(ids,list) or not ids or any(type(i)is not int or i not in rotation_start for i in ids) or len(set(ids))!=len(ids):raise ActionRejected('rotation requires distinct fixed material vertices')
     if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-10:raise ActionRejected('invalid rotation axis')
     if type(degrees) not in (int,float) or not np.isfinite(degrees) or abs(degrees)>360:raise ActionRejected('rotation degrees must be finite within 360')
     axis=axis/np.linalg.norm(axis);angle=np.deg2rad(degrees);w=np.array([rotation_start[i] for i in ids])-center
     if np.linalg.norm(np.cross(axis,w),axis=1).max()*abs(angle)/frames>.001:raise ActionRejected('rotation exceeds 1mm/frame; increase frames')
     if op=='rotate_soft_grips' and any(len(self.soft_controls[i])==7 for i in ids) and abs(degrees)/frames>2:raise ActionRejected('Directional rotation exceeds 2 degrees per frame')
    roll=None
    if 'roll_radii' in action:
     raw=action['roll_radii'];tangent=np.asarray(action.get('roll_tangent'),float);start_angle=action.get('roll_start_angle_deg')
     if not isinstance(raw,dict) or set(raw)!=set(map(str,ids)):raise ActionRejected('Rolling radii must cover exactly the moving grips')
     radii=np.array([raw[str(i)] for i in ids],float)
     if not np.isfinite(radii).all() or (radii<0).any() or (radii>.002).any():raise ActionRejected('Rolling radius outside validated 0..2mm range')
     if tangent.shape!=(3,) or not np.isfinite(tangent).all() or abs(np.linalg.norm(tangent)-1)>1e-6 or abs(tangent@axis)>1e-6:raise ActionRejected('Rolling tangent must be unit and perpendicular to axis')
     if type(start_angle)not in (int,float) or not np.isfinite(start_angle) or abs(start_angle)>180 or abs(start_angle+degrees)>180:raise ActionRejected('Rolling fold angle outside +/-180 degrees')
     roll=(tangent,radii,np.deg2rad(start_angle));roll0=rolling_offset(tangent,axis,roll[2],radii)
    soft_start=copy.deepcopy(self.soft_controls);soft_finish=copy.deepcopy(soft_start)
    if op=='move_soft_grips':
     raw=action.get('targets')
     if not isinstance(raw,dict) or not raw:raise ActionRejected('compliant targets required')
     for key,point in raw.items():
      if not isinstance(key,(str,int)) or isinstance(key,bool):raise ActionRejected('invalid compliant point id')
      i=int(key)
      if str(i)!=str(key) or i not in soft_start:raise ActionRejected('target must refer to an attached compliant grip')
      q=np.asarray(point,dtype=float)
      if q.shape!=(3,) or not np.isfinite(q).all() or np.linalg.norm(q-soft_start[i][:3])/frames>.001:raise ActionRejected('invalid compliant target or motion exceeds 1mm/frame')
      soft_finish[i][:3]=q.tolist()
    control_start=copy.deepcopy(self.controls);control_finish=copy.deepcopy(control_start)
    if op=='drive_creases':
     raw=action.get('targets')
     if not isinstance(raw,dict) or not raw:raise ActionRejected('crease targets required')
     for key,degrees in raw.items():
      if not isinstance(key,(str,int)) or isinstance(key,bool):raise ActionRejected('invalid hinge id')
      i=int(key)
      if str(i)!=str(key) or i not in control_start or type(degrees) not in (int,float) or not np.isfinite(degrees) or abs(degrees)>180:raise ActionRejected('invalid crease target')
      angle=float(np.deg2rad(degrees))
      if abs(angle-control_start[i][0])/frames>np.deg2rad(2):raise ActionRejected('crease change exceeds 2 degrees per frame')
      control_finish[i][0]=angle
    for k in range(1,frames+1):
     targets={i:(np.asarray(start[i])+(np.asarray(finish[i])-start[i])*k/frames).tolist() for i in start}
     if op=='rotate_grips':
      a=angle*k/frames;points=center+w*np.cos(a)+np.cross(axis,w)*np.sin(a)+np.outer(w@axis,axis)*(1-np.cos(a))
      if roll is not None:points+=rolling_offset(roll[0],axis,roll[2]+a,roll[1])-rotate_vectors(roll0,axis,a)
      targets.update({i:p.tolist() for i,p in zip(ids,points)})
     control_targets=None if op!='drive_creases' else {i:(np.array(control_start[i])+(np.array(control_finish[i])-control_start[i])*k/frames).tolist() for i in control_start}
     soft_targets=None if op not in ('move_soft_grips','rotate_soft_grips') else {i:(np.array(soft_start[i])+(np.array(soft_finish[i])-soft_start[i])*k/frames).tolist() for i in soft_start}
     if op=='rotate_soft_grips':
      a=angle*k/frames;points=center+w*np.cos(a)+np.cross(axis,w)*np.sin(a)+np.outer(w@axis,axis)*(1-np.cos(a))
      for i,p in zip(ids,points):soft_targets[i][:3]=p.tolist()
      directional=[i for i in ids if len(soft_start[i])==7]
      if directional:
       normals=rotate_vectors([soft_start[i][4:] for i in directional],axis,a)
       for i,n in zip(directional,normals):soft_targets[i][4:]=n.tolist()
     self._advance(targets,1/60,controls=control_targets,soft_controls=soft_targets)
  except Exception:
   self._restore(snap);raise
  self.undo_states.append(snap);self.trace.append(copy.deepcopy(action));return self.state()
 def save(self,path):
  p=Path(path);p.mkdir(parents=True,exist_ok=True);(p/'model-config.json').write_text(json.dumps(self.configuration,indent=2));np.savez_compressed(p/'trajectory.npz',vertices=np.array(self.frames),faces=self.faces);(p/'actions.json').write_text(json.dumps(self.trace,indent=2));(p/'metrics.json').write_text(json.dumps(self.metrics,indent=2))
