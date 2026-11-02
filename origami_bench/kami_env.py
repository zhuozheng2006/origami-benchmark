"""Experimental shared physics controller. No certification implied by numerical completion.
Rollback reconstructs the full simulator by deterministic action replay, not partial tensors.
"""
from pathlib import Path
import os,sys,copy,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'outputs/warp-cache'))
sys.path.insert(0,str(ROOT/'external/kami'))
import kami
REVISION='9e7754bd458395e46640385da511d7c94be8cffc'
class KamiEnvironment:
 def __init__(self,N=17):
  self.N=N;self.trace=[];self.on_frame=None;self._new()
 def _new(self):
  self.sheet=kami.Sheet(kami.Paper.origami_70gsm(),kami.Sim(N=self.N,device='cpu'),batch=1)
 def state(self):
  return dict(vertices=self.sheet.positions[0].cpu().numpy().copy(),faces=self.sheet.topo.tris.copy(),time=float(self.sheet.time[0]),grabs=self.sheet.grabbed[0].cpu().tolist())
 def _advance(self,frames=1):
  for _ in range(frames):
   self.sheet.step()
   if self.on_frame is not None:self.on_frame()
 def _ids(self,a):
  ids=a['vertices']
  if not isinstance(ids,list) or not ids or any(type(i)!=int or i<0 or i>=self.N*self.N for i in ids):raise ValueError('invalid material vertex ids')
  return ids
 def _point(self,a,key):
  p=np.asarray(a[key],dtype=float)
  if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('point must be finite xyz')
  return p
 def _apply(self,a):
  s=self.sheet;op=a['op']
  fields={'pin':{'vertices'},'unpin':{'vertices'},'move_pins':{'vertices','delta','frames'},'grab':{'slot','point'},'move_grab':{'slot','point','frames'},'release':{'slot'},'wait':{'frames'},'press':{'a','b','radius','force','frames'},'remove_tool':set(),'move_grasps':{'targets','frames'},'rotate_pins':{'vertices','center','axis','degrees','frames'}}
  if op not in fields or set(a)!=(fields[op]|{'op'}):raise ValueError('unsupported action or fields')
  frames=a.get('frames',1)
  if type(frames)!=int or not 1<=frames<=600:raise ValueError('frames must be 1..600')
  if 'slot' in a and (type(a['slot'])!=int or not 0<=a['slot']<s.K):raise ValueError('invalid grab slot')
  if op=='pin':s.pin(self._ids(a));self._advance()
  elif op=='unpin':s.unpin(self._ids(a));self._advance()
  elif op=='move_pins':
   ids=self._ids(a);d=self._point(a,'delta')
   if np.linalg.norm(d)/frames>.001:raise ValueError('motion exceeds 1 mm per frame; increase frames')
   if not all(int(s.T['pin_on'][i,0]) for i in ids):raise ValueError('pin selected vertices first')
   p=s.positions.clone()
   for i in range(frames):
    target=p.clone();target[:,ids]+=target.new_tensor(d*(i+1)/frames);s.move_pins(target,ids);self._advance()
  elif op=='grab':s.grab(a['slot'],self._point(a,'point'));self._advance()
  elif op=='move_grab':
   slot=a['slot'];p=self._point(a,'point');v=int(s.grabbed[0,slot])
   if v<0:raise ValueError('grab slot is empty')
   start=s.positions[0,v].cpu().numpy().copy()
   if np.linalg.norm(p-start)/frames>.001:raise ValueError('motion exceeds 1 mm per frame')
   for i in range(frames):s.move_grab(slot,start+(p-start)*(i+1)/frames);self._advance()
  elif op=='move_grasps':
   targets=a['targets']
   if not isinstance(targets,list) or not targets:raise ValueError('targets must be nonempty')
   motions=[];slots=set()
   for t in targets:
    if set(t)!={'slot','point'} or type(t['slot'])!=int or not 0<=t['slot']<s.K or t['slot'] in slots:raise ValueError('invalid or duplicate grab slot')
    slot=t['slot'];slots.add(slot);v=int(s.grabbed[0,slot])
    if v<0:raise ValueError('grab slot is empty')
    start=s.positions[0,v].cpu().numpy().copy();end=self._point(t,'point')
    if np.linalg.norm(end-start)/frames>.001:raise ValueError('motion exceeds 1 mm per frame')
    motions.append((slot,start,end))
   for i in range(frames):
    for slot,start,end in motions:s.move_grab(slot,start+(end-start)*(i+1)/frames)
    self._advance()
  elif op=='rotate_pins':
   ids=self._ids(a);center=self._point(a,'center');axis=self._point(a,'axis');norm=np.linalg.norm(axis)
   if norm<1e-10:raise ValueError('rotation axis is zero')
   axis/=norm;degrees=a['degrees']
   if not isinstance(degrees,(int,float)) or not math.isfinite(degrees) or abs(degrees)>180:raise ValueError('rotation must be within 180 degrees')
   if not all(int(s.T['pin_on'][i,0]) for i in ids):raise ValueError('pin selected vertices first')
   p0=s.positions.clone();r=p0[0,ids].cpu().numpy()-center;angle=math.radians(degrees)
   if np.linalg.norm(np.cross(axis,r),axis=1).max()*abs(angle)/frames>.001:raise ValueError('arc speed exceeds 1 mm per frame')
   for i in range(frames):
    phi=angle*(i+1)/frames
    pos=r*math.cos(phi)+np.cross(axis,r)*math.sin(phi)+np.outer(r@axis,axis)*(1-math.cos(phi))+center
    target=p0.clone();target[0,ids]=target.new_tensor(pos);s.move_pins(target,ids);self._advance()
  elif op=='release':s.release(a['slot']);self._advance()
  elif op=='wait':self._advance(frames)
  elif op=='press':
   pa=self._point(a,'a');pb=self._point(a,'b');r=a['radius'];force=a['force']
   if not isinstance(r,(int,float)) or not math.isfinite(r) or not .001<=r<=.03:raise ValueError('radius outside 1–30 mm')
   if not isinstance(force,(int,float)) or not math.isfinite(force) or not 0<=force<=20:raise ValueError('force outside 0–20 N')
   s.set_tool(0,pa,pb,radius=r,force=force);self._advance(frames)
  elif op=='remove_tool':s.remove_tool();self._advance()
  if not np.isfinite(self.state()['vertices']).all():raise RuntimeError('nonfinite simulation')
 def restore(self,trace):
  callback=self.on_frame;self.on_frame=None
  try:
   self._new()
   for a in trace:self._apply(a)
   self.trace=copy.deepcopy(trace)
  finally:self.on_frame=callback
 def step(self,action):
  a=copy.deepcopy(action)
  if a=={'op':'undo'}:
   if not self.trace:raise ValueError('nothing to undo')
   self.restore(self.trace[:-1]);return self.state()
  before=copy.deepcopy(self.trace)
  try:self._apply(a)
  except Exception:
   self.restore(before);raise
  self.trace.append(a);return self.state()
