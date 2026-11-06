"""Private kinematic annotation check; never imported by the agent controller."""
import numpy as np
from .layer_controller import rotation

class IdealTrace:
 def __init__(self,vertices,faces):
  self.vertices=np.array(vertices,float);self.faces=np.array(faces,int);self.folds={};self.checks=[]
 def apply(self,record,actual):
  d=record['details'];op=record['action']['op']
  if op=='fold_line':
   ids=np.unique(self.faces[d['selected_faces']]);pivot=np.array(d['axis_point']);axis=np.array(d['axis_direction']);angle=d['degrees']
   self.vertices[ids]=rotation(self.vertices[ids],pivot,axis,angle)
   self.folds[d['fold_id']]=dict(ids=ids,pivot=pivot,axis=axis,angle=angle)
  elif op=='unfold':
   fold=self.folds[d['fold_id']];amount=d['unfold_angle_deg'];delta=-np.copysign(amount,fold['angle'])
   self.vertices[fold['ids']]=rotation(self.vertices[fold['ids']],fold['pivot'],fold['axis'],delta)
   fold['angle']+=delta
   if abs(fold['angle'])<1e-8:del self.folds[d['fold_id']]
  elif op in ('turn_over','rotate_paper'):
   pivot=np.array(d['axis_point']);axis=np.array(d['axis_direction']);angle=d['degrees']
   self.vertices=rotation(self.vertices,pivot,axis,angle)
   for fold in self.folds.values():
    fold['pivot']=rotation(fold['pivot'][None],pivot,axis,angle)[0];fold['axis']=rotation(fold['axis'][None],np.zeros(3),axis,angle)[0]
  else:raise ValueError('Reference annotations must use forward physical actions')
  if actual is None:return None
  error=np.linalg.norm(np.asarray(actual)-self.vertices,axis=1)
  result=dict(op=op,rms_mm=float(np.sqrt(np.mean(error**2))*1000),max_mm=float(error.max()*1000))
  self.checks.append(result);return result
