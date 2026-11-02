"""Private, ordered shape checkpoints; no controller or policy dependencies.
Accept any action decomposition that reaches the diagram's illustrated states.
Search only retained forward physical frames, starting at the prepared state.
"""
import numpy as np
from .document_pose_audit import compare_in_plane

def ordered_checkpoints(vertices,checkpoints,start_frame=0):
 v=np.asarray(vertices,float);targets=np.asarray(checkpoints,float)
 if v.ndim!=3 or targets.ndim!=3 or v.shape[1:]!=targets.shape[1:] or not np.isfinite(v).all() or not np.isfinite(targets).all():raise ValueError('Finite matching trajectory/checkpoint geometry required')
 if type(start_frame)is not int or not 0<=start_frame<len(v):raise ValueError('Invalid prepared-state frame')
 cursor=start_frame;matches=[]
 for target in targets:
  best=None;found=False
  for j in range(cursor,len(v)):
   shape=compare_in_plane(v[j],target)
   if best is None or shape['rms_mm']<best['rms_mm']:best=dict(frame=j,rms_mm=shape['rms_mm'],max_mm=shape['max_mm'])
   if shape['rms_mm']<1 and shape['max_mm']<3:
    matches.append(dict(frame=j,rms_mm=shape['rms_mm'],max_mm=shape['max_mm']));cursor=j;found=True;break
  if not found:return dict(passed=False,matched=len(matches),required=len(targets),matches=matches,next_best=best,criterion='ordered physical states; RMS<1mm,max<3mm; XY rigid placement allowed')
 return dict(passed=True,matched=len(matches),required=len(targets),matches=matches,criterion='ordered physical states; RMS<1mm,max<3mm; XY rigid placement allowed')
