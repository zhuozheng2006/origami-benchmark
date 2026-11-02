"""Separate diagnostic: remove only global XY translation/proper z rotation.
The frozen primary grader and reports are never changed by this audit.
"""
import numpy as np

def compare_in_plane(actual,expected):
 a=np.asarray(actual,float);b=np.asarray(expected,float);ac=a[:,:2].mean(0);bc=b[:,:2].mean(0);u,_,vt=np.linalg.svd((a[:,:2]-ac).T@(b[:,:2]-bc));r=u@vt
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
 fit=np.column_stack(((a[:,:2]-ac)@r+bc,a[:,2]));d=np.linalg.norm(fit-b,axis=1);raw=np.linalg.norm(a-b,axis=1)
 return dict(rms_mm=float(np.sqrt(np.mean(d*d))*1000),max_mm=float(d.max()*1000),raw_rms_mm=float(np.sqrt(np.mean(raw*raw))*1000),rotation_degrees=float(np.rad2deg(np.arctan2(r[0,1],r[0,0]))),xy_translation_mm=((bc-ac@r)*1000).tolist(),reflection_allowed=False,z_translation_allowed=False)
