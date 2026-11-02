"""Independent signed dihedral tracking through the +/- pi branch cut."""
import numpy as np

def continuous_dihedral_degrees(trajectory, stencil):
 x=np.asarray(trajectory,float)[:,stencil]
 a,b,c,d=x[:,0],x[:,1],x[:,2],x[:,3]
 n=np.cross(b-a,c-a);m=np.cross(c-d,b-d);axis=b-c
 lengths=np.linalg.norm(axis,axis=1)
 if np.any(lengths<1e-12) or np.any(np.linalg.norm(n,axis=1)<1e-15) or np.any(np.linalg.norm(m,axis=1)<1e-15):raise ValueError('degenerate audit triangle')
 axis=axis/lengths[:,None]
 angles=np.arctan2(np.sum(np.cross(m,n)*axis,axis=1),np.sum(n*m,axis=1))
 return np.rad2deg(np.unwrap(angles))
