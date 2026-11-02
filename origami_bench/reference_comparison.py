"""Read-only target alignment for visualization; never used to drive the simulator."""
import numpy as np

def compare_reference(actual,reference,world_pose=False):
    actual=np.asarray(actual,dtype=float);reference=np.asarray(reference,dtype=float)
    if actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=3 or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Matching finite 3D material coordinates required')
    aligned=reference.copy()
    if not world_pose:
        x=reference-reference.mean(0);y=actual-actual.mean(0);u,_,vt=np.linalg.svd(x.T@y)
        d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);aligned=x@u@d@vt+actual.mean(0)
    error=np.linalg.norm(actual-aligned,axis=1)*1000
    return aligned,dict(rms_mm=float(np.sqrt(np.mean(error**2))),max_mm=float(error.max()),alignment='world' if world_pose else 'proper rigid rotation and translation')
