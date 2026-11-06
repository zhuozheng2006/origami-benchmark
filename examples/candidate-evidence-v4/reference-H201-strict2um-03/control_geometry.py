"""Geometry of physical compliant controls; no task information."""
import numpy as np

def rotate_vectors(vectors,axis,angle):
    v=np.asarray(vectors,float);axis=np.asarray(axis,float)
    return v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+np.outer(v@axis,axis)*(1-np.cos(angle))

def midpoint_controls(start,end):
    result={i:((np.array(start[i])+v)*.5).tolist() for i,v in end.items()}
    for c in result.values():
        if len(c)==7:
            n=np.linalg.norm(c[4:])
            if n<1e-12:raise ValueError('Opposing directional controls cannot be interpolated')
            c[4:]=(np.array(c[4:])/n).tolist()
    return result

def rolling_offset(tangent,axis,angle,radii):
    """Symmetric circular bend: reserve arc length in the free crease band.

    Compared with rotating around a fixed offset axis, the moving panel has
    transient offset -R*abs(theta)/2*(t0+t(theta)). It vanishes at 0 and pi.
    This is a control proposal, never a physical state assignment.
    """
    t=np.asarray(tangent,float)
    return -.5*abs(angle)*np.asarray(radii)[:,None]*(t+rotate_vectors(t[None],axis,angle)[0])
