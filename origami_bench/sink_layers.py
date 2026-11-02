"""Observable sink semantics; these necessary checks do not prove an origami type."""
import numpy as np


def cap_faces(uv, faces, boundary=(8, 9, 11), radius_l1=None):
    uv=np.asarray(uv); faces=np.asarray(faces)
    if radius_l1 is not None:return abs(uv[faces].mean(1)).sum(1)<radius_l1+1e-10
    tri=uv[list(boundary)]
    bc=np.linalg.solve(np.column_stack((tri[1]-tri[0],tri[2]-tri[0])),
                       (uv[faces].mean(1)-tri[0]).T).T
    return (bc.min(1)>=-1e-9)&(bc.sum(1)<=1+1e-9)


def normal_coherence(vertices, faces):
    triangles=np.asarray(vertices)[...,np.asarray(faces),:]
    area=np.cross(triangles[...,1,:]-triangles[...,0,:],triangles[...,2,:]-triangles[...,0,:])
    return np.linalg.norm(area.sum(-2),axis=-1)/np.maximum(np.linalg.norm(area,axis=-1).sum(-1),1e-20)


def enclosing_layers(vertices, faces, cap_mask):
    """Vertical rays through actual cap-face centroids; coordinates must be target-aligned."""
    triangles=np.asarray(vertices)[np.asarray(faces)]
    cap=triangles[cap_mask]; body=triangles[~cap_mask]
    weights=np.linalg.norm(np.cross(cap[:,1]-cap[:,0],cap[:,2]-cap[:,0]),axis=1)
    centres=cap.mean(1); enclosed=[]; counts=[]
    for p in centres:
        heights=[]
        for tri in body:
            A=np.column_stack((tri[1,:2]-tri[0,:2],tri[2,:2]-tri[0,:2]))
            if abs(np.linalg.det(A))<1e-14:continue
            u,v=np.linalg.solve(A,p[:2]-tri[0,:2])
            if min(u,v)>=-1e-8 and u+v<=1+1e-8:
                heights.append(float(tri[0,2]+u*(tri[1,2]-tri[0,2])+v*(tri[2,2]-tri[0,2])))
        above=int(sum(h>p[2]+1e-6 for h in heights));below=int(sum(h<p[2]-1e-6 for h in heights))
        enclosed.append(bool(above and below));counts.append([below,above])
    return dict(enclosed_area_fraction=float(np.clip(np.dot(weights,enclosed)/max(weights.sum(),1e-20),0,1)),
                cap_faces=len(cap),enclosed_faces=sum(enclosed),body_intersection_counts=counts,
                scope='Actual face-centroid rays bracketed by body surfaces; necessary containment check, not a complete layer-order certificate.')
