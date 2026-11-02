"""Read ARCSim's OBJ material coordinates and changing triangular meshes."""
from pathlib import Path
import numpy as np

def read_obj(path):
    vertices=[];uv=[];faces=[];material_faces=[];plastic=[];damage=[]
    for line in Path(path).read_text().splitlines():
        t=line.split()
        if not t:continue
        if t[0]=='v':vertices.append([float(x) for x in t[1:4]])
        elif t[0]=='vt':uv.append([float(x) for x in t[1:3]])
        elif t[0]=='f':
            ids=[x.split('/') for x in t[1:]]
            if len(ids)!=3:raise ValueError('Expected triangular ARCSim output')
            faces.append([int(x[0])-1 for x in ids]);material_faces.append([int(x[1])-1 for x in ids]);plastic.append([0.]*4);damage.append(0.)
        elif t[0] in ('fs','ts'):plastic[-1]=[float(x) for x in t[1:5]]
        elif t[0] in ('fd','td'):damage[-1]=float(t[1])
    return dict(vertices=np.array(vertices),faces=np.array(faces,dtype=np.int32),uv=np.array(uv),material_faces=np.array(material_faces,dtype=np.int32),plastic_curvature=np.array(plastic).reshape(-1,2,2),damage=np.array(damage))

def material_metrics(data):
    v,f,u,g=(data[k] for k in ('vertices','faces','uv','material_faces'))
    dm=np.stack((u[g[:,1]]-u[g[:,0]],u[g[:,2]]-u[g[:,0]]),axis=-1)
    ds=np.stack((v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=-1)
    sv=np.linalg.svd(ds@np.linalg.inv(dm),compute_uv=False)
    return dict(vertices=len(v),triangles=len(f),material_area_m2=float(abs(np.linalg.det(dm)).sum()/2),peak_stretch=float(sv.max()-1),peak_compression=float(1-sv.min()),max_plastic_curvature_per_m=float(np.linalg.norm(data['plastic_curvature'],axis=(1,2)).max()),max_damage=float(data['damage'].max()))

def sample_material(data, points):
    """Interpolate on the actual mesh, never substitute a Delaunay triangulation."""
    u,g,v,f=(data[k] for k in ('uv','material_faces','vertices','faces'))
    out=np.full((len(points),3),np.nan);remaining=np.ones(len(points),bool)
    for face,mat in zip(f,g):
        tri=u[mat];ids=np.flatnonzero(remaining & (points>=tri.min(0)-1e-9).all(1) & (points<=tri.max(0)+1e-9).all(1))
        if not len(ids):continue
        ab=np.linalg.solve((tri[1:]-tri[0]).T,(points[ids]-tri[0]).T).T
        w=np.column_stack((1-ab.sum(1),ab));ok=(w>=-1e-8).all(1);chosen=ids[ok]
        out[chosen]=w[ok]@v[face];remaining[chosen]=False
    if remaining.any():raise ValueError('Material samples outside saved mesh')
    return out
