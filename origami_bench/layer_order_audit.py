"""Read-only layer-order observation on simulated triangles, in a fixed reference plane.

Material triangles are clipped against coarse faces before projection, so an
artificial coarse diagonal never assigns an entire crossing fine face incorrectly.
No reference position is substituted for any simulated vertex.
"""
import numpy as np


def cross2(a,b):
    return a[0]*b[1]-a[1]*b[0]


def area(poly):
    p=np.asarray(poly)
    return abs(sum(cross2(a,b) for a,b in zip(p,np.roll(p,-1,axis=0))))/2 if len(p)>=3 else 0.


def clip_polygon(subject, boundary, eps=1e-12):
    p=np.asarray(subject,dtype=float);q=np.asarray(boundary,dtype=float)
    if len(p)<3:return np.empty((0,2))
    if sum(cross2(a,b) for a,b in zip(q,np.roll(q,-1,axis=0)))<0:q=q[::-1]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        if len(p)==0:break
        out=[];edge=b-a
        for s,e in zip(p,np.roll(p,-1,axis=0)):
            ds=cross2(edge,s-a);de=cross2(edge,e-a);inside_s=ds>=-eps;inside_e=de>=-eps
            if inside_s:out.append(s)
            if inside_s!=inside_e and abs(ds-de)>1e-25:out.append(s+(e-s)*(ds/(ds-de)))
        p=np.asarray(out,dtype=float).reshape(-1,2)
    return p


def material_pieces(uv,faces,xyz,coarse_uv,coarse_faces):
    pieces=[[] for _ in coarse_faces]
    for face in faces:
        tri=uv[face];actual=xyz[face]
        inv=np.linalg.inv(np.column_stack((tri[1]-tri[0],tri[2]-tri[0])))
        lo=tri.min(0);hi=tri.max(0)
        for c,ids in enumerate(coarse_faces):
            region=coarse_uv[ids]
            if np.any(region.max(0)<lo-1e-12) or np.any(region.min(0)>hi+1e-12):continue
            overlap=clip_polygon(tri,region)
            if area(overlap)<=1e-14:continue
            bary=(overlap-tri[0])@inv.T
            poly=actual[0]+bary@(actual[1:]-actual[0])
            if area(poly[:,:2])<=1e-14:continue
            plane=np.linalg.lstsq(np.column_stack((poly[:,:2],np.ones(len(poly)))),poly[:,2],rcond=None)[0]
            pieces[c].append((poly[:,:2],plane))
    return pieces


def audit_layer_orders(uv,faces,aligned,witness,tolerance=1e-6):
    """Orders 1=A above B, 2=A below B; convention calibrated with one M/V hinge."""
    origin=np.array(witness['projection_origin']);axes=np.array(witness['projection_axes'])
    xyz=(np.asarray(aligned)-origin)@axes.T
    coarse_uv=np.array(witness['coarse_uv']);coarse_faces=np.array(witness['coarse_faces']);reference=np.array(witness['folded_xy'])
    pieces=material_pieces(np.asarray(uv),np.asarray(faces),xyz,coarse_uv,coarse_faces)
    rows=[];irrelevant=[]
    for key,order in witness['orders'].items():
        a,b=map(int,key.split());ref_area=area(clip_polygon(reference[coarse_faces[a]],reference[coarse_faces[b]]))
        if ref_area<=1e-12:irrelevant.append(key);continue
        total=wrong=0.;minimum=float('inf');maximum=-float('inf');regions=0
        for pa,za in pieces[a]:
            for pb,zb in pieces[b]:
                if np.any(pa.max(0)<pb.min(0)-1e-12) or np.any(pb.max(0)<pa.min(0)-1e-12):continue
                overlap=clip_polygon(pa,pb);weight=area(overlap)
                if weight<=1e-14:continue
                heights=np.column_stack((overlap,np.ones(len(overlap))))@(za-zb)*(1 if order==1 else -1)
                total+=weight;regions+=1;minimum=min(minimum,float(heights.min()));maximum=max(maximum,float(heights.max()))
                if heights.min() < -tolerance or heights.max()<=tolerance:wrong+=weight
        rows.append(dict(faces=[a,b],expected_order=order,reference_overlap_area=ref_area,actual_overlap_area=total,
                         overlap_regions=regions,wrong_or_unresolved_area=wrong,
                         minimum_signed_gap_mm=minimum*1000 if regions else None,
                         maximum_signed_gap_mm=maximum*1000 if regions else None))
    missing=[r['faces'] for r in rows if not r['overlap_regions']]
    wrong=[r['faces'] for r in rows if r['wrong_or_unresolved_area']>1e-12]
    coverage=sum(min(r['actual_overlap_area'],r['reference_overlap_area']) for r in rows)/max(sum(r['reference_overlap_area'] for r in rows),1e-30)
    return dict(passed=bool(rows and not missing and not wrong and coverage>=.9),required_pairs=len(rows),
                missing_pairs=missing,wrong_or_unresolved_pairs=wrong,overlap_area_coverage=float(coverage),pairs=rows,
                nonoverlapping_reference_pairs=irrelevant,tolerance_m=tolerance,
                scope='Actual piecewise-linear surfaces, clipped in material coordinates, checked over every projected overlap polygon. Requires separate shape, strain, continuous contact and operation-history audits.')
