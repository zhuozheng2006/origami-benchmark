"""Experimental material-hinge actuation, derived solely from current geometry.

Commands apply finite torques through the existing physical crease actuators.
They never assign vertex positions or reference shapes.
"""
import numpy as np
from origami_bench.dihedral_audit import continuous_dihedral_degrees

def servo_fold(vertices, faces, info, configuration, hold_all=False, stiffness=.01):
    from .layer_controller import rotation
    v=np.asarray(vertices,float);f=np.asarray(faces,int)
    stencils=configuration['hinge_stencils']
    selected=set(info['selected_faces'])
    face_lookup={frozenset(map(int,face)):i for i,face in enumerate(f)}
    targets={};initial={};hold={}
    axis=np.asarray(info['axis_direction']);degrees=info['degrees']
    for i,st in enumerate(stencils):
        a,b,c,d=map(int,st)
        angle=float(continuous_dihedral_degrees(v[None],st)[0])
        left=face_lookup[frozenset((a,b,c))] in selected
        right=face_lookup[frozenset((b,c,d))] in selected
        if left != right:
            # A selected/unselected material edge must lie on the requested
            # physical hinge; folded shared seams within the packet are held.
            if max(np.linalg.norm(np.cross(v[j]-np.asarray(info['axis_point']),axis)) for j in (b,c))>.002:
                raise ValueError('Selected panel boundary is not the requested material crease')
            pivot=(v[b]+v[c])/2;edge=v[c]-v[b];edge/=np.linalg.norm(edge)
            if edge@axis<0:edge=-edge
            q=v.copy();opposite=a if left else d
            q[opposite]=rotation(v[[opposite]],pivot,edge,np.sign(degrees))[0]
            changed=float(continuous_dihedral_degrees(np.stack([v,q]),st)[1])
            direction=np.sign(changed-angle)
            if not direction:raise ValueError('Cannot resolve material hinge actuation direction')
            initial[str(i)]=angle
            targets[str(i)]=float(np.clip(angle+direction*abs(degrees),-180,180))
        elif hold_all or abs(angle)>45:hold[str(i)]=angle
    if not targets:raise ValueError('No material hinge separates the moving packet')
    low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='release_creases'),dict(op='set_grips',vertices=info['fixed_vertices'])]
    if hold:low.append(dict(op='attach_creases',hinges=list(map(int,hold)),stiffness=stiffness if hold_all else .002))
    low.append(dict(op='attach_creases',hinges=list(map(int,targets)),stiffness=stiffness))
    low.append(dict(op='drive_creases',targets={**hold,**targets},frames=max(1,int(np.ceil(abs(degrees)/.5)))))
    return low,dict(crease_servo_initial=initial,crease_servo_targets=targets,crease_servo_hold=hold,moving_grip_kind='crease_servo')
