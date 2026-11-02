"""Current-space, layer-aware boundary fixture controller (development v12).
No task/reference data. Physical reverse motion is distinct from snapshot undo.
"""
import copy,math
import numpy as np
from .skills import SkillUnsupported

VERSION='stack-boundary-v12-dev6'
SCALE=.15

def boundary_vertices(faces):
    counts={}
    for f in faces:
        for i,j in zip(f,np.roll(f,-1)):
            e=tuple(sorted((int(i),int(j))));counts[e]=counts.get(e,0)+1
    return set(i for e,n in counts.items() if n==1 for i in e)

def rotation(points,center,axis,degrees):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis);w=np.asarray(points)-center;a=np.deg2rad(degrees)
    return center+w*np.cos(a)+np.cross(axis,w)*np.sin(a)+np.outer(w@axis,axis)*(1-np.cos(a))

def motion(ids,center,axis,degrees,vertices):
    radius=np.linalg.norm(np.cross(np.asarray(vertices)[ids]-center,axis),axis=1).max()
    frames=max(1,int(math.ceil(abs(degrees)/.5)),int(math.ceil(radius*abs(math.radians(degrees))/.0008)))
    if frames>600:raise SkillUnsupported('Motion exceeds validated per-action integration budget')
    return dict(op='rotate_grips',vertices=list(map(int,ids)),axis_point=np.asarray(center).tolist(),axis_direction=np.asarray(axis).tolist(),degrees=float(degrees),frames=frames)

def point_hit(p,tri):
    a,b,c=tri[:,:2];m=np.column_stack([b-a,c-a])
    if abs(np.linalg.det(m))<1e-12:return None
    uv=np.linalg.solve(m,p-a)
    if min(*uv,1-uv.sum())< -1e-7:return None
    return float(tri[0,2]+uv@(tri[1:,2]-tri[0,2]))

def compile_stack_fold(vertices,faces,action):
    if set(action)!={'op','line','moving_point','layers','sense','angle_deg'} or action['op']!='fold_line':raise ValueError('fold_line requires line,moving_point,layers,sense,angle_deg')
    v=np.asarray(vertices,float);f=np.asarray(faces,int);line=np.asarray(action['line'],float);point=np.asarray(action['moving_point'],float)
    if line.shape!=(2,2) or point.shape!=(2,) or not np.isfinite(line).all() or not np.isfinite(point).all():raise ValueError('Finite board coordinates required')
    if (line<-.5).any() or (line>1.5).any() or (point<-.5).any() or (point>1.5).any():raise ValueError('Coordinates outside padded board [-.5,1.5]')
    degrees=action['angle_deg']
    if type(degrees) not in (float,int) or not math.isfinite(degrees) or not 0<degrees<=180:raise ValueError('angle_deg must be in (0,180]')
    if action['sense'] not in ('toward','away'):raise ValueError('sense must be toward or away from current +z camera')
    if action['layers'] not in ('all','top','bottom'):raise ValueError('layers must be all, top, bottom')
    line=(line-.5)*SCALE;point=(point-.5)*SCALE;d=line[1]-line[0]
    if np.linalg.norm(d)<.0075:raise ValueError('Crease line too short')
    d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
    if abs((point-line[0])@n)<.003:raise ValueError('moving_point must clearly select moving side')
    if (point-line[0])@n<0:d=-d;n=-n
    sd=(v[:,:2]-line[0])@n;tol=.0006;tri=sd[f]
    if ((tri.min(1)<-tol)&(tri.max(1)>tol)).any():raise SkillUnsupported('Current crease crosses mesh faces beyond 0.6mm instrument tolerance; runtime remeshing unavailable')
    candidate=np.flatnonzero((tri.mean(1)>tol)&(tri.min(1)>=-tol));candidate_set=set(candidate.tolist())
    if not len(candidate):raise ValueError('No paper on requested moving side')
    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
    # Sharp folded seams separate layers; do not flood through the folded hinge.
    edge_faces={};adj={int(i):[] for i in candidate}
    for i in candidate:
        for x,y in zip(f[i],np.roll(f[i],-1)):edge_faces.setdefault(tuple(sorted((x,y))),[]).append(int(i))
    for ids in edge_faces.values():
        if len(ids)==2 and normals[ids[0]]@normals[ids[1]]>np.cos(np.deg2rad(45)):
            adj[ids[0]].append(ids[1]);adj[ids[1]].append(ids[0])
    components=[];unseen=set(candidate_set)
    while unseen:
        seed=unseen.pop();component={seed};queue=[seed]
        while queue:
            for j in adj[queue.pop()]:
                if j in unseen:unseen.remove(j);component.add(j);queue.append(j)
        components.append(component)
    hits=[]
    for ci,comp in enumerate(components):
        heights=[z for i in comp if (z:=point_hit(point,v[f[i]])) is not None]
        if heights:hits.append((max(heights),ci))
    if not hits:raise ValueError('moving_point does not hit a moving paper layer')
    if action['layers']=='all':selected=set.union(*components)
    else:selected=components[sorted(hits)[-1 if action['layers']=='top' else 0][1]]
    moving_material=set(f[list(selected)].ravel().tolist());boundary=boundary_vertices(f)
    moving=sorted(i for i in moving_material&boundary if sd[i]>.008)
    # All unselected boundaries stay fixed, except the free crease band.
    fixed=sorted(i for i in boundary if i not in moving_material and abs(sd[i])>.008)
    fixed+=sorted(i for i in boundary&moving_material if sd[i]<-.008)
    fixed=sorted(set(fixed))
    # Hold distant outer boundaries, leaving near-axis boundary material free to
    # bend and slide. This is a geometry-only fixture rule, not a per-task fix.
    if moving and len(hits)>1 and action['layers']=='all':
        cutoff=.85*max(sd[moving]);ordered=sorted(moving,key=lambda i:sd[i],reverse=True);moving=sorted(set([i for i in moving if sd[i]>=cutoff]+ordered[:2]))
    if fixed and len(hits)>1 and action['layers']=='all':
        distances=np.abs(sd[fixed]);cutoff=.85*max(distances);ordered=sorted(fixed,key=lambda i:abs(sd[i]),reverse=True);fixed=sorted(set([i for i in fixed if abs(sd[i])>=cutoff]+ordered[:2]))
    if len(moving)<2 or len(fixed)<2:raise SkillUnsupported('Not enough distinct moving/fixed outer boundary grips')
    near=np.flatnonzero(np.abs(sd)<tol)
    if len(near)<2:raise SkillUnsupported('No current crease vertices')
    # Axis spans selected material hinge points; retain geometric thickness.
    touch=set()
    for i in selected:
        for j in f[i]:
            if abs(sd[j])<tol:touch.add(int(j))
    if len(touch)<2:raise SkillUnsupported('Selected layer does not touch requested crease')
    pts=v[sorted(touch)];t=(pts[:,:2]-line[0])@d
    first,last=pts[np.argmin(t)],pts[np.argmax(t)];axis=np.r_[d,0.]
    if np.ptp(pts[:,2])>.002:raise SkillUnsupported('Crease is not a near-planar stack axis; compound folding not implemented')
    z=float((pts[:,2].min()+pts[:,2].max())/2)
    sign=1 if action['sense']=='toward' else -1
    # Fixed policy depends only on current layer stack thickness, not task ID.
    z+=sign*(.00015+np.ptp(pts[:,2])/2)
    # Like the previously qualified flat-panel controller, follow the actual
    # current hinge rather than imposing an ideal horizontal material axis.
    # This fit only uses current geometry within the declared instrument band.
    center=pts.mean(axis=0)
    _,_,basis=np.linalg.svd(pts-center,full_matrices=False);axis=basis[0]
    if axis[:2]@d<0:axis=-axis
    deviation=np.linalg.norm(np.cross(pts-center,axis),axis=1).max()
    if deviation>.001:raise SkillUnsupported('Current selected crease is not sufficiently straight')
    center[2]+=sign*(.00015+np.ptp(pts[:,2])/2)
    # Choose a finite clearance from CURRENT stationary geometry at the
    # commanded endpoint. Meeting gate edges must not be driven to coincident
    # positions. This is not a reference target lookup or collision certificate.
    stationary=np.array([i for i in range(len(f)) if i not in selected],dtype=int)
    lift=0.
    for _ in range(3):
        predicted=rotation(v[moving],center,axis,sign*degrees);deficit=0.
        for vertex,q in zip(moving,predicted):
            heights=[]
            for face in stationary:
                if vertex in f[face]:continue
                zhit=point_hit(q[:2],v[f[face]])
                if zhit is not None:heights.append(zhit)
                for j in f[face]:
                    if np.linalg.norm(v[j,:2]-q[:2])<.0002:heights.append(v[j,2])
            if heights:
                surface=max(heights) if sign>0 else min(heights)
                deficit=max(deficit,.0002-sign*(q[2]-surface))
        if deficit<=1e-8:break
        delta=deficit/max(1-np.cos(np.deg2rad(degrees)),.1)
        if lift+delta>.001:raise SkillUnsupported('Endpoint needs more than validated 1mm fixture clearance offset')
        center[2]+=sign*delta;lift+=delta
    cmd=motion(moving,center,axis,sign*degrees,v)
    info=dict(endpoint_clearance_axis_lift_m=lift,selected_faces=sorted(selected),moving_vertices=moving,fixed_vertices=fixed,hit_layers=len(hits),selected_layer_mode=action['layers'],axis_point=cmd['axis_point'],axis_direction=cmd['axis_direction'],degrees=cmd['degrees'])
    return [dict(op='release'),dict(op='set_grips',vertices=sorted(moving+fixed)),cmd],info

class StackController:
    def __init__(self,backend):self.backend=backend;self.records=[];self.snapshots=[];self.fold_stack=[]
    def execute(self,action):
        if not isinstance(action,dict):raise ValueError('Action must be an object')
        state=self.backend.state();v=np.asarray(state['vertices']);f=np.asarray(state['faces']);op=action.get('op');info={};low=[]
        previous=copy.deepcopy(self.fold_stack)
        if op=='fold_line':low,info=compile_stack_fold(v,f,action)
        elif op=='unfold':
            if action!={'op':'unfold'}:raise ValueError('unfold takes no arguments; opens latest retained fold physically')
            if not self.fold_stack:raise ValueError('No retained fold to unfold')
            r=self.fold_stack[-1];cmd=motion(r['moving_vertices'],np.asarray(r['axis_point']),np.asarray(r['axis_direction']),-r['degrees'],v)
            low=[dict(op='release'),dict(op='set_grips',vertices=sorted(r['moving_vertices']+r['fixed_vertices'])),cmd];info=dict(unfolds_record=r['record_index'],physical_forward_time=True)
        elif op in ('turn_over','rotate_paper'):
            allowed={'op','axis'} if op=='turn_over' else {'op','degrees'}
            if set(action)!=allowed:raise ValueError('Invalid rigid paper action fields')
            if op=='turn_over':
                if action['axis'] not in ('x','y'):raise ValueError('turn_over axis is x or y')
                axis=np.array([1.,0,0]) if action['axis']=='x' else np.array([0.,1,0]);degrees=180.
            else:
                degrees=action['degrees']
                if type(degrees) not in (int,float) or not math.isfinite(degrees) or not 0<abs(degrees)<=180:raise ValueError('rotate_paper degrees must be nonzero within +/-180')
                axis=np.array([0.,0,1.])
            center=np.array([0.,0.,float(np.mean(v[:,2]))]);ids=sorted(boundary_vertices(f));cmd=motion(ids,center,axis,degrees,v)
            low=[dict(op='release'),dict(op='set_grips',vertices=ids),cmd];info=dict(axis_point=center.tolist(),axis_direction=axis.tolist(),degrees=float(degrees))
        elif op=='undo':
            if action!={'op':'undo'} or not self.records:raise ValueError('Nothing to undo or invalid fields')
            record=self.records[-1]
            for _ in record['low_level_actions']:self.backend.step({'op':'undo'})
            self.records.pop();self.fold_stack=self.snapshots.pop();return dict(undone=True,low_level_actions=[],forward_physical_motion=False)
        else:raise SkillUnsupported('Unsupported controller operation: '+str(op))
        completed=0
        try:
            for cmd in low:self.backend.step(cmd);completed+=1
        except Exception:
            for _ in range(completed):self.backend.step({'op':'undo'})
            if self.backend.state()['hash']!=state['hash']:raise RuntimeError('Composite rollback did not restore paper')
            raise
        if op=='fold_line':info['record_index']=len(self.records);self.fold_stack.append(copy.deepcopy(info))
        elif op=='unfold':self.fold_stack.pop()
        elif op in ('turn_over','rotate_paper'):
            for fold in self.fold_stack:
                fold['axis_point']=rotation(np.array([fold['axis_point']]),center,axis,degrees)[0].tolist()
                fold['axis_direction']=rotation(np.array([fold['axis_direction']]),np.zeros(3),axis,degrees)[0].tolist()
        record=dict(action=copy.deepcopy(action),low_level_actions=low,details=info,before=state['hash'],after=self.backend.state()['hash'],frame=self.backend.state()['steps'])
        self.snapshots.append(previous);self.records.append(record);return copy.deepcopy(record)
