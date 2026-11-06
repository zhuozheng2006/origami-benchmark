"""Current-space, layer-aware boundary fixture controller (development v12).
No task/reference data. Physical reverse motion is distinct from snapshot undo.
"""
import copy,math
import numpy as np
from .skills import SkillUnsupported

VERSION='layer-boundary-v14-dev27'
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

def panel_hull_grips(ids,vertices):
    """Enclose a deformable panel instead of choosing three interior supports."""
    ids=sorted(set(ids))
    if len(ids)<=3:return ids
    v=np.asarray(vertices)[ids];center=v.mean(0);_,_,basis=np.linalg.svd(v-center,full_matrices=False);p=(v-center)@basis[:2].T
    def cross(i,j,k):return float(np.linalg.det(np.stack([p[j]-p[i],p[k]-p[i]])))
    order=sorted(range(len(ids)),key=lambda i:tuple(p[i]));lower=[];upper=[]
    for seq,hull in [(order,lower),(order[::-1],upper)]:
        for i in seq:
            while len(hull)>=2 and cross(hull[-2],hull[-1],i)<=1e-14:hull.pop()
            hull.append(i)
    hull=lower[:-1]+upper[:-1]
    while len(hull)>3:
        errors=[abs(cross(hull[i-1],j,hull[(i+1)%len(hull)]))/max(np.linalg.norm(p[hull[(i+1)%len(hull)]]-p[hull[i-1]]),1e-12) for i,j in enumerate(hull)]
        at=int(np.argmin(errors))
        if errors[at]>=.0002:break
        hull.pop(at)
    middle=int(np.linalg.norm(p-p[hull].mean(0),axis=1).argmin())
    return sorted({ids[i] for i in hull+[middle]})

def compile_stack_fold(vertices,faces,action,fixture_mode='boundary'):
    thickness_panel=fixture_mode=='thickness_panel'
    force_panel=fixture_mode in ('full_panel','thickness_panel')
    if force_panel:fixture_mode='distributed_panel'
    soft_stiffness=1000.
    if fixture_mode.startswith('soft_distributed_'):
        soft_stiffness=float(fixture_mode.removeprefix('soft_distributed_'))
        if not np.isfinite(soft_stiffness) or soft_stiffness<=0:raise ValueError('Invalid soft fixture stiffness')
        fixture_mode='soft_distributed'
    if fixture_mode not in ('boundary','sparse','compliant','tripod','panel','adaptive_panel','balanced_panel','free_edge_panel','distributed_panel','soft_distributed','packet_panel','hull_panel'):
        raise ValueError('Unknown fixture mode')
    if set(action)!={'op','line','moving_point','layers','sense','angle_deg'} or action['op']!='fold_line':raise ValueError('fold_line requires line,moving_point,layers,sense,angle_deg')
    v=np.asarray(vertices,float);f=np.asarray(faces,int);line=np.asarray(action['line'],float);point=np.asarray(action['moving_point'],float)
    if line.shape!=(2,2) or point.shape!=(2,) or not np.isfinite(line).all() or not np.isfinite(point).all():raise ValueError('Finite board coordinates required')
    if (line<-.5).any() or (line>1.5).any() or (point<-.5).any() or (point>1.5).any():raise ValueError('Coordinates outside padded board [-.5,1.5]')
    degrees=action['angle_deg']
    if type(degrees) not in (float,int) or not math.isfinite(degrees) or not 0<degrees<=180:raise ValueError('angle_deg must be in (0,180]')
    if action['sense'] not in ('toward','away'):raise ValueError('sense must be toward or away from current +z camera')
    layer_spec=action['layers']
    if isinstance(layer_spec,dict):
        if set(layer_spec)!={'side','count'} or layer_spec['side'] not in ('top','bottom') or type(layer_spec['count']) is not int or layer_spec['count']<1:raise ValueError('Layer packet requires side top/bottom and positive integer count')
    elif layer_spec not in ('all','top','bottom'):raise ValueError('layers must be all, top, bottom, or an explicit layer packet')
    line=(line-.5)*SCALE;point=(point-.5)*SCALE;d=line[1]-line[0]
    if np.linalg.norm(d)<.0075:raise ValueError('Crease line too short')
    d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
    if abs((point-line[0])@n)<.003:raise ValueError('moving_point must clearly select moving side')
    if (point-line[0])@n<0:d=-d;n=-n
    sd=(v[:,:2]-line[0])@n;tol=.0006;tri=sd[f]
    if ((tri.min(1)<-tol)&(tri.max(1)>tol)).any():raise SkillUnsupported('Current crease crosses mesh faces beyond 0.6mm instrument tolerance; runtime remeshing unavailable')
    candidate=np.flatnonzero((tri.mean(1)>1e-10)&(tri.min(1)>=-tol));candidate_set=set(candidate.tolist())
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
    elif isinstance(layer_spec,dict):
        if layer_spec['count']>len(hits):raise ValueError('Requested packet contains more layers than the moving point intersects')
        chosen=sorted(hits)[-layer_spec['count']:] if layer_spec['side']=='top' else sorted(hits)[:layer_spec['count']]
        selected=set.union(*(components[ci] for _,ci in chosen))
    else:selected=components[sorted(hits)[-1 if action['layers']=='top' else 0][1]]
    moving_material=set(f[list(selected)].ravel().tolist());boundary=boundary_vertices(f)
    free_band=.008
    if thickness_panel:
        heights=v[[i for i in moving_material if abs(sd[i])<tol],2]
        if len(heights)<2:raise SkillUnsupported('No selected material near the requested hinge')
        # Half the largest current-stack semicircular bend length plus a
        # small transition margin. Uses current geometry, not crease labels.
        free_band=max(.001,float(np.pi*(np.ptp(heights)+.00015)/2+.00025))
    moving=sorted(i for i in moving_material&boundary if sd[i]>free_band)
    # All unselected boundaries stay fixed, except the free crease band.
    fixed=sorted(i for i in boundary if i not in moving_material and abs(sd[i])>free_band)
    fixed+=sorted(i for i in boundary&moving_material if sd[i]<-free_band)
    fixed=sorted(set(fixed));dense_fixed=fixed.copy();dense_moving=moving.copy()
    # Hold distant outer boundaries, leaving near-axis boundary material free to
    # bend and slide. This is a geometry-only fixture rule, not a per-task fix.
    if moving and len(hits)>1 and action['layers']=='all':
        cutoff=.85*max(sd[moving]);ordered=sorted(moving,key=lambda i:sd[i],reverse=True);moving=sorted(set([i for i in moving if sd[i]>=cutoff]+ordered[:2]))
    if fixed and len(hits)>1 and action['layers']=='all':
        distances=np.abs(sd[fixed]);cutoff=.85*max(distances);ordered=sorted(fixed,key=lambda i:abs(sd[i]),reverse=True);fixed=sorted(set([i for i in fixed if abs(sd[i])>=cutoff]+ordered[:2]))
    if fixture_mode in ('sparse','compliant'):
        def distant(ids):
            if not ids:return ids
            dist=np.abs(sd[ids]);band=[i for i in ids if abs(sd[i])>=.85*dist.max()]
            return sorted(set(band+sorted(ids,key=lambda i:abs(sd[i]),reverse=True)[:2]))
        moving=distant(moving);fixed=distant(fixed)
    if fixture_mode in ('tripod','compliant'):
        # Three widely separated stationary boundary anchors remove the free
        # rigid rotation left by a collinear two-grip fixture.
        ids=dense_fixed
        if len(ids)>=3:
            a=max(ids,key=lambda i:abs(sd[i]));b=max(ids,key=lambda i:np.linalg.norm(v[i]-v[a]))
            c=max(ids,key=lambda i:np.linalg.norm(np.cross(v[b]-v[a],v[i]-v[a])))
            fixed=sorted(set([a,b,c]))
        moving=dense_moving if fixture_mode=='compliant' else moving
    panel_override=False
    folded_stationary_support=fixture_mode=='hull_panel' and np.min(normals@normals[0])<np.cos(np.deg2rad(45))
    if force_panel or folded_stationary_support or fixture_mode=='panel' or (fixture_mode in ('adaptive_panel','balanced_panel','free_edge_panel','distributed_panel','soft_distributed','packet_panel','hull_panel') and (sum(bool(comp&selected) for comp in components)>1 or len(moving)<2 or len(fixed)<2)):
        panel_override=True
        # A packet's original paper boundary may lie entirely on old seams.
        # Grip at most three interior material vertices per planar layer,
        # excluding vertices shared by two moving layer components.
        owners={}
        for ci,comp in enumerate(components):
            for j in set(f[list(comp)].ravel().tolist()):owners.setdefault(j,set()).add(ci)
        moving=[]
        all_edges={}
        for fi,face in enumerate(f):
            for x,y in zip(face,np.roll(face,-1)):all_edges.setdefault(tuple(sorted((x,y))),[]).append(fi)
        def actual_hinge(group,other):
            return any(max(abs(sd[j]) for j in edge)<tol and any(fi in group for fi in ids) and any(fi in other for fi in ids) for edge,ids in all_edges.items())
        def triangle_grips(ids):
            if fixture_mode=='hull_panel':return panel_hull_grips(ids,v)
            if fixture_mode in ('distributed_panel','soft_distributed') or len(ids)<2:return ids
            a=max(ids,key=lambda i:abs(sd[i]));b=max(ids,key=lambda i:np.linalg.norm(v[i]-v[a]))
            c=max(ids,key=lambda i:np.linalg.norm(np.cross(v[b]-v[a],v[i]-v[a])))
            return sorted(set([a,b,c]))
        for ci,comp in enumerate(components):
            if not (comp&selected):continue
            band=-tol if fixture_mode in ('free_edge_panel','distributed_panel','soft_distributed','packet_panel','hull_panel') and not actual_hinge(comp,set(range(len(f)))-selected) else free_band
            ids=[i for i in set(f[list(comp)].ravel().tolist()) if sd[i]>band and (fixture_mode in ('distributed_panel','soft_distributed','packet_panel','hull_panel') or len(owners[i])==1)]
            moving.extend(triangle_grips(ids))
        moving=sorted(set(moving));fixed=triangle_grips(dense_fixed)
        if fixture_mode in ('balanced_panel','free_edge_panel','distributed_panel','soft_distributed','packet_panel','hull_panel'):
            # Stabilize every stationary planar component as well. Holding just
            # one outside flap leaves the underlying packet free to buckle.
            remaining=set(range(len(f)))-selected;edges={};links={i:[] for i in remaining}
            for i in remaining:
                for x,y in zip(f[i],np.roll(f[i],-1)):edges.setdefault(tuple(sorted((x,y))),[]).append(i)
            for ids in edges.values():
                if len(ids)==2 and normals[ids[0]]@normals[ids[1]]>np.cos(np.deg2rad(45)):
                    links[ids[0]].append(ids[1]);links[ids[1]].append(ids[0])
            groups=[];unseen=set(remaining)
            while unseen:
                seed=unseen.pop();group={seed};queue=[seed]
                while queue:
                    for j in links[queue.pop()]:
                        if j in unseen:unseen.remove(j);group.add(j);queue.append(j)
                groups.append(group)
            stationary_owners={}
            for ci,group in enumerate(groups):
                for j in set(f[list(group)].ravel().tolist()):stationary_owners.setdefault(j,set()).add(ci)
            fixed=[]
            for ci,group in enumerate(groups):
                band=-tol if fixture_mode in ('free_edge_panel','distributed_panel','soft_distributed','packet_panel','hull_panel') and not actual_hinge(group,selected) else free_band
                ids=[j for j in set(f[list(group)].ravel().tolist()) if j not in moving_material and abs(sd[j])>band and (fixture_mode in ('distributed_panel','soft_distributed','packet_panel','hull_panel') or len(stationary_owners[j])==1)]
                fixed.extend(triangle_grips(ids))
            fixed=sorted(set(fixed))

    if len(moving)<2 or len(fixed)<2:raise SkillUnsupported('Not enough distinct moving/fixed outer boundary grips')
    near=np.flatnonzero(np.abs(sd)<tol)
    if len(near)<2:raise SkillUnsupported('No current crease vertices')
    # Axis spans selected material hinge points; retain geometric thickness.
    touch=set()
    for i in selected:
        for j in f[i]:
            if abs(sd[j])<tol:touch.add(int(j))
    if len(touch)<2:raise SkillUnsupported('Selected layer does not touch requested crease')
    near_selected=v[sorted(touch)]
    # Use actual moving/stationary material adjacency. A tolerance band may
    # contain many refined rows; averaging all of them shifts the hinge.
    all_edges={}
    for fi,face in enumerate(f):
        for x,y in zip(face,np.roll(face,-1)):all_edges.setdefault(tuple(sorted((x,y))),[]).append(fi)
    touch={int(j) for edge,owners in all_edges.items() if len(owners)==2 and any(fi in selected for fi in owners) and any(fi not in selected for fi in owners) and max(abs(sd[j]) for j in edge)<tol for j in edge}
    if len(touch)<2:raise SkillUnsupported('No material hinge joins the selected packet to stationary paper at the requested crease')
    pts=v[sorted(touch)];t=(pts[:,:2]-line[0])@d
    first,last=pts[np.argmin(t)],pts[np.argmax(t)];axis=np.r_[d,0.]
    if np.ptp(near_selected[:,2])>.002:raise SkillUnsupported('Crease is not a near-planar stack axis; compound folding not implemented')
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
    center[2]=float(near_selected[:,2].max()+.00015 if sign>0 else near_selected[:,2].min()-.00015)
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
    info['fixture_mode']='thickness_panel' if thickness_panel else 'full_panel' if force_panel else fixture_mode;info['free_crease_band_halfwidth_m']=free_band;info['folded_stationary_support']=bool(folded_stationary_support);info['has_multiple_surface_orientations']=bool(np.min(normals@normals[0])<np.cos(np.deg2rad(45)));info['selected_planar_components']=sum(bool(comp&selected) for comp in components)
    info['moving_panel_groups']=[sorted(set(f[list(comp)].ravel().tolist())) for comp in components if comp&selected]
    if fixture_mode=='compliant' or (fixture_mode=='soft_distributed' and panel_override):
        cmd['op']='rotate_soft_grips';info['moving_grip_stiffness_N_m']=soft_stiffness;info['moving_grip_kind']='compliant'
        return [dict(op='release'),dict(op='release_soft_grips'),dict(op='set_grips',vertices=fixed),dict(op='attach_soft_grips',vertices=moving,stiffness=soft_stiffness),cmd],info
    return [dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=sorted(moving+fixed)),cmd],info

class LayerController:
    def __init__(self,backend,fixture_mode='boundary'):self.fixture_mode=fixture_mode;self.next_fold_id=0;self.backend=backend;self.records=[];self.snapshots=[];self.fold_stack=[];self.undo_floor=0;self.rejected_endpoints=[]
    def execute(self,action):
        if not isinstance(action,dict):raise ValueError('Action must be an object')
        state=self.backend.state();v=np.asarray(state['vertices']);f=np.asarray(state['faces']);op=action.get('op');info={};low=[]
        previous=copy.deepcopy(self.fold_stack)
        if op=='fold_line':
            servo=self.fixture_mode in ('hinge_servo','hinge_servo_panel','hinge_servo_strong')
            mode='packet_panel' if servo or self.fixture_mode=='packet_press' else self.fixture_mode
            guided_directional=mode=='guided_directional' or mode.startswith('guided_directional_')
            anchor_stiffness=float(mode.removeprefix('guided_directional_')) if mode.startswith('guided_directional_') else 100.
            if not math.isfinite(anchor_stiffness) or anchor_stiffness<=0:raise ValueError('Invalid tangential anchor stiffness')
            sliding_directional=mode=='sliding_directional'
            directional_thickness=mode=='directional_thickness' or guided_directional or sliding_directional
            if directional_thickness:mode='thickness_panel'
            directional=mode=='directional_panel' or mode.startswith('directional_panel_')
            if directional:mode='soft_distributed_'+(mode.removeprefix('directional_panel_') if mode.startswith('directional_panel_') else '1000')
            if mode=='rolling_panel':mode='distributed_panel'
            low,info=compile_stack_fold(v,f,action,mode)
            if directional_thickness and (info['selected_planar_components']>1 or ((guided_directional or sliding_directional) and info['has_multiple_surface_orientations'])):
                cmd=low[-1];cmd['op']='rotate_soft_grips'
                low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=info['fixed_vertices']),dict(op='attach_soft_grips',vertices=info['moving_vertices'],stiffness=1000.,normal=[0.,0.,1.]),cmd]
                info.update(moving_grip_kind='compliant',moving_grip_stiffness_N_m=1000.,directional_normal_start=[0.,0.,1.])
                if guided_directional:
                    axis=np.array(info['axis_direction']);center=np.array(info['axis_point']);anchors=[]
                    for group in info['moving_panel_groups']:
                        ids=sorted(set(group)&set(info['moving_vertices']))
                        if not ids:continue
                        radii=np.linalg.norm(np.cross(v[ids]-center,axis),axis=1);far=[i for i,radius in zip(ids,radii) if radius>=.85*radii.max()]
                        centroid=v[far].mean(0);anchors.append(min(far,key=lambda i:np.linalg.norm(v[i]-centroid)))
                    anchors=sorted(set(anchors));low.insert(-1,dict(op='attach_soft_grips',vertices=anchors,stiffness=anchor_stiffness))
                    info.update(tangential_anchor_vertices=anchors,tangential_anchor_stiffness_N_m=anchor_stiffness)
            if self.fixture_mode=='rolling_panel' and info['selected_planar_components']>1:
                axis=np.array(info['axis_direction']);normal=np.array([0.,0.,1.]);normal-=axis*(normal@axis);normal/=np.linalg.norm(normal)
                tangent=np.cross(normal,axis);center=np.array(info['axis_point']);radii={i:[] for i in info['moving_vertices']}
                for group in info['moving_panel_groups']:
                    radius=abs(float(np.median((center-v[group])@normal)))
                    for i in set(group)&set(radii):radii[i].append(radius)
                radii={str(i):float(np.mean(rs)) for i,rs in radii.items()}
                info.update(roll_radii=radii,roll_tangent=tangent.tolist())
                low[-1].update(roll_radii=radii,roll_tangent=tangent.tolist(),roll_start_angle_deg=0.)
            if directional:
                for command in low:
                    if command['op']=='attach_soft_grips':
                        command['normal']=[0.,0.,1.];info['directional_normal_start']=[0.,0.,1.]
            if self.fixture_mode=='packet_press' and info['selected_planar_components']>1:
                # First allow the packet to bend during rotation, then approach
                # the command's own endpoint with enclosing panel fixtures.
                # The targets are rotations of CURRENT material, never a task goal.
                _,enclosing=compile_stack_fold(v,f,action,'hull_panel')
                ids=sorted(set(enclosing['moving_vertices']+enclosing['fixed_vertices']))
                expected=v.copy();material=np.unique(f[info['selected_faces']])
                expected[material]=rotation(v[material],np.array(info['axis_point']),np.array(info['axis_direction']),info['degrees'])
                low.extend([dict(op='release'),dict(op='set_grips',vertices=ids),dict(op='move_grips',targets={str(i):expected[i].tolist() for i in ids},frames=120)])
                info['endpoint_finish_kind']='physical_hull_press';info['endpoint_finish_vertices']=ids
            if servo and info['selected_planar_components']>1:
                from .layer_servo import servo_fold
                low,extra=servo_fold(v,f,info,self.backend.configuration,self.fixture_mode!='hinge_servo',.1 if self.fixture_mode=='hinge_servo_strong' else .01);info.update(extra)
        elif op=='unfold':
            if set(action)-{'op','fold_id','angle_deg'}:raise ValueError('unfold accepts optional fold_id and angle_deg')
            if not self.fold_stack:raise ValueError('No active fold to unfold')
            fold_id=action.get('fold_id',self.fold_stack[-1]['fold_id'])
            if type(fold_id)is not int:raise ValueError('fold_id must be an integer')
            found=[i for i,r in enumerate(self.fold_stack) if r['fold_id']==fold_id]
            if not found:raise ValueError('fold_id is not active')
            at=found[0];r=self.fold_stack[at]
            if r.get('moving_grip_kind')=='crease_servo':raise SkillUnsupported('Experimental crease-servo unfolding not yet qualified')
            for later in self.fold_stack[at+1:]:
                if set(r['selected_faces'])&set(later['selected_faces']):raise SkillUnsupported('Open overlapping later folds first; dependent layer unfolding is not yet validated')
            amount=action.get('angle_deg',abs(r['degrees']))
            if type(amount)not in (int,float) or not math.isfinite(amount) or not 0<amount<=abs(r['degrees'])+1e-8:raise ValueError('Unfold angle must be positive and not exceed the remaining fold angle')
            delta=-math.copysign(amount,r['degrees']);cmd=motion(r['moving_vertices'],np.asarray(r['axis_point']),np.asarray(r['axis_direction']),delta,v)
            if 'roll_radii' in r:cmd.update(roll_radii=r['roll_radii'],roll_tangent=r['roll_tangent'],roll_start_angle_deg=r['degrees'])
            low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=sorted(r['moving_vertices']+r['fixed_vertices'])),cmd]
            if r.get('moving_grip_kind')=='compliant':
                cmd['op']='rotate_soft_grips'
                low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=r['fixed_vertices']),dict(op='attach_soft_grips',vertices=r['moving_vertices'],stiffness=r['moving_grip_stiffness_N_m']),cmd]
                if 'directional_normal_start' in r:
                    low[-2]['normal']=rotation(np.array([r['directional_normal_start']]),np.zeros(3),np.array(r['axis_direction']),r['degrees'])[0].tolist()
                if r.get('tangential_anchor_vertices'):
                    low.insert(-1,dict(op='attach_soft_grips',vertices=r['tangential_anchor_vertices'],stiffness=r['tangential_anchor_stiffness_N_m']))
            info=dict(unfolds_record=r['record_index'],fold_id=fold_id,unfold_angle_deg=float(amount),physical_forward_time=True)
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
            low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=ids),cmd];info=dict(axis_point=center.tolist(),axis_direction=axis.tolist(),degrees=float(degrees))
        elif op=='undo':
            if action!={'op':'undo'} or len(self.records)<=self.undo_floor:raise ValueError('Nothing to undo within this episode; prepared initial state cannot be undone')
            record=self.records[-1]
            for _ in record['low_level_actions']:self.backend.step({'op':'undo'})
            self.records.pop();self.fold_stack=self.snapshots.pop();return dict(undone=True,low_level_actions=[],forward_physical_motion=False)
        else:raise SkillUnsupported('Unsupported controller operation: '+str(op))
        completed=0
        try:
            for cmd in low:
                end=self.backend.step(cmd);completed+=1
                if cmd['op']=='rotate_soft_grips':
                    controls=end.get('soft_controls',{});actual=np.asarray(end['vertices'])
                    errors=[abs((actual[int(i)]-c[:3])@np.asarray(c[4:])) if len(c)==7 else np.linalg.norm(actual[int(i)]-c[:3]) for i,c in controls.items()]
                    error=max(errors,default=float('inf'));info['controlled_endpoint_error_mm']=float(error*1000)
                    info['unconstrained_3d_target_error_mm']=end.get('last_metrics',{}).get('soft_target_error',float('inf'))*1000
                    if error>.0005:
                        reason=f'Compliant controlled directions exceeded 0.5mm endpoint accuracy: {error*1000:.3f}mm'
                        self.rejected_endpoints.append(dict(action=copy.deepcopy(action),reason=reason,state=copy.deepcopy(end)));raise SkillUnsupported(reason)
                    if any(len(c)==7 for c in controls.values()):
                        details=info if op=='fold_line' else r;ids=np.unique(f[details['selected_faces']]);expected=v.copy()
                        expected[ids]=rotation(v[ids],np.array(cmd['axis_point']),np.array(cmd['axis_direction']),cmd['degrees'])
                        distance=np.linalg.norm(actual-expected,axis=1);error_shape=dict(rms_mm=float(np.sqrt(np.mean(distance**2))*1000),max_mm=float(distance.max()*1000));info['command_endpoint_shape']=error_shape
                        if error_shape['rms_mm']>=1 or error_shape['max_mm']>=3:
                            reason=f'Physical motion did not reach its requested endpoint geometry: RMS {error_shape["rms_mm"]:.3f}mm, max {error_shape["max_mm"]:.3f}mm'
                            self.rejected_endpoints.append(dict(action=copy.deepcopy(action),reason=reason,state=copy.deepcopy(end)));raise SkillUnsupported(reason)
            if info.get('endpoint_finish_kind')=='physical_hull_press':
                distance=np.linalg.norm(np.asarray(end['vertices'])-expected,axis=1)
                shape=dict(rms_mm=float(np.sqrt(np.mean(distance**2))*1000),max_mm=float(distance.max()*1000));info['command_endpoint_shape']=shape
                if shape['rms_mm']>=1 or shape['max_mm']>=3:
                    reason=f'Physical endpoint pressing did not reach command geometry: RMS {shape["rms_mm"]:.3f}mm, max {shape["max_mm"]:.3f}mm'
                    self.rejected_endpoints.append(dict(action=copy.deepcopy(action),reason=reason,state=copy.deepcopy(end)));raise SkillUnsupported(reason)
        except Exception:
            if getattr(self.backend,'poisoned',False):raise RuntimeError('Simulation transport lost; rollback cannot be verified and episode must stop')
            for _ in range(completed):self.backend.step({'op':'undo'})
            if self.backend.state()['hash']!=state['hash']:raise RuntimeError('Composite rollback did not restore paper')
            raise
        if op=='fold_line':
            info['record_index']=len(self.records);info['fold_id']=self.next_fold_id;self.next_fold_id+=1;self.fold_stack.append(copy.deepcopy(info))
        elif op=='unfold':
            remaining=abs(r['degrees'])-amount
            if remaining<1e-8:self.fold_stack.pop(at)
            else:self.fold_stack[at]['degrees']=math.copysign(remaining,r['degrees'])
        elif op in ('turn_over','rotate_paper'):
            for fold in self.fold_stack:
                fold['axis_point']=rotation(np.array([fold['axis_point']]),center,axis,degrees)[0].tolist()
                fold['axis_direction']=rotation(np.array([fold['axis_direction']]),np.zeros(3),axis,degrees)[0].tolist()
                if 'roll_tangent' in fold:fold['roll_tangent']=rotation(np.array([fold['roll_tangent']]),np.zeros(3),axis,degrees)[0].tolist()
                if 'directional_normal_start' in fold:fold['directional_normal_start']=rotation(np.array([fold['directional_normal_start']]),np.zeros(3),axis,degrees)[0].tolist()
        record=dict(action=copy.deepcopy(action),low_level_actions=low,details=info,before=state['hash'],after=self.backend.state()['hash'],frame=self.backend.state()['steps'])
        self.snapshots.append(previous);self.records.append(record);return copy.deepcopy(record)
