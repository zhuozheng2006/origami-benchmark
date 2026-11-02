"""Sparse ideal material-point alignment, never a rigid platen simulation."""
import numpy as np

def public_panel_points(rest,targets,points,regions):
 """Barycentric tool positions from public corner targets, without a hidden target mesh."""
 uv=np.asarray(regions['vertices'])[:,:2];faces=np.asarray(regions['faces']);ids=[]
 for point in uv:
  distance=np.linalg.norm(rest[:,:2]-point,axis=1);i=int(distance.argmin())
  if distance[i]>2e-6 or i not in targets:raise ValueError('Every panel corner needs a public target')
  ids.append(i)
 positions=[]
 for i in points:
  weights=[np.linalg.solve(np.vstack([uv[f].T,np.ones(3)]),np.r_[rest[i,:2],1]) for f in faces]
  j=int(np.argmax([w.min() for w in weights]))
  if weights[j].min()<-1e-6:raise ValueError('Material point outside panels')
  positions.append(weights[j]@np.array([targets[ids[k]] for k in faces[j]]))
 return np.array(positions)

def target_plane(targets):
 points=np.asarray(targets,float);origin=points.mean(0);_,singular,V=np.linalg.svd(points-origin,full_matrices=False)
 if len(singular)<3 or singular[1]<1e-5:raise ValueError('Control targets do not span a plane')
 normal=V[-1]
 if abs((points-origin)@normal).max()>.0005:raise ValueError('Control targets are not sufficiently planar')
 return origin,normal

def control_planes(rest,targets,points,regions=None):
 """Fit only public actuator targets, using rest-material triangles for membership."""
 if regions is None:
  origin,normal=target_plane(list(targets.values()))
  return np.tile(origin,(len(points),1)),np.tile(normal,(len(points),1))
 uv=np.asarray(regions['vertices'])[:,:2];faces=np.asarray(regions['faces']);ids=[]
 for point in uv:
  distance=np.linalg.norm(rest[:,:2]-point,axis=1);i=int(distance.argmin())
  if distance[i]>2e-6 or i not in targets:raise ValueError('Every region corner needs an existing public control target')
  ids.append(i)
 candidates=[]
 for face in faces:
  origin,normal=target_plane([targets[ids[j]] for j in face])
  matrix=np.vstack([uv[face].T,np.ones(3)]);candidates.append((matrix,origin,normal))
 origins=[];normals=[]
 for i in points:
  memberships=[np.linalg.solve(matrix,np.r_[rest[i,:2],1]).min() for matrix,_,_ in candidates]
  j=int(np.argmax(memberships))
  if memberships[j]<-1e-6:raise ValueError('Selected material point is outside the controlled regions')
  _,origin,normal=candidates[j];origins.append(origin);normals.append(normal)
 return np.array(origins),np.array(normals)

def press_normal(env,count,stiffness,steps,on_step=None,directional=False,regions=None,minimum_crease_distance=0.,exclude_vertices=(),plane_control_vertices=None):
 if type(count)is not int or count<1 or not np.isfinite(stiffness) or stiffness<=0 or type(steps)is not int or steps<30:raise ValueError('Invalid normal control parameters')
 if not np.isfinite(minimum_crease_distance) or minimum_crease_distance<0:raise ValueError('Invalid crease margin')
 existing=sorted(set(env.grips)|set(env.soft_controls));targets={**env.grips,**{i:c[:3] for i,c in env.soft_controls.items()}}
 excluded=set(exclude_vertices)
 if any(type(i)is not int or i<0 or i>=len(env.rest) for i in excluded):raise ValueError('Invalid excluded material vertex')
 available=[i for i in range(len(env.rest)) if i not in existing and i not in excluded]
 if minimum_crease_distance:
  creases=[c['vertices'] for c in env.meta['crease_edges'] if abs(c['angle_degrees'])>2]
  if creases:
   edge=np.array(creases);a=env.rest[edge[:,0],:2];delta=env.rest[edge[:,1],:2]-a;points=env.rest[available,:2]
   t=np.clip(np.einsum('nej,ej->ne',points[:,None]-a,delta)/np.einsum('ej,ej->e',delta,delta),0,1)
   distance=np.linalg.norm(points[:,None]-a-t[:,:,None]*delta,axis=-1).min(axis=1)
   available=[i for i,d in zip(available,distance) if d>=minimum_crease_distance]
 if not available or not existing:raise ValueError('Need controlled anchors and free material points')
 uv=env.rest[:,:2];selected=[];anchors=existing.copy()
 for _ in range(min(count,len(available))):
  distances=np.linalg.norm(uv[available,None]-uv[anchors][None],axis=-1).min(1);i=available[int(np.argmax(distances))];selected.append(i);anchors.append(i);available.remove(i)
 plane_ids=None
 if plane_control_vertices is not None:
  plane_ids=list(plane_control_vertices)
  if regions is not None or len(plane_ids)<3 or len(set(plane_ids))!=len(plane_ids) or any(type(i)is not int or i not in targets for i in plane_ids):raise ValueError('Plane controls must be distinct existing public target vertices; cannot combine with regions')
  plane_targets={i:targets[i] for i in plane_ids}
 else:plane_targets=targets
 origins,normals=control_planes(env.rest,plane_targets,selected,regions)
 if directional:
  for i,normal in zip(selected,normals):env.step(dict(op='attach_soft_grips',vertices=[i],stiffness=float(stiffness),normal=normal.tolist()))
 else:env.step(dict(op='attach_soft_grips',vertices=selected,stiffness=float(stiffness)))
 for k in range(1,steps+1):
  current=env.positions()[selected];gap=np.einsum('ij,ij->i',current-origins,normals)
  targets=current-gap[:,None]*min(1,k/120)*normals
  frames=max(1,int(np.ceil(max(np.linalg.norm(p-env.soft_controls[i][:3]) for i,p in zip(selected,targets))/.00075)))
  env.step(dict(op='move_soft_grips',targets={i:p.tolist() for i,p in zip(selected,targets)},frames=frames))
  if on_step:on_step(k)
 return dict(vertices=selected,plane_control_vertices=plane_ids,origins=origins.tolist(),normals=normals.tolist(),stiffness_N_m=stiffness,steps=steps,directional=directional,crease_exclusion_margin_m=minimum_crease_distance,excluded_vertices=sorted(excluded),region_count=len(regions['faces']) if regions else 1,scope='Sparse ideal material-point springs use planes fitted to public actuator targets. '+('Rank-one springs leave both tangential directions free throughout the nonlinear solve. ' if directional else 'Isotropic springs follow current tangential positions between solves. ')+'Not a rigid platen or externally accessible gripper certificate; unsampled nodes remain physically solved.')
