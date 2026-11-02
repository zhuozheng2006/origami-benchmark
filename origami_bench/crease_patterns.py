"""Initial, uncut crease-aligned material meshes and an independent rigid-panel oracle."""
import math,xml.etree.ElementTree as ET
import numpy as np,triangle

def from_svg(path):
 root=ET.parse(path).getroot();segments=[]
 for el in root.iter():
  tag=el.tag.split('}')[-1];color=el.get('stroke','').upper();assignment={'#000000':'B','#FF0000':'M','#0000FF':'V'}.get(color)
  if not assignment:continue
  angle=(1 if assignment=='M' else -1)*180*float(el.get('opacity','1')) if assignment!='B' else 0
  if tag=='line':pairs=[([float(el.get('x1')),float(el.get('y1'))],[float(el.get('x2')),float(el.get('y2'))])]
  elif tag=='rect':
   x,y,w,h=[float(el.get(k)) for k in ['x','y','width','height']];p=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]];pairs=[(p[i],p[(i+1)%4]) for i in range(4)]
  else:raise ValueError('Unsupported SVG geometry: '+tag)
  segments.extend(dict(a=a,b=b,assignment=assignment,angle=angle) for a,b in pairs)
 boundary=np.array([p for s in segments if s['assignment']=='B' for p in [s['a'],s['b']]])
 low=boundary.min(0);high=boundary.max(0);size=high-low
 if abs(size[0]-size[1])/max(size)>1e-4:raise ValueError('Expected an uncut square boundary')
 for s in segments:
  for key in ['a','b']:
   p=(np.array(s[key])-(low+high)/2)*(.15/max(size));p[1]*=-1;s[key]=np.round(p,6).tolist()
  if s['assignment']=='B' and not any(abs(abs(s['a'][k])-.075)<1e-6 and abs(s['a'][k]-s['b'][k])<1e-6 for k in [0,1]):raise ValueError('Boundary contains a cut or a non-square outline')
 return segments

def mesh_segments(segments,area=.00012,quality=True,extra_vertices=None):
 lines=[(np.array(s['a']),np.array(s['b'])) for s in segments];cuts=[[0.,1.] for _ in lines]
 def cross(a,b):return a[0]*b[1]-a[1]*b[0]
 for i,(a,b) in enumerate(lines):
  for j,(c,d) in enumerate(lines[:i]):
   u=b-a;v=d-c;den=cross(u,v)
   if abs(den)>1e-12:
    t=cross(c-a,v)/den;q=cross(c-a,u)/den
    if -1e-8<=t<=1+1e-8 and -1e-8<=q<=1+1e-8:cuts[i].append(float(np.clip(t,0,1)));cuts[j].append(float(np.clip(q,0,1)))
   else:
    for point in [c,d]:
     t=(point-a)@u/(u@u)
     if -1e-8<=t<=1+1e-8 and abs(cross(point-a,u))<1e-10:cuts[i].append(float(np.clip(t,0,1)))
    for point in [a,b]:
     q=(point-c)@v/(v@v)
     if -1e-8<=q<=1+1e-8 and abs(cross(point-c,v))<1e-10:cuts[j].append(float(np.clip(q,0,1)))
 vertices=[];lookup={};edges={}
 def index(p):
  key=tuple(np.round(p,6))
  if key not in lookup:lookup[key]=len(vertices);vertices.append(key)
  return lookup[key]
 for k,((a,b),ts) in enumerate(zip(lines,cuts)):
  ids=[index(a+t*(b-a)) for t in sorted(set(ts))]
  for i,j in zip(ids,ids[1:]):
   if i==j:continue
   key=tuple(sorted((i,j)))
   if key in edges and abs(segments[edges[key]-1]['angle']-segments[k]['angle'])>1e-8:raise ValueError('Conflicting crease assignments')
   edges[key]=k+1
 for point in extra_vertices or []:
  point=np.asarray(point,float)
  if point.shape!=(2,) or not np.isfinite(point).all() or max(abs(point))>=.075:raise ValueError('Extra refinement points must lie inside the square')
  for a,b in lines:
   t=np.clip((point-a)@(b-a)/((b-a)@(b-a)),0,1)
   if np.linalg.norm(point-a-t*(b-a))<2e-6:raise ValueError('Refinement point is too close to an unsplit segment')
  index(point)
 result=triangle.triangulate(dict(vertices=np.array(vertices),segments=np.array(list(edges)),segment_markers=np.array(list(edges.values()))),'pq25a'+format(area,'.12f') if quality else 'p')
 uv=result['vertices'];faces=result['triangles'];crease=[]
 if quality:
  measured=[abs(cross(uv[b]-uv[a],uv[c]-uv[a]))/2 for a,b,c in faces]
  if max(measured)>area*(1+1e-6):raise ValueError('Triangulator did not enforce requested maximum area')
 for e,marker in zip(result['segments'],result['segment_markers'].ravel()):
  s=segments[int(marker)-1]
  if s['assignment']!='B':crease.append(dict(vertices=e.tolist(),assignment=s['assignment'],angle_degrees=s['angle'],segment_index=int(marker)-1))
 unique_edges=set(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
 assert len(uv)-len(unique_edges)+len(faces)==1
 area_sum=sum(abs(cross(uv[b]-uv[a],uv[c]-uv[a]))/2 for a,b,c in faces)
 assert abs(area_sum-.15**2)<1e-7
 return dict(vertices=uv.tolist(),faces=faces.tolist(),crease_edges=crease,segments=segments,initial_remesh_only=True,uncut_square_area_m2=float(area_sum),coordinate_rounding_m=1e-6)

def rigid_target(meta):
 uv=np.array(meta['vertices']);rest=np.c_[uv,np.zeros(len(uv))];faces=np.array(meta['faces']);angles={tuple(sorted(e['vertices'])):np.deg2rad(e['angle_degrees']) for e in meta['crease_edges']};owners={}
 for i,f in enumerate(faces):
  for a,b in zip(f,np.roll(f,-1)):owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
 transforms={0:(np.eye(3),np.zeros(3))};queue=[0];closure=[]
 while queue:
  i=queue.pop(0);R,t=transforms[i];face=faces[i]
  for a,b in zip(face,np.roll(face,-1)):
   key=tuple(sorted((int(a),int(b))));phi=angles.get(key,0.);p=R@rest[a]+t;axis=R@(rest[b]-rest[a]);axis/=np.linalg.norm(axis);x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);Q=np.eye(3)+np.sin(phi)*K+(1-np.cos(phi))*(K@K);nextR=Q@R;nextt=Q@(t-p)+p
   for j in owners[key]:
    if j==i:continue
    if j not in transforms:transforms[j]=(nextR,nextt);queue.append(j)
    else:
     oldR,oldt=transforms[j];closure.append(np.linalg.norm((rest[faces[j]]@nextR.T+nextt)-(rest[faces[j]]@oldR.T+oldt),axis=1).max())
 assert len(transforms)==len(faces)
 predictions=[[] for _ in rest]
 for i,(R,t) in transforms.items():
  for j in faces[i]:predictions[j].append(R@rest[j]+t)
 target=np.array([np.mean(p,axis=0) for p in predictions]);error=max(closure,default=0.)
 return target,float(error)
