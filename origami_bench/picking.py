"""Select a visible material vertex using the same current mesh shown to the user."""
import numpy as np
def ray_hits(vertices,faces,origin,direction):
 v=np.asarray(vertices,float);f=np.asarray(faces,int);o=np.asarray(origin,float);d=np.asarray(direction,float);d=d/np.linalg.norm(d)
 a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]];e1=b-a;e2=c-a;h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h);valid=abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);s=o-a;u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);w=inv*(q@d);t=inv*np.einsum('ij,ij->i',e2,q)
 valid&=(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)&(t>1e-9)
 return np.where(valid,t,np.inf)
def pick_visible_vertex(vertices,faces,origin,direction):
 v=np.asarray(vertices,float);f=np.asarray(faces,int);o=np.asarray(origin,float);d=np.asarray(direction,float);d=d/np.linalg.norm(d);dist=ray_hits(v,f,o,d);face=int(dist.argmin())
 if not np.isfinite(dist[face]):return None
 hit=o+d*dist[face]
 for index in sorted(f[face],key=lambda i:np.linalg.norm(v[i]-hit)):
  delta=v[index]-o;length=np.linalg.norm(delta);occluders=ray_hits(v,f,o,delta);incident=np.any(f==index,axis=1)
  if not np.any(occluders[~incident]<length-1e-7):return int(index)
 return None
