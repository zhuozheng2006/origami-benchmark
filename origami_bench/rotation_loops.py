"""Vertex rotation closure and its analytic Jacobian for geometry-only planning."""
import numpy as np

class RotationLoops:
 def __init__(self,mesh):
  uv=np.array(mesh['vertices']);edges=mesh['crease_edges'];self.n=len(edges);self.loops=[];self.last=None
  for i,x in enumerate(uv):
   if np.any(np.isclose(abs(x),.075,atol=1e-8)):continue
   loop=[]
   for j,e in enumerate(edges):
    if i in e['vertices']:
     k=next(k for k in e['vertices'] if k!=i);d=uv[k]-x;d/=np.linalg.norm(d);a=np.r_[d,0];K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0.]])
     loop.append((np.arctan2(d[1],d[0]),j,K,K@K))
   if loop:self.loops.append(sorted(loop,key=lambda z:z[0]))
 def evaluate(self,theta):
  if self.last is not None and np.array_equal(theta,self.last):return self.value,self.jacobian
  I=np.eye(3);rows=[];J=np.zeros((len(self.loops)*9,self.n))
  for n,loop in enumerate(self.loops):
   matrices=[I+np.sin(theta[j])*K+(1-np.cos(theta[j]))*K2 for _,j,K,K2 in loop]
   prefix=[I]
   for R in matrices:prefix.append(prefix[-1]@R)
   suffix=[I]*(len(loop)+1)
   for k in range(len(loop)-1,-1,-1):suffix[k]=matrices[k]@suffix[k+1]
   rows.extend((prefix[-1]-I).ravel())
   for k,(_,j,K,K2) in enumerate(loop):J[n*9:(n+1)*9,j]=(prefix[k]@(np.cos(theta[j])*K+np.sin(theta[j])*K2)@suffix[k+1]).ravel()
  self.last=np.array(theta,copy=True);self.value=np.array(rows);self.jacobian=J
  return self.value,self.jacobian
