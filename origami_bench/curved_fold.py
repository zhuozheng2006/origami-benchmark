"""A continuous, isometric parabolic-crease folding family.

Extrude an arclength curve (X(u), lambda*g(u)) along v. Reflect v<g(u)
through the plane lambda*y-z=0. The crease is fixed in material coordinates
g(u)=k*u**2. Each side is a generalized cylinder. At lambda=0 the map is flat.
The polygonal guide uses chord lengths; the independent smooth reference uses
the exact arclength integral. No state of a simulator is assigned here.
"""
import numpy as np
L=.075
def material_mesh(nx=21,half_rows=5,k=3.):
 us=np.linspace(-L,L,nx);uv=[]
 for j in range(2*half_rows+1):
  for u in us:
   g=k*u*u
   v=-L+(g+L)*j/half_rows if j<=half_rows else g+(L-g)*(j-half_rows)/half_rows
   uv.append([u,v])
 faces=[]
 for j in range(2*half_rows):
  for i in range(nx-1):
   a=j*nx+i;faces.extend([[a,a+1,a+nx+1],[a,a+nx+1,a+nx]])
 return np.array(uv),np.array(faces,dtype=np.int32),[[half_rows*nx+i,half_rows*nx+i+1] for i in range(nx-1)]
def shape(uv,lam,k=3.,polygon=False):
 u,v=np.asarray(uv).T;g=k*u*u
 if lam==0:return np.c_[u,v,np.zeros_like(u)]
 if 2*k*L*lam>=1:raise ValueError('This developable family requires |lambda*g_prime|<1')
 if polygon:
  us=np.unique(u);gs=k*us**2;xx=np.r_[-L,-L+np.cumsum(np.sqrt(np.diff(us)**2-lam**2*np.diff(gs)**2))];x=xx[np.searchsorted(us,u)]
 else:
  c=2*k*lam
  def integral(t):return .5*(t*np.sqrt(1-(c*t)**2)+np.arcsin(c*t)/c)
  x=-L+integral(u)-integral(-L)
 out=np.c_[x,v,lam*g];lower=v<g;d=v[lower]-g[lower]
 out[lower,1]-=2*lam*lam*d/(1+lam*lam);out[lower,2]+=2*lam*d/(1+lam*lam)
 return out
def crease_angles(vertices,faces,edges):
 owners={}
 for f in faces:
  for a,b in zip(f,np.roll(f,-1)):owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(f)
 result=[]
 for edge in edges:
  f,g=owners[tuple(sorted(edge))];a,b=edge
  if not any(i==a and j==b for i,j in zip(f,np.roll(f,-1))):a,b=b,a
  n=np.cross(vertices[f[1]]-vertices[f[0]],vertices[f[2]]-vertices[f[0]]);n/=np.linalg.norm(n)
  m=np.cross(vertices[g[1]]-vertices[g[0]],vertices[g[2]]-vertices[g[0]]);m/=np.linalg.norm(m)
  axis=vertices[b]-vertices[a];axis/=np.linalg.norm(axis);result.append(np.rad2deg(np.arctan2(axis@np.cross(n,m),n@m)))
 return np.array(result)
