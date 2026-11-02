"""Constructive rigid-panel references for named folding vertices.
Only sparse points from these references are used as physical grips.
"""
import numpy as np

def rabbit_corners(degrees,gap=.0003):
 L=.075*np.sqrt(2);c=L*(np.sqrt(2)-1);r=L-c;t=c/L;a=-np.deg2rad(degrees)
 if abs(a)<1e-12:T=[0,L,0]
 else:
  cot=1/np.tan(a/2);x=2*r*t/(t*t*(1+cot*cot)+1);T=[x,c+r-x/t,-x*cot+.25*gap*(1-np.cos(a))]
 p=np.array([[0,-L,0],[L*np.cos(a),0,L*np.sin(a)-.5*gap*(1-np.cos(a))],T,[-L,0,0],[0,c,0]],float)
 R=np.array([[1,-1],[1,1]])/np.sqrt(2);p[:,:2]=p[:,:2]@R.T;return p

def star_map(uv,rest_corners,folded_corners):
 out=np.zeros((len(uv),3));assigned=np.zeros(len(uv),bool)
 for j in range(4):
  ids=[4,j,(j+1)%4];tri=rest_corners[ids,:2];weights=np.linalg.solve(np.vstack([tri.T,np.ones(3)]),np.vstack([np.asarray(uv).T,np.ones(len(uv))])).T;mask=(weights.min(axis=1)>-1e-7)&(~assigned);out[mask]=weights[mask]@folded_corners[ids];assigned[mask]=True
 if not assigned.all():raise ValueError('Material points outside star panels')
 return out

def square_base(uv,degrees,gap=.0003):
 L=.075;R=np.array([[1,-1],[1,1]])/np.sqrt(2);u=np.asarray(uv)@R;a=np.deg2rad(degrees);sn=np.sin(a);co=np.cos(a);q=L*co*co/(1+sn*sn);h=2*L*sn/(1+sn*sn)
 rays=np.array([[-L,-L],[L,-L],[L,0],[L,L],[-L,L],[-L,0]])
 targets=np.array([[-L,-L*co-gap*sn,L*sn],[L,-L*co-gap*sn,L*sn],[q,-gap*sn/3,h],[L,L*co+gap*sn,L*sn],[-L,L*co+gap*sn,L*sn],[-q,gap*sn/3,h]])
 out=np.zeros((len(u),3));assigned=np.zeros(len(u),bool)
 for j in range(6):
  weights=np.linalg.solve(rays[[j,(j+1)%6]].T,u.T).T;mask=(weights.min(axis=1)>-1e-7)&(~assigned);out[mask]=weights[mask]@targets[[j,(j+1)%6]];assigned[mask]=True
 assert assigned.all();out[:,:2]=out[:,:2]@R.T;return out
