"""Arc-length preserving polygonal fold band, used for sparse grip path planning.
This geometric recipe is NOT a physical trajectory. The shell solver still advances
all ungripped vertices and certifies each accepted physical substep independently.
"""
import numpy as np

def fold_x(vertices,center,halfwidth,segments,degrees,side):
 if side not in (-1,1) or halfwidth<=0 or segments<2 or not 0<=degrees<=180:raise ValueError('invalid fold band')
 v=np.asarray(vertices,float);out=v.copy();u=-side*(v[:,0]-center);s=np.maximum(halfwidth-u,0);length=2*halfwidth/segments;phi=np.deg2rad(degrees);dx=np.zeros(len(v));dz=np.zeros(len(v))
 for j in range(segments):
  traveled=np.clip(s-j*length,0,length);angle=(j+.5)*phi/segments;dx+=traveled*np.cos(angle);dz+=traveled*np.sin(angle)
 extra=np.maximum(s-2*halfwidth,0);dx+=extra*np.cos(phi);dz+=extra*np.sin(phi);active=u<halfwidth;out[active,0]=center-side*(halfwidth-dx[active]);out[active,2]+=dz[active]
 return out

def letter_mesh():
 first=dict(center=-.0253,halfwidth=.0003,segments=3,side=-1);second=dict(center=.025,halfwidth=.0006,segments=6,side=1)
 x=sorted(set([-.075,-.06,-.05,-.04,-.01,0,.01,.04,.05,.06,.075]+[round(p['center']-p['halfwidth']+j*2*p['halfwidth']/p['segments'],9) for p in [first,second] for j in range(p['segments']+1)]));y=np.linspace(-.075,.075,7);uv=np.array([[a,b] for b in y for a in x]);faces=[]
 for j in range(len(y)-1):
  for i in range(len(x)-1):
   a=j*len(x)+i;b=a+1;c=a+len(x);d=c+1;faces.extend([[a,b,d],[a,d,c]])
 return dict(vertices=uv.tolist(),faces=faces,name='rounded_letter',initial_remesh_only=True,fold_bands=[first,second],scope='uncut square; three-fold sparse grip planning with finite-width crease bands; left fold offset 0.3mm for layer clearance')

def letter_reference(rest,first_degrees=180,second_degrees=180):
 m=letter_mesh();a,b=m['fold_bands'];v=fold_x(rest,**a,degrees=first_degrees);mask=rest[:,0]>=b['center']-b['halfwidth']-1e-10;v[mask]=fold_x(rest[mask],**b,degrees=second_degrees);return v
