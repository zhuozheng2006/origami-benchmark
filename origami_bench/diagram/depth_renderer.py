"""Orthographic per-pixel z-buffer observations; no private task access."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
FRONT=np.array([188,219,239.]);BACK=np.array([255,221,155.])
def raster(vertices,faces,size=512,view='top'):
 v=np.asarray(vertices);f=np.asarray(faces);limit=.115
 if view=='top':u=np.array([1.,0,0]);up=np.array([0.,1,0]);camera=np.array([0.,0,1]);limit=.12
 else:
  e,a=np.deg2rad([28,-65]);camera=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]);u=np.array([-np.sin(a),np.cos(a),0]);up=np.cross(camera,u)
 p=np.column_stack([v@u,v@up]);screen=(p/limit/2+.5)*(size-1);screen[:,1]=(size-1)-screen[:,1];z=v@camera
 pixels=np.full((size,size,3),249.,dtype=float);depth=np.full((size,size),-np.inf);owner=np.full((size,size),-1,int)
 for i,tri in enumerate(f):
  q=screen[tri];xmin=max(0,int(np.floor(q[:,0].min())));xmax=min(size-1,int(np.ceil(q[:,0].max())));ymin=max(0,int(np.floor(q[:,1].min())));ymax=min(size-1,int(np.ceil(q[:,1].max())))
  if xmax<xmin or ymax<ymin:continue
  m=np.column_stack([q[1]-q[0],q[2]-q[0]])
  if abs(np.linalg.det(m))<1e-10:continue
  yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1];xy=np.stack([xx+.5,yy+.5],axis=-1);bary=(xy-q[0])@np.linalg.inv(m).T;valid=(bary[...,0]>=-1e-8)&(bary[...,1]>=-1e-8)&(bary.sum(-1)<=1+1e-8)
  h=z[tri[0]]+bary@(z[tri[1:]]-z[tri[0]]);part=depth[ymin:ymax+1,xmin:xmax+1];valid&=h>part
  normal=np.cross(v[tri[1]]-v[tri[0]],v[tri[2]]-v[tri[0]]);facing=normal@camera;shade=.78+.22*abs(facing)/max(np.linalg.norm(normal),1e-20)
  pixels[ymin:ymax+1,xmin:xmax+1][valid]=(FRONT if facing>=0 else BACK)*shade;part[valid]=h[valid];owner[ymin:ymax+1,xmin:xmax+1][valid]=i
 return Image.fromarray(np.clip(pixels,0,255).astype('uint8')),owner

def render_public(state,path):
 size=512;canvas=Image.new('RGB',(1120,620),'white');draw=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
 except OSError:font=ImageFont.load_default()
 for i,view in enumerate(['top','oblique']):
  im,_=raster(state['vertices'],state['faces'],size,view);x=42+i*560;y=48;canvas.paste(im,(x,y));draw.text((x,y-30),'ACTUAL PAPER - '+view.upper(),fill='#253f50',font=font)
  if i==0:
   for t in [-.25,0,.25,.5,.75,1,1.25]:
    pos=((t-.5)*.15/.24+.5)*(size-1);draw.text((x+pos-10,y+size+7),str(t),fill='#294456',font=font);draw.text((x-32,y+size-1-pos-8),str(t),fill='#294456',font=font)
   draw.text((x+180,590),'board x (right), y (up)',fill='#294456',font=font)
 draw.text((620,580),'Blue: initial front | cream: initial back',fill='#294456',font=font);canvas.save(path)
