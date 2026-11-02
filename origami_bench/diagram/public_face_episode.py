"""Versioned public renderer: actual face orientation, no private targets."""
import numpy as np
from .episode import Episode
FRONT='#d5e5f1';BACK='#ffe1a3'
def face_colors(vertices,faces,elev,azim):
 v=np.asarray(vertices);f=np.asarray(faces);n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
 e,a=np.deg2rad([elev,azim]);camera=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
 return [FRONT if x>=0 else BACK for x in n@camera]
def render_public(state,path):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 v=np.asarray(state['vertices']);f=np.asarray(state['faces']);fig=plt.figure(figsize=(8,4),facecolor='white')
 for k,(e,a) in enumerate([(90,-90),(25,-65)]):
  ax=fig.add_subplot(1,2,k+1,projection='3d');colors=face_colors(v,f,e,a)
  ax.add_collection3d(Poly3DCollection(v[f]*1000,facecolors=colors,edgecolors=colors,linewidths=.15,alpha=1))
  ax.set(xlim=(-85,85),ylim=(-85,85),zlim=(-85,85));ax.set_box_aspect((1,1,1));ax.view_init(e,a)
  ax.set_xlabel('x / mm');ax.set_ylabel('y / mm');ax.set_zlabel('z / mm');ax.set_title('Top view' if k==0 else 'Oblique view')
 fig.suptitle('ACTUAL PAPER | Blue: front; cream: back',fontsize=10);fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
class PublicFaceEpisode(Episode):
 def observation(self):
  obs=super().observation();render_public(self.backend.state(),obs['current_image']);return obs
