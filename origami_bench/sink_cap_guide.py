"""Isometric four-triangle cap guide; geometry only, never physical acceptance.

Vertex order is A, B, D, C, S. Opening DC from 20 to 40 mm permits the
internal folds to reach a flat state and then reverse. Merely swinging S
around a fixed DC axis cannot perform that intrinsic crease reversal.
"""
import numpy as np

FACES = np.array([[4,0,2],[4,2,1],[4,1,3],[4,3,0]])
SEGMENT_EDGES = {8:(4,0), 9:(4,1), 10:(4,3), 11:(4,2)}
PREPARED_ANGLES = {8:180., 9:180., 10:180., 11:-180.}


def cap_positions(stage, t):
    if stage not in ('lift','invert','lower') or not np.isfinite(t) or not 0<=t<=1:
        raise ValueError('Invalid cap stage or progress')
    vertices = np.array([[0.,-.03,0.],[0.,-.03,0.],[0.,0.,0.],[0.,.02,0.],[.015,0.,0.]])
    if stage != 'invert':
        beta = np.pi/2*(t if stage=='lift' else 1+t)
        vertices[4] = [.015*np.cos(beta),0.,(.015 if stage=='lift' else -.015)*np.sin(beta)]
        return vertices
    if t==.5:
        return np.array([[-.03,0.,0.],[.03,0.,0.],[0.,0.,0.],[0.,.04,0.],[0.,.015,0.]])
    if t in (0,1):
        vertices[4] = [0.,0.,.015 if t==0 else -.015]
        return vertices
    distance = .02+.02*np.sin(np.pi*t)**2
    qa = (.03**2+distance**2-.05**2)/(2*distance)
    qs = (.015**2+distance**2-.025**2)/(2*distance)
    ra = np.sqrt(max(0.,.03**2-qa**2))
    rs = np.sqrt(max(0.,.015**2-qs**2))
    cosine = np.clip(-qa*qs/(ra*rs),-1,1) if ra*rs>1e-15 else 0.
    sine = np.sqrt(max(0.,1-cosine**2))
    branch = 1 if t<=.5 else -1
    vertices[0] = [ra*(2*cosine**2-1),qa,branch*2*ra*cosine*sine]
    vertices[1] = [ra,qa,0.]
    vertices[3] = [0.,distance,0.]
    vertices[4] = [rs*cosine,qs,branch*rs*sine]
    return vertices


def cap_angles(stage, t):
    if stage=='lift' or (stage=='invert' and t==0):
        return PREPARED_ANGLES.copy()
    if stage=='lower' or (stage=='invert' and t==1):
        return {key:-angle for key,angle in PREPARED_ANGLES.items()}
    vertices = cap_positions(stage,t)
    result = {}
    for segment,(i,j) in SEGMENT_EDGES.items():
        f,g = [face.tolist() for face in FACES if i in face and j in face]
        if f[(f.index(i)+1)%3]!=j:
            i,j = j,i
        x = next(k for k in f if k not in (i,j))
        y = next(k for k in g if k not in (i,j))
        n = np.cross(vertices[i]-vertices[x],vertices[j]-vertices[x])
        m = np.cross(vertices[j]-vertices[y],vertices[i]-vertices[y])
        direction = vertices[j]-vertices[i]
        direction /= np.linalg.norm(direction)
        result[segment] = float(np.rad2deg(np.arctan2(direction@np.cross(n,m),n@m)))
    return result
