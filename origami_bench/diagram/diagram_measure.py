"""Public-pixel measurement for colored, numbered, front-view diagram strips.

No data files or task identifiers are accepted. Dashed creases are groups of
collinear short components; arrow heads are thick filled parts of a curved
connected component. Uncertain/unsupported styles reject instead of guessing.
"""
import numpy as np
from scipy import ndimage as ndi
from PIL import ImageDraw
from .line_fold_agent import clip_line


def measure(image, panel, steps):
    rgb=np.asarray(image.convert('RGB')).astype(float);h,w=rgb.shape[:2]
    if not 0<=panel<steps:raise ValueError('No active numbered panel')
    first=rgb[:,:round(w/(steps+1))]
    color=(first.max(2)-first.min(2)>18)&(first.mean(2)>120)
    yy,xx=np.where(color)
    if len(xx)<.05*first.shape[0]*first.shape[1]:raise ValueError('No calibrated colored initial sheet')
    left,right,top,bottom=float(xx.min())-1.5,float(xx.max())+1.5,float(yy.min())-1.5,float(yy.max())+1.5
    scale=np.array([right-left,bottom-top])
    if abs(scale[0]/scale[1]-1)>.03:raise ValueError('Initial sheet is not an axis-aligned square')
    # Labels locate panel origins without using step-specific crop constants.
    dark=rgb.max(2)<130;header=dark.copy();header[max(0,int(top)-6):]=False
    lab,n=ndi.label(header);boxes=[]
    for sl in ndi.find_objects(lab):
        if sl and (sl[1].stop-sl[1].start)*(sl[0].stop-sl[0].start)>15 and sl[1].stop-sl[1].start<.09*scale[0] and sl[0].stop-sl[0].start<.12*scale[1]:
            boxes.append((sl[1].start,sl[1].stop))
    groups=[]
    for x0,x1 in sorted(boxes):
        if groups and x0-groups[-1][1]<.08*scale[0]:groups[-1][1]=x1
        else:groups.append([x0,x1])
    if len(groups)!=steps+1:raise ValueError('Ambiguous numbered panel layout')
    offset=groups[panel][0]-groups[0][0];left+=offset;right+=offset
    x0,x1=int(np.floor(left))+3,int(np.ceil(right))-2
    y0,y1=int(np.floor(top))+3,int(np.ceil(bottom))-2
    roi=dark[y0:y1,x0:x1];labels,n=ndi.label(roi,np.ones((3,3)))
    components=[];segments=[]
    for k,sl in enumerate(ndi.find_objects(labels),1):
        if sl is None:continue
        ys,xs=np.where(labels[sl]==k)
        p=np.column_stack([xs+sl[1].start+x0,ys+sl[0].start+y0]).astype(float)
        if len(p)<12:continue
        c=p.mean(0);_,sv,vh=np.linalg.svd(p-c,full_matrices=False)
        direction=vh[0];extent=np.ptp((p-c)@direction)
        components.append((k,p,sl))
        if 5<extent<.13*scale[0] and sv[0]/max(sv[1],1e-8)>2.5:
            segments.append((p,c,direction))
    candidates=[]
    for p,c,d in segments:
        normal=np.array([-d[1],d[0]])
        cluster=[s for s in segments if abs((s[1]-c)@normal)<2.5 and abs(s[2]@d)>.96]
        if len(cluster)<4:continue
        points=np.concatenate([s[0] for s in cluster]);center=points.mean(0)
        _,_,vh=np.linalg.svd(points-center,full_matrices=False);axis=vh[0]
        norm=np.array([-axis[1],axis[0]]);residual=np.sqrt(np.mean(((points-center)@norm)**2))
        extent=np.ptp((points-center)@axis)
        if residual<1.8 and extent>.25*scale[0]:candidates.append((len(cluster),extent,center,axis,residual))
    if not candidates:raise ValueError('No confident collinear dashed crease')
    count,extent,center,axis,residual=max(candidates,key=lambda c:(c[0],c[1]))
    normal=np.array([-axis[1],axis[0]])
    for c in candidates:
        if c[0]>=count-1 and (abs((c[2]-center)@normal)>3 or abs(c[3]@axis)<.98):
            raise ValueError('Multiple plausible dashed creases')
    # A filled triangular arrowhead is wider than its shaft or paper outline.
    arrow_candidates=[]
    for k,p,sl in components:
        if len(p)<80:continue
        component=labels[sl]==k
        distance=ndi.distance_transform_edt(np.pad(component,1))[1:-1,1:-1]
        thick=float(distance.max())
        if thick<3:continue
        ys,xs=np.where(distance>=thick*.7)
        head=np.array([xs.mean()+sl[1].start+x0,ys.mean()+sl[0].start+y0])
        tail=p[np.argmax(np.linalg.norm(p-head,axis=1))]
        if np.linalg.norm(head-tail)<.12*scale[0]:continue
        if ((head-center)@normal)*((tail-center)@normal)>=0:continue
        arrow_candidates.append((thick,head,tail))
    if not arrow_candidates:raise ValueError('No arrow crossing the measured crease')
    arrow_candidates.sort(key=lambda c:c[0],reverse=True)
    if len(arrow_candidates)>1 and arrow_candidates[1][0]>=arrow_candidates[0][0]-.5:
        raise ValueError('Ambiguous arrowheads')
    thick,head,tail=arrow_candidates[0]
    def uv(p):return np.array([(p[0]-left)/scale[0],1-(p[1]-top)/scale[1]])
    origin=uv(center);d=np.array([axis[0]/scale[0],-axis[1]/scale[1]]);d/=np.linalg.norm(d)
    crease=clip_line(origin,d);point=uv(tail)
    if (point<0).any() or (point>1).any():raise ValueError('Arrow tail outside material frame')
    result=dict(line=crease.round(7).tolist(),moving_point=point.round(7).tolist(),arrow_head=uv(head).round(7).tolist(),
                evidence=dict(panel=panel+1,sheet_frame_pixels=[left,top,right,bottom],dash_components=count,
                              fit_rms_pixels=float(residual),span_pixels=float(extent),arrowhead_radius_pixels=thick),
                scope='Measured only from public pixels; requires numbered aligned colored strip, straight dashed crease and isolated filled arrow')
    annotated=image.copy();draw=ImageDraw.Draw(annotated)
    def xy(p):return (float(left+p[0]*scale[0]),float(top+(1-p[1])*scale[1]))
    draw.line([xy(p) for p in crease],fill='#00a0b0',width=2)
    for p,col in [(tail,'#d000a0'),(head,'#00a040')]:
        x,y=p;draw.ellipse((x-7,y-7,x+7,y+7),outline=col,width=3)
    return result,annotated
