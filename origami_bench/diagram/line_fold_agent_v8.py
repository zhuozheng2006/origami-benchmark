"""State-aware public protocol. No task IDs, measurements or target access."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .line_fold_agent import to_fold
from .skills import validate_fold

VERSION='line-fold-v8'
GUIDE='''Image 1 shows only the CURRENT numbered origami instruction. Image 2 shows the actual paper, NOT a requested action.
Return ONE JSON action. Do not copy an earlier action simply because it succeeded.
Read the arrow from its TAIL to its ARROWHEAD. The tail lies on the moving flap; the arrowhead points toward its destination.
Choose the whole OUTER EDGE carried by that flap, not whichever edge happens to be nearest the drawn arrow tail. A vertical crease moves the LEFT or RIGHT side; a horizontal crease moves TOP or BOTTOM. For an oblique crease, identify which original outer edge reflects onto the destination line and check that its bisector matches the drawn crease.
Use {"op":"align_lines","source":EDGE,"target":LINE,"branch":"near","sense":SENSE}.
EDGE: top, bottom, left, right. LINE: top, bottom, left, right, horizontal_middle, vertical_middle, diagonal_rising, diagonal_falling.
Names always refer to the ORIGINAL square: x points RIGHT, y points UP. Do not renormalize the current folded outline.
horizontal_middle: y=0.5; vertical_middle: x=0.5. diagonal_rising: original bottom-left to top-right (y=x); diagonal_falling: top-left to bottom-right (y=1-x).
The source edge is reflected ONTO the target line. The drawn crease lies BETWEEN them; it is not itself the target line.
SENSE: toward for V / dashed / valley; away for M / dash-dot / mountain. Each action completes a 180-degree fold.
For another straight crease use {"op":"fold_line","line":[[x1,y1],[x2,y2]],"moving_point":[x,y],"sense":SENSE,"angle_deg":180} in ORIGINAL [0,1] coordinates.
Undo: {"op":"undo"}. If unreadable: {"op":"cannot_read"}. Internal simulator op "fold" is NOT part of your API.
Completed history below is a record of YOUR actions, not a suggestion or a certified correct answer. Read the current instruction; do not repeat a completed full fold. Rejected actions did not change the paper.'''

def public_result(obs):
    r=obs.get('previous_action_result')
    return {k:r[k] for k in ('status','category','detail','rollback_verified') if k in r} if r else None

def prompt(obs):
    n=len(obs['accepted_actions']);total=obs['task']['step_count']
    if n>=total:
        # Do not echo internal fold commands, and do not show old instruction arrows.
        return ('REVIEW PHASE. All '+str(total)+' numbered instructions have been executed. Image 1 is the END panel of the public diagram; Image 2 is the actual paper. '
                'This is NOT another fold instruction. If the paper matches the diagram, return exactly {"op":"finish"}. '
                'If your last fold was wrong, return exactly {"op":"undo"}. These are the ONLY two valid actions in this phase. '
                'Do not repeat or reformat earlier folds. Acceptance does not certify diagram correctness.\n'+json.dumps(dict(phase='review',executed_instructions=n,total_instructions=total,last_result=public_result(obs))))
    history=[dict(instruction=i+1,action={**a,'op':'fold_line'}) for i,a in enumerate(obs['accepted_actions'])]
    return GUIDE+'\nCURRENT STATE: '+json.dumps(dict(phase='execute',current_instruction=n+1,total_instructions=total,completed=history,paper_frame=obs['task']['paper_frame'],last_result=public_result(obs)))

def prepare_inputs(obs,output,turn):
    total=obs['task']['step_count'];n=min(len(obs['accepted_actions']),total)
    im=Image.open(obs['diagram']).convert('RGB');w,h=im.size
    if not .65<=w/(total+1)/h<=1.3:raise ValueError('Unsupported public panel layout')
    box=(round(w*n/(total+1)),0,round(w*(n+1)/(total+1)),h)
    p=Path(output)/f'policy-panel-{turn:03d}.png';im.crop(box).save(p)
    p.with_suffix('.json').write_text(json.dumps(dict(source=obs['diagram'],crop_pixels=box,accepted_instruction_count=len(obs['accepted_actions']),public_step_count=total,phase='review' if n==total else 'execute')))
    return [str(p),obs['current_image']]

def same_full_fold(a,b):
    la,_,na,aa=validate_fold(a);lb,_,nb,ab=validate_fold(b)
    return abs(aa)==180 and aa==ab and na@nb>1-1e-10 and np.max(np.abs((la-lb[0])@nb))<1e-9

def prepare_action(action,obs):
    n=len(obs['accepted_actions']);total=obs['task']['step_count']
    if action=={'op':'cannot_read'}:raise ValueError('model_abstention: diagram unreadable')
    if n>=total and action not in ({'op':'finish'},{'op':'undo'}):
        raise ValueError('phase_error: all numbered folds have already executed; return finish or undo, not another fold')
    if n<total and action=={'op':'finish'}:
        raise ValueError('phase_error: numbered instructions remain; read the current panel or undo a wrong action')
    fold=to_fold(action)
    if fold.get('op')=='fold' and any(same_full_fold(fold,a) for a in obs['accepted_actions']):
        raise ValueError('duplicate_action: this same complete material fold was already applied; no state changed. Read the CURRENT panel; undo first only if you intend to correct that earlier fold')
    return fold

def record_rejection(episode,action,error):
    message=str(error);prefix=message.split(':',1)[0]
    category=prefix if prefix in ('model_abstention','phase_error','duplicate_action') else 'invalid_action_geometry' if any(x in message for x in ('straddles','Constructed crease','Coincident','moving_point')) else 'model_protocol_error'
    state=episode.backend.state()['hash']
    result=dict(status='rejected',category=category,detail=message,rollback_verified=True)
    episode.events.append(dict(index=len(episode.events),instruction_index=len(episode.actions),action=action,before=state,after=state,low_level_actions=[],result=result))
    from .episode import write_json
    write_json(episode.output/'events.json',episode.events)
    return result
