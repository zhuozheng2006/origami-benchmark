"""Same geometric API, focused public-panel observation and shorter instructions.

Uniform numbered horizontal strip layout is an explicit input-format assumption.
Cropping uses only public step_count and accepted instruction count, never IDs.
"""
from pathlib import Path
import json
from PIL import Image
from .line_fold_agent import to_fold

VERSION='line-fold-v3'
GUIDE='''Read the curved arrow and dashed crease in Image 1, the CURRENT instruction panel.
Image 2 is the current simulator state, not the instruction.
Choose the OUTER EDGE that moves, and the line on which that edge lands.
Return exactly {"op":"align_lines","source":EDGE,"target":LINE,"branch":"near","sense":SENSE}.
EDGE names: "top", "bottom", "left", "right". These are edges, NOT corners.
LINE names: "top", "bottom", "left", "right", "horizontal_middle", "vertical_middle", "diagonal_rising", "diagonal_falling".
horizontal_middle is halfway between original top and bottom. vertical_middle is halfway between original left and right.
diagonal_rising runs from original bottom-left to top-right. diagonal_falling runs from original top-left to bottom-right.
The action reflects the source edge ONTO the target line. Its crease is the midline/angle bisector BETWEEN them. The dashed crease is NOT the target line.
SENSE: "toward" for V / dashed / valley; "away" for M / dash-dot / mountain.
All names refer to the ORIGINAL sheet, not the changing folded outline. The diagram's x is right and y is up.
Each action performs one complete 180-degree fold. Do not repeat a completed instruction.
For other straight creases the same API accepts {"op":"fold_line","line":[[x1,y1],[x2,y2]],"moving_point":[x,y],"sense":SENSE,"angle_deg":180} in original [0,1] material coordinates.
Return {"op":"cannot_read"} if unclear. Undo is {"op":"undo"}.'''


def prompt(obs):
    count=len(obs['accepted_actions']);total=obs['task']['step_count']
    if count>=total:
        return (f'The simulator has accepted ALL {total} complete numbered folds. Image 1 is the complete diagram, Image 2 is the actual current paper. '
                'If done return {"op":"finish"}; if the last move was wrong return {"op":"undo"}. Do not repeat a finished fold.\n'+
                json.dumps({'accepted_actions':obs['accepted_actions'],'last_result':obs['previous_action_result']}))
    return GUIDE+f'\nYou are executing instruction {count+1} of {total}. '+json.dumps({'completed_instructions':count,'paper_frame':obs['task']['paper_frame'],'last_result':obs['previous_action_result']})


def prepare_inputs(obs, output, turn):
    count=len(obs['accepted_actions']);total=obs['task']['step_count']
    if count>=total:return [obs['diagram'],obs['current_image']]
    im=Image.open(obs['diagram']).convert('RGB');width,height=im.size
    # Declared strip format: equal-width numbered panels plus END.
    if width/(total+1) < height*.65 or width/(total+1)>height*1.3:
        raise ValueError('Unsupported public panel layout; no guessed crop')
    region=(round(width*count/(total+1)),0,round(width*(count+1)/(total+1)),height)
    path=Path(output)/f'policy-panel-{turn:03d}.png';im.crop(region).save(path)
    (Path(output)/f'policy-panel-{turn:03d}.json').write_text(json.dumps({'source':str(obs['diagram']),'crop_pixels':region,'accepted_instruction_count':count,'public_step_count':total}))
    return [str(path),obs['current_image']]
