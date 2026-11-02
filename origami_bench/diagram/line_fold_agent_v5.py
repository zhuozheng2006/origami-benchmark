"""Explicit tool-assisted VLM baseline. Geometry comes ONLY from public pixels.

The model chooses the emitted command and fold sense. No reference/target reads,
no response repair, and no automatic execution of the measured geometry.
"""
import json
from pathlib import Path
from PIL import Image
from .line_fold_agent import to_fold
from .line_fold_agent_v3 import prepare_inputs, prompt as progress_prompt
from .diagram_measure import measure

VERSION='line-fold-v5-tools'
def prompt(obs):return progress_prompt(obs)

def act(agent,obs,output,turn,inputs):
    count=len(obs['accepted_actions']);total=obs['task']['step_count']
    if count>=total:return agent.request_public(prompt(obs),inputs)
    measured,overlay=measure(Image.open(obs['diagram']),count,total)
    out=Path(output)
    (out/f'pixel-measurement-{turn:03d}.json').write_text(json.dumps(measured,ensure_ascii=False,indent=2))
    overlay.save(out/f'pixel-measurement-{turn:03d}.png')
    text='''You operate a physical origami simulator with a public-image measurement tool.
Image 1 is the current numbered diagram panel. Image 2 is the actual paper.
The tool measured the dashed crease and arrow tail DIRECTLY from pixels. Its coordinates refer to the ORIGINAL sheet, x right, y up, normalized [0,1]. The arrow tail selects the moving material side. Tool measurements are NOT reference answers and may be rejected by the controller.
Read the fold sense from the diagram: V / toward / dashed means "toward"; M / away / dash-dot means "away".
To execute the measured straight fold, output one JSON {"op":"fold_line","line":MEASURED_LINE,"moving_point":MEASURED_ARROW_TAIL,"sense":"toward or away","angle_deg":180}.
Copy the tool coordinates accurately; do not reinterpret them as image pixels or move the crease to the middle. Do not silently retry a previous instruction. You may instead return {"op":"cannot_read"} if the diagram and measurement disagree.
'''+json.dumps({'current_instruction':count+1,'total_instructions':total,'measured_line':measured['line'],
                 'measured_arrow_tail':measured['moving_point'],'measurement_evidence':measured['evidence'],
                 'last_result':obs['previous_action_result']})
    response=agent.request_public(text,inputs)
    return {**response,'measurement_file':f'pixel-measurement-{turn:03d}.json','policy_track':'public_pixel_tool_assisted_vlm'}
