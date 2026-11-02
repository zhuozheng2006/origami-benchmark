"""Two model readings, not answer repair: arrow motion then destination line."""
import json
from pathlib import Path
from .line_fold_agent import to_fold as geometric_to_fold
from .line_fold_agent_v3 import prepare_inputs, prompt as progress_prompt

VERSION='line-fold-v4'
EDGE={'down':'top','up':'bottom','left':'right','right':'left'}
TARGETS=('top','bottom','left','right','horizontal_middle','vertical_middle','diagonal_rising','diagonal_falling')
ARROW_PROMPT='''Read ONLY the curved arrow in this single numbered origami instruction panel. Locate its pointed ARROWHEAD and the opposite tail. What is the dominant direction of motion from tail to arrowhead? Choose up, down, left, or right. The edge where the arrow is DRAWN is not the direction. Ignore already folded yellow regions; follow the arrow in THIS panel.
Also read V / toward / dashed as toward, M / away / dash-dot as away.
Return only JSON {"direction":"up or down or left or right","sense":"toward or away"}. If unreadable return {"op":"cannot_read"}.'''


def prompt(obs):return progress_prompt(obs)


def to_fold(action):
    if isinstance(action,dict) and action.get('op')=='fold_edge':
        if set(action)!={'op','direction','target','sense'} or action['direction'] not in EDGE:
            raise ValueError('Invalid fold_edge fields/direction')
        return geometric_to_fold(dict(op='align_lines',source=EDGE[action['direction']],target=action['target'],branch='near',sense=action['sense']))
    return geometric_to_fold(action)


def act(agent,obs,output,turn,inputs):
    if len(obs['accepted_actions'])>=obs['task']['step_count']:
        return agent.request_public(prompt(obs),inputs)
    out=Path(output)
    arrow=agent.request_public(ARROW_PROMPT,[inputs[0]])
    (out/f'perception-arrow-{turn:03d}.json').write_text(json.dumps(arrow,ensure_ascii=False,indent=2))
    a=arrow['action']
    if not isinstance(a,dict) or set(a)!={'direction','sense'} or a['direction'] not in EDGE or a['sense'] not in ('toward','away'):
        return {**arrow,'action':{'op':'cannot_read'},'assembly_note':'Malformed or unreadable first reading; no default motion'}
    source=EDGE[a['direction']]
    text=f'''Image 1 is ONLY the current numbered instruction; Image 2 is the current physical paper.
The separate arrow reading says motion {a['direction']}, corresponding to the original {source} edge moving.
Where should that moving edge LAND after reflecting across the dashed crease? Identify the destination line, NOT the crease itself.
Choose target from: top, bottom, left, right, horizontal_middle, vertical_middle, diagonal_rising, diagonal_falling.
horizontal_middle joins original left/right edge midpoints. vertical_middle joins original bottom/top edge midpoints. diagonal_rising runs original bottom-left to top-right; diagonal_falling runs original top-left to bottom-right.
All names refer to the ORIGINAL square, unchanged after earlier folds. Read the active panel, not a prior instruction or the yellow region.
Return only JSON {{"target":"one line name"}}. If unclear return {{"op":"cannot_read"}}.
Public progress: {len(obs['accepted_actions'])} of {obs['task']['step_count']} complete instructions accepted. Current panel {len(obs['accepted_actions'])+1}.'''
    target=agent.request_public(text,inputs)
    (out/f'perception-target-{turn:03d}.json').write_text(json.dumps(target,ensure_ascii=False,indent=2))
    t=target['action']
    action=({'op':'fold_edge','direction':a['direction'],'target':t['target'],'sense':a['sense']}
            if isinstance(t,dict) and set(t)=={'target'} and t['target'] in TARGETS else {'op':'cannot_read'})
    return {**target,'action':action,'assembly_note':'Fields copied from two saved model responses; no answer lookup or correction',
            'arrow_response_file':f'perception-arrow-{turn:03d}.json','target_response_file':f'perception-target-{turn:03d}.json',
            'seconds':arrow['seconds']+target['seconds']}
