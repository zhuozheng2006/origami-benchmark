"""Restricted half-fold skill: no task IDs, targets, references or inference fixes.

Direction names the movement, not the location where an arrow is drawn.
The same mapping applies to all square sheets and supports all four directions.
"""
import json

HALF_GUIDE='''Available skill: half_fold. It folds a square completely in half (180 degrees) along its horizontal or vertical midline.
Return ONE JSON action: {"op":"half_fold","direction":"up or down or left or right","sense":"toward or away"}.
Choose direction from the curved arrow's motion: its pointed head is the destination. The arrow's placement near a paper edge is not the fold direction. Up/down use a horizontal middle crease; left/right use a vertical middle crease.
Sense: V / valley / dashed crease means toward; M / mountain / dash-dot crease means away.
Inspect the numbered instruction panel in Image 1, not the END panel. Image 2, if present, is the current simulator state.
Only use this skill if the image shows a full middle half-fold. If the diagram is missing, unclear or requires another skill, return {"op":"cannot_read"}. Do not guess a default direction.
No numeric coordinates are requested.'''

def half_to_fold(action):
    if action in ({'op':'finish'},{'op':'undo'}):return dict(action)
    if not isinstance(action,dict) or set(action)!={'op','direction','sense'} or action['op']!='half_fold':
        raise ValueError('Expected half_fold, direction, sense; unreadable diagrams do not execute')
    if action['sense'] not in ('toward','away'):raise ValueError('Unknown fold sense')
    geometry={
        'down':([[0,.5],[1,.5]],[.5,.75]),
        'up':([[0,.5],[1,.5]],[.5,.25]),
        'left':([[.5,0],[.5,1]],[.75,.5]),
        'right':([[.5,0],[.5,1]],[.25,.5]),
    }
    if action['direction'] not in geometry:raise ValueError('Unknown direction')
    line,point=geometry[action['direction']]
    return dict(op='fold',line=line,moving_point=point,sense=action['sense'],angle_deg=180)

def half_prompt(observation):
    completed=len(observation['accepted_actions'])
    total=observation['task']['step_count']
    if completed>=total:
        return ('Image 1 is the instruction diagram with its END panel. Image 2 is the actual current paper. '
                f'The simulator accepted {completed} complete fold instruction(s), of {total} numbered instruction(s). '
                'Each accepted half_fold already completed the full 180-degree motion, not one small time step. '
                'Check the result. Return {"op":"finish"} if the requested folding is done, '
                'or {"op":"undo"} if the last move is wrong. Do not repeat an already completed instruction. '
                'Finishing does not bypass private scoring.\n'+json.dumps({'accepted_actions':observation['accepted_actions'],'last_result':observation['previous_action_result']}))
    return HALF_GUIDE+'\n'+json.dumps({'instruction':observation['task']['instruction'],'accepted_instruction_count':completed,'last_result':observation['previous_action_result']})
