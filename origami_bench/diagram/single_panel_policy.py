"""Generic support for one public instruction image with no result panel.

Selection uses only public step count and image layout; no task ID or target.
The existing strip policy remains unchanged for multi-panel diagrams.
"""
from PIL import Image
from . import line_fold_agent_v9 as strip
VERSION='single-panel-v1-history'
prepare_action=strip.prepare_action

def select_policy(task,diagram):
    with Image.open(diagram) as im:w,h=im.size
    total=task['step_count']
    if .65<=w/(total+1)/h<=1.3:return strip
    if total==1 and .65<=w/h<=1.3:
        import sys
        return sys.modules[__name__]
    raise ValueError('Unsupported public panel layout')

def prepare_inputs(obs,output,turn):
    return [obs['diagram'],obs['current_image']]

def prompt(obs):
    text=strip.prompt(obs)
    if len(obs['accepted_actions'])>=obs['task']['step_count']:
        text=text.replace('Image 1 is the END panel of the public diagram;',
            'Image 1 repeats the original single instruction; this source has NO result panel. Compare your completed action with that instruction;')
    return text
