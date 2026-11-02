"""Single prepared-state v15 policy with 3-view depth observation description, identical for all sources and models."""
import json
from .document_policy import prepare_action, prepare_inputs
from .layer_policy import GUIDE as V14_GUIDE

VERSION = 'prepared-layers-v15-1'

GUIDE = V14_GUIDE.replace(
    'image 2 is the ACTUAL current paper with a fixed board coordinate scale.',
    'image 2 shows the ACTUAL current paper in three depth-rendered views: TOP (+z with the fixed board coordinate grid), OBLIQUE (+z), and BOTTOM OBLIQUE (-z underside view so lower-layer flaps are visible).',
)


def prompt(obs):
    return GUIDE + '\nPUBLIC STATE: ' + json.dumps(
        dict(
            task=obs['task'],
            document=obs['document'],
            paper_state_hash=obs['current']['state_hash'],
            preparation_actions=obs['preparation_actions'],
            active_folds=obs['active_folds'],
            retained_paper_actions=obs['accepted_actions'],
            last_result=obs['previous_action_result'],
            history=obs['attempt_history'],
            budget=obs['attempt_budget'],
        ),
        ensure_ascii=False,
    )


__all__ = ['VERSION', 'GUIDE', 'prompt', 'prepare_inputs', 'prepare_action']
