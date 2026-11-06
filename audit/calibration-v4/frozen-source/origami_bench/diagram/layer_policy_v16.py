"""Uniform layer policy with explicit scope precedence and public motion bounds."""
import json
from .layer_policy_v15 import GUIDE as PRIOR_GUIDE, prepare_action, prepare_inputs

VERSION = 'prepared-layers-v16-scope-and-motion-bounds1'

GUIDE = PRIOR_GUIDE.replace(
    'Follow the illustrated intermediate states in order as well as the final shape. Different action decompositions are allowed, but skipping a demonstrated fold-and-open sequence is not full diagram completion.',
    'Follow the explicit task scope and completion rules. By default, follow illustrated intermediate states in order as well as the final shape. If the task explicitly permits equivalent action orders or moving halves, that permission takes precedence. Required fold-and-open operations must still be physically completed; an unfolded outline alone does not prove that precreases were made.',
)
GUIDE += '\nPublic controller capability: each rotation is discretized to at most 600 integration frames, with at most 0.5 degree rotation and 0.8 mm outer-grip arc travel per frame. Thus a 180 degree rotation of a moving region whose farthest gripped point is more than about 153 mm from the axis can exceed this limit. A large complementary half may be unsupported even when the equivalent smaller moving half is supported. When the task permits equivalent halves, choose a supported one. A capability_limit rejection is an interface bound, consumes an attempt, and does not mean the intended crease is wrong. Physical contact can also reject otherwise valid intended folds. These bounds are shared by all tasks and models.'
GUIDE += '\nPaper views may expand to keep the actual paper visible. Read the TOP coordinate ticks: one board unit is always 15 cm, and coordinates are never renormalized to the visible outline. Only the view extent changes.'


def prompt(obs):
    return GUIDE + '\nPUBLIC STATE: ' + json.dumps(dict(
        task=obs['task'], document=obs['document'],
        paper_state_hash=obs['current']['state_hash'],
        preparation_actions=obs['preparation_actions'], active_folds=obs['active_folds'],
        retained_paper_actions=obs['accepted_actions'],
        last_result=obs['previous_action_result'], history=obs['attempt_history'],
        budget=obs['attempt_budget'],
    ), ensure_ascii=False)


__all__ = ['VERSION', 'GUIDE', 'prompt', 'prepare_inputs', 'prepare_action']
