"""Admission rules for the next dataset, independent of model outcomes.

This is a data/protocol guard, not a new physics controller or executable runner.
No candidate becomes a scored task solely because its source looks difficult.
"""
TIERS = ('easy', 'medium', 'hard')
ATTEMPT_CAPS = {'easy': 40, 'medium': 100, 'hard': 400}

def admission_errors(task):
    errors=[]
    tier=task.get('tier')
    if tier not in TIERS:errors.append('unknown_tier')
    if task.get('no_cut_tear_glue_verified') is not True:errors.append('operation_scope_not_verified')
    if task.get('readability_verified') is not True:errors.append('diagram_not_verified')
    if task.get('annotation_verified') is not True:errors.append('action_annotation_not_verified')
    if task.get('renderer_depth_verified') is not True:errors.append('observation_renderer_not_verified')
    if task.get('family_split_verified') is not True:errors.append('family_split_not_verified')
    if task.get('historical_dedup_verified') is not True:errors.append('historical_dedup_not_verified')
    required=task.get('required_operations');supported=task.get('controller_verified_operations',[])
    if not required or not set(required)<=set(supported):errors.append('controller_capability_missing')
    length=task.get('expanded_reference_action_count')
    if type(length) is not int or length<1:errors.append('reference_length_unknown')
    elif tier in TIERS and 2*length+task.get('document_navigation_allowance',0)+1>ATTEMPT_CAPS[tier]:
        errors.append('attempt_budget_inadequate')
    runs=task.get('reference_replays',[])
    ids=[r.get('run_id') for r in runs]
    if len(runs)<2 or not all(ids) or len(set(ids))!=len(ids):errors.append('need_two_distinct_reference_replays')
    for r in runs:
        if not all(r.get(k) is True for k in ('shape_pass','strain_pass','contact_pass','grip_pass','same_controller','same_initialization','trace_hash_verified')):
            errors.append('reference_physical_evidence_not_passed');break
    if task.get('experiment_freeze_verified') is not True:errors.append('experiment_not_frozen')
    return errors

def validate_group_split(rows):
    """Variants/windows/pages of the same connected source family stay together."""
    seen={}
    for row in rows:
        split=row.get('split')
        if split not in ('development','test'):raise ValueError('Split is not assigned')
        groups=row.get('split_group_ids',[])
        if not groups:raise ValueError('Missing design/source/variant family groups')
        for group in groups:
            if group in seen and seen[group]!=split:raise ValueError('Family leakage: '+group)
            seen[group]=split
    return True
