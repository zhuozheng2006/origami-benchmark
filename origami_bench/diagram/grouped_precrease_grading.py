"""Candidate grouped-v4 contract for real-source precreasing tasks.

Group order follows explicit diagram stages; distinct targets inside one group
may be reached in either order. Flat re-depictions are NOT motion targets. The
same material-labelled state cannot count twice, even if copied to more frames.
This module only grades retained evidence. It never simulates or changes paper.
"""
import numpy as np

from .structural_grading_v3 import (
    RMS_MM, MAX_MM, initial_yaw, compare_placement, combine_success,
)

VERSION = 'grouped-precrease-structure-and-memory-v4-candidate1'


def _targets(groups, vertex_shape):
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError('Nonempty ordered group list required')
    result=[]
    for gi, group in enumerate(groups):
        if not isinstance(group, dict) or 'targets' not in group:
            raise ValueError('Each group requires a targets array')
        array=np.asarray(group['targets'],float)
        if array.ndim!=3 or array.shape[1:]!=vertex_shape or len(array)==0 or not np.isfinite(array).all():
            raise ValueError('Groups require nonempty finite labelled target states')
        result.append(dict(id=str(group.get('id',gi)),targets=array))
    if len({g['id'] for g in result})!=len(result):
        raise ValueError('Group identifiers must be unique')
    return result


def _distance_distinct(a,b):
    score=compare_placement(a,b,np.eye(2))
    return dict(distinct=score['rms_mm']>2*RMS_MM or score['max_mm']>2*MAX_MM,
                rms_mm=score['rms_mm'],max_mm=score['max_mm'])


def _trajectory_scores(vertices,target,yaw):
    """Fixed-yaw + XYZ placement distances with bounded working memory."""
    target=target-target.mean(0)
    rms=np.empty(len(vertices));maximum=np.empty(len(vertices))
    for lo in range(0,len(vertices),512):
        hi=min(lo+512,len(vertices));actual=vertices[lo:hi].copy()
        actual[:,:,:2]=actual[:,:,:2]@yaw
        actual-=actual.mean(axis=1,keepdims=True)
        error=np.linalg.norm(actual-target[None,:,:],axis=2)*1000
        rms[lo:hi]=np.sqrt(np.mean(error*error,axis=1));maximum[lo:hi]=error.max(axis=1)
    return rms,maximum


def grouped_motion_checkpoints(vertices,groups,initial_target,start_frame=0):
    """Ordered groups, injective unordered target coverage within each group.

All target acceptance regions must be disjoint under the labelled fixed-yaw
metric (RMS separation >2mm OR maximum >6mm). This version rejects duplicate
targets across groups as well: repeated depictions must be explicitly coalesced
by annotation, not padded with static frames. Legitimate non-repeated sampled
paths and multi-action decompositions can still reach the required states.
    """
    frames=np.asarray(vertices,float)
    if frames.ndim!=3 or frames.shape[2]!=3 or frames.shape[1]<3 or len(frames)==0 or not np.isfinite(frames).all():
        raise ValueError('Nonempty finite labelled XYZ trajectory required')
    if type(start_frame) is not int or not 0<=start_frame<len(frames):
        raise ValueError('Invalid prepared-state frame')
    prepared=np.asarray(initial_target,float)
    if prepared.shape!=frames.shape[1:] or not np.isfinite(prepared).all():
        raise ValueError('Prepared state does not match labelled trajectory')
    target_groups=_targets(groups,frames.shape[1:])
    total=sum(len(g['targets']) for g in target_groups)
    yaw=initial_yaw(frames[start_frame],prepared)
    initial_shape=compare_placement(frames[start_frame],prepared,yaw)
    base=dict(required=total,matched=0,matches=[],group_matches=[],required_groups=len(groups),
              completed_groups=0,contract_valid=True,fixed_initial_yaw_degrees=initial_shape['yaw_degrees'])
    if not initial_shape['passed']:
        return dict(base,passed=False,reason='prepared_state_mismatch',initial_shape=initial_shape)
    flattened=[(gi,ti,target) for gi,g in enumerate(target_groups) for ti,target in enumerate(g['targets'])]
    separations=[]
    for i,(gi,ti,target) in enumerate(flattened):
        distinct=_distance_distinct(target,prepared)
        if not distinct['distinct']:
            return dict(base,passed=False,contract_valid=False,reason='target_duplicates_prepared_state',
                        ambiguous_target=dict(group=gi,target=ti),separation=distinct)
        for gj,tj,other in flattened[:i]:
            distinct=_distance_distinct(target,other)
            separations.append(dict(group=gi,target=ti,other_group=gj,other_target=tj,**distinct))
            if not distinct['distinct']:
                return dict(base,passed=False,contract_valid=False,reason='overlapping_target_acceptance_regions',
                            ambiguous_target_pair=[[gj,tj],[gi,ti]],separation=distinct)
    cursor=start_frame+1;matches=[];group_matches=[]
    for gi,group in enumerate(target_groups):
        scores=[_trajectory_scores(frames,target,yaw) for target in group['targets']]
        found=[];missing=[];next_best=[]
        for ti,(rms,maximum) in enumerate(scores):
            ids=np.flatnonzero((rms[cursor:]<RMS_MM)&(maximum[cursor:]<MAX_MM))+cursor
            if not len(ids):
                missing.append(ti)
                if cursor<len(frames):
                    j=cursor+int(np.argmin(rms[cursor:]));next_best.append(dict(target=ti,frame=j,rms_mm=float(rms[j]),max_mm=float(maximum[j])))
                continue
            j=int(ids[0]);found.append(dict(group=gi,group_id=group['id'],target=ti,frame=j,rms_mm=float(rms[j]),max_mm=float(maximum[j])))
        # Disjoint tolerance balls imply unique matching frame indices. Check
        # explicitly as a fail-closed guard against future metric changes.
        if len({m['frame'] for m in found})!=len(found):
            return dict(base,passed=False,contract_valid=False,reason='noninjective_target_match')
        matches.extend(sorted(found,key=lambda m:m['frame']))
        group_matches.append(dict(group=gi,group_id=group['id'],matched=len(found),required=len(group['targets']),
                                  frames=[m['frame'] for m in sorted(found,key=lambda m:m['frame'])],missing_targets=missing))
        if missing:
            return dict(base,passed=False,reason='required_group_not_covered',matched=len(matches),matches=matches,
                        completed_groups=gi,group_matches=group_matches,first_incomplete_group=gi,
                        next_best=next_best,separations=separations)
        cursor=max(m['frame'] for m in found)+1
    return dict(base,passed=True,reason='all_ordered_groups_covered',matched=len(matches),matches=matches,
                completed_groups=len(groups),group_matches=group_matches,separations=separations)


def grade_grouped_precrease(actual_frames,initial_frame,initial_target,final_target,groups,*,
                            physical_pass,memory_pass,finished,environment_error=False):
    """Conjoin geometry, retained process, signed memory, physical gates, finish.

The caller computes memory_pass from retained physical history and a nonempty
qualified signed-hinge/band specification; this module cannot infer it from
flat final vertices. Unknown gates fail. Placement invariance is not evidence
of physical translation or contact safety.
    """
    try:
        path=grouped_motion_checkpoints(actual_frames,groups,initial_target,initial_frame)
        shape=compare_placement(np.asarray(actual_frames,float)[-1],final_target)
    except (TypeError,ValueError,IndexError,np.linalg.LinAlgError) as error:
        return dict(grading_version=VERSION,task_success=False,structure_success=False,input_valid=False,
                    error_type=type(error).__name__,reason=str(error),
                    diagram_path=dict(passed=False,contract_valid=False,reason='invalid_evidence_or_contract'),
                    physical_pass=physical_pass,memory_pass=memory_pass,finished=finished,environment_error=environment_error)
    return dict(grading_version=VERSION,input_valid=True,shape=shape,diagram_path=path,
                structure_success=shape['passed'],physical_pass=physical_pass,memory_pass=memory_pass,
                finished=finished,environment_error=environment_error,
                task_success=combine_success(shape,path,finished,physical_pass,memory_pass,environment_error))
