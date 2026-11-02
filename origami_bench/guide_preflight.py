"""Reject malformed or explicitly unfinished guides before starting a physical run.

This check does not certify contact, material response or a folding operation.
"""
import numpy as np


def validate_flat_start_guide(report,vertices):
    v=np.asarray(vertices,dtype=float);uv=np.asarray(report['coarse']['vertices'],dtype=float)
    if uv.ndim!=2 or uv.shape[1]!=2 or v.ndim!=3 or v.shape[1:]!=(len(uv),3) or len(v)<2:
        raise ValueError('Guide dimensions do not match material control points')
    if not np.isfinite(v).all() or not np.isfinite(uv).all():
        raise ValueError('Guide contains non-finite positions')
    initial_error=float(np.linalg.norm(v[0]-np.column_stack((uv,np.zeros(len(uv)))),axis=1).max())
    if initial_error>1e-6:
        raise ValueError('This driver starts from flat material; guide begins in a different state')
    if report.get('reached_destination') is False or report.get('valid') is False:
        raise ValueError('Planner explicitly reports an unfinished or invalid path')
    records=report.get('records',[])
    if not records:raise ValueError('Guide continuity records are missing')
    values=np.asarray([[r.get('closure_m',np.nan),r.get('jump_degrees',np.nan)] for r in records])
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Guide continuity metrics are invalid')
    closure,jump=values.max(axis=0)
    if closure>1e-6 or jump>5:raise ValueError('Guide closure or angular jump exceeds its bound')
    if report.get('nonrigid_guide'):
        strain=report.get('guide_principal_strain_max',np.nan)
        if not np.isfinite(strain) or not 0<=strain<.01:raise ValueError('Nonrigid guide strain is missing or exceeds its bound')
    return dict(flat_start_max_error_m=initial_error,max_closure_m=float(closure),max_angle_jump_degrees=float(jump),
                planner_destination_claim=report.get('reached_destination'),physical_certified=False,
                scope='Input consistency and declared geometric prerequisites only; independent physical acceptance still required.')
