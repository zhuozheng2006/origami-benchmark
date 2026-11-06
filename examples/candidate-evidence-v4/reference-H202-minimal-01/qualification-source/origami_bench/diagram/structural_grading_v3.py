"""Candidate structural + diagram-motion contract, preserving material vertex IDs.

XYZ translation is nuisance placement, not shape. A single proper yaw is fitted
at the prepared state and kept for the whole illustrated process; therefore a
required paper rotation is still observable. No reflection, scale or SO(3) fit.
This grades evidence; it does not certify the physical validity of that evidence.
"""
import numpy as np

VERSION = 'structure-and-diagram-motion-v3-candidate1'
RMS_MM = 1.0
MAX_MM = 3.0


def _vertices(value):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError('At least three finite labelled XYZ material vertices required')
    return a


def initial_yaw(actual, expected):
    a, b = _vertices(actual), _vertices(expected)
    if a.shape != b.shape:
        raise ValueError('Material correspondence mismatch')
    aa, bb = a[:, :2] - a[:, :2].mean(0), b[:, :2] - b[:, :2].mean(0)
    u, s, vt = np.linalg.svd(aa.T @ bb)
    if s[-1] < 1e-12:
        raise ValueError('Prepared state does not determine a unique board yaw')
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def compare_placement(actual, expected, yaw=None):
    a, b = _vertices(actual), _vertices(expected)
    if a.shape != b.shape:
        raise ValueError('Material correspondence mismatch')
    yaw = initial_yaw(a, b) if yaw is None else np.asarray(yaw, float)
    if yaw.shape != (2, 2) or not np.isfinite(yaw).all() or not np.allclose(yaw.T @ yaw, np.eye(2), atol=1e-10) or not np.isclose(np.linalg.det(yaw), 1):
        raise ValueError('Only a proper board yaw is allowed')
    fit = a.copy()
    fit[:, :2] = fit[:, :2] @ yaw
    offset = b.mean(0) - fit.mean(0)
    fit += offset
    d = np.linalg.norm(fit - b, axis=1) * 1000
    rms, maximum = float(np.sqrt(np.mean(d*d))), float(d.max())
    return dict(rms_mm=rms, max_mm=maximum, passed=rms < RMS_MM and maximum < MAX_MM,
                placement_offset_mm=(1000 * offset).tolist(),
                yaw_degrees=float(np.rad2deg(np.arctan2(yaw[0, 1], yaw[0, 0]))),
                translation_allowed='XYZ', reflection_allowed=False, arbitrary_3d_rotation_allowed=False)


def ordered_motion_checkpoints(vertices, checkpoints, initial_target, start_frame=0):
    """Distinct semantic states must have disjoint acceptance neighbourhoods.

    Separation uses the SAME fixed-yaw material-coordinate metric as matching.
    If RMS distance > 2 mm OR max distance > 6 mm, no single state can satisfy
    both strict (<1 mm RMS, <3 mm max) balls. This rejects near-duplicate task
    specifications, not merely repeated frame indices. Static frame padding
    cannot turn an ambiguous specification into a valid multi-step process.
    """
    v, targets = np.asarray(vertices, float), np.asarray(checkpoints, float)
    if v.ndim != 3 or targets.ndim != 3 or len(v) == 0 or len(targets) == 0 or v.shape[1:] != targets.shape[1:] or not np.isfinite(v).all() or not np.isfinite(targets).all():
        raise ValueError('Nonempty finite matching trajectory/checkpoints required')
    if type(start_frame) is not int or not 0 <= start_frame < len(v):
        raise ValueError('Invalid prepared-state frame')
    initial = _vertices(initial_target)
    yaw = initial_yaw(v[start_frame], initial)
    initial_shape = compare_placement(v[start_frame], initial, yaw)
    if not initial_shape['passed']:
        return dict(passed=False, contract_valid=True, reason='prepared_state_mismatch', matched=0,
                    required=len(targets), matches=[], initial_shape=initial_shape)
    separations, previous = [], initial
    for i, target in enumerate(targets):
        distance = compare_placement(target, previous, np.eye(2))
        distinct = distance['rms_mm'] > 2*RMS_MM or distance['max_mm'] > 2*MAX_MM
        separations.append(dict(checkpoint=i, distinct=distinct, rms_mm=distance['rms_mm'], max_mm=distance['max_mm']))
        if not distinct:
            return dict(passed=False, contract_valid=False, reason='ambiguous_checkpoint_transition',
                        ambiguous_checkpoint=i, matched=0, required=len(targets), matches=[], separations=separations)
        previous = target
    cursor, matches = start_frame + 1, []
    for i, target in enumerate(targets):
        best = None
        for j in range(cursor, len(v)):
            shape = compare_placement(v[j], target, yaw)
            item = dict(checkpoint=i, frame=j, rms_mm=shape['rms_mm'], max_mm=shape['max_mm'])
            if best is None or shape['rms_mm'] < best['rms_mm']:
                best = item
            if shape['passed']:
                matches.append(item)
                cursor = j + 1
                break
        else:
            return dict(passed=False, contract_valid=True, reason='required_state_not_reached_in_order',
                        matched=len(matches), required=len(targets), matches=matches, next_best=best,
                        separations=separations, fixed_initial_yaw_degrees=initial_shape['yaw_degrees'])
    return dict(passed=True, contract_valid=True, reason='all_distinct_states_reached', matched=len(matches),
                required=len(targets), matches=matches, separations=separations,
                fixed_initial_yaw_degrees=initial_shape['yaw_degrees'])


def combine_success(shape, path, finished, physical_pass, memory_pass, environment_error):
    # Require explicit True evidence; missing/unknown validity never passes.
    return bool(shape.get('passed') is True and path.get('passed') is True and
                path.get('contract_valid') is True and finished is True and
                physical_pass is True and memory_pass is True and environment_error is False)


def physical_evidence_pass(metrics):
    """Only complete finite retained evidence can certify the scoped gates."""
    if not isinstance(metrics, list) or not metrics:
        return False
    for m in metrics:
        if not isinstance(m, dict) or m.get('clear') is not True or not isinstance(m.get('clearance'), dict) or m['clearance'].get('clear') is not True:
            return False
        for name, limit in [('stretch', .01), ('compression', .01), ('grip_error', .0005)]:
            value = m.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < limit:
                return False
    return True
