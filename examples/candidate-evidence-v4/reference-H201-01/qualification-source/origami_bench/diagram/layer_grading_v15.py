"""Private v15 grading: ordered geometric checkpoints + band-distributed crease memory.
Never imported by prompts or the agent controller.
"""
import numpy as np
from .document_grading import crease_check, reference_creases, retained_history
from .layer_path_grading import ordered_checkpoints


def _canonical_dir_and_offset(p0, p1):
    vec = np.asarray(p1[:2], float) - np.asarray(p0[:2], float)
    length = float(np.linalg.norm(vec))
    if length < 1e-12:
        return None
    d = vec / length
    if d[0] < -1e-9 or (abs(d[0]) <= 1e-9 and d[1] < 0):
        d = -d
    n = np.array([-d[1], d[0]], float)
    mid = 0.5 * (np.asarray(p0[:2], float) + np.asarray(p1[:2], float))
    s = float(mid @ n)
    t = float(mid @ d)
    return d, n, s, t, length


def extract_band_creases(history, rest_vertices, band_halfwidth_m=0.0022, threshold_degrees=140.0):
    """Identify crease bands whose cumulative signed dihedral across parallel strip hinges exceeds threshold."""
    branch = retained_history(history)
    if not branch or not all('angles' in h and 'hinges' in h for h in branch):
        return []
    angles = np.asarray([h['angles'] for h in branch], float)
    hinges = branch[0]['hinges']
    rest = np.asarray(rest_vertices, float)

    meta = []
    for hi, st in enumerate(hinges):
        res = _canonical_dir_and_offset(rest[int(st[1])], rest[int(st[2])])
        if res is not None:
            d, n, s, t, length = res
            meta.append((hi, d, s, t, length))

    # Cluster hinges by canonical angle (rounded to 0.01 rad) and transverse offset s (within band_halfwidth_m)
    clusters = []
    for hi, d, s, t, length in meta:
        placed = False
        for cl in clusters:
            if abs(d @ cl['d']) > np.cos(np.deg2rad(2.0)) and abs(s - cl['s']) <= band_halfwidth_m:
                cl['items'].append((hi, s, t, length))
                placed = True
                break
        if not placed:
            clusters.append(dict(d=d, s=s, items=[(hi, s, t, length)]))

    required_bands = []
    threshold = np.deg2rad(threshold_degrees)
    for cl in clusters:
        items = cl['items']
        h_ids = [hi for hi, _, _, _ in items]
        weights = np.array([length for _, _, _, length in items], float)
        total_t_span = max(t for _, _, t, _ in items) - min(t for _, _, t, _ in items) + max(weights)
        if total_t_span < 0.01:
            continue
        # Across a band of parallel strip rows at the same longitudinal span,
        # the weighted sum of dihedral angles per unit longitudinal span gives the net bend angle.
        # Approximate longitudinal span by the max single-row length sum along the centre line:
        s_vals = np.array([s for _, s, _, _ in items], float)
        s_center = float(np.median(s_vals))
        center_rows = [length for _, s, _, length in items if abs(s - s_center) < 1e-6]
        span = sum(center_rows) if center_rows else max( total_t_span, 1e-6 )
        band_angles = (angles[:, h_ids] * weights[None, :]).sum(axis=1) / max(span, 1e-6)
        signs = [int(sign) for sign in (-1, 1) if float(np.max(band_angles * sign)) > threshold]
        if signs:
            required_bands.append(
                dict(
                    direction=np.round(cl['d'], 6).tolist(),
                    offset_m=round(s_center, 6),
                    hinge_ids=h_ids,
                    weights=np.round(weights / max(span, 1e-6), 6).tolist(),
                    signs=signs,
                )
            )
    return required_bands


def band_crease_memory_check(history, required_bands, threshold_degrees=130.0):
    """Verify that every required crease band achieved the required signed bend angle along the retained branch."""
    if not required_bands:
        return True
    branch = retained_history(history)
    if not branch or not all('angles' in h for h in branch):
        return False
    angles = np.asarray([h['angles'] for h in branch], float)
    threshold = np.deg2rad(threshold_degrees)
    for band in required_bands:
        h_ids = [int(i) for i in band['hinge_ids']]
        weights = np.asarray(band['weights'], float)
        band_angles = (angles[:, h_ids] * weights[None, :]).sum(axis=1)
        for sign in band['signs']:
            if float(np.max(band_angles * int(sign))) <= threshold:
                return False
    return True


__all__ = [
    'ordered_checkpoints',
    'crease_check',
    'reference_creases',
    'retained_history',
    'extract_band_creases',
    'band_crease_memory_check',
]
