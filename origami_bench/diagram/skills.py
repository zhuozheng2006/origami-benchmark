"""Small action compiler over the SAME Codim controller used by the GUI.

No task IDs, target vertices, reference actions or recorded trajectories are read.
Only crease-aligned, initially flat-panel folds are in the v1 skill domain.
"""
import math
import numpy as np


class SkillUnsupported(ValueError):
    pass


def validate_fold(action):
    if not isinstance(action, dict) or set(action) != {
        'op', 'line', 'moving_point', 'sense', 'angle_deg'
    } or action['op'] != 'fold':
        raise ValueError('Expected fold with line, moving_point, sense, angle_deg')
    line = np.asarray(action['line'], float)
    point = np.asarray(action['moving_point'], float)
    angle = action['angle_deg']
    if line.shape != (2, 2) or point.shape != (2,) or not np.isfinite(line).all() or not np.isfinite(point).all():
        raise ValueError('Expected finite normalized material coordinates')
    if (line < 0).any() or (line > 1).any() or (point < 0).any() or (point > 1).any():
        raise ValueError('Material coordinates must be in [0,1]')
    if type(angle) not in (int, float) or not math.isfinite(angle) or not 0 < angle <= 180:
        raise ValueError('angle_deg must be in (0,180]')
    if action['sense'] not in ('toward', 'away'):
        raise ValueError('sense must be toward or away from initial +z viewer')
    d = line[1] - line[0]
    if np.linalg.norm(d) < .05:
        raise ValueError('Crease endpoints are too close')
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    signed = (point - line[0]) @ n
    if abs(signed) < .02:
        raise ValueError('moving_point must clearly select one side of the crease')
    if signed < 0:
        d, n = -d, -n
    return line, d, n, float(angle) * (1 if action['sense'] == 'toward' else -1)


def compile_fold(rest, vertices, faces, action):
    """Compile straight-line material folding to boundary grasps + arc rotation.

    This is a fixed ideal-fixture skill, not robot finger planning. The crease
    must exist in the frozen mesh. No snapping to a hidden expected crease.
    """
    line, d, n, degrees = validate_fold(action)
    rest, vertices, faces = np.asarray(rest), np.asarray(vertices), np.asarray(faces)
    uv = rest[:, :2]
    origin = (line[0] - .5) * .15
    signed = (uv - origin) @ n
    tri = signed[faces]
    if ((tri.min(1) < -1e-7) & (tri.max(1) > 1e-7)).any():
        # Public instrument quantization: snap at most 0.5 mm / 0.5 degrees
        # to a complete line ALREADY in the material mesh. No answer access.
        candidates=[]
        for face in faces:
            for i,j in zip(face,np.roll(face,-1)):
                ed=uv[j]-uv[i];length=np.linalg.norm(ed)
                if length<1e-8:continue
                ed/=length
                if ed@d<0:ed=-ed
                if ed@d<np.cos(np.deg2rad(.5)):continue
                en=np.array([-ed[1],ed[0]])
                error=np.max(np.abs(((line-.5)*.15-uv[i])@en))
                if error>.0005:continue
                sd=(uv-uv[i])@en;ft=sd[faces]
                if ((ft.min(1)<-1e-7)&(ft.max(1)>1e-7)).any():continue
                candidates.append((float(error),uv[i].copy(),ed.copy(),en.copy(),sd))
        if not candidates:raise SkillUnsupported('Crease crosses mesh faces and no complete mesh line is within 0.5mm / 0.5 degrees')
        _,origin,d,n,signed=min(candidates,key=lambda c:c[0])
    # Boundary-only controls; the interior remains free in the physical solver.
    boundary = np.max(np.abs(uv), axis=1) > .074999
    moving = np.flatnonzero(boundary & (signed > .008)).tolist()
    fixed = np.flatnonzero(boundary & (signed < -.008)).tolist()
    if len(moving) < 2 or len(fixed) < 2:
        raise SkillUnsupported('Insufficient boundary grips for this fold')
    crease = np.flatnonzero(np.abs(signed) < 1e-7)
    if len(crease) < 2:
        raise SkillUnsupported('No material crease vertices')
    # Align the axis to the *current* material crease, never a reference state.
    t = (uv[crease] - origin) @ d
    first, last = crease[np.argmin(t)], crease[np.argmax(t)]
    axis = vertices[last] - vertices[first]
    axis /= np.linalg.norm(axis)
    center = vertices[first].copy()
    distance = np.linalg.norm(np.cross(vertices[crease] - center, axis), axis=1)
    if distance.max() > .001:
        raise SkillUnsupported('Current crease is not sufficiently straight for this skill')
    # Supported panel is still in its initial plane (up to numerical motion).
    if np.ptp(vertices[moving, 2]) > .001 or np.max(np.abs(vertices[moving, 2])) > .002:
        raise SkillUnsupported('Folded multilayer flap needs a different, unimplemented skill')
    center[2] += math.copysign(.00015, degrees)
    radius = np.linalg.norm(np.cross(vertices[moving] - center, axis), axis=1).max()
    frames = max(int(math.ceil(abs(degrees) / .5)), int(math.ceil(radius * abs(math.radians(degrees)) / .0008)))
    if frames > 600:
        raise SkillUnsupported('Motion exceeds per-action frame budget')
    return [
        {'op': 'release'},
        {'op': 'set_grips', 'vertices': sorted(moving + fixed)},
        {'op': 'rotate_grips', 'vertices': moving, 'axis_point': center.tolist(),
         'axis_direction': axis.tolist(), 'degrees': degrees, 'frames': frames},
    ]


ACTION_GUIDE = {
    'coordinates': '15cm square. Initial material xy normalized to [0,1]^2; x right, y up, initial viewer at +z. Coordinates always refer to the original sheet.',
    'fold': {'op': 'fold', 'line': [[0, .3], [1, .3]], 'moving_point': [.25, .1], 'sense': 'away', 'angle_deg': 90},
    'semantics': 'line gives two crease points; moving_point selects the material side to move. toward lifts that side toward the initial viewer; away moves behind. A flat fold is 180 degrees. Example is illustrative, not the answer.',
    'also_available': [{'op': 'undo'}, {'op': 'finish'}],
    'limits': 'Pre-aligned mesh, ideal boundary grips, straight folds of still-flat panels only. Requested lines may quantize to an existing complete mesh line within 0.5mm and 0.5 degrees, based ONLY on mesh geometry. No runtime remeshing. Unknown/unimplemented actions explicitly reject. The controller never loads an answer trajectory.',
}
