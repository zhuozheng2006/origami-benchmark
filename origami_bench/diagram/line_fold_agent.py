"""General straight-fold constructions; no dataset, task IDs or target access.

Line alignment is origami axiom 3: parallel midline or angular bisector.
All geometry is in original material coordinates, independent of the renderer.
The existing physical controller still decides whether a request is executable.
"""
import json
import numpy as np
from .skills import validate_fold

VERSION = 'line-fold-v2'
LINES = {
    'bottom': [[0, 0], [1, 0]], 'top': [[0, 1], [1, 1]],
    'left': [[0, 0], [0, 1]], 'right': [[1, 0], [1, 1]],
    'horizontal_middle': [[0, .5], [1, .5]],
    'vertical_middle': [[.5, 0], [.5, 1]],
    'diagonal_rising': [[0, 0], [1, 1]],
    'diagonal_falling': [[0, 1], [1, 0]],
}

GUIDE = '''You control full straight folds from numbered origami diagrams.
Image 1 is the instruction diagram. Image 2 is the current physical paper.
Read ONLY the next unexecuted numbered panel; END is a comparison, not another instruction.
Return one JSON action, no prose. Available operations:
1. {"op":"align_lines","source":LINE,"target":LINE,"branch":"near","sense":SENSE}
This folds the source edge onto the target line by 180 degrees. It computes a midline for parallel lines and an angle bisector for intersecting lines. The source edge selects the moving flap. Choose the construction that matches the DASHED crease and curved arrow in the image. The crease itself is NOT the target line: target is where the moving edge lands.
LINE may be a name or two numeric points [[x1,y1],[x2,y2]]. Names: bottom, top, left, right, horizontal_middle, vertical_middle, diagonal_rising, diagonal_falling. diagonal_rising connects original bottom-left to top-right; diagonal_falling connects original top-left to bottom-right.
branch="near" selects the bisector of the smaller angle between the two unoriented lines (the unique midline for parallels). branch="far" selects the other perpendicular bisector for intersecting lines. Do not default every fold to the middle of the paper.
2. {"op":"fold_line","line":[[x1,y1],[x2,y2]],"moving_point":[x,y],"sense":SENSE,"angle_deg":180}
Use this for any explicit straight crease. moving_point is clearly INSIDE the moving side. Numeric coordinates refer to the original square: bottom-left (0,0), bottom-right (1,0), top-right (1,1), top-left (0,1). x right, y up. They do NOT renormalize after a fold. Named lines also ALWAYS refer to the original material sheet.
SENSE is "toward" for V / dashed / valley (toward initial front viewer) and "away" for M / dash-dot / mountain. Read the arrowhead to identify where the moving flap goes.
Every accepted action completes the full fold, not one simulation time step. After ALL numbered instructions are accepted, return {"op":"finish"}; use {"op":"undo"} to undo a wrong accepted action. If the diagram cannot be read, return {"op":"cannot_read"}. Unsupported geometry is rejected, never silently repaired.'''


def line_value(value):
    if isinstance(value, str):
        if value not in LINES:
            raise ValueError('Unknown line name')
        value = LINES[value]
    a = np.asarray(value, dtype=float)
    if a.shape != (2, 2) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Line must contain two finite points in original [0,1] square')
    if np.linalg.norm(a[1]-a[0]) < .05:
        raise ValueError('Line endpoints are too close')
    return a


def direction(line):
    d = line[1]-line[0]
    d = d / np.linalg.norm(d)
    if d[0] < -1e-10 or (abs(d[0]) <= 1e-10 and d[1] < 0):
        d = -d
    return d


def clip_line(origin, d):
    hits = []
    for k in (0, 1):
        if abs(d[k]) < 1e-10:
            continue
        for edge in (0, 1):
            p = origin + (edge-origin[k])/d[k]*d
            if (p >= -1e-9).all() and (p <= 1+1e-9).all() and all(np.linalg.norm(p-h) > 1e-8 for h in hits):
                hits.append(np.clip(p, 0, 1))
    if len(hits) != 2 or np.linalg.norm(hits[0]-hits[1]) < .05:
        raise ValueError('Constructed crease does not cross the sheet')
    return np.asarray(hits)


def to_fold(action):
    if action in ({'op': 'finish'}, {'op': 'undo'}):
        return dict(action)
    if not isinstance(action, dict):
        raise ValueError('Expected one JSON action')
    if action.get('op') == 'fold_line':
        if set(action) != {'op', 'line', 'moving_point', 'sense', 'angle_deg'}:
            raise ValueError('Invalid fold_line fields')
        result = {**action, 'op': 'fold'}
        validate_fold(result)
        return result
    if action.get('op') != 'align_lines' or set(action) != {'op', 'source', 'target', 'branch', 'sense'}:
        raise ValueError('Expected align_lines with source, target, branch, sense')
    if action['branch'] not in ('near', 'far') or action['sense'] not in ('toward', 'away'):
        raise ValueError('Invalid branch or sense')
    s, t = line_value(action['source']), line_value(action['target'])
    ds, dt = direction(s), direction(t)
    if ds @ dt < 0:
        dt = -dt
    normal = np.array([-ds[1], ds[0]])
    cross = ds[0]*dt[1]-ds[1]*dt[0]
    if abs(cross) < 1e-9:
        distance = (t[0]-s[0]) @ normal
        if abs(distance) < .04 or action['branch'] != 'near':
            raise ValueError('Coincident/too close parallel lines or nonexistent far branch')
        origin, d = s[0]+normal*distance/2, ds
    else:
        u = np.linalg.solve(np.column_stack([ds, -dt]), t[0]-s[0])[0]
        origin = s[0]+u*ds
        d = ds+dt if action['branch'] == 'near' else ds-dt
        d /= np.linalg.norm(d)
    line = clip_line(origin, d)
    n = np.array([-d[1], d[0]])
    side = (s-origin) @ n
    if side.min() < -1e-8 and side.max() > 1e-8:
        raise ValueError('Source segment straddles crease; specify fold_line moving_point explicitly')
    moving = s[np.argmax(np.abs(side))]
    result = dict(op='fold', line=line.tolist(), moving_point=moving.tolist(), sense=action['sense'], angle_deg=180)
    validate_fold(result)
    return result


def prompt(observation):
    completed = len(observation['accepted_actions'])
    total = observation['task']['step_count']
    public = dict(instruction=observation['task']['instruction'], paper_frame=observation['task']['paper_frame'],
                  completed_instructions=completed, total_instructions=total,
                  next_numbered_panel=completed+1 if completed < total else None,
                  accepted_actions=observation['accepted_actions'], last_result=observation['previous_action_result'])
    state = (f'All {total} complete folds have been accepted. Inspect the current state and return finish or undo.'
             if completed >= total else f'Execute numbered panel {completed+1} of {total}. Do not repeat earlier panels.')
    return GUIDE+'\nCURRENT PROGRESS: '+state+'\n'+json.dumps(public, ensure_ascii=False)
