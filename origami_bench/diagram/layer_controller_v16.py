"""Current-space, layer-aware boundary & cross-seam fixture controller (v15).
No task/reference data. Physical reverse motion is distinct from snapshot undo.
"""
import copy
import math
import numpy as np
from .skills import SkillUnsupported
from .layer_controller import (
    SCALE,
    boundary_vertices,
    motion,
    panel_hull_grips,
    point_hit,
    rotation,
)

VERSION = 'layer-boundary-v16-dev1'


def compile_stack_fold_v15(vertices, faces, action, fixture_mode='boundary'):
    thickness_panel = fixture_mode == 'thickness_panel'
    force_panel = fixture_mode in ('full_panel', 'thickness_panel')
    if force_panel:
        fixture_mode = 'distributed_panel'
    soft_stiffness = 1000.0
    if fixture_mode.startswith('soft_distributed_'):
        soft_stiffness = float(fixture_mode.removeprefix('soft_distributed_'))
        if not np.isfinite(soft_stiffness) or soft_stiffness <= 0:
            raise ValueError('Invalid soft fixture stiffness')
        fixture_mode = 'soft_distributed'
    if fixture_mode not in (
        'boundary',
        'sparse',
        'compliant',
        'tripod',
        'panel',
        'adaptive_panel',
        'balanced_panel',
        'free_edge_panel',
        'distributed_panel',
        'soft_distributed',
        'packet_panel',
        'hull_panel',
    ):
        raise ValueError('Unknown fixture mode')
    if set(action) != {'op', 'line', 'moving_point', 'layers', 'sense', 'angle_deg'} or action['op'] != 'fold_line':
        raise ValueError('fold_line requires line,moving_point,layers,sense,angle_deg')
    v = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    line = np.asarray(action['line'], float)
    point = np.asarray(action['moving_point'], float)
    if line.shape != (2, 2) or point.shape != (2,) or not np.isfinite(line).all() or not np.isfinite(point).all():
        raise ValueError('Finite board coordinates required')
    if (line < -0.5).any() or (line > 1.5).any() or (point < -0.5).any() or (point > 1.5).any():
        raise ValueError('Coordinates outside padded board [-.5,1.5]')
    degrees = action['angle_deg']
    if type(degrees) not in (float, int) or not math.isfinite(degrees) or not 0 < degrees <= 180:
        raise ValueError('angle_deg must be in (0,180]')
    if action['sense'] not in ('toward', 'away'):
        raise ValueError('sense must be toward or away from current +z camera')
    layer_spec = action['layers']
    if isinstance(layer_spec, dict):
        if set(layer_spec) != {'side', 'count'} or layer_spec['side'] not in ('top', 'bottom') or type(layer_spec['count']) is not int or layer_spec['count'] < 1:
            raise ValueError('Layer packet requires side top/bottom and positive integer count')
    elif layer_spec not in ('all', 'top', 'bottom'):
        raise ValueError('layers must be all, top, bottom, or an explicit layer packet')

    line = (line - 0.5) * SCALE
    point = (point - 0.5) * SCALE
    d = line[1] - line[0]
    if np.linalg.norm(d) < 0.0075:
        raise ValueError('Crease line too short')
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    if abs((point - line[0]) @ n) < 0.003:
        raise ValueError('moving_point must clearly select moving side')
    if (point - line[0]) @ n < 0:
        d = -d
        n = -n
    sd = (v[:, :2] - line[0]) @ n
    tol = 0.0006
    tri = sd[f]
    if ((tri.min(1) < -tol) & (tri.max(1) > tol)).any():
        raise SkillUnsupported('Current crease crosses mesh faces beyond 0.6mm instrument tolerance; runtime remeshing unavailable')
    candidate = np.flatnonzero((tri.mean(1) > 1e-10) & (tri.min(1) >= -tol))
    candidate_set = set(candidate.tolist())
    if not len(candidate):
        raise ValueError('No paper on requested moving side')

    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    # Sharp folded seams separate layers; do not flood through the folded hinge.
    edge_faces = {}
    adj = {int(i): [] for i in candidate}
    for i in candidate:
        for x, y in zip(f[i], np.roll(f[i], -1)):
            edge_faces.setdefault(tuple(sorted((x, y))), []).append(int(i))
    for ids in edge_faces.values():
        if len(ids) == 2 and normals[ids[0]] @ normals[ids[1]] > np.cos(np.deg2rad(45)):
            adj[ids[0]].append(ids[1])
            adj[ids[1]].append(ids[0])
    components = []
    unseen = set(candidate_set)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = [seed]
        while queue:
            for j in adj[queue.pop()]:
                if j in unseen:
                    unseen.remove(j)
                    component.add(j)
                    queue.append(j)
        components.append(component)

    hits = []
    for ci, comp in enumerate(components):
        heights = [z for i in comp if (z := point_hit(point, v[f[i]])) is not None]
        if heights:
            hits.append((max(heights), ci))
    if not hits:
        raise ValueError('moving_point does not hit a moving paper layer')
    if action['layers'] == 'all':
        selected = set.union(*components)
    elif isinstance(layer_spec, dict):
        if layer_spec['count'] > len(hits):
            raise ValueError('Requested packet contains more layers than the moving point intersects')
        chosen = sorted(hits)[-layer_spec['count']:] if layer_spec['side'] == 'top' else sorted(hits)[:layer_spec['count']]
        selected = set.union(*(components[ci] for _, ci in chosen))
    else:
        selected = components[sorted(hits)[-1 if action['layers'] == 'top' else 0][1]]

    moving_material = set(f[list(selected)].ravel().tolist())
    boundary = boundary_vertices(f)
    free_band = 0.008
    if thickness_panel:
        heights = v[[i for i in moving_material if abs(sd[i]) < tol], 2]
        if len(heights) < 2:
            raise SkillUnsupported('No selected material near the requested hinge')
        free_band = max(0.001, float(np.pi * (np.ptp(heights) + 0.00015) / 2 + 0.00025))

    moving = sorted(i for i in moving_material & boundary if sd[i] > free_band)
    fixed = sorted(i for i in boundary if i not in moving_material and abs(sd[i]) > free_band)
    fixed += sorted(i for i in boundary & moving_material if sd[i] < -free_band)
    fixed = sorted(set(fixed))
    dense_fixed = fixed.copy()
    dense_moving = moving.copy()

    if moving and len(hits) > 1 and action['layers'] == 'all':
        cutoff = 0.85 * max(sd[moving])
        ordered = sorted(moving, key=lambda i: sd[i], reverse=True)
        moving = sorted(set([i for i in moving if sd[i] >= cutoff] + ordered[:2]))
    if fixed and len(hits) > 1 and action['layers'] == 'all':
        distances = np.abs(sd[fixed])
        cutoff = 0.85 * max(distances)
        ordered = sorted(fixed, key=lambda i: abs(sd[i]), reverse=True)
        fixed = sorted(set([i for i in fixed if abs(sd[i]) >= cutoff] + ordered[:2]))

    panel_override = False
    folded_stationary_support = fixture_mode == 'hull_panel' and np.min(normals @ normals[0]) < np.cos(np.deg2rad(45))
    selected_comp_ids = {ci for ci, comp in enumerate(components) if comp & selected}
    owners = {}
    for ci in selected_comp_ids:
        for j in set(f[list(components[ci])].ravel().tolist()):
            owners.setdefault(j, set()).add(ci)
    shared_moving_seam = {j for j, cset in owners.items() if len(cset) > 1}
    crosses_folded_seam = any(abs(sd[j]) < free_band for j in shared_moving_seam)
    seam_corner_band = set()
    seam_moving_band = set(shared_moving_seam)
    if crosses_folded_seam:
        for fi in selected:
            if any(j in shared_moving_seam for j in f[fi]):
                seam_moving_band.update(int(j) for j in f[fi])
        seam_hinge_verts = [j for j in shared_moving_seam if abs(sd[j]) < 2.0 * free_band]
        if seam_hinge_verts:
            sh_pts = v[seam_hinge_verts, :2]
            corner_r = max(0.0038, 1.5 * free_band)
            dists = np.min(np.linalg.norm(v[:, None, :2] - sh_pts[None, :, :], axis=2), axis=1)
            seam_corner_band = set(np.flatnonzero(dists < corner_r).tolist())

    if (
        force_panel
        or folded_stationary_support
        or fixture_mode == 'panel'
        or (
            fixture_mode in ('adaptive_panel', 'balanced_panel', 'free_edge_panel', 'distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel')
            and (len(selected_comp_ids) > 1 or len(moving) < 2 or len(fixed) < 2)
        )
    ):
        panel_override = True
        moving = []
        all_edges = {}
        for fi, face in enumerate(f):
            for x, y in zip(face, np.roll(face, -1)):
                all_edges.setdefault(tuple(sorted((x, y))), []).append(fi)

        def actual_hinge(group, other):
            g_verts = {int(j) for fi in group for j in f[fi] if abs(sd[j]) < tol}
            if not g_verts:
                return False
            o_verts = {int(j) for fi in other for j in f[fi] if abs(sd[j]) < tol}
            return bool(g_verts & o_verts)

        def triangle_grips(ids):
            if fixture_mode == 'hull_panel':
                return panel_hull_grips(ids, v)
            if fixture_mode in ('distributed_panel', 'soft_distributed') or len(ids) < 2:
                return ids
            a = max(ids, key=lambda i: abs(sd[i]))
            b = max(ids, key=lambda i: np.linalg.norm(v[i] - v[a]))
            c = max(ids, key=lambda i: np.linalg.norm(np.cross(v[b] - v[a], v[i] - v[a])))
            return sorted(set([a, b, c]))

        non_selected = set(range(len(f))) - selected
        for ci in sorted(selected_comp_ids):
            comp = components[ci]
            band = (
                -tol
                if fixture_mode in ('free_edge_panel', 'distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel')
                and not actual_hinge(comp, non_selected)
                else free_band
            )
            comp_verts = set(f[list(comp)].ravel().tolist())
            ids = [
                i for i in comp_verts
                if sd[i] > band
                and i not in seam_corner_band
                and i not in seam_moving_band
                and (
                    (fixture_mode in ('distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel') and not crosses_folded_seam)
                    or len(owners[i]) == 1
                )
            ]
            if len(ids) < 2:
                ids = [i for i in comp_verts if sd[i] > band]
            moving.extend(triangle_grips(ids))
        moving = sorted(set(moving))
        fixed = triangle_grips(dense_fixed)

        if fixture_mode in ('balanced_panel', 'free_edge_panel', 'distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel'):
            remaining = set(range(len(f))) - selected
            edges = {}
            links = {i: [] for i in remaining}
            for i in remaining:
                for x, y in zip(f[i], np.roll(f[i], -1)):
                    edges.setdefault(tuple(sorted((x, y))), []).append(i)
            for ids in edges.values():
                if len(ids) == 2 and normals[ids[0]] @ normals[ids[1]] > np.cos(np.deg2rad(45)):
                    links[ids[0]].append(ids[1])
                    links[ids[1]].append(ids[0])
            groups = []
            unseen = set(remaining)
            while unseen:
                seed = unseen.pop()
                group = {seed}
                queue = [seed]
                while queue:
                    for j in links[queue.pop()]:
                        if j in unseen:
                            unseen.remove(j)
                            group.add(j)
                            queue.append(j)
                groups.append(group)
            stationary_owners = {}
            hinging_stationary_verts = set()
            group_hinges = []
            for ci, group in enumerate(groups):
                gv = set(f[list(group)].ravel().tolist())
                for j in gv:
                    stationary_owners.setdefault(j, set()).add(ci)
                is_hinge = actual_hinge(group, selected)
                group_hinges.append((group, gv, is_hinge))
                if is_hinge:
                    hinging_stationary_verts |= gv
            shared_stationary_seam = {j for j, cset in stationary_owners.items() if len(cset) > 1}
            seam_stationary_band = set(shared_stationary_seam)
            stat_corner_band = set(seam_corner_band)
            if crosses_folded_seam and shared_stationary_seam:
                for fi in remaining:
                    if any(j in shared_stationary_seam for j in f[fi]):
                        seam_stationary_band.update(int(j) for j in f[fi])
                stat_hinge_verts = [j for j in (shared_moving_seam | shared_stationary_seam) if abs(sd[j]) < 2.0 * free_band]
                if stat_hinge_verts:
                    sh_pts = v[stat_hinge_verts, :2]
                    corner_r = max(0.0038, 1.5 * free_band)
                    dists = np.min(np.linalg.norm(v[:, None, :2] - sh_pts[None, :, :], axis=2), axis=1)
                    stat_corner_band |= set(np.flatnonzero(dists < corner_r).tolist())
            fixed = []
            for group, group_verts, is_hinge in group_hinges:
                use_free_band = is_hinge or (
                    fixture_mode not in ('free_edge_panel', 'distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel')
                )
                ids = [
                    j for j in group_verts
                    if j not in moving_material
                    and j not in stat_corner_band
                    and j not in seam_stationary_band
                    and abs(sd[j]) > (free_band if (use_free_band or j in hinging_stationary_verts) else -tol)
                    and (
                        (fixture_mode in ('distributed_panel', 'soft_distributed', 'packet_panel', 'hull_panel') and not crosses_folded_seam)
                        or len(stationary_owners[j]) == 1
                    )
                ]
                fixed.extend(triangle_grips(ids))
            fixed = sorted(set(fixed))

    if len(moving) < 2 or len(fixed) < 2:
        raise SkillUnsupported('Not enough distinct moving/fixed outer boundary grips')
    near = np.flatnonzero(np.abs(sd) < tol)
    if len(near) < 2:
        raise SkillUnsupported('No current crease vertices')
    touch = set()
    for i in selected:
        for j in f[i]:
            if abs(sd[j]) < tol:
                touch.add(int(j))
    if len(touch) < 2:
        raise SkillUnsupported('Selected layer does not touch requested crease')
    near_selected = v[sorted(touch)]
    all_edges = {}
    for fi, face in enumerate(f):
        for x, y in zip(face, np.roll(face, -1)):
            all_edges.setdefault(tuple(sorted((x, y))), []).append(fi)
    touch = {
        int(j)
        for edge, edge_owners in all_edges.items()
        if len(edge_owners) == 2
        and any(fi in selected for fi in edge_owners)
        and any(fi not in selected for fi in edge_owners)
        and max(abs(sd[j]) for j in edge) < tol
        for j in edge
    }
    if len(touch) < 2:
        raise SkillUnsupported('No material hinge joins the selected packet to stationary paper at the requested crease')
    pts = v[sorted(touch)]
    if np.ptp(near_selected[:, 2]) > 0.002:
        raise SkillUnsupported('Crease is not a near-planar stack axis; compound folding not implemented')
    sign = 1 if action['sense'] == 'toward' else -1
    center = pts.mean(axis=0)
    _, _, basis = np.linalg.svd(pts - center, full_matrices=False)
    axis = basis[0]
    if axis[:2] @ d < 0:
        axis = -axis
    deviation = np.linalg.norm(np.cross(pts - center, axis), axis=1).max()
    if deviation > 0.001:
        raise SkillUnsupported('Current selected crease is not sufficiently straight')
    center[2] = float(near_selected[:, 2].max() + 0.00015 if sign > 0 else near_selected[:, 2].min() - 0.00015)

    if crosses_folded_seam and panel_override:
        def prune_hinge_locks(grip_list, candidate_faces):
            gset = set(grip_list)
            to_drop = set()
            for fi in candidate_faces:
                tri = [int(x) for x in f[fi]]
                in_g = [x for x in tri if x in gset]
                out_g = [x for x in tri if x not in gset]
                if len(in_g) == 2 and len(out_g) == 1 and abs(sd[out_g[0]]) <= free_band + 1e-4:
                    a, b = in_g
                    eb = v[b] - v[a]
                    nb = np.linalg.norm(eb)
                    if nb > 1e-12 and abs((eb / nb) @ axis) < 0.98:
                        drop_v = a if abs(sd[a]) <= abs(sd[b]) else b
                        if abs(sd[drop_v]) < 0.0020:
                            to_drop.add(drop_v)
            if to_drop and len(gset - to_drop) >= 3:
                gset -= to_drop
            adj_v = {}
            for fi in candidate_faces:
                tri = [int(x) for x in f[fi]]
                for x in tri:
                    adj_v.setdefault(x, set()).update(y for y in tri if y != x)
            for j, nbrs in adj_v.items():
                if j in gset or abs(sd[j]) > free_band + 1e-4:
                    continue
                gn = sorted(nbrs & gset, key=lambda x: abs(sd[x]))
                if len(gn) >= 3:
                    vecs = [(v[x] - v[j]) / max(np.linalg.norm(v[x] - v[j]), 1e-12) for x in gn]
                    if any(abs(vecs[p] @ vecs[q]) < 0.99 for p in range(len(vecs)) for q in range(p + 1, len(vecs))):
                        for x in gn[:-2]:
                            if len(gset) > 3 and abs(sd[x]) < 0.0020:
                                gset.discard(x)
            return sorted(gset)

        moving = prune_hinge_locks(moving, selected)
        fixed = prune_hinge_locks(fixed, set(range(len(f))) - selected)

    stationary = np.array([i for i in range(len(f)) if i not in selected], dtype=int)
    stat_faces = f[stationary] if len(stationary) else np.zeros((0, 3), dtype=int)
    stat_tris = v[stat_faces] if len(stationary) else np.zeros((0, 3, 3), dtype=float)
    stat_bb_min = stat_tris[:, :, :2].min(axis=1) - 0.00025 if len(stationary) else np.zeros((0, 2), dtype=float)
    stat_bb_max = stat_tris[:, :, :2].max(axis=1) + 0.00025 if len(stationary) else np.zeros((0, 2), dtype=float)

    def calc_deficit(c_try):
        if not len(stationary):
            return 0.0
        predicted = rotation(v[moving], c_try, axis, sign * degrees)
        deficit = 0.0
        for vertex, q in zip(moving, predicted):
            heights = []
            cand_idx = np.flatnonzero(
                (stat_bb_min[:, 0] <= q[0])
                & (q[0] <= stat_bb_max[:, 0])
                & (stat_bb_min[:, 1] <= q[1])
                & (q[1] <= stat_bb_max[:, 1])
            )
            for idx in cand_idx:
                sf = stat_faces[idx]
                if vertex in sf:
                    continue
                tri = stat_tris[idx]
                zhit = point_hit(q[:2], tri)
                if zhit is not None:
                    heights.append(zhit)
                for j in sf:
                    if np.linalg.norm(v[j, :2] - q[:2]) < 0.0002:
                        heights.append(v[j, 2])
            if heights:
                surface = max(heights) if sign > 0 else min(heights)
                deficit = max(deficit, 0.0002 - sign * (q[2] - surface))
        return deficit

    # Side-by-side horizontal gate clearance: when a ~180-deg fold brings a moving edge
    # to meet a stationary edge at the same layer height (where shifting the fold axis
    # by 0.12-0.18 mm toward the moving side resolves the 0.2 mm conflict without needing
    # to stack one gate door vertically on top of the other), use the horizontal offset.
    horiz_shift = 0.0
    if degrees > 150.0 and calc_deficit(center) > 1e-8:
        for s_try in (0.00012, 0.00018):
            c_try = center + np.r_[s_try * n, 0.0]
            if calc_deficit(c_try) <= 1e-8:
                center = c_try
                horiz_shift = s_try
                break

    lift = 0.0
    for _ in range(3):
        deficit = calc_deficit(center)
        if deficit <= 1e-8:
            break
        delta = deficit / max(1.0 - np.cos(np.deg2rad(degrees)), 0.1)
        if lift + delta > 0.001:
            raise SkillUnsupported('Endpoint needs more than validated 1mm fixture clearance offset')
        center[2] += sign * delta
        lift += delta

    cmd = motion(moving, center, axis, sign * degrees, v)
    info = dict(
        endpoint_clearance_axis_lift_m=lift,
        endpoint_clearance_axis_horiz_shift_m=horiz_shift,
        selected_faces=sorted(selected),
        moving_vertices=moving,
        fixed_vertices=fixed,
        hit_layers=len(hits),
        selected_layer_mode=action['layers'],
        axis_point=cmd['axis_point'],
        axis_direction=cmd['axis_direction'],
        degrees=cmd['degrees'],
    )
    info['fixture_mode'] = 'thickness_panel' if thickness_panel else 'full_panel' if force_panel else fixture_mode
    info['free_crease_band_halfwidth_m'] = free_band
    info['folded_stationary_support'] = bool(folded_stationary_support)
    info['has_multiple_surface_orientations'] = bool(np.min(normals @ normals[0]) < np.cos(np.deg2rad(45)))
    info['selected_planar_components'] = len(selected_comp_ids)
    info['crosses_folded_seam'] = bool(crosses_folded_seam)
    info['moving_panel_groups'] = [
        sorted(set(f[list(components[ci])].ravel().tolist()) - (shared_moving_seam if crosses_folded_seam else set()))
        for ci in sorted(selected_comp_ids)
    ]
    if fixture_mode == 'compliant' or (fixture_mode == 'soft_distributed' and panel_override):
        cmd['op'] = 'rotate_soft_grips'
        info['moving_grip_stiffness_N_m'] = soft_stiffness
        info['moving_grip_kind'] = 'compliant'
        return [
            dict(op='release'),
            dict(op='release_soft_grips'),
            dict(op='set_grips', vertices=fixed),
            dict(op='attach_soft_grips', vertices=moving, stiffness=soft_stiffness),
            cmd,
        ], info
    return [
        dict(op='release_soft_grips'),
        dict(op='release'),
        dict(op='set_grips', vertices=sorted(moving + fixed)),
        cmd,
    ], info


class LayerControllerV16:
    def __init__(self, backend, fixture_mode='boundary'):
        self.fixture_mode = fixture_mode
        self.next_fold_id = 0
        self.backend = backend
        self.records = []
        self.snapshots = []
        self.fold_stack = []
        self.undo_floor = 0
        self.rejected_endpoints = []

    def execute(self, action):
        if not isinstance(action, dict):
            raise ValueError('Action must be an object')
        state = self.backend.state()
        v = np.asarray(state['vertices'])
        f = np.asarray(state['faces'])
        op = action.get('op')
        info = {}
        low = []
        previous = copy.deepcopy(self.fold_stack)
        if op == 'fold_line':
            servo = self.fixture_mode in ('hinge_servo', 'hinge_servo_panel', 'hinge_servo_strong')
            mode = 'packet_panel' if servo or self.fixture_mode == 'packet_press' else self.fixture_mode
            guided_directional = mode == 'guided_directional' or mode.startswith('guided_directional_')
            anchor_stiffness = float(mode.removeprefix('guided_directional_')) if mode.startswith('guided_directional_') else 100.0
            if not math.isfinite(anchor_stiffness) or anchor_stiffness <= 0:
                raise ValueError('Invalid tangential anchor stiffness')
            sliding_directional = mode == 'sliding_directional'
            directional_thickness = mode == 'directional_thickness' or guided_directional or sliding_directional
            if directional_thickness:
                mode = 'thickness_panel'
            directional = mode == 'directional_panel' or mode.startswith('directional_panel_')
            if directional:
                mode = 'soft_distributed_' + (mode.removeprefix('directional_panel_') if mode.startswith('directional_panel_') else '1000')
            if mode == 'rolling_panel':
                mode = 'distributed_panel'
            low, info = compile_stack_fold_v15(v, f, action, mode)
            axis_dir = np.array(info.get('axis_direction', [1.0, 0.0, 0.0]))
            is_diagonal = abs(abs(axis_dir[0]) - abs(axis_dir[1])) < 0.1
            if directional_thickness and (
                info['selected_planar_components'] > 1
                or ((guided_directional or sliding_directional) and (info['has_multiple_surface_orientations'] or is_diagonal))
            ):
                cmd = low[-1]
                cmd['op'] = 'rotate_soft_grips'
                low = [
                    dict(op='release_soft_grips'),
                    dict(op='release'),
                    dict(op='set_grips', vertices=info['fixed_vertices']),
                    dict(op='attach_soft_grips', vertices=info['moving_vertices'], stiffness=1000.0, normal=[0.0, 0.0, 1.0]),
                    cmd,
                ]
                info.update(moving_grip_kind='compliant', moving_grip_stiffness_N_m=1000.0, directional_normal_start=[0.0, 0.0, 1.0])
                if guided_directional:
                    axis = np.array(info['axis_direction'])
                    center = np.array(info['axis_point'])
                    group_candidates = []
                    for group in info['moving_panel_groups']:
                        ids = sorted(set(group) & set(info['moving_vertices']))
                        if not ids:
                            continue
                        radii = np.linalg.norm(np.cross(v[ids] - center, axis), axis=1)
                        far = [i for i, radius in zip(ids, radii) if radius >= 0.85 * radii.max()]
                        centroid = v[far].mean(0)
                        anchor_id = min(far, key=lambda i: np.linalg.norm(v[i] - centroid))
                        group_candidates.append((abs(float(v[ids, 2].mean()) - float(center[2])), anchor_id))
                    if info.get('crosses_folded_seam') and len(group_candidates) > 1:
                        anchors = [min(group_candidates, key=lambda item: item[0])[1]]
                    else:
                        anchors = sorted({anchor_id for _, anchor_id in group_candidates})
                    low.insert(-1, dict(op='attach_soft_grips', vertices=anchors, stiffness=anchor_stiffness))
                    info.update(tangential_anchor_vertices=anchors, tangential_anchor_stiffness_N_m=anchor_stiffness)
        elif op == 'unfold':
            if set(action) - {'op', 'fold_id', 'angle_deg'}:
                raise ValueError('unfold accepts optional fold_id and angle_deg')
            if not self.fold_stack:
                raise ValueError('No active fold to unfold')
            fold_id = action.get('fold_id', self.fold_stack[-1]['fold_id'])
            if type(fold_id) is not int:
                raise ValueError('fold_id must be an integer')
            found = [i for i, r in enumerate(self.fold_stack) if r['fold_id'] == fold_id]
            if not found:
                raise ValueError('fold_id is not active')
            at = found[0]
            r = self.fold_stack[at]
            if r.get('moving_grip_kind') == 'crease_servo':
                raise SkillUnsupported('Experimental crease-servo unfolding not yet qualified')
            for later in self.fold_stack[at + 1:]:
                if set(r['selected_faces']) & set(later['selected_faces']):
                    raise SkillUnsupported('Open overlapping later folds first; dependent layer unfolding is not yet validated')
            amount = action.get('angle_deg', abs(r['degrees']))
            if type(amount) not in (int, float) or not math.isfinite(amount) or not 0 < amount <= abs(r['degrees']) + 1e-8:
                raise ValueError('Unfold angle must be positive and not exceed the remaining fold angle')
            delta = -math.copysign(amount, r['degrees'])
            cmd = motion(r['moving_vertices'], np.asarray(r['axis_point']), np.asarray(r['axis_direction']), delta, v)
            if 'roll_radii' in r:
                cmd.update(roll_radii=r['roll_radii'], roll_tangent=r['roll_tangent'], roll_start_angle_deg=r['degrees'])
            low = [
                dict(op='release_soft_grips'),
                dict(op='release'),
                dict(op='set_grips', vertices=sorted(r['moving_vertices'] + r['fixed_vertices'])),
                cmd,
            ]
            if r.get('moving_grip_kind') == 'compliant':
                cmd['op'] = 'rotate_soft_grips'
                low = [
                    dict(op='release_soft_grips'),
                    dict(op='release'),
                    dict(op='set_grips', vertices=r['fixed_vertices']),
                    dict(op='attach_soft_grips', vertices=r['moving_vertices'], stiffness=r['moving_grip_stiffness_N_m']),
                    cmd,
                ]
                if 'directional_normal_start' in r:
                    low[-2]['normal'] = rotation(
                        np.array([r['directional_normal_start']]),
                        np.zeros(3),
                        np.array(r['axis_direction']),
                        r['degrees'],
                    )[0].tolist()
                if r.get('tangential_anchor_vertices'):
                    low.insert(-1, dict(op='attach_soft_grips', vertices=r['tangential_anchor_vertices'], stiffness=r['tangential_anchor_stiffness_N_m']))
            info = dict(unfolds_record=r['record_index'], fold_id=fold_id, unfold_angle_deg=float(amount), physical_forward_time=True)
        elif op in ('turn_over', 'rotate_paper'):
            allowed = {'op', 'axis'} if op == 'turn_over' else {'op', 'degrees'}
            if set(action) != allowed:
                raise ValueError('Invalid rigid paper action fields')
            if op == 'turn_over':
                if action['axis'] not in ('x', 'y'):
                    raise ValueError('turn_over axis is x or y')
                axis = np.array([1.0, 0.0, 0.0]) if action['axis'] == 'x' else np.array([0.0, 1.0, 0.0])
                degrees = 180.0
            else:
                degrees = action['degrees']
                if type(degrees) not in (int, float) or not math.isfinite(degrees) or not 0 < abs(degrees) <= 180:
                    raise ValueError('rotate_paper degrees must be nonzero within +/-180')
                axis = np.array([0.0, 0.0, 1.0])
            center = np.array([0.0, 0.0, float(np.mean(v[:, 2]))])
            ids = sorted(boundary_vertices(f))
            cmd = motion(ids, center, axis, degrees, v)
            low = [dict(op='release_soft_grips'), dict(op='release'), dict(op='set_grips', vertices=ids), cmd]
            info = dict(axis_point=center.tolist(), axis_direction=axis.tolist(), degrees=float(degrees))
        elif op == 'undo':
            if action != {'op': 'undo'} or len(self.records) <= self.undo_floor:
                raise ValueError('Nothing to undo within this episode; prepared initial state cannot be undone')
            record = self.records[-1]
            for _ in record['low_level_actions']:
                self.backend.step({'op': 'undo'})
            self.records.pop()
            self.fold_stack = self.snapshots.pop()
            return dict(undone=True, low_level_actions=[], forward_physical_motion=False)
        else:
            raise SkillUnsupported('Unsupported controller operation: ' + str(op))

        completed = 0
        try:
            for cmd in low:
                end = self.backend.step(cmd)
                completed += 1
                if cmd['op'] == 'rotate_soft_grips':
                    controls = end.get('soft_controls', {})
                    actual = np.asarray(end['vertices'])
                    errors = [
                        abs((actual[int(i)] - c[:3]) @ np.asarray(c[4:])) if len(c) == 7 else np.linalg.norm(actual[int(i)] - c[:3])
                        for i, c in controls.items()
                    ]
                    error = max(errors, default=float('inf'))
                    info['controlled_endpoint_error_mm'] = float(error * 1000.0)
                    info['unconstrained_3d_target_error_mm'] = end.get('last_metrics', {}).get('soft_target_error', float('inf')) * 1000.0
                    if error > 0.0005:
                        reason = f'Compliant controlled directions exceeded 0.5mm endpoint accuracy: {error * 1000:.3f}mm'
                        self.rejected_endpoints.append(dict(action=copy.deepcopy(action), reason=reason, state=copy.deepcopy(end)))
                        raise SkillUnsupported(reason)
                    if any(len(c) == 7 for c in controls.values()):
                        details = info if op == 'fold_line' else r
                        ids = np.unique(f[details['selected_faces']])
                        expected = v.copy()
                        expected[ids] = rotation(v[ids], np.array(cmd['axis_point']), np.array(cmd['axis_direction']), cmd['degrees'])
                        distance = np.linalg.norm(actual - expected, axis=1)
                        error_shape = dict(rms_mm=float(np.sqrt(np.mean(distance ** 2)) * 1000.0), max_mm=float(distance.max() * 1000.0))
                        info['command_endpoint_shape'] = error_shape
                        if error_shape['rms_mm'] >= 1.0 or error_shape['max_mm'] >= 3.0:
                            reason = f'Physical motion did not reach its requested endpoint geometry: RMS {error_shape["rms_mm"]:.3f}mm, max {error_shape["max_mm"]:.3f}mm'
                            self.rejected_endpoints.append(dict(action=copy.deepcopy(action), reason=reason, state=copy.deepcopy(end)))
                            raise SkillUnsupported(reason)
        except Exception:
            if getattr(self.backend, 'poisoned', False):
                raise RuntimeError('Simulation transport lost; rollback cannot be verified and episode must stop')
            for _ in range(completed):
                self.backend.step({'op': 'undo'})
            if self.backend.state()['hash'] != state['hash']:
                raise RuntimeError('Composite rollback did not restore paper')
            raise

        if op == 'fold_line':
            info['record_index'] = len(self.records)
            info['fold_id'] = self.next_fold_id
            self.next_fold_id += 1
            self.fold_stack.append(copy.deepcopy(info))
        elif op == 'unfold':
            remaining = abs(r['degrees']) - amount
            if remaining < 1e-8:
                self.fold_stack.pop(at)
            else:
                self.fold_stack[at]['degrees'] = math.copysign(remaining, r['degrees'])
        elif op in ('turn_over', 'rotate_paper'):
            for fold in self.fold_stack:
                fold['axis_point'] = rotation(np.array([fold['axis_point']]), center, axis, degrees)[0].tolist()
                fold['axis_direction'] = rotation(np.array([fold['axis_direction']]), np.zeros(3), axis, degrees)[0].tolist()
                if 'roll_tangent' in fold:
                    fold['roll_tangent'] = rotation(np.array([fold['roll_tangent']]), np.zeros(3), axis, degrees)[0].tolist()
                if 'directional_normal_start' in fold:
                    fold['directional_normal_start'] = rotation(np.array([fold['directional_normal_start']]), np.zeros(3), axis, degrees)[0].tolist()
        record = dict(
            action=copy.deepcopy(action),
            low_level_actions=low,
            details=info,
            before=state['hash'],
            after=self.backend.state()['hash'],
            frame=self.backend.state()['steps'],
        )
        self.snapshots.append(previous)
        self.records.append(record)
        return copy.deepcopy(record)
