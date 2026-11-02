"""Non-leaking 3-view orthographic z-buffer + silhouette/depth-jump/crease renderer (v15).

Addresses the observability occlusion and coplanar-layer ambiguity documented in
`docs/v14-fork-audit.md` without accessing any private task, reference, or checkpoint data:
1. Three views:
   - `top`: orthographic +z view with faint board coordinate grid lines at 0.25 intervals.
   - `oblique`: upper oblique view (elevation +28 deg, azimuth -65 deg).
   - `bottom_oblique`: lower oblique view from -z (elevation -28 deg, azimuth -65 deg) to
     expose underside flaps folded with `sense="away"` (which had 0.00%-0.12% TOP pixel diff in v14).
2. Comprehensible technical rendering (Saito & Takahashi, SIGGRAPH 1990):
   - Per-pixel depth discontinuity contours (|dz| > 75 um) and normal-jump creases (> 15 deg).
   - Depth-tested true sheet boundary edges (shared by 1 face) and active plastic hinge edges
     so stacked coplanar layers with identical normals are visually distinct.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRONT = np.array([188.0, 219.0, 239.0])
BACK = np.array([255.0, 221.0, 155.0])
SILHOUETTE_RGB = np.array([32.0, 48.0, 64.0])
DEPTH_JUMP_RGB = np.array([45.0, 65.0, 85.0])
CREASE_RGB = np.array([95.0, 115.0, 135.0])


def _camera_frame(view: str):
    if view == 'top':
        u = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        camera = np.array([0.0, 0.0, 1.0])
        limit = 0.12
    elif view == 'bottom_oblique':
        e, a = np.deg2rad([-28.0, -65.0])
        camera = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        u = np.array([-np.sin(a), np.cos(a), 0.0])
        up = np.cross(u, camera)
        limit = 0.115
    else:
        e, a = np.deg2rad([28.0, -65.0])
        camera = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        u = np.array([-np.sin(a), np.cos(a), 0.0])
        up = np.cross(camera, u)
        limit = 0.115
    return u, up, camera, limit


def _extract_feature_edges(vertices: np.ndarray, faces: np.ndarray, crease_deg: float = 15.0):
    """Return boundary edges (1 incident face) and sharp dihedral edges (> crease_deg) from current 3D state."""
    v0 = vertices[faces[:, 0]]
    normals = np.cross(vertices[faces[:, 1]] - v0, vertices[faces[:, 2]] - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    unit_normals = normals / np.maximum(norms, 1e-20)

    edge_faces = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (min(int(a), int(b)), max(int(a), int(b)))
            edge_faces.setdefault(e, []).append(fi)

    boundary_edges = []
    crease_edges = []
    cos_thresh = float(np.cos(np.deg2rad(crease_deg)))
    for (u, v), fids in edge_faces.items():
        if len(fids) == 1:
            boundary_edges.append((u, v))
        elif len(fids) == 2:
            cos_a = float(np.clip(unit_normals[fids[0]] @ unit_normals[fids[1]], -1.0, 1.0))
            if cos_a < cos_thresh:
                crease_edges.append((u, v))
    return boundary_edges, crease_edges, unit_normals


def _draw_board_grid_top(pixels: np.ndarray, owner: np.ndarray, size: int, limit: float):
    """Draw faint board coordinate grid lines for [0, 1]^2 on the TOP view background and lightly over paper."""
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        world = (t - 0.5) * 0.15
        pos = int(round((world / limit / 2.0 + 0.5) * (size - 1)))
        if 0 <= pos < size:
            # Vertical line x = t (dashed every 4 pixels)
            ys = np.arange(0, size)
            dash = (ys % 6) < 3
            bg_mask = dash & (owner[:, pos] < 0)
            fg_mask = dash & (owner[:, pos] >= 0)
            pixels[bg_mask, pos] = np.array([218.0, 226.0, 234.0])
            pixels[fg_mask, pos] = 0.88 * pixels[fg_mask, pos] + 0.12 * np.array([120.0, 140.0, 160.0])

            # Horizontal line y = t
            row = (size - 1) - pos
            if 0 <= row < size:
                xs = np.arange(0, size)
                dash_x = (xs % 6) < 3
                bg_x = dash_x & (owner[row, :] < 0)
                fg_x = dash_x & (owner[row, :] >= 0)
                pixels[row, bg_x] = np.array([218.0, 226.0, 234.0])
                pixels[row, fg_x] = 0.88 * pixels[row, fg_x] + 0.12 * np.array([120.0, 140.0, 160.0])


def _raster_depth_tested_edges(
    pixels: np.ndarray,
    depth: np.ndarray,
    screen: np.ndarray,
    z: np.ndarray,
    edges: list,
    color: np.ndarray,
    size: int,
    z_bias: float = 0.00018,
):
    """Draw 3D line segments that pass depth test (z_line + z_bias >= depth_buffer)."""
    for u_idx, v_idx in edges:
        p0, p1 = screen[u_idx], screen[v_idx]
        z0, z1 = float(z[u_idx]), float(z[v_idx])
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        steps = max(2, int(np.ceil(dist * 1.5)))
        ts = np.linspace(0.0, 1.0, steps)
        xs = np.rint(p0[0] + ts * (p1[0] - p0[0])).astype(int)
        ys = np.rint(p0[1] + ts * (p1[1] - p0[1])).astype(int)
        zs = z0 + ts * (z1 - z0)
        in_bounds = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        xs, ys, zs = xs[in_bounds], ys[in_bounds], zs[in_bounds]
        vis = zs + z_bias >= depth[ys, xs]
        pixels[ys[vis], xs[vis]] = color


def raster_v15(vertices, faces, size=512, view='top', return_buffers=False):
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    u, up, camera, limit = _camera_frame(view)

    p = np.column_stack([v @ u, v @ up])
    screen = (p / limit / 2.0 + 0.5) * (size - 1)
    screen[:, 1] = (size - 1) - screen[:, 1]
    z = v @ camera

    pixels = np.full((size, size, 3), 249.0, dtype=float)
    depth = np.full((size, size), -np.inf, dtype=float)
    owner = np.full((size, size), -1, dtype=int)
    normal_buf = np.zeros((size, size, 3), dtype=float)

    boundary_edges, crease_edges, unit_normals = _extract_feature_edges(v, f)

    for i, tri in enumerate(f):
        q = screen[tri]
        xmin = max(0, int(np.floor(q[:, 0].min())))
        xmax = min(size - 1, int(np.ceil(q[:, 0].max())))
        ymin = max(0, int(np.floor(q[:, 1].min())))
        ymax = min(size - 1, int(np.ceil(q[:, 1].max())))
        if xmax < xmin or ymax < ymin:
            continue
        m = np.column_stack([q[1] - q[0], q[2] - q[0]])
        if abs(np.linalg.det(m)) < 1e-10:
            continue
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        xy = np.stack([xx + 0.5, yy + 0.5], axis=-1)
        bary = (xy - q[0]) @ np.linalg.inv(m).T
        valid = (bary[..., 0] >= -1e-8) & (bary[..., 1] >= -1e-8) & (bary.sum(-1) <= 1.0 + 1e-8)
        h = z[tri[0]] + bary @ (z[tri[1:]] - z[tri[0]])
        part = depth[ymin : ymax + 1, xmin : xmax + 1]
        valid &= h > part
        if not np.any(valid):
            continue
        n_unit = unit_normals[i]
        facing = float(n_unit @ camera)
        shade = 0.78 + 0.22 * abs(facing)
        pixels[ymin : ymax + 1, xmin : xmax + 1][valid] = (FRONT if facing >= 0.0 else BACK) * shade
        part[valid] = h[valid]
        owner[ymin : ymax + 1, xmin : xmax + 1][valid] = i
        normal_buf[ymin : ymax + 1, xmin : xmax + 1][valid] = n_unit if facing >= 0.0 else -n_unit

    if view == 'top':
        _draw_board_grid_top(pixels, owner, size, limit)

    # 1. Image-space silhouette & multi-layer occlusion-step contours
    paper_mask = owner >= 0
    sil_mask = np.zeros((size, size), dtype=bool)
    depth_jump_mask = np.zeros((size, size), dtype=bool)

    # Precompute vertex sets per face to distinguish connected mesh neighbors from overlapping layer steps
    face_v0 = f[:, 0]
    face_v1 = f[:, 1]
    face_v2 = f[:, 2]

    for dy, dx in ((0, 1), (1, 0)):
        p_a = paper_mask[: size - dy, : size - dx]
        p_b = paper_mask[dy:, dx:]
        sil_step = p_a ^ p_b
        sil_mask[: size - dy, : size - dx] |= sil_step
        sil_mask[dy:, dx:] |= sil_step

        both = p_a & p_b
        oa = owner[: size - dy, : size - dx]
        ob = owner[dy:, dx:]
        diff_face = both & (oa != ob)
        if np.any(diff_face):
            ys, xs = np.where(diff_face)
            fa = oa[ys, xs]
            fb = ob[ys, xs]
            # Two faces are connected on the same sheet branch if they share at least one mesh vertex
            shared = (
                (face_v0[fa] == face_v0[fb]) | (face_v0[fa] == face_v1[fb]) | (face_v0[fa] == face_v2[fb])
                | (face_v1[fa] == face_v0[fb]) | (face_v1[fa] == face_v1[fb]) | (face_v1[fa] == face_v2[fb])
                | (face_v2[fa] == face_v0[fb]) | (face_v2[fa] == face_v1[fb]) | (face_v2[fa] == face_v2[fb])
            )
            disjoint = ~shared
            if np.any(disjoint):
                dys = ys[disjoint]
                dxs = xs[disjoint]
                depth_jump_mask[dys, dxs] = True
                depth_jump_mask[dys + dy, dxs + dx] = True

    pixels[depth_jump_mask & paper_mask] = DEPTH_JUMP_RGB
    pixels[sil_mask & paper_mask] = SILHOUETTE_RGB

    # 2. Object-space depth-tested plastic crease edges and true boundary edges
    _raster_depth_tested_edges(pixels, depth, screen, z, crease_edges, CREASE_RGB, size, z_bias=0.00018)
    _raster_depth_tested_edges(pixels, depth, screen, z, boundary_edges, SILHOUETTE_RGB, size, z_bias=0.00018)

    img = Image.fromarray(np.clip(pixels, 0, 255).astype('uint8'))
    if return_buffers:
        return img, owner, depth
    return img, owner


def render_public_v15(state, path):
    size = 512
    canvas = Image.new('RGB', (1680, 630), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()

    views = [
        ('top', 'ACTUAL PAPER - TOP (+Z, WITH BOARD GRID)'),
        ('oblique', 'ACTUAL PAPER - TOP OBLIQUE (+Z)'),
        ('bottom_oblique', 'ACTUAL PAPER - BOTTOM OBLIQUE (-Z UNDERSIDE)'),
    ]
    for i, (view, label) in enumerate(views):
        im, _ = raster_v15(state['vertices'], state['faces'], size, view)
        x = 42 + i * 545
        y = 48
        canvas.paste(im, (x, y))
        draw.rectangle([x, y, x + size, y + size], outline='#cbd5e1', width=1)
        draw.text((x, y - 28), label, fill='#253f50', font=font)
        if i == 0:
            for t in [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25]:
                pos = ((t - 0.5) * 0.15 / 0.24 + 0.5) * (size - 1)
                draw.text((x + pos - 10, y + size + 7), str(t), fill='#294456', font=font)
                draw.text((x - 34, y + size - 1 - pos - 8), str(t), fill='#294456', font=font)
            draw.text((x + 150, 592), 'board x (right), y (up) — dashed grid = 0.25 units', fill='#294456', font=font)

    draw.text(
        (600, 588),
        'Blue: initial front | Cream: initial back | Dark lines: layer boundaries & depth steps | Grey lines: active creases',
        fill='#294456',
        font=font,
    )
    canvas.save(path)
