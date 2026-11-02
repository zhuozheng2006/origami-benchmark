"""Experimental reference-space contact policy; never culls collision pairs.

For initially flat shell material coordinates only. Inspired by Gomez-Nogales
et al., arXiv:2510.10256, sections 3.2-3.3. This is an independent certifier,
separate from the native simulation implementation. Local stencils
retain zero-offset CCD; all other stencils retain the declared physical offset.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import ipctk
from .contact_certification import candidate_plane_certificate

KINDS = (('vv_candidates', 1), ('ev_candidates', 1),
         ('fv_candidates', 1), ('ee_candidates', 2))


def stencil_key(kind, ids):
    ids = [int(i) for i in ids if i >= 0]
    if kind == 'ee_candidates':
        return (kind, *sum(sorted([sorted(ids[:2]), sorted(ids[2:])]), []))
    if kind == 'vv_candidates':
        return (kind, *sorted(ids))
    return (kind, ids[0], *sorted(ids[1:]))


@dataclass(frozen=True)
class ReferenceContactPolicy:
    local_keys: frozenset
    offset: float
    activation: float
    local_activation: float
    reference_sha256: str

    @classmethod
    def from_flat_mesh(cls, mesh, reference, offset=.0001, barrier_width=.00005):
        x = np.asarray(reference, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError('finite 3D reference coordinates required')
        if np.ptp(x[:, 2]) > 1e-12:
            raise ValueError('policy requires flat material coordinates in XY')
        if not np.isfinite([offset, barrier_width]).all() or min(offset, barrier_width) <= 0:
            raise ValueError('positive finite offset and barrier width required')
        # Separate material components never receive the local self-contact filter.
        # Production tasks use one uncut sheet; this also prevents accidental
        # weakening of contact between independent objects.
        parent = list(range(len(x)))
        def component(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i, j in np.asarray(mesh.edges):
            a, b = component(int(i)), component(int(j))
            if a != b: parent[b] = a
        labels = [component(i) for i in range(len(x))]
        activation = offset + barrier_width
        candidates = ipctk.Candidates()
        candidates.build(mesh, x, inflation_radius=activation)
        if len(candidates.pv_candidates):
            raise ValueError('plane contacts are outside this shell policy')
        local = set(); minimum = activation
        for kind, _ in KINDS:
            for c in getattr(candidates, kind):
                ids = [int(i) for i in c.vertex_ids(mesh.edges, mesh.faces) if i >= 0]
                if len({labels[i] for i in ids}) != 1:
                    continue
                d2 = c.compute_distance(x, mesh.edges, mesh.faces)
                if not np.isfinite(d2) or d2 <= 0:
                    raise ValueError('intersecting or degenerate material reference')
                distance = float(np.sqrt(d2))
                if distance < activation:
                    local.add(stencil_key(kind, c.vertex_ids(mesh.edges, mesh.faces)))
                    minimum = min(minimum, distance)
        digest = hashlib.sha256(x.tobytes() + np.asarray(mesh.faces).tobytes()).hexdigest()
        return cls(frozenset(local), offset, activation, minimum, digest)

    def distance_for(self, kind, ids):
        return 0. if stencil_key(kind, ids) in self.local_keys else self.offset

    def metadata(self):
        return dict(policy='experimental-reference-filter-v1',
                    nonlocal_clearance_m=self.offset,
                    local_clearance_m=0., local_activation_m=self.local_activation,
                    original_activation_m=self.activation,
                    local_stencil_count=len(self.local_keys),
                    reference_sha256=self.reference_sha256,
                    scope='nonintersection for local material stencils; physical offset for all other stencils; not uniform thickness certification')

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(json.dumps(dict(**self.metadata(), local_keys=sorted(self.local_keys)), indent=2))


def certify_filtered_step(mesh, before, after, policy, ccd=None):
    """Certify every candidate; zero-offset local pairs are not skipped."""
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('invalid motion')
    ccd = ccd or ipctk.TightInclusionCCD(tolerance=1e-8, max_iterations=1000000, conservative_rescaling=1.)
    candidates = ipctk.Candidates()
    candidates.build(mesh, before, after, inflation_radius=policy.offset)
    count = 0; locals_checked = 0; total = 0
    if len(candidates.pv_candidates):
        return dict(clear=False, method='unsupported-plane-candidate')
    for kind, split in KINDS:
        for c in getattr(candidates, kind):
            total += 1
            ids = [i for i in c.vertex_ids(mesh.edges, mesh.faces) if i >= 0]
            distance = policy.distance_for(kind, ids)
            locals_checked += distance == 0
            hit, _ = c.ccd(c.dof(before, mesh.edges, mesh.faces), c.dof(after, mesh.edges, mesh.faces), min_distance=distance, narrow_phase_ccd=ccd)
            if not hit:
                continue
            if not candidate_plane_certificate(c, before, after, mesh.edges, mesh.faces, split, distance):
                return dict(clear=False, method='uncertified-filtered-contact', failed_stencil=list(stencil_key(kind, ids)), required_clearance_m=distance)
            count += 1
    return dict(clear=True, method='filtered-tight-inclusion+exact-planes', exact_plane_certificates=count, local_candidates_checked=int(locals_checked), total_candidates=total)
