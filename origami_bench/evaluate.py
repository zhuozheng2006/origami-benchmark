"""Conservative endpoint regression metrics, not a general state-equivalence oracle."""
import numpy as np

def compare_states(actual, target, tolerance=1e-7):
    # Current metric assumes matching material vertex IDs and face subdivision.
    # Different valid subdivisions are unscorable, never automatically failures.
    keys = set(actual['vertices'])
    if keys != set(target['vertices']) or actual['faces'] != target['faces']:
        return dict(status='unscorable_topology', geometric_match=None, physical_validity='unknown', benchmark_success=None)
    errors = [np.linalg.norm(np.array(actual['vertices'][k])-target['vertices'][k]) for k in keys]
    error = float(max(errors, default=0))
    layers_match = actual['layers'] == target['layers']
    orientation_match = actual['faces_orientations'] == target['faces_orientations']
    # Edge direction and order are representation details; assignment matters.
    def edges(s): return sorted((min(a,b),max(a,b),c) for a,b,c in s['edges'])
    assignments_match = edges(actual)==edges(target)
    return dict(status='scored_geometry_only', max_vertex_error=error,
                layers_match=layers_match, orientations_match=orientation_match,
                crease_assignments_match=assignments_match,
                geometric_match=error<=tolerance and layers_match and orientation_match and assignments_match,
                physical_validity='unknown', benchmark_success=None)

