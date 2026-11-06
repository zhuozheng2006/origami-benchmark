"""Explicit precrease-artifact-v4 contract; no reference action order is scored.

This measures a retained *directed precrease history*, not an exact final plastic
rest-angle field. Material labels, face orientation and required signs matter.
Placement invariance never certifies contacts, strain or physical translation.
"""
import numpy as np

from .document_grading import retained_history
from .layer_grading_v15 import extract_band_creases
from .structural_grading_v3 import compare_placement

CONTRACT = 'precrease-artifact-v4'
VERSION = 'precrease-artifact-signed-memory-v4-candidate2'
HINGE_THRESHOLD_DEGREES = 140.0
BAND_THRESHOLD_DEGREES = 130.0
EXTRA_BAND_THRESHOLD_DEGREES = 140.0


def _signs(value):
    if not isinstance(value, (list, tuple)) or not value or any(type(s) is not int or s not in (-1, 1) for s in value) or len(set(value)) != len(value):
        raise ValueError('Nonempty unique integer signs -1/+1 required')
    return value


def _validated_history(history, rest):
    if not isinstance(history, list) or not history:
        raise ValueError('Nonempty material history required')
    branch = retained_history(history)
    if not branch:
        raise ValueError('Retained material history is empty')
    angles = np.asarray([h['angles'] for h in branch], float)
    hinges = np.asarray(branch[0]['hinges'])
    if angles.ndim != 2 or not angles.shape[1] or not np.isfinite(angles).all():
        raise ValueError('Finite hinge angles required')
    if hinges.shape != (angles.shape[1], 4) or not np.issubdtype(hinges.dtype, np.integer) or np.any(hinges < 0) or np.any(hinges >= len(rest)):
        raise ValueError('Material hinge topology does not match vertices')
    if any(not np.array_equal(np.asarray(h['hinges']), hinges) for h in branch):
        raise ValueError('Material hinge topology changed in retained branch')
    return branch, angles, hinges


def _band_key(band):
    return tuple(int(i) for i in band['hinge_ids'])


def signed_precrease_check(history, specification, rest_vertices):
    """Required coverage, opposite-sign veto and obvious extra-line veto.

Requirements come from a physically accepted, source-reviewed reference and
must contain signs absent from preparation. Both signs are allowed only where
the source/reference specification explicitly lists both. Extra line detection
uses the existing 140 degree band extractor; smaller incidental bending is not
claimed to be an exact CP mismatch. Missing individual material hinges are
reported in addition to missing bands. Thresholds are version constants.
    """
    try:
        rest = np.asarray(rest_vertices, float)
        if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
            raise ValueError('Finite rest vertices required')
        branch, angles, hinges = _validated_history(history, rest)
        if not np.array_equal(np.asarray(specification['hinge_stencils']), hinges):
            raise ValueError('Material hinge identities differ from the frozen source specification')
        required = specification['required_signed_hinges']
        prepared = specification['prepared_signed_hinges']
        added = specification['task_added_signed_hinges']
        bands = specification['required_band_creases']
        if not isinstance(required, dict) or not required or not isinstance(added, dict) or not added or not bands:
            raise ValueError('Nonempty required and task-added signed creases required')
        normalized = {}
        for key, signs in required.items():
            i = int(key)
            if str(i) != str(key) or not 0 <= i < angles.shape[1]:
                raise ValueError('Invalid required hinge index')
            normalized[i] = _signs(signs)
        for key, signs in added.items():
            _signs(signs)
            if any(s not in required.get(str(key), []) or s in prepared.get(str(key), []) for s in signs):
                raise ValueError('Task-added requirements must be absent from preparation')
        missing, opposite, hinge_peaks = [], [], []
        for i, signs in normalized.items():
            peaks = {str(s): float(np.rad2deg(np.max(angles[:, i] * s))) for s in (-1, 1)}
            hinge_peaks.append(dict(hinge=i, required_signs=signs, peak_degrees=peaks))
            for s in signs:
                if peaks[str(s)] <= HINGE_THRESHOLD_DEGREES:
                    missing.append(dict(hinge=i, sign=s, peak_degrees=peaks[str(s)]))
            if len(signs) == 1 and peaks[str(-signs[0])] > HINGE_THRESHOLD_DEGREES:
                opposite.append(dict(hinge=i, prohibited_sign=-signs[0], peak_degrees=peaks[str(-signs[0])]))
        missing_bands, opposite_bands, band_peaks, band_keys = [], [], [], set()
        for bi, band in enumerate(bands):
            ids = np.asarray(band['hinge_ids'])
            weights = np.asarray(band['weights'], float)
            signs = _signs(band['signs'])
            if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= angles.shape[1]) or weights.shape != ids.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError('Malformed crease band')
            key = _band_key(band)
            if key in band_keys:
                raise ValueError('Duplicate required crease band')
            band_keys.add(key)
            values = angles[:, ids] @ weights
            peaks = {str(s): float(np.rad2deg(np.max(values * s))) for s in (-1, 1)}
            meta = dict(band=bi, direction=band['direction'], offset_m=band['offset_m'])
            band_peaks.append(dict(meta, required_signs=signs, peak_degrees=peaks))
            for s in signs:
                if peaks[str(s)] <= BAND_THRESHOLD_DEGREES:
                    missing_bands.append(dict(meta, sign=s, peak_degrees=peaks[str(s)]))
            if len(signs) == 1 and peaks[str(-signs[0])] > BAND_THRESHOLD_DEGREES:
                opposite_bands.append(dict(meta, prohibited_sign=-signs[0], peak_degrees=peaks[str(-signs[0])]))
        observed = extract_band_creases(history, rest, threshold_degrees=EXTRA_BAND_THRESHOLD_DEGREES)
        unexpected = [b for b in observed if _band_key(b) not in band_keys]
        unexpected_hinges = [dict(hinge=i, signs=[s for s in (-1, 1) if float(np.max(angles[:, i]*s)) > np.deg2rad(HINGE_THRESHOLD_DEGREES)])
                             for i in range(angles.shape[1]) if i not in normalized and float(np.max(np.abs(angles[:, i]))) > np.deg2rad(HINGE_THRESHOLD_DEGREES)]
        return dict(passed=not (missing or missing_bands or opposite or opposite_bands or unexpected), input_valid=True,
                    required_hinges=len(required), required_hinge_signs=sum(map(len, required.values())),
                    task_added_hinge_signs=sum(map(len, added.values())), required_bands=len(bands),
                    missing_hinges=missing, missing_bands=missing_bands,
                    opposite_sign_hinges=opposite, opposite_sign_bands=opposite_bands,
                    unexpected_bands=unexpected, unexpected_hinges_diagnostic=unexpected_hinges,
                    hinge_peaks=hinge_peaks, band_peaks=band_peaks, observed_bands=observed,
                    hinge_threshold_degrees=HINGE_THRESHOLD_DEGREES,
                    band_threshold_degrees=BAND_THRESHOLD_DEGREES,
                    extra_band_threshold_degrees=EXTRA_BAND_THRESHOLD_DEGREES,
                    scope='Retained directed bend history; not exact final plastic rest-angle/CP equality')
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        return dict(passed=False, input_valid=False, reason=str(error), error_type=type(error).__name__)


def grade_precrease_artifact(final_vertices, final_target, *, physical_pass, memory_pass,
                            finished, initial_state_pass, environment_error=False):
    """Final labelled face/shape + memory + unchanged physical gates + finish."""
    base = dict(grading_version=VERSION, grading_contract=CONTRACT,
                physical_pass=physical_pass, memory_pass=memory_pass, finished=finished,
                environment_error=environment_error, initial_state_pass=initial_state_pass,
                process_order_scored=False, moving_half_scored=False,
                diagram_path=dict(applicable=False, reason='artifact_contract_does_not_score_arbitrary_reference_order'))
    try:
        shape = compare_placement(final_vertices, final_target)
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError) as error:
        return dict(base, input_valid=False, task_success=False, shape=dict(passed=False), reason=str(error))
    passed = all(v is True for v in (shape['passed'], physical_pass, memory_pass, finished, initial_state_pass)) and environment_error is False
    return dict(base, input_valid=True, shape=shape, structure_success=shape['passed'], task_success=passed)
