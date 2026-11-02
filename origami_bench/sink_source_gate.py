"""Separate permission to experiment from a strict benchmark acceptance claim."""
import math


def validate_sink_source(result,goal,audit,layers,allow_approximate=False):
    if not result.get('completed') or not audit.get('completed'):
        raise ValueError('Source trajectory is incomplete')
    if 'bad_intervals' not in audit or audit['bad_intervals']:
        raise ValueError('Source continuous clearance is missing or failed')
    strain=[audit.get(k,float('nan')) for k in ('peak_stretch','peak_compression')]
    if not all(math.isfinite(x) and 0<=x<.01 for x in strain):
        raise ValueError('Source strain is missing or failed')
    seq=goal.get('open_sink_sequence',{})
    edges=seq.get('edges',[])
    if not seq.get('passed') or not edges or not all(e.get('correct_final_sign') is True for e in edges):
        raise ValueError('Source actual sink sequence and MV must pass')
    if not seq.get('inner_total') or seq.get('inner_reversed')!=seq['inner_total']:
        raise ValueError('Source internal reversal is incomplete')
    enclosed=layers.get('final_containment',{}).get('enclosed_area_fraction',float('nan'))
    if not math.isfinite(enclosed) or enclosed<.95:
        raise ValueError('Source cap containment is missing or insufficient')
    shape=goal.get('held_shape',{})
    if not all(math.isfinite(shape.get(k,float('nan'))) and shape[k]>=0 for k in ('rms_mm','max_mm')):
        raise ValueError('Source shape must be measured even for an experiment')
    if goal.get('passed'):return 'strictly_accepted'
    if not allow_approximate:
        raise ValueError('Source shape acceptance failed; explicit approximate-source experiment required')
    return 'approximate_shape_experiment'
