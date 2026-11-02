"""Explicit experimental line-hinge material, configured only at flat initialization.

This substitutes a declared crease's elastic coefficient, not a control torque.
It is not a model of how a previously uncreased sheet acquires that material law.
"""
import copy
import numpy as np


def configure_line_material(env, spec):
    if env.time != 0 or len(env.frames) != 1 or env.trace or env.controls or env.grips or env.soft_controls:
        raise ValueError('Line material must be configured before all actions')
    if 'line_crease_material' in env.configuration:
        raise ValueError('Line material is already configured')
    if env.configuration['backend_variant'] != 'codim-ipc-directional-density':
        raise ValueError('Line material requires the verified directional density backend')
    if not isinstance(spec,dict) or set(spec) != {'edges','stiffness_per_length_N'}:
        raise ValueError('Expected material edges and stiffness_per_length_N')
    value=spec['stiffness_per_length_N'];edges=spec['edges']
    if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
        raise ValueError('Positive finite line material stiffness required')
    if not isinstance(edges,list) or not edges or any(not isinstance(e,list) or len(e)!=2 or any(type(v) is not int for v in e) or e[0]==e[1] for e in edges):
        raise ValueError('Expected nonempty material edge list')
    keys=[tuple(sorted(e)) for e in edges]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate material edge')
    native={tuple(sorted((st[1],st[2]))):i for i,st,_ in env.hinge_data()}
    if any(e not in native for e in keys):raise ValueError('Material crease must be an internal mesh edge')
    cfg=env.configuration
    k=cfg['youngs_modulus_Pa']*cfg['material_thickness_m']**3/(24*(1-cfg['poisson_ratio']**2))
    width=2*k/value
    if not np.isfinite(width) or width<=0:raise ValueError('Invalid equivalent coefficient width')
    staged=type(env.sim.edgeInfo)(env.sim.edgeInfo);records=[]
    for edge in keys:
        i=native[edge];old=[staged[i][j] for j in range(3)]
        staged[i]=env.J.Vector3d(old[0],old[1],width)
        records.append(dict(edge=list(edge),hinge=i,original_native_width_m=old[2],equivalent_width_m=width,elastic_coefficient_Nm=value*old[1]/2))
    for item in records:
        v=staged[item['hinge']]
        if not np.isclose(k*v[1]/v[2],item['elastic_coefficient_Nm'],rtol=1e-12,atol=0):
            raise ValueError('Native coefficient readback mismatch')
    env.sim.edgeInfo=staged
    env.configuration['line_crease_material']=dict(specification=copy.deepcopy(spec),edges=records,
        law='E_elastic=0.5*kappa_line*L*(theta-theta_plastic)^2; yield moment remains density*L',
        scope='Declared line material from initial flat state; uncalibrated. Equivalent width is a coefficient parameter, not changed contact geometry. No spontaneous crease formation claim.')
