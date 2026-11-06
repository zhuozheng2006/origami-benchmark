"""Comparable physics metadata; excludes process IDs, CPU leases and clocks."""
import hashlib,json
KEYS=('backend_binary_sha256','layer_env_sha256','document_worker_sha256','continuous_clearance_verifier','contact_barrier_parameter_m','effective_activation_distance_m','newton_velocity_RMS_tolerance_m_s','max_newton_iterations','maximum_step_seconds','youngs_modulus_Pa','poisson_ratio','density_kg_m3','material_thickness_m','gravity','friction','yield_moment_Nm_per_hinge','contact_clearance_m','principal_strain_limit','compression_barrier','native_singular_value_limits','native_strain_barrier_stiffness','grip_solver_tolerance_m','grip_solver_metric','grip_acceptance_max_error_m')
def signature(configuration):
 missing=[k for k in KEYS if k not in configuration]
 if missing:raise ValueError('Incomplete physical protocol: '+','.join(missing))
 result={k:configuration[k] for k in KEYS}
 for key in ['control_geometry_sha256','contact_certification_sha256']:result[key]=configuration[key]
 return result
def digest(configuration):return hashlib.sha256(json.dumps(signature(configuration),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def execution_options(configuration):
 return dict(newton_tolerance=configuration['newton_velocity_RMS_tolerance_m_s'],backend_variant=configuration['backend_variant'],barrier_distance=configuration['contact_barrier_parameter_m'],compression_barrier=configuration['compression_barrier'],grip_solve_tolerance=configuration['grip_solver_tolerance_m'],grip_solve_metric=configuration['grip_solver_metric'])
