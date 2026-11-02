"""Tight-Inclusion CCD plus exact separating-plane certificates.
A fixed plane separates two linearly moving convex primitives throughout a step
if its projected gap exceeds dmin at BOTH endpoints. All final dot products and
squared comparisons use exact rational arithmetic on the input binary floats.
No sampled-distance or AdditiveCCD result is accepted as a certificate.
"""
from fractions import Fraction as F
import numpy as np
import ipctk

def separating_plane(a,b,group_size,dmin,axes):
 return _plane_interval(a,b,group_size,dmin,axes,F(0),F(1))

def _plane_interval(a,b,group_size,dmin,axes,lo,hi):
 """Exact proof on a dyadic interval of the ORIGINAL endpoint motion."""
 threshold=F(float(dmin))**2
 for axis in axes:
  n=np.asarray(axis,float)
  if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)==0:continue
  # Cheap rejection only; every accepted direction is rechecked exactly below.
  pa0=a@n;pb0=b@n;pa=pa0*(1-float(lo))+pb0*float(lo);pb=pa0*(1-float(hi))+pb0*float(hi);norm=np.linalg.norm(n)
  forward=min(pa[:group_size].min()-pa[group_size:].max(),pb[:group_size].min()-pb[group_size:].max())
  backward=min(pa[group_size:].min()-pa[:group_size].max(),pb[group_size:].min()-pb[:group_size].max())
  if max(forward,backward)<dmin*norm*.999999:continue
  q=[F(float(x)) for x in n];norm2=sum(x*x for x in q)
  def projections(x):return [sum(F(float(v))*w for v,w in zip(row,q)) for row in x]
  xa,xb=projections(a),projections(b)
  x=[(1-lo)*u+lo*v for u,v in zip(xa,xb)]
  y=[(1-hi)*u+hi*v for u,v in zip(xa,xb)]
  for first,second in [(slice(0,group_size),slice(group_size,None)),(slice(group_size,None),slice(0,group_size))]:
   gap=min(min(x[first])-max(x[second]),min(y[first])-max(y[second]))
   if gap>0 and gap*gap>threshold*norm2:return True
 return False

def candidate_plane_certificate(candidate,before,after,edges,faces,group_size,dmin):
 """Search for a plane, but prove its gap over the entire step exactly.

 Interior positions supply only candidate normal directions. Acceptance still
 uses the ORIGINAL endpoints in exact rational arithmetic; it never accepts a
 step from sampled distances. Every vertex projection is affine in time, so
 strict endpoint separation proves separation for all intermediate times.
 """
 ids=[i for i in candidate.vertex_ids(edges,faces) if i>=0]
 a,b=before[ids],after[ids]
 axes=[candidate.compute_distance_vector(x,edges,faces) for x in (before,after)]
 if separating_plane(a,b,group_size,dmin,axes):return True
 axes=[candidate.compute_distance_vector(before*(1-t)+after*t,edges,faces) for t in (.25,.5,.75)]
 if separating_plane(a,b,group_size,dmin,axes):return True
 # A single plane need not separate an otherwise safe moving pair. Cover the
 # original linear motion with dyadic intervals, each proved by its own plane.
 # Rounded interpolated coordinates are used ONLY to propose directions; the
 # proof projects original binary floats and interpolates with exact Fractions.
 pending=[(F(0),F(1),0)]
 while pending:
  lo,hi,depth=pending.pop()
  times=[lo,(lo+hi)/2,hi]
  normals=[candidate.compute_distance_vector(before*(1-float(t))+after*float(t),edges,faces) for t in times]
  if _plane_interval(a,b,group_size,dmin,normals,lo,hi):continue
  if depth>=6:return False
  mid=(lo+hi)/2
  pending.extend([(mid,hi,depth+1),(lo,mid,depth+1)])
 return True

def certify_step(mesh,before,after,min_distance=.0001,ccd=None):
 ccd=ccd or ipctk.TightInclusionCCD(tolerance=1e-8,max_iterations=1000000,conservative_rescaling=1.)
 if ipctk.is_step_collision_free(mesh,before,after,min_distance=min_distance,narrow_phase_ccd=ccd):return dict(clear=True,method='tight-inclusion',exact_plane_certificates=0)
 candidates=ipctk.Candidates();candidates.build(mesh,before,after,inflation_radius=min_distance)
 edges,faces=mesh.edges,mesh.faces;count=0
 if len(candidates.pv_candidates):return dict(clear=False,method='unsupported-plane-candidate',exact_plane_certificates=0)
 for attr,split in [('vv_candidates',1),('ev_candidates',1),('fv_candidates',1),('ee_candidates',2)]:
  for c in getattr(candidates,attr):
   hit,_=c.ccd(c.dof(before,edges,faces),c.dof(after,edges,faces),min_distance=min_distance,narrow_phase_ccd=ccd)
   if not hit:continue
   if not candidate_plane_certificate(c,before,after,edges,faces,split,min_distance):return dict(clear=False,method='uncertified',exact_plane_certificates=count)
   count+=1
 return dict(clear=True,method='tight-inclusion-plus-exact-separating-planes',exact_plane_certificates=count)
