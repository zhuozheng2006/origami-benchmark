"""Generic source-aware crease grading; unknown M/V assignments are not invented."""
import json
import numpy as np
from .document_episode import DocumentEpisode
from .document_grading import retained_history
from .attempt_ledger import atomic_json
from .document_pose_audit import compare_in_plane
def unsigned_crease_check(history,required,threshold=140):
 if not required:return True
 h=retained_history(history)
 if not h:return False
 angles=np.asarray([x['angles'] for x in h])
 return all(np.max(np.abs(angles[:,int(i)]))>np.deg2rad(threshold) for i in required)
class HardEpisode(DocumentEpisode):
 def finish(self,agent_kind):
  r=super().finish(agent_kind)
  r['unsigned_memory_pass']=unsigned_crease_check(json.loads((self.output/'material-history.json').read_text()),self.private.get('required_unsigned_hinges',[]))
  r['grading_version']='source-assignment-v13-1'
  r['fixed_board_shape']=r['shape']
  if self.private.get('goal'):
   state=json.loads((self.output/'final-state.json').read_text())
   r['shape']=compare_in_plane(state['vertices'],np.load(self.private['goal'])['vertices'])
   # A proper XY rotation/translation is harmless placement, not a reflection,
   # scaling, z-offset, deformation, or teleportation of the actual paper.
   r['task_success']=bool(r['finished'] and r['physical_pass'] and r['memory_pass'] and r['unsigned_memory_pass'] and r['shape']['rms_mm']<1 and r['shape']['max_mm']<3 and r['status']!='environment_error')
   if r['status']!='environment_error':r['status']='passed_scoped' if r['task_success'] else 'not_passed'
  atomic_json(self.output/'report.json',r)
  return r
