"""Private grading only; neither prompts nor the controller import this module."""
import numpy as np

def retained_history(history):
 """Replay the paper branch, leaving the immutable attempt/history files intact."""
 branch=[]
 for h in history:
  if h['op']=='undo':
   if len(branch)>1:branch.pop()
  else:branch.append(h)
 return branch

def crease_check(history,required,threshold_degrees=140):
 if not required:return True
 branch=retained_history(history)
 if not branch or not all('angles' in h for h in branch):return False
 angles=np.asarray([h['angles'] for h in branch]);threshold=np.deg2rad(threshold_degrees)
 return all(bool(np.max(angles[:,int(i)]*sign)>threshold) for i,signs in required.items() for sign in signs)

def reference_creases(history,threshold_degrees=150):
 angles=np.asarray([h['angles'] for h in retained_history(history)]);threshold=np.deg2rad(threshold_degrees)
 return {str(i):[sign for sign in (-1,1) if np.max(angles[:,i]*sign)>threshold] for i in range(angles.shape[1]) if np.max(np.abs(angles[:,i]))>threshold}
