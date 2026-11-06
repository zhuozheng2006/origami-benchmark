from pathlib import Path
import subprocess,selectors,os,fcntl,json,time,re
from ..remote_codim import RemoteCodimClient,ROOT
class LayerRemoteClient(RemoteCodimClient):
 def _request(self,a):
  if getattr(self,'poisoned',False):raise RuntimeError('Simulation transport lost; further commands are disabled and paper state is unverified')
  complete=False;absolute=time.monotonic()+600;deadline=time.monotonic()+120
  try:
   self.proc.stdin.write((json.dumps(a)+'\n').encode());self.proc.stdin.flush()
   while True:
    remaining=min(absolute,deadline)-time.monotonic()
    if remaining<=0:raise TimeoutError('Simulation request exceeded 600s total or 120s without a frame; end state unverified')
    if b'\n' not in self.buffer:
     if not self.sel.select(remaining):raise TimeoutError('Simulation transport timed out; end state unverified')
     chunk=os.read(self.proc.stdout.fileno(),65536)
     if not chunk:raise EOFError('Simulation connection ended; end state unverified')
     self.buffer+=chunk;continue
    line,self.buffer=self.buffer.split(b'\n',1);r=json.loads(line)
    if 'frame' in r:
     self.cached=r['frame'];deadline=time.monotonic()+120
     if self.on_frame:self.on_frame()
     continue
    complete=True
    if 'state' in r:self.cached=r['state']
    if not r.get('ok'):raise RuntimeError(r.get('error','Unknown simulator rejection'))
    return r
  except BaseException:
   if not complete:self.poisoned=True
   raise
 def __init__(self,mesh,backend_variant='codim-ipc',cpu_group=None,barrier_distance=.001,compression_barrier=False,grip_solve_tolerance=.00002,grip_solve_metric='l2',newton_tolerance=1e-4):
  if backend_variant not in ('codim-ipc','codim-ipc-soft','codim-ipc-directional'):raise ValueError('Unsupported development backend')
  self.on_frame=None;self.trace=[];self.mesh=mesh;self.log=(ROOT/'outputs/layer-paths-v14/remote.log').open('a')
  preferred=int(os.environ.get('ORIGAMI_DOC_CPU_GROUP','4')) if cpu_group is None else cpu_group
  groups=range(4,7) if preferred==0 else [preferred];self.cpu_lease=None
  for group in groups:
   assert 4<=group<=6
   handle=(ROOT/'.local'/f'document-cpu-group-{group}.lock').open('a')
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:handle.close();continue
   self.cpu_lease=handle;break
  if self.cpu_lease is None:raise RuntimeError('All requested document simulation slots are busy; retry after the active run finishes')
  py='<REMOTE_ROOT>/__outside_backend__/miniconda3/envs/origami-codim/bin/python';command=f'cd <REMOTE_ROOT>/__outside_backend__/origami-benchmark && {py} scripts/run_bounded.py --cpus 4 --group {group} {py} scripts/layer_worker.py'
  self.proc=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ServerAliveInterval=30','4090B',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
  self.sel=selectors.DefaultSelector();self.sel.register(self.proc.stdout,selectors.EVENT_READ);self.buffer=b''
  try:r=self._request(dict(mesh=mesh,backend_variant=backend_variant,barrier_distance=barrier_distance,compression_barrier=compression_barrier,grip_solve_tolerance=grip_solve_tolerance,grip_solve_metric=grip_solve_metric,newton_tolerance=newton_tolerance))
  except BaseException:self.close();raise
  self.backend_variant=backend_variant;self.material=r['material'];self.configuration=r['configuration'];self.committed_state=r['state'];self.configuration['cpu_group']=group;self.configuration['backend_variant']=backend_variant
 def recover(self,*args,**kwargs):raise RuntimeError('Document bridge recovery must not switch to the legacy worker; start a separately logged episode')
 def save(self):
  local=super().save();remote='<REMOTE_ROOT>/__outside_backend__/origami-benchmark/outputs/interactive-codim/'+local.name+'/material-history.json'
  subprocess.run(['rsync','-az','-e','ssh -o ClearAllForwardings=yes','4090B:'+remote,str(local)+'/'],check=True,timeout=30);
  subprocess.run(['rsync','-az','-e','ssh -o ClearAllForwardings=yes','4090B:'+remote.rsplit('/',1)[0]+'/rejected-candidates',str(local)+'/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
  self._saved_folder=local
  return local

 def close(self):
  try:
   if getattr(self,'poisoned',False):
    cfg=getattr(self,'configuration',{});pid=cfg.get('worker_process_id');folder=cfg.get('worker_artifact_folder','')
    if type(pid)is int and re.fullmatch('<REMOTE_ROOT>/__outside_backend__/origami-benchmark/outputs/interactive-codim/[a-f0-9]{32}',folder):
     # Only terminate the exact worker that completed our initialization.
     script="import os,signal\nfrom pathlib import Path\np=Path('/proc')/"+repr(str(pid))+"\nif p.exists() and b'scripts/layer_worker.py' in (p/'cmdline').read_bytes().split(b'\\0') and str((p/'fd/1').resolve())=="+repr(folder+'/solver.log')+":os.kill(int(p.name),signal.SIGTERM)\n"
     try:subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','4090B','python3 -'],input=script.encode(),stdout=subprocess.DEVNULL,stderr=self.log,timeout=10,check=False)
     except subprocess.TimeoutExpired:pass
    self.proc.terminate()
    try:self.proc.wait(timeout=5)
    except subprocess.TimeoutExpired:self.proc.kill()
    self.sel.close();self.log.close()
   else:super().close()
  finally:
   if getattr(self,'cpu_lease',None) is not None:self.cpu_lease.close();self.cpu_lease=None
