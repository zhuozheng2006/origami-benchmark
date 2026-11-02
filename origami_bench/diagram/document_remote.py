from pathlib import Path
import subprocess,selectors,os,fcntl
from ..remote_codim import RemoteCodimClient,ROOT
class DocumentRemoteClient(RemoteCodimClient):
 def __init__(self,mesh,backend_variant='codim-ipc',cpu_group=None):
  if backend_variant!='codim-ipc':raise ValueError('Only unchanged qualified material/backend allowed')
  self.on_frame=None;self.trace=[];self.mesh=mesh;self.log=(ROOT/'outputs/document-autonomy-v12/remote.log').open('a')
  preferred=int(os.environ.get('ORIGAMI_DOC_CPU_GROUP','4')) if cpu_group is None else cpu_group
  groups=range(4,7) if preferred==0 else [preferred];self.cpu_lease=None
  for group in groups:
   assert 4<=group<=6
   handle=(ROOT/'.local'/f'document-cpu-group-{group}.lock').open('a')
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:handle.close();continue
   self.cpu_lease=handle;break
  if self.cpu_lease is None:raise RuntimeError('All requested document simulation slots are busy; retry after the active run finishes')
  py='/data/ckpt/zhengzhuo/miniconda3/envs/origami-codim/bin/python';command=f'cd /data/ckpt/zhengzhuo/origami-benchmark && {py} scripts/run_bounded.py --cpus 4 --group {group} {py} scripts/document_worker.py'
  self.proc=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ServerAliveInterval=30','4090B',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
  self.sel=selectors.DefaultSelector();self.sel.register(self.proc.stdout,selectors.EVENT_READ);self.buffer=b''
  try:r=self._request(dict(mesh=mesh))
  except BaseException:self.close();raise
  self.backend_variant=backend_variant;self.material=r['material'];self.configuration=r['configuration'];self.committed_state=r['state'];self.configuration['cpu_group']=group
 def recover(self,*args,**kwargs):raise RuntimeError('Document bridge recovery must not switch to the legacy worker; start a separately logged episode')
 def save(self):
  local=super().save();remote='/data/ckpt/zhengzhuo/origami-benchmark/outputs/interactive-codim/'+local.name+'/material-history.json'
  subprocess.run(['rsync','-az','-e','ssh -o ClearAllForwardings=yes','4090B:'+remote,str(local)+'/'],check=True,timeout=30);return local

 def close(self):
  try:super().close()
  finally:
   if getattr(self,'cpu_lease',None) is not None:self.cpu_lease.close();self.cpu_lease=None
