"""Local client for the same CPU shell controller used by automated checks."""
from pathlib import Path
import subprocess,json,selectors,time,os
ROOT=Path(__file__).resolve().parents[1]
class RemoteCodimClient:
 def __init__(self,mesh='horizontal',backend_variant='codim-ipc'):
  self.on_frame=None;self.trace=[];self.mesh=mesh;logdir=ROOT/'outputs/viewer';logdir.mkdir(parents=True,exist_ok=True);self.log=(logdir/'remote-codim.log').open('a')
  command='cd /data/ckpt/zhengzhuo/origami-benchmark && /data/ckpt/zhengzhuo/miniconda3/envs/origami-codim/bin/python scripts/run_bounded.py --cpus 4 --group 4 /data/ckpt/zhengzhuo/miniconda3/envs/origami-codim/bin/python scripts/codim_worker.py'
  self.proc=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o','BatchMode=yes','4090B',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0);self.sel=selectors.DefaultSelector();self.sel.register(self.proc.stdout,selectors.EVENT_READ)
  self.buffer=b'';r=self._request({'mesh':mesh,'backend_variant':backend_variant});self.backend_variant=backend_variant;self.material=r['material'];self.committed_state=r['state'];self.configuration=r.get('configuration')
 def _request(self,a):
  self.proc.stdin.write((json.dumps(a)+'\n').encode());self.proc.stdin.flush();deadline=time.monotonic()+120
  while True:
   if b'\n' not in self.buffer:
    if not self.sel.select(max(0,deadline-time.monotonic())):raise TimeoutError('远端计算超时；请重建纸张')
    chunk=os.read(self.proc.stdout.fileno(),65536)
    if not chunk:raise RuntimeError('远端连接结束；请重建纸张')
    self.buffer+=chunk
    continue
   line,self.buffer=self.buffer.split(b'\n',1)
   r=json.loads(line)
   if 'frame' in r:
    deadline=time.monotonic()+120
    self.cached=r['frame']
    if self.on_frame:self.on_frame()
    continue
   if 'state' in r:self.cached=r['state']
   if not r.get('ok'):raise RuntimeError(r.get('error','未知错误'))
   return r
 def state(self):return self.cached
 def step(self,a):
  r=self._request(a);self.committed_state=r['state']
  if a.get('op')=='undo':self.trace.pop()
  else:self.trace.append(a)
  return r['state']
 def recover(self):
  replacement=RemoteCodimClient(self.mesh,self.backend_variant)
  try:
   if self.configuration is None or replacement.configuration!=self.configuration:raise RuntimeError('求解器版本或材料配置改变，不能安全重放；原记录已保留')
   if replacement.material!=self.material:raise RuntimeError('材料网格改变，不能安全重放；原记录已保留')
   for action in self.trace:replacement.step(action)
   if replacement.state()['hash']!=self.committed_state['hash']:raise RuntimeError('重放状态与上次确认状态不一致，已保留原记录')
  except Exception:
   replacement.close();raise
  return replacement
 def save(self):
  r=self._request({'op':'save'});remote=r['path'];prefix='/data/ckpt/zhengzhuo/origami-benchmark/outputs/interactive-codim/'
  if not remote.startswith(prefix) or not remote[len(prefix):].isalnum():raise ValueError('unexpected output path')
  local=ROOT/'outputs/interactive-codim'/remote.rsplit('/',1)[-1];local.mkdir(parents=True,exist_ok=True)
  sources=['4090B:'+remote+'/'+name for name in ['trajectory.npz','actions.json','metrics.json','model-config.json']]
  subprocess.run(['rsync','-az','-e','ssh -o ClearAllForwardings=yes',*sources,str(local)+'/'],check=True,timeout=30);return local
 def close(self):
  try:self._request({'op':'close'});self.proc.wait(timeout=5)
  except Exception:self.proc.terminate()
  self.sel.close();self.log.close()
