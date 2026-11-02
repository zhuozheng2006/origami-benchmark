"""Same frozen solver/transport, separate CPU groups for independent episodes."""
from pathlib import Path
import os,subprocess,selectors
from ..remote_codim import RemoteCodimClient,ROOT
class SuiteRemoteClient(RemoteCodimClient):
 def __init__(self,mesh='horizontal',backend_variant='codim-ipc'):
  self.on_frame=None;self.trace=[];self.mesh=mesh;logdir=ROOT/'outputs/viewer';logdir.mkdir(exist_ok=True);self.log=(logdir/'small-suite-remote.log').open('a')
  group=int(os.environ.get('ORIGAMI_SUITE_CPU_GROUP','4'));assert 4<=group<=9
  py='/data/ckpt/zhengzhuo/miniconda3/envs/origami-codim/bin/python'
  command=f'cd /data/ckpt/zhengzhuo/origami-benchmark && {py} scripts/run_bounded.py --cpus 4 --group {group} {py} scripts/codim_worker.py'
  self.proc=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o','BatchMode=yes','4090B',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
  self.sel=selectors.DefaultSelector();self.sel.register(self.proc.stdout,selectors.EVENT_READ);self.buffer=b''
  r=self._request({'mesh':mesh,'backend_variant':backend_variant});self.backend_variant=backend_variant;self.material=r['material'];self.committed_state=r['state'];self.configuration=r.get('configuration')
