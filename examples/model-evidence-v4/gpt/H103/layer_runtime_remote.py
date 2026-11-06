"""Explicit wall-time-only follow-up. Original frozen transport stays unchanged."""
import os,selectors,json,time,hashlib
from pathlib import Path
from .layer_remote import LayerRemoteClient
VERSION='layer-runtime-budget-v14b-1'
class RuntimeLayerRemoteClient(LayerRemoteClient):
 total_timeout=int(os.environ.get('ORIGAMI_TOTAL_TIMEOUT','3600'))
 idle_timeout=int(os.environ.get('ORIGAMI_IDLE_TIMEOUT','600'))
 def _request(self,a):
  if getattr(self,'poisoned',False):raise RuntimeError('Simulation transport lost; further commands are disabled and paper state is unverified')
  complete=False;absolute=time.monotonic()+self.total_timeout;deadline=time.monotonic()+self.idle_timeout
  try:
   self.proc.stdin.write((json.dumps(a)+'\n').encode());self.proc.stdin.flush()
   while True:
    remaining=min(absolute,deadline)-time.monotonic()
    if remaining<=0:raise TimeoutError(f'Simulation request exceeded {self.total_timeout}s total or {self.idle_timeout}s without a frame; end state unverified')
    if b'\n' not in self.buffer:
     if not self.sel.select(remaining):raise TimeoutError('Simulation transport timed out; end state unverified')
     chunk=os.read(self.proc.stdout.fileno(),65536)
     if not chunk:raise EOFError('Simulation connection ended; end state unverified')
     self.buffer+=chunk;continue
    line,self.buffer=self.buffer.split(b'\n',1);r=json.loads(line)
    if 'frame' in r:
     self.cached=r['frame'];deadline=time.monotonic()+self.idle_timeout
     if self.on_frame:self.on_frame()
     continue
    complete=True
    if 'state' in r:self.cached=r['state']
    if not r.get('ok'):raise RuntimeError(r.get('error','Unknown simulator rejection'))
    return r
  except BaseException:
   if not complete:self.poisoned=True
   raise
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.configuration.update(transport_variant=VERSION,transport_total_timeout_seconds=self.total_timeout,transport_idle_timeout_seconds=self.idle_timeout,transport_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
