"""Private stdio bridge. Simulation logs never mix with the JSON protocol."""
from pathlib import Path
import os,sys,json,uuid,re
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));protocol=os.fdopen(os.dup(1),'w',buffering=1)
run=root/'outputs/interactive-codim'/uuid.uuid4().hex;run.mkdir(parents=True)
log=open(run/'solver.log','w');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
def send(obj):protocol.write(json.dumps(obj,allow_nan=False)+'\n');protocol.flush()
try:
 init=json.loads(sys.stdin.readline());mesh=init.get('mesh','horizontal')
 if not re.fullmatch('[a-z0-9_]+',mesh):raise ValueError('invalid mesh name')
 from origami_bench.codim_env import CodimEnvironment
 env=CodimEnvironment(mesh,run,backend_variant=init.get('backend_variant','codim-ipc'));env.on_frame=lambda st:send({'frame':st});send({'ok':True,'state':env.state(),'material':env.rest.tolist(),'configuration':env.configuration})
 for line in sys.stdin:
  request=json.loads(line)
  try:
   if request.get('op')=='close':send({'ok':True});break
   if request.get('op')=='save':env.save(run);send({'ok':True,'path':str(run)});continue
   state=env.step(request);send({'ok':True,'state':state})
  except Exception as e:send({'ok':False,'error':type(e).__name__+': '+str(e),'state':env.state()})
 env.save(run)
except Exception as e:send({'ok':False,'error':type(e).__name__+': '+str(e)})
