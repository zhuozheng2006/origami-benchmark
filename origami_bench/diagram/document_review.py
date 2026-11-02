"""Local read-only review and a human adapter through the same accounted harness."""
from pathlib import Path
import json,re,sqlite3,uuid,threading
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/document-autonomy-v12';DATA=ROOT/'data/document-pilot-v12'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def summary():
 tasks=read(DATA/'public/index.json',{}).get('tasks',[]);records=[]
 for p in sorted((OUT/'runs').glob('*/report.json')):
  r=read(p);events=read(p.parent/'events.json',[]);records.append(dict(id=p.parent.name,report=r,rejections=sum(e['result']['status']=='rejected' for e in events),model=r.get('model',r.get('agent_kind'))))
 controls=[]
 for p in sorted(OUT.glob('control-*/report.json')):
  r=read(p);events=read(p.parent/'events.json',[]);controls.append(dict(id=p.parent.name,report=r,events=[dict(action=e['action'],status=e['result']['status'],error=e['result'].get('error')) for e in events]))
 cost=read(OUT/'cost-reconciliation.json',{});db=sqlite3.connect('file:'+str(ROOT/'outputs/multimodel-small-v1/costs.sqlite3')+'?mode=ro',uri=True)
 try:cost['accounted_upper_cny']=db.execute('select sum(coalesce(charge,reserve)) from calls').fetchone()[0]
 finally:db.close()
 return dict(tasks=tasks,runs=records,controls=controls,analysis=read(OUT/'analysis.json'),qualification=read(OUT/'qualification.json',{}),freeze=read(OUT/'policy-freeze.json'),cost=cost,claude_preflight=read(OUT/'claude-new-group-preflight/result.json',{}).get('image_verified',False))
def folder(name):
 if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('Invalid run ID')
 p=(OUT/name) if name.startswith('control-') else OUT/'runs'/name
 if not p.exists():raise FileNotFoundError()
 return p
def replay(name):
 p=folder(name)
 if not (p/'trajectory.npz').exists():return dict(vertices=[],faces=[],frame_ids=[],events=read(p/'events.json',[]),ledger=read(p/'attempt-ledger.json',{}),report=read(p/'report.json'),config=read(p/'run-config.json'),images=[dict(name=f.name,url='/document-assets/'+name+'/'+f.name) for f in sorted(p.glob('*.png'))])
 d=np.load(p/'trajectory.npz');v=d['vertices'];events=read(p/'events.json',[]);config=read(p/'run-config.json',{});initial=config.get('initialization',[]);cursor=initial[-1]['frame'] if initial else 0;frames=[]
 for e in events:
  cursor=e.get('frame',e.get('after',{}).get('steps',cursor) if isinstance(e.get('after'),dict) else cursor);e['frame']=cursor;frames.append(cursor)
 ids=np.unique(np.r_[np.arange(0,len(v),max(1,len(v)//280)),len(v)-1,frames]).astype(int)
 ledger=read(p/'attempt-ledger.json',{});clean=[]
 for e in events:
  clean.append({k:e[k] for k in ['index','action','result','frame','time','material_memory','document'] if k in e})
 return dict(vertices=v[ids].round(8).tolist(),faces=d['faces'].tolist(),frame_ids=ids.tolist(),events=clean,ledger=ledger,report=read(p/'report.json'),config=read(p/'run-config.json'),images=[dict(name=f.name,url='/document-assets/'+name+'/'+f.name) for f in sorted(p.glob('*.png'))])

MANUAL=None
LOCK=threading.Lock()
class Human:
 def __init__(self,action):self.action=action
 def request_public(self,text,images):return dict(action=self.action,raw_output=json.dumps(self.action),output_tokens=0,input_tokens=0,seconds=0,public_prompt=text)
def _manual(command):
 global MANUAL
 from .document_episode import DocumentEpisode
 from .document_remote import DocumentRemoteClient
 from .accounted_episode import AccountedEpisode
 from . import document_policy
 op=command.get('op')
 if op=='start':
  if MANUAL is not None:raise ValueError('Close the existing manual session first')
  task=next(t for t in read(DATA/'public/index.json')['tasks'] if t['id']==command.get('task'));private=read(DATA/'private/tasks.json')[task['id']];name='manual-'+uuid.uuid4().hex[:12];ep=DocumentEpisode(task,private,OUT/'runs'/name,lambda mesh:DocumentRemoteClient(mesh,cpu_group=0),40);MANUAL=AccountedEpisode(ep,4096);obs=ep.observation();return dict(run=name,observation='/document-assets/'+name+'/'+Path(obs['current_image']).name)
 if MANUAL is None:raise ValueError('Start a manual session first')
 if op=='close':
  report=MANUAL.finish('human_same_harness');MANUAL.close();MANUAL=None;return dict(report=report)
 if op!='step':raise ValueError('Unknown manual command')
 result,stop=MANUAL.turn(Human(command['action']),document_policy);obs=MANUAL.ep.observation();reply=dict(result=result,budget=MANUAL.ledger.budget(),observation='/document-assets/'+MANUAL.ep.output.name+'/'+Path(obs['current_image']).name)
 if stop:
  reply['report']=MANUAL.finish('human_same_harness');MANUAL.close();MANUAL=None
 return reply

def manual(command):
 with LOCK:return _manual(command)
