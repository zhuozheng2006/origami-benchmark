"""Credential-local multimodal API transport with durable global cost reservations."""
from pathlib import Path
import base64,hashlib,json,sqlite3,time,urllib.request,urllib.error,ssl,math
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
class BudgetExceeded(RuntimeError):pass
class NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  raise urllib.error.HTTPError(req.full_url,code,'Credential-bearing redirects are disabled',headers,fp)
def usage_counts(usage):
 if not isinstance(usage,dict):return None
 values=[usage.get('prompt_tokens'),usage.get('completion_tokens')]
 if any(type(x) is not int or x<0 for x in values):return None
 return tuple(values)
class CostBook:
 def __init__(self,path,limit=100):
  Path(path).parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(path,timeout=30)
  self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, ceiling REAL)')
  self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?)',(limit,));assert self.db.execute('SELECT ceiling FROM config').fetchone()[0]==limit
  self.db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, created REAL, model TEXT, stage TEXT, reserve REAL, charge REAL, status TEXT, meta TEXT)');self.db.commit();self.limit=limit
 def total(self):return self.db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls').fetchone()[0]
 def reserve(self,model,stage,amount,stage_limit=None):
  self.db.execute('BEGIN IMMEDIATE')
  stage_used=self.db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls WHERE stage=?',(stage,)).fetchone()[0]
  if self.total()+amount>min(self.limit,90) or (stage_limit is not None and stage_used+amount>stage_limit):self.db.rollback();raise BudgetExceeded('Cost ceiling would be exceeded')
  cur=self.db.execute('INSERT INTO calls(created,model,stage,reserve,status) VALUES(?,?,?,?,?)',(time.time(),model,stage,amount,'reserved'));self.db.commit();return cur.lastrowid
 def finish(self,i,charge,status,meta):
  self.db.execute('UPDATE calls SET charge=?,status=?,meta=? WHERE id=? AND status=?',(charge,status,json.dumps(meta,ensure_ascii=False),i,'reserved'));self.db.commit()
 def summary(self):
  names=['id','created','model','stage','reserve_cny','upper_charge_cny','status','metadata'];rows=[dict(zip(names,r)) for r in self.db.execute('SELECT * FROM calls')]
  for r in rows:r['metadata']=json.loads(r['metadata']) if r['metadata'] else None
  return dict(limit_cny=self.limit,accounted_upper_cny=self.total(),calls=rows,method='Reserve before request; upper-bound configured prices; missing usage retains full reservation. Gateway credits 1 USD unit = 1 CNY per user; no exchange conversion applied to gateway.')
class ApiPolicy:
 def __init__(self,spec,costbook,out,stage='preflight'):
  self.spec=spec;self.book=costbook;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.stage=stage
  credentials=json.loads((ROOT/'.local/api-credentials.json').read_text())['connections'][spec['connection']]
  expected='https://api.deepseek.com' if spec['connection']=='deepseek' else 'https://aicodelink.top/v1' if spec['connection'].startswith('gateway') else None
  if expected is None:raise ValueError('Unknown credential origin')
  if credentials['base_url'].rstrip('/')!=expected:raise ValueError('Configured endpoint differs from authorized provider')
  self.url=expected+'/chat/completions';self.key=credentials['api_key'];self.metadata={k:v for k,v in spec.items() if k!='api_key'}
 def request_public(self,text,image_paths):
  spec=self.spec;image_blocks=[];image_info=[];bound=len(text.encode())+4096
  for p in image_paths:
   content=Path(p).read_bytes();im=Image.open(p);bound+=math.ceil(im.width*im.height/16)+4096
   image_info.append(dict(sha256=hashlib.sha256(content).hexdigest(),size=list(im.size)))
   image_blocks.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(content).decode()}})
  cap=spec['max_output_tokens'];rates=spec['upper_cny_per_million'];reserve=(bound*rates[0]+cap*rates[1])/1e6
  i=self.book.reserve(spec['model'],self.stage,reserve,5 if self.stage=='preflight' else None)
  req=dict(model=spec['model'],messages=[dict(role='user',content=[{'type':'text','text':text}]+image_blocks)])
  req.update(spec.get('parameters',{}));req[spec.get('output_parameter','max_tokens')]=cap
  meta=dict(call_id=i,model=spec['model'],origin=self.url,stage=self.stage,public_prompt=text,images=image_info,settings={k:v for k,v in req.items() if k!='messages'},reserve_cny=reserve,input_token_upper_bound=bound)
  (self.out/f'call-{i:05d}-request.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
  start=time.monotonic();status='unknown';charge=None;response={}
  try:
   request=urllib.request.Request(self.url,data=json.dumps(req).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'OrigamiBenchmark/0.11'},method='POST')
   with urllib.request.build_opener(NoCredentialRedirect).open(request,timeout=180) as res:response=json.loads(res.read());status='response'
  except urllib.error.HTTPError as e:
   status='http_'+str(e.code);response={'error':{'http_status':e.code,'message':e.read(8192).decode(errors='replace').replace(self.key,'[REDACTED]')}}
  except Exception as e:status='transport_error';response={'error':{'type':type(e).__name__}}
  safe=json.loads(json.dumps(response,ensure_ascii=False).replace(self.key,'[REDACTED]'))
  usage=safe.get('usage');input_tokens=output_tokens=None
  counts=usage_counts(usage)
  if counts is not None:
   input_tokens,output_tokens=counts;charge=(input_tokens*rates[0]+output_tokens*rates[1])/1e6
  if safe.get('error') and status=='response':status='provider_error'
  seconds=time.monotonic()-start
  (self.out/f'call-{i:05d}-response.json').write_text(json.dumps(dict(status=status,response=safe,seconds=seconds),ensure_ascii=False,indent=2))
  self.book.finish(i,charge,status,dict(input_tokens=input_tokens,output_tokens=output_tokens,usage=usage,seconds=seconds,origin=self.url,request_path=str(self.out/f'call-{i:05d}-request.json')))
  if status!='response':raise RuntimeError('API '+status+'; immutable call record '+str(i))
  choice=safe.get('choices',[{}])[0];raw=choice.get('message',{}).get('content') or '';action=None
  try:
   cleaned=raw.strip()
   if cleaned.startswith('```'):cleaned=cleaned.split('\n',1)[1].rsplit('```',1)[0].strip()
   action=json.loads(cleaned)
  except (ValueError,IndexError):pass
  return dict(ok=True,raw=raw,raw_output=raw,action=action,input_tokens=input_tokens,output_tokens=output_tokens,seconds=seconds,public_prompt=text,api_call_id=i,usage=usage,returned_model=safe.get('model'),finish_reason=choice.get('finish_reason'),reasoning=choice.get('message',{}).get('reasoning_content'),cost_upper_cny=charge)
 def close(self):pass
