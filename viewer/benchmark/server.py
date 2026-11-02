"""Local review UI. Only public diagrams and completed run artifacts are served."""
from pathlib import Path
import sys,json,mimetypes,re,sqlite3
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse,unquote
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from origami_bench.diagram.episode import DEFAULT_DATA
WEB=Path(__file__).resolve().parent;RUNS=ROOT/'outputs/diagram2fold-v1.1'


def read_attempt_ledger(folder):
    path=folder/'attempts.sqlite3'
    if not path.exists():return None
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:
        config=json.loads(db.execute('SELECT value FROM config').fetchone()[0]);rows={}
        for i,kind,payload in db.execute('SELECT attempt,kind,payload FROM journal ORDER BY seq'):
            rows.setdefault(i,dict(attempt_id=i))[kind]=json.loads(payload)
        return dict(config=config,attempts=list(rows.values()))
    finally:db.close()


def attempt_counts(folder, report):
    path=folder/'events.json'
    if not path.exists():return None
    events=json.loads(path.read_text())
    rejected=sum(e.get('result',{}).get('status')=='rejected' for e in events)
    undos=sum(e.get('action')=={'op':'undo'} for e in events)
    observation=folder/'observation-000.json'
    budget=json.loads(observation.read_text()).get('max_decisions') if observation.exists() else None
    direct=bool(report['task_success'] and not rejected and not undos and not report.get('perception_error_count'))
    ledger=read_attempt_ledger(folder)
    return dict(decisions=len(ledger['attempts']) if ledger else len(events),budget=ledger['config']['max_attempts'] if ledger else budget,rejected=rejected,undo_requests=undos,direct_success=direct)


def index():
    data=json.loads((DEFAULT_DATA/'public/index.json').read_text());runs=[]
    for p in sorted(RUNS.glob('*/report.json')):
        r=json.loads(p.read_text())
        if not {'task_id','agent_kind','task_success'}<=r.keys():continue
        runs.append(dict(id=p.parent.name,report=r,accounting=attempt_counts(p.parent,r)))
    data['runs']=runs
    context=ROOT/'outputs/decision-context-v10'
    for key,file in [('context_study','summary.json'),('context_audit','audit.json'),('context_probes','probe-audit.json')]:
        data[key]=json.loads((context/file).read_text()) if (context/file).exists() else None
    history=ROOT/'outputs/attempt-history-v9'
    data['history_study']=json.loads((history/'summary.json').read_text()) if (history/'summary.json').exists() else None
    data['history_audit']=json.loads((history/'audit.json').read_text()) if (history/'audit.json').exists() else None
    data['qualification']=json.loads((RUNS/'qualification-manifest.json').read_text())['tasks']
    diagnostic=ROOT/'outputs/diagram2fold-d001-debug/assessment.json'
    data['diagnostic']=json.loads(diagnostic.read_text()) if diagnostic.exists() else None
    validation=ROOT/'outputs/half-fold-validation-v1/summary.json'
    data['validation']=json.loads(validation.read_text()) if validation.exists() else None
    reverse=ROOT/'outputs/half-fold-validation-v1/reverse-summary.json'
    if data['validation'] is not None:data['validation']['reverse_physical']=json.loads(reverse.read_text()) if reverse.exists() else None
    data['line_validation']=[json.loads(p.read_text()) for p in sorted((ROOT/'outputs').glob('line-fold-validation-v*/summary.json'))]
    progress=ROOT/'outputs/model-comparison-32b/progress.json'
    data['model_progress']=json.loads(progress.read_text()) if progress.exists() else None
    heldout=ROOT/'data/diagram2fold-generalization-v1/public/index.json'
    assessment=ROOT/'outputs/generalization-32b-v1/summary.json'
    data['generalization']=json.loads(assessment.read_text()) if assessment.exists() else None
    if heldout.exists():
        extra=json.loads(heldout.read_text())['tasks']
        for t in extra:t['image']='heldout/'+t['image']
        data['tasks']+=extra
        if data['generalization']:
            data['qualification'] += [r['qualification'] for r in data['generalization']['rows'] if r['qualification']]
    heldout2=ROOT/'data/diagram2fold-generalization-v2/public/index.json'
    paired=ROOT/'outputs/generalization-32b-v2/summary.json'
    data['paired']=json.loads(paired.read_text()) if paired.exists() else None
    analysis=ROOT/'outputs/generalization-32b-v2/analysis.json'
    data['paired_analysis']=json.loads(analysis.read_text()) if analysis.exists() else None
    progress2=ROOT/'outputs/generalization-32b-v2/thinking-progress.json'
    data['paired_progress']=json.loads(progress2.read_text()) if progress2.exists() else None
    diagnosis=ROOT/'outputs/interface-v8-development/failure-diagnosis.json'
    data['failure_diagnosis']=json.loads(diagnosis.read_text()) if diagnosis.exists() else None
    if heldout2.exists():
        extra=json.loads(heldout2.read_text())['tasks']
        for t in extra:t['image']='heldout-v2/'+t['image']
        data['tasks']+=extra
        if data['paired']:data['qualification'] += [r['qualification'] for r in data['paired']['rows'] if r['qualification']]
    small=ROOT/'data/diagram2fold-small-v1/public/index.json'
    if small.exists():
        extra=json.loads(small.read_text())['tasks']
        for t in extra:t['image']='small-v1/'+t['image']
        data['tasks']+=extra
        q=ROOT/'outputs/multimodel-small-v1/qualification.json'
        if q.exists():data['qualification']+=json.loads(q.read_text())
    return data


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in ('/api/document/manual','/api/hard/manual'):return self.send_error(404)
        origin=self.headers.get('Origin')
        if origin and origin != 'http://'+self.headers.get('Host',''):return self.send_error(403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<16384:raise ValueError('Invalid request length')
            from origami_bench.diagram.document_review import manual
            if self.path=='/api/hard/manual':
                from origami_bench.diagram.hard_review import manual
            return self.json(manual(json.loads(self.rfile.read(length))))
        except Exception as e:return self.json(dict(error=str(e)))

    def do_GET(self):
        route=unquote(urlparse(self.path).path)
        try:
            if route=='/api/hard':
                from origami_bench.diagram.hard_review import summary
                return self.json(summary())
            hard_match=re.fullmatch(r'/api/hard/run/([A-Za-z0-9_-]+)',route)
            if hard_match:
                from origami_bench.diagram.hard_review import replay
                return self.json(replay(hard_match.group(1)))
            if route=='/api/document':
                from origami_bench.diagram.document_review import summary
                return self.json(summary())
            match=re.fullmatch(r'/api/document/run/([A-Za-z0-9_-]+)',route)
            if match:
                from origami_bench.diagram.document_review import replay
                return self.json(replay(match[1]))
            if route=='/api/tiers':
                payload=json.loads((ROOT/'data/diagram2fold-tiered-v1/catalog.json').read_text())
                payload['design']=json.loads((ROOT/'configs/tiered-suite-v1.design.json').read_text())
                payload['harness']=json.loads((ROOT/'outputs/tiered-design-v1/harness-audit.json').read_text())
                return self.json(payload)
            if route=='/api/suite':
                folder=ROOT/'outputs/multimodel-small-v1';payload=json.loads((folder/'summary.json').read_text())
                payload['tasks']=json.loads((ROOT/'data/diagram2fold-small-v1/public/index.json').read_text())['tasks']
                payload['qualification']=json.loads((folder/'qualification.json').read_text())
                payload['qualified']=sum(q['eligible'] for q in payload['qualification']);payload['qualification_complete']=len(payload['qualification'])==120
                frozen=json.loads((folder/'policy-freeze.json').read_text()) if (folder/'policy-freeze.json').exists() else {}
                payload['test_size']=len(frozen.get('tasks',[]));payload['test_ids']=frozen.get('tasks',[])
                payload['model_roster']=[s['model'] for s in frozen.get('model_specs',[])]+([frozen['local_model_spec']['model']] if frozen.get('local_model_spec') else [])
                payload['audit']=json.loads((folder/'audit.json').read_text()) if (folder/'audit.json').exists() else None
                payload['novelty']=json.loads((folder/'novelty-audit.json').read_text()) if (folder/'novelty-audit.json').exists() else None
                db=sqlite3.connect('file:'+str(folder/'costs.sqlite3')+'?mode=ro',uri=True)
                try:payload['cost']['accounted_upper_cny']=db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls').fetchone()[0]
                finally:db.close()
                return self.json(payload)
            if route=='/api/index': return self.json(index())
            ledger_match=re.fullmatch(r'/api/ledger/([A-Za-z0-9_-]+)',route)
            if ledger_match:return self.json(read_attempt_ledger(RUNS/ledger_match[1]))
            match=re.fullmatch(r'/api/run/([A-Za-z0-9_-]+)',route)
            if match:
                folder=RUNS/match[1]
                d=np.load(folder/'trajectory.npz');v=d['vertices']
                checkpoints=json.loads((folder/'checkpoints.json').read_text()) if (folder/'checkpoints.json').exists() else []
                ids=np.unique(np.r_[np.arange(0,len(v),max(1,len(v)//220)),len(v)-1,[c['frame'] for c in checkpoints]]).astype(int)
                events=json.loads((folder/'events.json').read_text()) if (folder/'events.json').exists() else []
                policy=[json.loads(p.read_text()) for p in sorted(folder.glob('adapter-turn-*.json'))]
                inputs=[dict(name=p.name,url=f'/run-assets/{folder.name}/{p.name}',kind='model_input',phase=json.loads(p.with_suffix('.json').read_text()).get('phase')) for p in sorted(folder.glob('policy-panel-*.png'))]
                inputs += [dict(name=p.name,url=f'/run-assets/{folder.name}/{p.name}',kind='measurement_overlay') for p in sorted(folder.glob('pixel-measurement-*.png'))]
                observations=[dict(name=p.name,url=f'/run-assets/{folder.name}/{p.name}') for p in sorted(folder.glob('observation-*.png'))]
                readings=[dict(file=p.name,**json.loads(p.read_text())) for p in sorted(list(folder.glob('perception-*.json'))+list(folder.glob('model-turn-*.json'))+list(folder.glob('pixel-measurement-*.json')))]
                report=json.loads((folder/'report.json').read_text())
                return self.json(dict(vertices=v[ids].round(7).tolist(),faces=d['faces'].tolist(),frame_ids=ids.tolist(),total_frames=len(v),events=events,policy=policy,checkpoints=checkpoints,inputs=inputs,observations=observations,readings=readings,ledger=read_attempt_ledger(folder),accounting=attempt_counts(folder,report),report=report))
            if route in ('/hard-summary/comparison.png','/hard-summary/analysis.json','/hard-summary/results.csv','/hard-summary/crease-audit.png'):
                path=ROOT/'outputs/document-hard-v13'/route.rsplit('/',1)[1]
            elif route=='/hard':path=WEB/'hard.html'
            elif re.fullmatch(r'/hard-assets/[A-Za-z0-9_-]+/(final|observation-[0-9]{3}|document-view-[0-9]{3})\.png',route):
                from origami_bench.diagram.hard_review import folder
                parts=route.split('/');path=folder(parts[2])/parts[3]
            elif route in ('/document-summary/comparison.png','/document-summary/analysis.json','/document-summary/results.csv'):
                path=ROOT/'outputs/document-autonomy-v12'/route.rsplit('/',1)[1]
            elif route=='/autonomy':path=WEB/'document.html'
            elif re.fullmatch(r'/document-assets/[A-Za-z0-9_-]+/(final|observation-[0-9]{3}|document-view-[0-9]{3})\.png',route):
                from origami_bench.diagram.document_review import folder
                _,_,name,filename=route.split('/');path=folder(name)/filename
            elif route=='/tiers':path=WEB/'tiers.html'
            elif re.fullmatch(r'/tiered-assets/T[0-9]{3}/(source\.pdf|page-[0-9]{3}\.png)',route):
                _,_,source_id,filename=route.split('/')
                allowed={x['id'] for x in json.loads((ROOT/'data/diagram2fold-tiered-v1/catalog.json').read_text())['sources']}
                if source_id not in allowed:raise FileNotFoundError()
                path=ROOT/'.local/diagram-sources-tiered-v1'/source_id/filename
            elif route=='/suite':path=WEB/'suite.html'
            elif route in ('/suite-assets/comparison.png','/suite-assets/task-matrix.png','/suite-assets/results.csv','/suite-assets/audit.json'):
                path=ROOT/'outputs/multimodel-small-v1'/route.rsplit('/',1)[1]
            elif route.startswith('/public/small-v1/'):
                public=ROOT/'data/diagram2fold-small-v1/public';path=(public/route[len('/public/small-v1/'):]).resolve()
                if not path.is_relative_to(public.resolve()):raise FileNotFoundError()
            elif route=='/': path=WEB/'index.html'
            elif route in ('/paired-assets/comparison-final-states.png','/paired-assets/results.csv'):
                path=ROOT/'outputs/generalization-32b-v2'/route.rsplit('/',1)[1]
            elif route in ('/context-assets/comparison.png','/context-assets/diagnostic-evidence.png','/context-assets/results.csv','/context-assets/probe-audit.json'):
                path=ROOT/'outputs/decision-context-v10'/route.rsplit('/',1)[1]
            elif route in ('/history-assets/comparison.png','/history-assets/results.csv'):
                path=ROOT/'outputs/attempt-history-v9'/route.rsplit('/',1)[1]
            elif route=='/validation-assets/reversed-arrow.png':path=ROOT/'outputs/diagram2fold-d001-debug/reversed-arrow.png'
            elif re.fullmatch(r'/run-assets/[A-Za-z0-9_-]+/(policy-panel|pixel-measurement|observation)-[0-9]{3}\.png',route):
                _,_,run_id,filename=route.split('/');path=RUNS/run_id/filename
            elif route.startswith('/public/heldout-v2/'):
                public=ROOT/'data/diagram2fold-generalization-v2/public'
                path=(public/route[len('/public/heldout-v2/'):]).resolve()
                if not path.is_relative_to(public.resolve()):raise FileNotFoundError()
            elif route.startswith('/public/heldout/'):
                public=ROOT/'data/diagram2fold-generalization-v1/public'
                path=(public/route[len('/public/heldout/'):]).resolve()
                if not path.is_relative_to(public.resolve()):raise FileNotFoundError()
            elif route.startswith('/public/'):
                path=(DEFAULT_DATA/'public'/route[8:]).resolve()
                if not path.is_relative_to((DEFAULT_DATA/'public').resolve()): raise FileNotFoundError()
            elif route.startswith('/vendor/'):
                path=(WEB/route.lstrip('/')).resolve()
                if not path.is_relative_to(WEB/'vendor'): raise FileNotFoundError()
            else: raise FileNotFoundError()
            body=path.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(body)
        except (FileNotFoundError,ValueError,KeyError): self.send_error(404)
    def json(self,obj):
        b=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);args=p.parse_args()
    print(f'Diagram2Fold review: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
