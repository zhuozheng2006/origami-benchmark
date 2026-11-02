"""Durable attempt accounting, independent of undoable simulation state.

The SQLite journal is authoritative. JSON is a derived, atomically replaced export.
An unfinished attempt blocks execution after reopening: never silently refund or
replay an operation whose physical outcome may be unknown.
"""
from pathlib import Path
import json, sqlite3, datetime, math, os

class BudgetExhausted(RuntimeError): pass
class UnresolvedAttempt(RuntimeError): pass

def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0

class AttemptLedger:
    def __init__(self,path,max_attempts=8,output_cap_per_call=384):
        if type(max_attempts) is not int or max_attempts<1 or type(output_cap_per_call) is not int or output_cap_per_call<1:raise ValueError('Positive integer budgets required')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=10)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY, attempt INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('started','response','result')), payload TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(attempt,kind));
          CREATE TRIGGER IF NOT EXISTS journal_no_update BEFORE UPDATE ON journal BEGIN SELECT RAISE(ABORT,'append-only journal'); END;
          CREATE TRIGGER IF NOT EXISTS journal_no_delete BEFORE DELETE ON journal BEGIN SELECT RAISE(ABORT,'append-only journal'); END;
          CREATE TRIGGER IF NOT EXISTS config_no_update BEFORE UPDATE ON config BEGIN SELECT RAISE(ABORT,'immutable budget'); END;
          CREATE TRIGGER IF NOT EXISTS config_no_delete BEFORE DELETE ON config BEGIN SELECT RAISE(ABORT,'immutable budget'); END;
        ''')
        self.config=dict(max_attempts=max_attempts,output_cap_per_call=output_cap_per_call,max_output_tokens=max_attempts*output_cap_per_call)
        encoded=json.dumps(self.config,sort_keys=True)
        self.db.execute('INSERT OR IGNORE INTO config VALUES(1,?)',(encoded,))
        if self.db.execute('SELECT value FROM config').fetchone()[0]!=encoded:
            self.db.close();raise ValueError('Cannot change an existing budget')
    def _append(self,i,kind,payload):
        self.db.execute('INSERT INTO journal(attempt,kind,payload,created) VALUES(?,?,?,?)',(i,kind,json.dumps(payload,ensure_ascii=False,allow_nan=False),datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def attempts(self):
        rows={}
        for i,kind,payload,created in self.db.execute('SELECT attempt,kind,payload,created FROM journal ORDER BY seq'):
            r=rows.setdefault(i,dict(attempt_id=i));r[kind]=json.loads(payload);r[kind+'_at']=created
        return list(rows.values())
    def budget(self):
        rows=self.attempts();costs={};unknown={}
        for key in ('input_tokens','output_tokens','seconds'):
            vals=[r.get('response',{}).get(key) for r in rows]
            costs[key]=sum(v for v in vals if numeric(v));unknown[key]=sum(not numeric(v) for v in vals)
        reserved=sum(self.config['output_cap_per_call'] for r in rows if not numeric(r.get('response',{}).get('output_tokens')))
        return dict(known_wall_seconds=sum(r.get('result',{}).get('wall_seconds',0) for r in rows),**self.config,attempts_used=len(rows),attempts_remaining=self.config['max_attempts']-len(rows),known_costs=costs,unknown_cost_attempts=unknown,output_tokens_reserved_for_unknown=reserved,output_tokens_remaining=self.config['max_output_tokens']-costs['output_tokens']-reserved)
    def begin(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows=self.attempts()
            if any('result' not in r for r in rows):raise UnresolvedAttempt('Unfinished attempt remains charged; reconcile paper state before continuing')
            if rows and (rows[-1]['result']['result'].get('category')=='environment_error' or rows[-1]['result'].get('episode_finished')):raise UnresolvedAttempt('Episode is terminal; no further calls')
            b=self.budget()
            if b['attempts_remaining']<=0 or b['output_tokens_remaining']<self.config['output_cap_per_call']:raise BudgetExhausted('Attempt or output reservation budget exhausted')
            i=len(rows);self._append(i,'started',dict(output_tokens_reserved=self.config['output_cap_per_call']));self.db.commit();return i
        except BaseException:self.db.rollback();raise
    def response(self,i,response):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r=next((r for r in self.attempts() if r['attempt_id']==i),None)
            if r is None or 'response' in r or 'result' in r:raise ValueError('Response must belong to an open attempt exactly once')
            self._append(i,'response',response);self.db.commit()
        except BaseException:self.db.rollback();raise
        if numeric(response.get('output_tokens')) and response['output_tokens']>self.config['output_cap_per_call']:raise ValueError('Worker exceeded frozen output cap; actual cost was retained')
    def finish(self,i,payload):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r=next((r for r in self.attempts() if r['attempt_id']==i),None)
            if r is None or 'result' in r:raise ValueError('Result must belong to an open attempt exactly once')
            self._append(i,'result',payload);self.db.commit()
        except BaseException:self.db.rollback();raise
    def public_history(self):
        history=[]
        for r in self.attempts():
            if 'result' not in r:continue
            response=r.get('response',{});result=r['result']
            history.append(dict(attempt_id=r['attempt_id'],action=response.get('action'),raw_output=response.get('raw_output'),result={k:result['result'][k] for k in ('status','category','detail','rollback_verified') if k in result['result']},before=result.get('before'),after=result.get('after'),cost={k:response.get(k) for k in ('input_tokens','output_tokens','seconds')}))
        return history
    def export(self):
        value=dict(config=self.config,budget=self.budget(),attempts=self.attempts());atomic_json(self.path.with_name('attempt-ledger.json'),value);return value
    def close(self):self.db.close()
