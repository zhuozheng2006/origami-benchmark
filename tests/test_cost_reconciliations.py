"""Effective-budget controls with synthetic approved accounting records."""
import concurrent.futures
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from origami_bench.diagram.cost_book import CostBook, calls_digest, accounting_digest, decimal_amount


def approve_fixture(db, call_id, provider_value):
    """Create only the columns consumed by the runtime, using a verified row."""
    db.execute('CREATE TABLE IF NOT EXISTS cost_reconciliations (call_id INTEGER, effective_cny_decimal TEXT, original_row_sha256 TEXT, original_row_json TEXT)')
    row=db.execute('SELECT id,created,model,stage,reserve,charge,status,meta FROM calls WHERE id=?',(call_id,)).fetchone()
    fingerprint=hashlib.sha256(json.dumps([row],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    db.execute('INSERT INTO cost_reconciliations VALUES(?,?,?,?)',
               (call_id,provider_value,fingerprint,json.dumps(row,ensure_ascii=False,separators=(',',':'))))


class CostReconciliationTests(unittest.TestCase):
    def fixture(self,path):
        b=CostBook(path,150,140)
        paid=b.reserve('M','stage',10);b.finish(paid,5,'response',{})
        zero=b.reserve('M','stage',.4);b.finish(zero,None,'http_524',{})
        unknown=b.reserve('M','stage',.9);b.finish(unknown,None,'http_500',{})
        b.reserve('M','stage',.7)
        before=calls_digest(b.db)
        approve_fixture(b.db,paid,'.25');approve_fixture(b.db,zero,'0')
        self.assertEqual(calls_digest(b.db),before)
        return b,paid

    def test_effective_cost_replaces_estimate_and_preserves_unmatched_reserves(self):
        with tempfile.TemporaryDirectory() as d:
            b,_=self.fixture(Path(d)/'budget.db')
            self.assertAlmostEqual(b.total(),1.85)
            accounting=b.summary()['accounting']
            self.assertAlmostEqual(accounting['original_accounted_cny'],7.)
            self.assertAlmostEqual(accounting['provider_verified_cny'],.25)
            self.assertAlmostEqual(accounting['unresolved_reserves_cny'],1.6)
            b.db.close()

    def test_upward_correction_and_accounting_hash_preserve_raw_history(self):
        with tempfile.TemporaryDirectory() as d:
            b=CostBook(Path(d)/'budget.db',150,140)
            cid=b.reserve('M','stage',1);b.finish(cid,.249979,'response',{})
            raw=calls_digest(b.db);before=accounting_digest(b.db)
            approve_fixture(b.db,cid,'.25')
            self.assertEqual(b.total(),.25)
            self.assertEqual(calls_digest(b.db),raw)
            self.assertNotEqual(accounting_digest(b.db),before)
            b.db.close()

    def test_invalid_decimal_amounts_are_rejected(self):
        for value in ('NaN','Infinity','-0.01','',None,True):
            with self.assertRaises(ValueError): decimal_amount(value)
        self.assertEqual(decimal_amount('0.000021'),Decimal('0.000021'))

    def test_invalid_reconciled_amount_blocks_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            b,cid=self.fixture(Path(d)/'budget.db')
            b.db.execute("UPDATE cost_reconciliations SET effective_cny_decimal='NaN' WHERE call_id=?",(cid,))
            with self.assertRaises(ValueError): b.reserve('M','stage',.01)
            b.db.close()

    def test_raw_row_change_duplicate_credit_or_inflight_credit_fail_closed(self):
        for scenario in ('raw_changed','duplicate','inflight'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory() as d:
                b,cid=self.fixture(Path(d)/'budget.db')
                if scenario=='raw_changed': b.db.execute('UPDATE calls SET charge=0 WHERE id=?',(cid,))
                elif scenario=='duplicate': approve_fixture(b.db,cid,'.25')
                else:
                    inflight=b.reserve('M','stage',.1)
                    approve_fixture(b.db,inflight,'0')
                with self.assertRaises(ValueError): b.reserve('M','stage',.01)
                b.db.close()

    def race(self,amount,stage_limit):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.db';b,_=self.fixture(path);b.db.close()
            code='''import sys
sys.path.insert(0,sys.argv[1])
from origami_bench.diagram.cost_book import CostBook,BudgetExceeded
b=CostBook(sys.argv[2],150,140)
try:
 b.reserve('M','stage',float(sys.argv[3]),float(sys.argv[4]));print('reserved')
except BudgetExceeded:print('blocked')
finally:b.db.close()
'''
            def invoke(_):
                return subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(path),str(amount),str(stage_limit)],text=True).strip()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(invoke,range(2)))
            self.assertEqual(sorted(results),['blocked','reserved'])

    def test_cross_process_global_limit_uses_effective_total(self):
        self.race(70,150)

    def test_cross_process_stage_limit_uses_effective_total(self):
        self.race(.1,2.)


if __name__=='__main__':unittest.main()
