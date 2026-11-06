import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from origami_bench.diagram.cost_book import CostBook,BudgetExceeded,calls_digest
spec=importlib.util.spec_from_file_location('audit_cost_ledger',ROOT/'scripts/audit_cost_ledger.py')
audit_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def migrate_fixture(path, backup):
    initial=audit_module.audit(path)
    return audit_module.migrate(path,100,150,140,initial['calls_sha256'],backup,'Synthetic unit-test authorization')


class CostBookTests(unittest.TestCase):
    def test_audited_migration_preserves_rows_and_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.db';backup=Path(tmp)/'before.db'
            b=CostBook(path,100);cid=b.reserve('model','old',7);b.finish(cid,None,'http_524',{});b.db.close()
            before=audit_module.audit(path);result=migrate_fixture(path,backup)
            self.assertEqual(before['calls_sha256'],result['after']['calls_sha256'])
            self.assertEqual(audit_module.audit(backup)['configured_ceiling_cny'],100)
            self.assertEqual(result['after']['configured_ceiling_cny'],150)
            self.assertEqual(result['after']['operating_ceiling_cny'],140)
            self.assertEqual(result['after']['unresolved_reserves_cny'],7)
            b=CostBook(path,150,140);self.assertEqual(b.total(),7);b.db.close()

    def test_migration_refuses_changed_ledger_or_existing_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.db';backup=Path(tmp)/'before.db'
            b=CostBook(path,100);old=audit_module.audit(path);b.reserve('A','stage',1);b.db.close()
            with self.assertRaises(ValueError):audit_module.migrate(path,100,150,140,old['calls_sha256'],backup,'authorized')
            self.assertFalse(backup.exists())
            migrate_fixture(path,backup)
            with self.assertRaises(FileExistsError):migrate_fixture(path,backup)

    def test_constructor_cannot_raise_authorized_or_operating_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.db';b=CostBook(path,100);b.db.close()
            with self.assertRaises(ValueError):CostBook(path,150,140)
            with self.assertRaises(ValueError):CostBook(path,100,100)
            self.assertEqual(audit_module.audit(path)['configured_ceiling_cny'],100)

    def test_unknown_charge_and_duplicate_finalization_do_not_refund(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=CostBook(Path(tmp)/'costs.db',100)
            cid=b.reserve('A','stage',5);b.finish(cid,None,'http_502',{})
            self.assertEqual(b.total(),5)
            with self.assertRaises(ValueError):b.finish(cid,0,'response',{})
            self.assertEqual(b.total(),5);b.db.close()

    def test_invalid_amounts_do_not_poison_transactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=CostBook(Path(tmp)/'costs.db',100)
            for x in [-1,float('nan'),float('inf'),True]:
                with self.assertRaises(ValueError):b.reserve('A','stage',x)
            cid=b.reserve('A','stage',1)
            with self.assertRaises(ValueError):b.finish(cid,-1,'response',{})
            b.finish(cid,1,'response',{});self.assertEqual(b.total(),1);b.db.close()

    def test_provider_overage_is_preserved_and_blocks_next_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=CostBook(Path(tmp)/'costs.db',100)
            cid=b.reserve('A','stage',1);b.finish(cid,110,'response',{})
            self.assertEqual(b.total(),110)
            with self.assertRaises(BudgetExceeded):b.reserve('A','stage',0)
            b.db.close()

    def test_stale_process_reopens_after_ceiling_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.db';b=CostBook(path,100)
            migrate_fixture(path,Path(tmp)/'before.db')
            with self.assertRaises(BudgetExceeded):b.reserve('A','stage',1)
            b.db.close()

    def race(self, initial, reservation, stage_limit, expected_wins, expected_total):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.db';b=CostBook(path,100);b.db.close()
            migrate_fixture(path,Path(tmp)/'before.db')
            b=CostBook(path,150,140)
            if initial:
                cid=b.reserve('historical','old',initial);b.finish(cid,None,'http_524',{})
            b.db.close()
            code='''import sys\nsys.path.insert(0,sys.argv[1])\nfrom origami_bench.diagram.cost_book import CostBook,BudgetExceeded\nb=CostBook(sys.argv[2],150,140)\ntry:\n b.reserve('synthetic','pilot',float(sys.argv[3]),float(sys.argv[4])); print('reserved')\nexcept BudgetExceeded: print('blocked')\nfinally: b.db.close()\n'''
            def call(_):
                return subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(path),str(reservation),str(stage_limit)],text=True).strip()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(call,range(8)))
            self.assertEqual(results.count('reserved'),expected_wins)
            self.assertEqual(audit_module.audit(path)['accounted_upper_cny'],expected_total)

    def test_cross_process_stage_reservations_are_atomic(self):
        self.race(0,7,30,4,28)

    def test_cross_process_global_reservations_are_atomic(self):
        self.race(130,4,100,2,138)


if __name__=='__main__':unittest.main()
