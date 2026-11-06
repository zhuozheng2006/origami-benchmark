"""Read-only shared ledger audit and explicit, backed-up ceiling migration.

No credential reads, API calls, call-row edits, reserve releases or billing claims.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid
from decimal import Decimal
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if Path(__file__).with_name('cost_book.py').is_file():
    from cost_book import amount, calls_digest, accounting_digest, limits, accounting_rows, accounting_totals
else:
    from origami_bench.diagram.cost_book import amount, calls_digest, accounting_digest, limits, accounting_rows, accounting_totals


def audit(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        ceiling, operating = limits(db)
        effective_rows = accounting_rows(db)
        totals = accounting_totals(db)
        grouped = {}
        for row in effective_rows:
            item = grouped.setdefault(row['status'], dict(status=row['status'], count=0,
                original_accounted_cny=0., provider_verified_cny=0., accounted_upper_cny=0., unresolved_reserves_cny=0.))
            item['count'] += 1
            item['original_accounted_cny'] += float(row['original_accounted'])
            item['provider_verified_cny'] += float(row['provider_charge'] or 0)
            item['accounted_upper_cny'] += float(row['effective_accounted'])
            item['unresolved_reserves_cny'] += float(row['original_reserve']) if row['unresolved'] else 0
        n, charged, reserved = db.execute('SELECT COUNT(*),COALESCE(SUM(charge),0),COALESCE(SUM(CASE WHEN charge IS NULL THEN reserve ELSE 0 END),0) FROM calls').fetchone()
        fingerprint = calls_digest(db)
        accounting_fingerprint = accounting_digest(db)
    total=float(totals['effective_accounted_cny'])
    return dict(ledger_mutated=False, calls=n, calls_sha256=fingerprint, accounting_sha256=accounting_fingerprint,
        configured_ceiling_cny=ceiling, operating_ceiling_cny=operating,
        estimated_charges_cny=charged, original_unresolved_reserves_cny=reserved,
        original_accounted_cny=float(totals['original_accounted_cny']),
        reconciled_calls=totals['reconciled_calls'], provider_verified_cny=float(totals['provider_verified_cny']),
        unresolved_reserves_cny=float(totals['unresolved_reserves_cny']), accounted_upper_cny=total,
        operating_headroom_cny=operating-total,
        by_status=[grouped[k] for k in sorted(grouped)],
        caveat='Verified provider charges replace only uniquely reconciled calls. Original estimates and all unmatched unknown reserves remain visible. Not an external invoice.')


def migrate(path, expected_ceiling, new_ceiling, operating_limit, expected_calls_sha256, backup, authorization_note):
    path, backup = Path(path).resolve(), Path(backup).resolve()
    cap=amount(new_ceiling,'new ceiling');op=amount(operating_limit,'operating limit')
    if not 0 < op <= cap or not authorization_note.strip():
        raise ValueError('Positive operating limit <= ceiling and explicit authorization note required')
    if path == backup or backup.exists():
        raise FileExistsError('Backup must be a new distinct file')
    db=sqlite3.connect(path.as_uri()+'?mode=rw',uri=True,isolation_level=None,timeout=30)
    db.execute('PRAGMA synchronous=FULL')
    try:
        db.execute('BEGIN IMMEDIATE')
        old_cap,old_op=limits(db)
        fingerprint=calls_digest(db)
        total=float(accounting_totals(db)['effective_accounted_cny'])
        if old_cap != expected_ceiling or fingerprint != expected_calls_sha256:
            raise ValueError('Ledger changed since audit; re-audit before explicit migration')
        if total > op:
            raise ValueError('New operating limit is below preserved accounted total')
        # A separate read connection snapshots while this writer holds RESERVED
        # lock. Calling backup on the writer itself can wait on its own transaction.
        backup.parent.mkdir(parents=True,exist_ok=True)
        with backup.open('xb'):
            pass
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(backup) as dest:
            source.backup(dest)
        with sqlite3.connect(backup.as_uri()+'?mode=ro',uri=True) as saved:
            if calls_digest(saved) != fingerprint or limits(saved)!=(old_cap,old_op):
                raise RuntimeError('Backup verification failed; ceiling unchanged')
        backup_sha=hashlib.sha256(backup.read_bytes()).hexdigest()
        db.execute('CREATE TABLE IF NOT EXISTS budget_policy (id INTEGER PRIMARY KEY CHECK(id=1), operating_ceiling REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS budget_migrations (id TEXT PRIMARY KEY, created TEXT, old_ceiling REAL, new_ceiling REAL, old_operating REAL, new_operating REAL, authorization_note TEXT, backup_path TEXT, backup_sha256 TEXT, calls_sha256 TEXT)')
        event=dict(id=str(uuid.uuid4()),created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            old_ceiling=old_cap,new_ceiling=cap,old_operating=old_op,new_operating=op,
            authorization_note=authorization_note,backup_path=str(backup),backup_sha256=backup_sha,calls_sha256=fingerprint)
        db.execute('UPDATE config SET ceiling=? WHERE id=1',(cap,))
        db.execute('INSERT OR REPLACE INTO budget_policy VALUES (1,?)',(op,))
        db.execute('INSERT INTO budget_migrations VALUES (?,?,?,?,?,?,?,?,?,?)',tuple(event.values()))
        if calls_digest(db)!=fingerprint:
            raise RuntimeError('Historical calls changed; rollback migration')
        db.commit()
        return dict(migration=event,historical_calls_unchanged=True,after=audit(path))
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true',help='Explicit audited migration; otherwise read-only')
    p.add_argument('--expected-ceiling',type=float)
    p.add_argument('--new-ceiling',type=float)
    p.add_argument('--operating-limit',type=float)
    p.add_argument('--expected-calls-sha256')
    p.add_argument('--backup',type=Path)
    p.add_argument('--authorization-note')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep old audits; choose a new output')
    if a.output.resolve()==a.ledger.resolve() or (a.backup and a.output.resolve()==a.backup.resolve()):raise ValueError('Audit output must be distinct')
    if a.apply:
        if any(x is None for x in (a.expected_ceiling,a.new_ceiling,a.operating_limit,a.expected_calls_sha256,a.backup,a.authorization_note)):
            p.error('--apply requires expected/new/operating limits, calls digest, new backup and authorization note')
        result=migrate(a.ledger,a.expected_ceiling,a.new_ceiling,a.operating_limit,a.expected_calls_sha256,a.backup,a.authorization_note)
    else:
        result=audit(a.ledger)
        if a.new_ceiling is not None:
            result['proposed_ceiling_cny']=a.new_ceiling
            result['proposed_operating_cny']=a.operating_limit
            result['proposed_operating_headroom_cny']=a.operating_limit-result['accounted_upper_cny'] if a.operating_limit is not None else None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
