"""Durable shared cost accounting; explicit cap migration, atomic reservations."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from decimal import Decimal


class BudgetExceeded(RuntimeError):
    pass


def amount(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    return float(value)


def default_operating_limit(limit):
    # Preserve the former 10-CNY guard for a 100-CNY ledger without a fixed cap.
    return limit - min(10.0, limit / 10.0)


def calls_digest(db):
    rows = db.execute('SELECT id,created,model,stage,reserve,charge,status,meta FROM calls ORDER BY id').fetchall()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def accounting_digest(db):
    """Bind a future experiment freeze to both raw calls and appended evidence."""
    value = {'calls_sha256': calls_digest(db)}
    for table in ('provider_evidence', 'reconciliation_batches', 'cost_reconciliations'):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        value[table] = db.execute('SELECT * FROM ' + table + ' ORDER BY 1').fetchall() if exists else []
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def limits(db):
    ceiling = float(db.execute('SELECT ceiling FROM config WHERE id=1').fetchone()[0])
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_policy'").fetchone()
    row = db.execute('SELECT operating_ceiling FROM budget_policy WHERE id=1').fetchone() if exists else None
    return ceiling, float(row[0]) if row else default_operating_limit(ceiling)


def decimal_amount(value, name='amount'):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a nonnegative finite decimal')
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(name + ' must be decimal') from exc
    if not result.is_finite() or result < 0:
        raise ValueError(name + ' must be a nonnegative finite decimal')
    return result


def accounting_rows(db, stage=None):
    """Return effective costs without rewriting any historical call.

    Approved reconciliation is a replacement, including upward corrections.
    Row tampering or an invalid reconciliation fails closed before reserving.
    """
    rows = db.execute('SELECT id,created,model,stage,reserve,charge,status,meta FROM calls ORDER BY id').fetchall()
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cost_reconciliations'").fetchone()
    reconciled = {}
    if exists:
        for cid, value, old_hash, old_json in db.execute('SELECT call_id,effective_cny_decimal,original_row_sha256,original_row_json FROM cost_reconciliations'):
            if cid in reconciled:
                raise ValueError('Multiple reconciliations for one call')
            reconciled[cid] = (decimal_amount(value, 'reconciled charge'), old_hash, old_json)
    ids = {row[0] for row in rows}
    if set(reconciled) - ids:
        raise ValueError('Reconciliation references a missing call')
    result = []
    for row in rows:
        cid, _, model, call_stage, reserve, charge, status, _ = row
        original = decimal_amount(charge if charge is not None else reserve)
        item = reconciled.get(cid)
        if item:
            current = json.dumps([row], ensure_ascii=False, separators=(',', ':'))
            if status == 'reserved' or hashlib.sha256(current.encode()).hexdigest() != item[1] or json.loads(item[2]) != list(row):
                raise ValueError('Reconciled call changed or is still reserved')
        if stage is None or stage == call_stage:
            result.append(dict(id=cid, model=model, stage=call_stage, status=status,
                               original_charge=charge, original_reserve=reserve,
                               original_accounted=original, provider_charge=item[0] if item else None,
                               effective_accounted=item[0] if item else original, reconciled=bool(item),
                               unresolved=charge is None and item is None))
    return result


def accounting_totals(db, stage=None):
    rows = accounting_rows(db, stage)
    total = lambda key: sum((r[key] for r in rows if r[key] is not None), Decimal(0))
    return dict(calls=len(rows), reconciled_calls=sum(r['reconciled'] for r in rows),
                original_accounted_cny=total('original_accounted'), provider_verified_cny=total('provider_charge'),
                effective_accounted_cny=total('effective_accounted'),
                unresolved_reserves_cny=sum((decimal_amount(r['original_reserve']) for r in rows if r['unresolved']), Decimal(0)),
                original_unresolved_reserves_cny=sum((decimal_amount(r['original_reserve']) for r in rows if r['original_charge'] is None), Decimal(0)))


class CostBook:
    def __init__(self, path, limit=100, operating_limit=None):
        self.limit = amount(limit, 'limit')
        if self.limit <= 0:
            raise ValueError('limit must be positive')
        self.operating_limit = None if operating_limit is None else amount(operating_limit, 'operating_limit')
        if self.operating_limit is not None and self.operating_limit > self.limit:
            raise ValueError('Operating limit cannot exceed authorized ceiling')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute('PRAGMA synchronous=FULL')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, ceiling REAL)')
            self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?)', (self.limit,))
            self.db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, created REAL, model TEXT, stage TEXT, reserve REAL, charge REAL, status TEXT, meta TEXT)')
            stored_ceiling, stored_operating = limits(self.db)
            if stored_ceiling != self.limit:
                raise ValueError('Existing ledger ceiling differs; explicit audited migration required')
            if self.operating_limit is None:
                self.operating_limit = stored_operating
            # Runtime options may LOWER a shared operating limit, never raise it.
            if self.operating_limit > stored_operating:
                raise ValueError('Operating limit increase requires explicit audited migration')
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise

    def total(self):
        return float(accounting_totals(self.db)['effective_accounted_cny'])

    def reserve(self, model, stage, amount_cny, stage_limit=None):
        requested = amount(amount_cny, 'reservation')
        if stage_limit is not None:
            stage_limit = amount(stage_limit, 'stage_limit')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            ceiling, operating = limits(self.db)
            if ceiling != self.limit:
                raise BudgetExceeded('Ledger ceiling changed; reopen with audited settings')
            operating = min(operating, self.operating_limit, ceiling)
            # Both totals and INSERT share this writer transaction. Independent
            # processes cannot each spend the same newly reconciled headroom.
            stage_used = accounting_totals(self.db, stage)['effective_accounted_cny']
            total = accounting_totals(self.db)['effective_accounted_cny']
            if total + decimal_amount(requested) > decimal_amount(operating) or (stage_limit is not None and stage_used + decimal_amount(requested) > decimal_amount(stage_limit)):
                raise BudgetExceeded('Shared cumulative or stage cost ceiling would be exceeded')
            cursor = self.db.execute('INSERT INTO calls(created,model,stage,reserve,status) VALUES(?,?,?,?,?)',
                                     (time.time(), model, stage, requested, 'reserved'))
            self.db.commit()
            return cursor.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, call_id, charge, status, meta):
        if charge is not None:
            charge = amount(charge, 'charge')
        if not isinstance(status, str) or not status or status == 'reserved':
            raise ValueError('A terminal billing status is required')
        encoded = json.dumps(meta, ensure_ascii=False, allow_nan=False)
        try:
            self.db.execute('BEGIN IMMEDIATE')
            cursor = self.db.execute('UPDATE calls SET charge=?,status=?,meta=? WHERE id=? AND status=?',
                                     (charge, status, encoded, call_id, 'reserved'))
            if cursor.rowcount != 1:
                raise ValueError('Only one open reservation may be finalized once')
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def summary(self):
        names = ['id','created','model','stage','reserve_cny','upper_charge_cny','status','metadata']
        rows = [dict(zip(names,r)) for r in self.db.execute('SELECT id,created,model,stage,reserve,charge,status,meta FROM calls ORDER BY id')]
        for row in rows:
            row['metadata'] = json.loads(row['metadata']) if row['metadata'] else None
        ceiling, operating = limits(self.db)
        accounting = {key: float(value) if isinstance(value, Decimal) else value for key, value in accounting_totals(self.db).items()}
        return dict(limit_cny=ceiling, operating_limit_cny=min(operating,self.operating_limit),
                    accounted_upper_cny=self.total(), calls=rows, accounting=accounting,
                    method='Verified provider reconciliation replaces a matched estimate; unmatched unknown usage retains its full reservation. Historical calls are preserved.')
