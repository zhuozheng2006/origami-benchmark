"""Durable shared cost accounting; explicit cap migration, atomic reservations."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


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


def limits(db):
    ceiling = float(db.execute('SELECT ceiling FROM config WHERE id=1').fetchone()[0])
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_policy'").fetchone()
    row = db.execute('SELECT operating_ceiling FROM budget_policy WHERE id=1').fetchone() if exists else None
    return ceiling, float(row[0]) if row else default_operating_limit(ceiling)


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
        return float(self.db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls').fetchone()[0])

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
            stage_used = self.db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls WHERE stage=?', (stage,)).fetchone()[0]
            if self.total() + requested > operating or (stage_limit is not None and stage_used + requested > stage_limit):
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
        return dict(limit_cny=ceiling, operating_limit_cny=min(operating,self.operating_limit),
                    accounted_upper_cny=self.total(), calls=rows,
                    method='Shared append-preserving ledger; unknown usage retains full reservation. Estimated charges are not provider invoices.')
