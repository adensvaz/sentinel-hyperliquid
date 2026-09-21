"""Persistent storage — normalized SQLite schema with WAL for a concurrent reader (dashboard) +
writer (engine loop).

Schema (v2):
  account          one row per mode: capital, realized/fees/funding pnl, peak, paused_until
  positions        current open book per (mode, symbol)            [PK(mode,symbol)]
  scores           latest signal score per (mode, symbol)          [PK(mode,symbol)]
  equity_curve     equity/gross/net/drawdown time series           [ix ts]
  fills            order fills                                      [ix ts, ix symbol]
  trades           closed round-trips                               [ix ts, ix symbol]
  whale_consensus  blended whale bias per coin (latest snapshot)    [ix ts]
  whale_wallets    per-wallet summary (latest snapshot)            [ix ts]
  meta             singletons (schema_version, whales_ts)

WAL + synchronous=NORMAL gives fast, durable, concurrent access with zero ops. The schema is plain SQL
and normalized, so it ports to Postgres/Timescale unchanged if multi-host scale is ever needed.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account(
  mode TEXT PRIMARY KEY, starting_capital REAL, realized_pnl REAL, fees_paid REAL,
  funding_pnl REAL, last_funding_ts REAL, peak_equity REAL, paused_until REAL, updated_ts INTEGER);
CREATE TABLE IF NOT EXISTS positions(
  mode TEXT, symbol TEXT, contracts INTEGER, avg_price REAL, multiplier REAL,
  opened_ts REAL, updated_ts INTEGER, PRIMARY KEY(mode, symbol));
CREATE TABLE IF NOT EXISTS scores(
  mode TEXT, symbol TEXT, score REAL, updated_ts INTEGER, PRIMARY KEY(mode, symbol));
CREATE TABLE IF NOT EXISTS equity_curve(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, mode TEXT,
  equity REAL, gross REAL, net REAL, drawdown_pct REAL);
CREATE INDEX IF NOT EXISTS ix_equity_ts ON equity_curve(mode, ts);
CREATE TABLE IF NOT EXISTS fills(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, mode TEXT, symbol TEXT, side TEXT,
  open_close TEXT, volume INTEGER, price REAL, fee REAL, order_id TEXT, status TEXT);
CREATE INDEX IF NOT EXISTS ix_fills_ts ON fills(mode, ts);
CREATE INDEX IF NOT EXISTS ix_fills_sym ON fills(symbol);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, mode TEXT, symbol TEXT, side TEXT,
  entry REAL, exit REAL, contracts INTEGER, pnl REAL, opened_ts REAL, duration_s REAL);
CREATE INDEX IF NOT EXISTS ix_trades_ts ON trades(mode, ts);
CREATE INDEX IF NOT EXISTS ix_trades_sym ON trades(symbol);
CREATE TABLE IF NOT EXISTS whale_consensus(ts INTEGER, coin TEXT, bias REAL);
CREATE INDEX IF NOT EXISTS ix_wc_ts ON whale_consensus(ts);
CREATE TABLE IF NOT EXISTS whale_wallets(ts INTEGER, address TEXT, equity REAL, gross REAL, net REAL);
CREATE INDEX IF NOT EXISTS ix_ww_ts ON whale_wallets(ts);
CREATE TABLE IF NOT EXISTS funding_history(ts INTEGER, mode TEXT, symbol TEXT, current REAL, next REAL);
CREATE INDEX IF NOT EXISTS ix_fh_ts ON funding_history(mode, ts);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS treasury(
  mode TEXT PRIMARY KEY, vault REAL, hwm REAL, last_sweep_ts REAL, total_swept REAL, updated_ts INTEGER);
-- Execution decisions, the training set for exec_policy. `arm` is what we actually did, `shadow`
-- is what the learned policy WOULD have done while it runs alongside without authority. Comparing
-- the two over weeks is what earns it the right to take over. arrival_mid is captured BEFORE the
-- order goes out, because it is the only non-circular benchmark for slippage.
CREATE TABLE IF NOT EXISTS exec_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, mode TEXT, symbol TEXT, side TEXT,
  context TEXT, arm TEXT, shadow TEXT, arrival_mid REAL, fill_price REAL, notional REAL,
  cost_bps REAL, filled INTEGER, order_id TEXT, ref_price REAL);
CREATE INDEX IF NOT EXISTS ix_exec_ts ON exec_log(mode, ts);
CREATE INDEX IF NOT EXISTS ix_exec_ctx ON exec_log(context, arm);
"""


class Store:
    def __init__(self, db_path: str, equity_csv: Optional[str] = None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.equity_csv = equity_csv
        self.conn = sqlite3.connect(db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self._add_column("exec_log", "ref_price", "REAL")
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                          (str(SCHEMA_VERSION),))
        self.conn.commit()

    def _add_column(self, table: str, col: str, decl: str) -> None:
        """CREATE TABLE IF NOT EXISTS is a no-op on a database that already has the table, so a
        column added to the schema never reaches an existing deployment. This closes that gap."""
        try:
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if have and col not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except sqlite3.Error:
            pass

    # -- meta -----------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        import json
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        import json
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    # -- account --------------------------------------------------------------
    def load_account(self, mode: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM account WHERE mode=?", (mode,)).fetchone()
        return dict(r) if r else None

    def save_account(self, mode: str, *, starting_capital: float, realized_pnl: float, fees_paid: float,
                     funding_pnl: float, last_funding_ts: float) -> None:
        cur = self.load_account(mode) or {}
        peak = cur.get("peak_equity") or 0.0
        paused = cur.get("paused_until") or 0.0
        self.conn.execute(
            """INSERT INTO account(mode,starting_capital,realized_pnl,fees_paid,funding_pnl,
                 last_funding_ts,peak_equity,paused_until,updated_ts) VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(mode) DO UPDATE SET starting_capital=excluded.starting_capital,
                 realized_pnl=excluded.realized_pnl, fees_paid=excluded.fees_paid,
                 funding_pnl=excluded.funding_pnl, last_funding_ts=excluded.last_funding_ts,
                 updated_ts=excluded.updated_ts""",
            (mode, starting_capital, realized_pnl, fees_paid, funding_pnl, last_funding_ts,
             peak, paused, int(time.time())))
        self.conn.commit()

    def peak_equity(self, mode: str, default: float = 0.0) -> float:
        a = self.load_account(mode)
        return float(a["peak_equity"]) if a and a.get("peak_equity") is not None else default

    def update_peak_equity(self, mode: str, equity: float) -> float:
        a = self.load_account(mode)
        peak = max(equity, float(a["peak_equity"]) if a and a.get("peak_equity") else 0.0)
        if a:
            self.conn.execute("UPDATE account SET peak_equity=?, updated_ts=? WHERE mode=?",
                              (peak, int(time.time()), mode))
        else:
            self.conn.execute("INSERT INTO account(mode,peak_equity,updated_ts) VALUES(?,?,?)",
                              (mode, peak, int(time.time())))
        self.conn.commit()
        return peak

    # -- treasury (money-manager vault) --------------------------------------
    def load_treasury(self, mode: str) -> dict:
        """Return the treasury state for a book (zeros if none yet)."""
        cur = self.conn.execute(
            "SELECT vault, hwm, last_sweep_ts, total_swept FROM treasury WHERE mode=?", (mode,))
        row = cur.fetchone()
        if not row:
            return {"vault": 0.0, "hwm": 0.0, "last_sweep_ts": 0.0, "total_swept": 0.0}
        return {"vault": float(row[0] or 0), "hwm": float(row[1] or 0),
                "last_sweep_ts": float(row[2] or 0), "total_swept": float(row[3] or 0)}

    def save_treasury(self, mode: str, vault: float, hwm: float,
                      last_sweep_ts: float, total_swept: float) -> None:
        self.conn.execute(
            """INSERT INTO treasury(mode,vault,hwm,last_sweep_ts,total_swept,updated_ts)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(mode) DO UPDATE SET vault=excluded.vault, hwm=excluded.hwm,
                 last_sweep_ts=excluded.last_sweep_ts, total_swept=excluded.total_swept,
                 updated_ts=excluded.updated_ts""",
            (mode, vault, hwm, last_sweep_ts, total_swept, int(time.time())))
        self.conn.commit()

    def paused_until(self, mode: str) -> float:
        a = self.load_account(mode)
        return float(a["paused_until"]) if a and a.get("paused_until") else 0.0

    def set_paused_until(self, mode: str, ts: float) -> None:
        if self.load_account(mode):
            self.conn.execute("UPDATE account SET paused_until=?, updated_ts=? WHERE mode=?",
                              (ts, int(time.time()), mode))
        else:
            self.conn.execute("INSERT INTO account(mode,paused_until,updated_ts) VALUES(?,?,?)",
                              (mode, ts, int(time.time())))
        self.conn.commit()

    # -- positions ------------------------------------------------------------
    def load_positions(self, mode: str) -> dict:
        rows = self.conn.execute("SELECT * FROM positions WHERE mode=?", (mode,)).fetchall()
        return {r["symbol"]: {"contracts": int(r["contracts"]), "avg_price": r["avg_price"],
                              "multiplier": r["multiplier"], "opened_ts": r["opened_ts"]} for r in rows}

    def save_positions(self, mode: str, positions: dict) -> None:
        self.conn.execute("DELETE FROM positions WHERE mode=?", (mode,))
        rows = [(mode, s, int(p["contracts"]), p["avg_price"], p.get("multiplier", 1.0),
                 p.get("opened_ts", 0.0), int(time.time()))
                for s, p in positions.items() if int(p["contracts"]) != 0]
        if rows:
            self.conn.executemany(
                "INSERT INTO positions(mode,symbol,contracts,avg_price,multiplier,opened_ts,updated_ts) "
                "VALUES(?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    # -- scores ---------------------------------------------------------------
    def save_scores(self, mode: str, scores: dict) -> None:
        ts = int(time.time())
        rows = [(mode, s, float(v), ts) for s, v in scores.items()]
        if rows:
            self.conn.executemany(
                "INSERT INTO scores(mode,symbol,score,updated_ts) VALUES(?,?,?,?) "
                "ON CONFLICT(mode,symbol) DO UPDATE SET score=excluded.score, updated_ts=excluded.updated_ts",
                rows)
            self.conn.commit()

    def scores_map(self, mode: str) -> dict:
        rows = self.conn.execute("SELECT symbol,score FROM scores WHERE mode=?", (mode,)).fetchall()
        return {r["symbol"]: r["score"] for r in rows}

    # -- equity curve ---------------------------------------------------------
    def record_equity(self, mode: str, equity: float, gross: float, net: float,
                      drawdown_pct: float, ts: Optional[int] = None) -> None:
        ts = ts or int(time.time())
        self.conn.execute("INSERT INTO equity_curve(ts,mode,equity,gross,net,drawdown_pct) VALUES(?,?,?,?,?,?)",
                          (ts, mode, equity, gross, net, drawdown_pct))
        self.conn.commit()
        if self.equity_csv:
            p = Path(self.equity_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            with p.open("a") as f:
                if new:
                    f.write("ts,iso,mode,equity,gross,net,drawdown_pct\n")
                iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
                f.write(f"{ts},{iso},{mode},{equity:.4f},{gross:.4f},{net:.4f},{drawdown_pct:.4f}\n")

    def equity_series(self, mode: str, full_days: float = 8.0) -> list[dict]:
        """The equity curve for the dashboard: every tick for the last `full_days`, one point per
        day (that day's last tick) for everything older.

        This used to be `ORDER BY ts ASC LIMIT 5000`, which returns the OLDEST 5,000 rows. At ~96
        ticks a day every book crossed that line around day 52, after which the newest data was
        silently cut off. Champion's 7D view then showed a single point — the synthetic "now" the
        dashboard appends — while the table underneath held 673 rows for that week. The headline
        numbers stayed correct because they are computed live, which is exactly what made it hard
        to notice: the chart died and nothing else did.

        Down-sampling the old tail instead of truncating it means the payload stays small forever
        (~1 row/day plus ~96/day for the recent window) and the ALL view keeps its full history.
        The chart already collapses 30D/ALL to daily client-side, so it loses nothing."""
        cutoff = time.time() - full_days * 86_400
        old = self.conn.execute(
            "SELECT ts,equity,gross,net,drawdown_pct,MAX(ts) AS _m FROM equity_curve "
            "WHERE mode=? AND ts<? GROUP BY CAST(ts/86400 AS INTEGER) ORDER BY ts ASC",
            (mode, cutoff)).fetchall()
        recent = self.conn.execute(
            "SELECT ts,equity,gross,net,drawdown_pct FROM equity_curve "
            "WHERE mode=? AND ts>=? ORDER BY ts ASC",
            (mode, cutoff)).fetchall()
        keep = ("ts", "equity", "gross", "net", "drawdown_pct")
        return [{k: r[k] for k in keep} for r in old] + [dict(r) for r in recent]

    def last_equity(self, mode: str) -> Optional[float]:
        r = self.conn.execute("SELECT equity FROM equity_curve WHERE mode=? ORDER BY ts DESC LIMIT 1",
                              (mode,)).fetchone()
        return r["equity"] if r else None

    def daily_returns(self, mode: str, days: int = 21) -> list:
        """Per-day returns from the last day-end equity points (downsamples the 60s ticks to one/day),
        so volatility targeting is on a consistent daily basis regardless of tick frequency."""
        rows = self.conn.execute("SELECT ts, equity FROM equity_curve WHERE mode=? ORDER BY ts ASC",
                                 (mode,)).fetchall()
        by_day = {}
        for r in rows:
            by_day[time.strftime("%Y-%m-%d", time.localtime(r["ts"]))] = r["equity"]
        eqs = [by_day[d] for d in sorted(by_day)][-(days + 1):]
        return [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] > 0]

    # -- fills ----------------------------------------------------------------
    def record_fills(self, mode: str, fills, ts: Optional[int] = None) -> None:
        ts = ts or int(time.time())
        self.conn.executemany(
            "INSERT INTO fills(ts,mode,symbol,side,open_close,volume,price,fee,order_id,status) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            [(ts, mode, f.symbol, f.side, f.open, f.volume, f.price, f.fee, f.order_id, f.status) for f in fills])
        self.conn.commit()

    def recent_fills(self, mode: str, limit: int = 25) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM fills WHERE mode=? ORDER BY id DESC LIMIT ?",
                                 (mode, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- execution learning ---------------------------------------------------
    def record_exec(self, mode: str, *, symbol: str, side: str, context: str, arm: str,
                    shadow: str, arrival_mid: float, notional: float, order_id: str = "",
                    ts: Optional[int] = None, ref_price: float = 0.0) -> int:
        """Log the decision at send time. cost_bps stays NULL until the outcome is known —
        an order whose result was never observed must never be fed to the learner as a zero.

        `ref_price` is the price a PASSIVE order would have rested at (the touch). Without it a
        POST decision cannot be scored later: knowing the mid at arrival says nothing about whether
        the market subsequently traded through our resting level."""
        ts = ts or int(time.time())
        cur = self.conn.execute(
            "INSERT INTO exec_log(ts,mode,symbol,side,context,arm,shadow,arrival_mid,fill_price,"
            "notional,cost_bps,filled,order_id,ref_price) "
            "VALUES(?,?,?,?,?,?,?,?,NULL,?,NULL,0,?,?)",
            (ts, mode, symbol, side, context, arm, shadow, arrival_mid, notional, order_id,
             ref_price))
        self.conn.commit()
        return int(cur.lastrowid)

    def resolve_exec(self, row_id: int, *, fill_price: float, cost_bps: float,
                     filled: bool = True) -> None:
        self.conn.execute(
            "UPDATE exec_log SET fill_price=?, cost_bps=?, filled=? WHERE id=?",
            (fill_price, cost_bps, 1 if filled else 0, row_id))
        self.conn.commit()

    def unresolved_exec(self, mode: str, older_than_s: float = 0.0) -> list[dict]:
        """Decisions still awaiting an outcome. The reconciler settles these next cycle."""
        cutoff = int(time.time() - older_than_s)
        rows = self.conn.execute(
            "SELECT * FROM exec_log WHERE mode=? AND cost_bps IS NULL AND ts<=? ORDER BY id",
            (mode, cutoff)).fetchall()
        return [dict(r) for r in rows]

    def exec_stats(self, mode: str, days: float = 30.0) -> dict:
        """Realised cost by arm, plus the shadow policy's scorecard. This is the report that
        decides whether the learned policy has earned the right to take over from the default."""
        since = int(time.time() - days * 86_400)
        rows = [dict(r) for r in self.conn.execute(
            "SELECT arm, shadow, cost_bps, notional FROM exec_log "
            "WHERE mode=? AND ts>=? AND cost_bps IS NOT NULL", (mode, since)).fetchall()]
        out: dict = {"n": len(rows), "by_arm": {}, "agreement": 0.0, "weighted_bps": 0.0}
        if not rows:
            return out
        for r in rows:
            a = out["by_arm"].setdefault(r["arm"], {"n": 0, "sum": 0.0})
            a["n"] += 1; a["sum"] += r["cost_bps"]
        for a in out["by_arm"].values():
            a["mean_bps"] = round(a["sum"] / a["n"], 2)
        notional = sum(abs(r["notional"] or 0) for r in rows) or 1.0
        out["weighted_bps"] = round(
            sum((r["cost_bps"] or 0) * abs(r["notional"] or 0) for r in rows) / notional, 3)
        out["agreement"] = round(
            sum(1 for r in rows if r["arm"] == r["shadow"]) / len(rows) * 100, 1)
        return out

    # -- closed trades --------------------------------------------------------
    def record_closed_trades(self, mode: str, trades) -> None:
        rows = []
        for t in trades:
            ct = int(t.get("closed_ts") or time.time())
            dur = ct - float(t.get("opened_ts", ct) or ct)
            rows.append((ct, mode, t["symbol"], t["side"], float(t["entry"]), float(t["exit"]),
                         int(t["contracts"]), float(t["pnl"]), float(t.get("opened_ts", 0.0)), dur))
        if rows:
            self.conn.executemany(
                "INSERT INTO trades(ts,mode,symbol,side,entry,exit,contracts,pnl,opened_ts,duration_s) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def recent_closed_trades(self, mode: str, limit: int = 40) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM trades WHERE mode=? ORDER BY ts DESC LIMIT ?",
                                 (mode, limit)).fetchall()
        return [dict(r) for r in rows]

    def closed_trades_page(self, mode: str, offset: int = 0, limit: int = 25):
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE mode=? ORDER BY ts DESC LIMIT ? OFFSET ?",
            (mode, limit, offset)).fetchall()
        total = self.conn.execute("SELECT COUNT(*) c FROM trades WHERE mode=?", (mode,)).fetchone()["c"]
        return [dict(r) for r in rows], total

    def fills_page(self, mode: str, offset: int = 0, limit: int = 25):
        rows = self.conn.execute(
            "SELECT * FROM fills WHERE mode=? ORDER BY id DESC LIMIT ? OFFSET ?",
            (mode, limit, offset)).fetchall()
        total = self.conn.execute("SELECT COUNT(*) c FROM fills WHERE mode=?", (mode,)).fetchone()["c"]
        return [dict(r) for r in rows], total

    def trade_stats(self, mode: str) -> dict:
        r = self.conn.execute("""SELECT COUNT(*) n,
            SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END) losses,
            SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) gw,
            SUM(CASE WHEN pnl<0 THEN pnl ELSE 0 END) gl,
            MAX(pnl) mx, MIN(pnl) mn, AVG(duration_s) ad, SUM(pnl) tot
            FROM trades WHERE mode=?""", (mode,)).fetchone()
        n, wins, losses = r["n"] or 0, r["wins"] or 0, r["losses"] or 0
        gw, gl = r["gw"] or 0.0, r["gl"] or 0.0
        return {"n": n, "wins": wins, "losses": losses,
                "win_rate": (100.0 * wins / n) if n else 0.0,
                "avg_win": (gw / wins) if wins else 0.0, "avg_loss": (gl / losses) if losses else 0.0,
                "profit_factor": (gw / abs(gl)) if gl < 0 else 0.0,
                "max_win": r["mx"] or 0.0, "max_loss": r["mn"] or 0.0,
                "avg_duration_s": r["ad"] or 0.0, "total_pnl": r["tot"] or 0.0}

    # -- whales (latest snapshot only; older pruned to stay bounded) ----------
    def save_whales(self, consensus: list, wallets: list, ts: Optional[int] = None) -> None:
        ts = ts or int(time.time())
        if consensus:
            self.conn.executemany("INSERT INTO whale_consensus(ts,coin,bias) VALUES(?,?,?)",
                                  [(ts, c["coin"], c["bias"]) for c in consensus])
        if wallets:
            self.conn.executemany("INSERT INTO whale_wallets(ts,address,equity,gross,net) VALUES(?,?,?,?,?)",
                                  [(ts, w["address"], w["equity"], w["gross"], w["net"]) for w in wallets])
        self.conn.execute("DELETE FROM whale_consensus WHERE ts<?", (ts,))
        self.conn.execute("DELETE FROM whale_wallets WHERE ts<?", (ts,))
        self.set_meta("whales_ts", ts)
        self.conn.commit()

    def record_funding(self, mode: str, funding: dict, next_funding: dict, ts: Optional[int] = None) -> None:
        """Snapshot per-symbol funding (current + next) each cycle so funding factors become
        backtestable from real history later. Bounded retention (~180d of daily snapshots)."""
        if not funding:
            return
        ts = ts or int(time.time())
        self.conn.executemany(
            "INSERT INTO funding_history(ts,mode,symbol,current,next) VALUES(?,?,?,?,?)",
            [(ts, mode, s, float(funding.get(s, 0.0)), float(next_funding.get(s, 0.0))) for s in funding])
        self.conn.execute("DELETE FROM funding_history WHERE ts < ?", (ts - 180 * 86_400,))
        self.conn.commit()

    def recent_funding_avg(self, mode: str, cycles: int = 5) -> dict:
        """Per-symbol average of the last `cycles` recorded funding snapshots. Funding is persistent,
        so the trailing average is a cleaner carry signal than a single instantaneous rate."""
        ts_rows = self.conn.execute(
            "SELECT DISTINCT ts FROM funding_history WHERE mode=? ORDER BY ts DESC LIMIT ?",
            (mode, cycles)).fetchall()
        if not ts_rows:
            return {}
        oldest = ts_rows[-1][0]
        rows = self.conn.execute(
            "SELECT symbol, AVG(current) FROM funding_history WHERE mode=? AND ts>=? GROUP BY symbol",
            (mode, oldest)).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def latest_whales(self) -> dict:
        ts = self.get_meta("whales_ts")
        if not ts:
            return {}
        cons = [dict(r) for r in self.conn.execute(
            "SELECT coin,bias FROM whale_consensus WHERE ts=? ORDER BY ABS(bias) DESC", (ts,)).fetchall()]
        wals = [dict(r) for r in self.conn.execute(
            "SELECT address,equity,gross,net FROM whale_wallets WHERE ts=?", (ts,)).fetchall()]
        return {"n_wallets": len(wals), "consensus": cons, "wallets": wals, "ts": ts}

    def close(self) -> None:
        self.conn.close()
