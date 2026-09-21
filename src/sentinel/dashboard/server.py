"""Tiny local dashboard (stdlib HTTP server + Chart.js, no extra deps).

Reads the same SQLite the loop writes to and serves:
  GET /              -> the HTML dashboard
  GET /api/state     -> JSON snapshot (equity curve, PnL, positions, smart-money whales)
  GET /signals       -> live signal page (semantic HTML, LLM-readable, no JS required)
  GET /signals.md    -> same content as clean Markdown (ideal for AI agent consumption)
  GET /signals.json  -> JSON-LD structured data feed (schema.org DataFeed)
  GET /llms.txt      -> AI agent discovery (static + live hints)
  GET /robots.txt    -> crawl policy (exposes /signals* for indexing)
  GET /.well-known/ai-plugin.json -> ChatGPT/agent plugin manifest
Run via:  ./sentinel dashboard   (or  python run.py dashboard --port 8787)
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..state.store import Store

log = logging.getLogger("sentinel")
_CFG: Config = None  # set by serve()


def _loop_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "run.py loop"], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


_marks = {"ts": 0.0, "px": {}, "fr": {}}
_marks_lock = threading.Lock()
_fx_client = None


def _get_marks(cfg, symbols, ttl: float = 12.0) -> dict:
    """Current tagPrice + funding rate per symbol from public /fapi/v1/index, cached server-side (TTL)
    so many 5s browser polls share one fetch and we don't hammer the exchange."""
    global _fx_client
    if not symbols:
        return {}
    now = time.time()
    with _marks_lock:
        if now - _marks["ts"] < ttl and all(s in _marks["px"] for s in symbols):
            return dict(_marks["px"])
    from ..exchange.factory import make_futures
    if _fx_client is None:
        _fx_client = make_futures(cfg)   # venue-aware (Hyperliquid | KoinBay)
    from ..util.concurrency import pmap

    def _one(s):
        idx = _fx_client.index(s)
        p = float(idx.get("tagPrice") or 0)
        return (s, p, float(idx.get("currentFundRate") or 0)) if p > 0 else None

    px, fr = {}, {}
    for r in pmap(_one, symbols, workers=12):
        if r:
            px[r[0]], fr[r[0]] = r[1], r[2]
    with _marks_lock:
        _marks["px"].update(px)
        _marks["fr"].update(fr)
        _marks["ts"] = now
        return dict(_marks["px"])


def _funding_rates() -> dict:
    """Latest cached per-symbol funding rates (populated by _get_marks)."""
    with _marks_lock:
        return dict(_marks["fr"])


_regime_cache = {"ts": 0.0, "ma": None, "ma_period": None, "series": None, "last_close": None}
_regime_lock = threading.Lock()


def _get_regime(cfg, ttl: float = 300.0):
    """CHAMPION only: where BTC sits vs its regime-MA (the risk-on/off gate), with a daily sparkline.
    Explains *why* the book is in cash and how close it is to flipping.

    Two-speed refresh so the gauge feels live: the 100-day MA + sparkline come from daily candles and
    are cached ~5min (they barely move intraday), but the live BTC price is re-read every call (its mark
    is cached ~12s), so the marker + distance update on every poll."""
    if getattr(cfg, "strategy", "neutral") != "champion":
        return None
    now = time.time()
    ma_n = cfg.champion.regime_ma
    btc_sym = "BTC" if getattr(cfg.exchange, "venue", "koinbay") == "hyperliquid" else "E-BTC-USDT"
    with _regime_lock:
        fresh = _regime_cache["ma"] is not None and now - _regime_cache["ts"] < ttl
    if not fresh:
        global _fx_client
        try:
            from ..exchange.factory import make_futures
            from ..engine.marketdata import _parse_klines
            if _fx_client is None:
                _fx_client = make_futures(cfg)   # venue-aware (Hyperliquid | KoinBay)
            spark = 90
            _, closes, _ = _parse_klines(_fx_client.klines(btc_sym, "1day", ma_n + spark + 5))
            if len(closes) >= ma_n + 2:
                series = [{"c": round(closes[i], 2),
                           "ma": round(sum(closes[i - ma_n + 1:i + 1]) / ma_n, 2)}
                          for i in range(len(closes) - spark, len(closes)) if i >= ma_n - 1]
                with _regime_lock:
                    _regime_cache.update(ts=now, ma=sum(closes[-ma_n:]) / ma_n, ma_period=ma_n,
                                         series=series, last_close=closes[-1])
        except Exception:
            pass
    ma = _regime_cache["ma"]
    if ma is None:
        return None
    try:
        live = _get_marks(cfg, [btc_sym]).get(btc_sym)
    except Exception:
        live = None
    price = float(live) if live else float(_regime_cache["last_close"] or ma)
    return {
        "on": price > ma, "price": round(price, 2), "ma": round(ma, 2),
        "ma_period": _regime_cache["ma_period"],
        "dist_pct": round((price / ma - 1.0) * 100.0, 2),       # +above / -below the line
        "to_cross_pct": round((ma / price - 1.0) * 100.0, 2),   # % BTC must move to reach the line
        "series": _regime_cache["series"], "updated": int(now),
    }


_whales_cache = {"ts": 0.0, "data": None}
_whales_lock = threading.Lock()
_whale_tracker = None


def _get_whales(cfg, ttl: float = 60.0) -> dict:
    """Live blended whale bias for the dashboard, recomputed from current on-chain positions
    (cached ~60s). Falls back to the cached value on error."""
    global _whale_tracker
    if not cfg.whales.enabled:
        return {}
    now = time.time()
    with _whales_lock:
        if _whales_cache["data"] is not None and now - _whales_cache["ts"] < ttl:
            return _whales_cache["data"]
    import json
    import os
    addrs = list(cfg.whales.addresses)
    path = cfg.whales.source_file
    if path and os.path.exists(path):
        try:
            a = json.loads(open(path).read()).get("addresses")
            if a:
                addrs = a
        except Exception:
            pass
    if not addrs:
        return {}
    from ..signal.whale_tracker import WhaleTracker
    if _whale_tracker is None or set(_whale_tracker.addresses) != set(addrs):
        if _whale_tracker:
            _whale_tracker.close()
        _whale_tracker = WhaleTracker(addrs, cfg.whales.info_url, cfg.whales.weighting)
    try:
        snaps = _whale_tracker.fetch()
        bias = _whale_tracker.blended_bias(snaps)
    except Exception:
        return _whales_cache["data"] or {}
    consensus = [{"coin": c, "bias": round(v, 4)}
                 for c, v in sorted(bias.items(), key=lambda x: -abs(x[1])) if abs(v) >= 0.005][:14]
    wallets = [{"address": s.address, "equity": s.equity, "gross": s.gross, "net": s.net} for s in snaps]
    data = {"n_wallets": len(snaps), "consensus": consensus, "wallets": wallets, "ts": int(now)}
    with _whales_lock:
        _whales_cache["data"] = data
        _whales_cache["ts"] = now
    return data


def gather_state(cfg: Config) -> dict:
    store = Store(cfg.state.db_path, cfg.state.equity_csv)
    try:
        mode = cfg.mode
        series = store.equity_series(mode)
        acct = store.load_account(mode) or {}
        starting = float(acct.get("starting_capital") or cfg.capital_usdt or 10_000.0)
        realized = float(acct.get("realized_pnl") or 0.0)
        fees = float(acct.get("fees_paid") or 0.0)
        funding = float(acct.get("funding_pnl") or 0.0)
        whales = _get_whales(cfg) or store.latest_whales() or {}  # live, fallback to last cycle
        scores = store.scores_map(mode)
        lev = cfg.portfolio.per_name_leverage

        # ---- LIVE mark-to-market from the normalized positions table ----
        positions = store.load_positions(mode)
        marks = _get_marks(cfg, list(positions))
        book = []
        gross = net = live_unreal = 0.0
        frates = _funding_rates()
        for sym, p in positions.items():
            c = int(p["contracts"])
            if c == 0:
                continue
            entry = float(p["avg_price"])
            mult = float(p.get("multiplier", 1.0))
            mark = marks.get(sym) or entry
            notional = abs(c) * mark * mult
            upnl = (mark - entry) * c * mult
            gross += notional
            net += notional if c > 0 else -notional
            live_unreal += upnl
            # per-position funding for THIS interval: long pays positive funding, short earns it
            fr = frates.get(sym)
            signed_notional = notional if c > 0 else -notional
            fpay = round(-fr * signed_notional, 4) if fr is not None else None  # +earn / -pay this interval
            book.append({"symbol": sym, "side": "LONG" if c > 0 else "SHORT", "contracts": c,
                         "entry": round(entry, 6), "mark": round(mark, 6),
                         "notional": round(notional, 2), "upnl": round(upnl, 2),
                         "score": round(scores[sym], 3) if sym in scores else None, "leverage": lev,
                         "entry_notional": round(abs(c) * entry * mult, 2),
                         "base_size": round(abs(c) * mult, 8),
                         "opened_ts": p.get("opened_ts") or None,
                         "funding_rate": round(fr * 100, 4) if fr is not None else None, "funding_pay": fpay})
        book.sort(key=lambda x: -x["notional"])
        if book:
            eq = starting + realized + funding + live_unreal
        else:
            last_eq = store.last_equity(mode)
            eq = float(last_eq) if last_eq is not None else starting
        if cfg.strategy == "funding":
            # Delta-neutral funding harvest: every short perp is hedged by a simulated SPOT LONG that is NOT in the
            # positions table, so marking only the short legs invents a phantom open PnL (the hedge would offset it).
            # The engine also stores realized_pnl GROSS of fees. So ignore the naked-leg MtM and use the engine's own
            # booked equity: cap + realized + funding − fees. Open PnL is ~0 by construction; PnL is the booked carry.
            net = 0.0
            live_unreal = 0.0
            eq = starting + realized + funding - fees

        peak = max(float(store.peak_equity(mode, eq)), eq)
        pnl = eq - starting
        dd = max(0.0, (peak - eq) / peak * 100.0) if peak else 0.0
        _treasury = store.load_treasury(mode)   # money-manager vault (safe, swept profits)
        # transient "now" point so the chart ticks live (not persisted to the DB)
        live_series = list(series) + [{"ts": int(time.time()), "equity": round(eq, 4),
                                       "gross": round(gross, 2), "net": round(net, 2),
                                       "drawdown_pct": round(dd, 4)}]
        return {
            "mode": cfg.mode,
            "strategy": cfg.strategy,
            "regime": _get_regime(cfg),   # champion: BTC vs its regime-MA gate (None for neutral)
            "running": _loop_running(),
            "rebalance_minutes": cfg.schedule.rebalance_minutes,
            "rebalance_hour_utc": cfg.schedule.rebalance_hour_utc,
            "next_rebalance_ts": store.get_meta("next_rebalance"),
            "paused_until": store.paused_until(mode) or None,
            "starting_capital": starting,
            "equity": eq,
            "pnl": pnl,
            "pnl_pct": (pnl / starting * 100.0) if starting else 0.0,
            "realized": realized,
            # funding book is delta-neutral -> open MtM is ~0 (the short-leg drift is hedged by the simulated spot long)
            "unrealized": 0.0 if cfg.strategy == "funding" else eq - starting - realized - funding,
            "funding": funding,
            "fees": fees,
            "peak": peak,
            "drawdown_pct": dd,
            "gross": gross,
            "net": net,
            "leverage": (gross / eq if eq else 0.0),
            "max_gross_leverage": cfg.risk.max_gross_leverage,
            "long_notional": (gross + net) / 2.0,
            "short_notional": (gross - net) / 2.0,
            "n_positions": len(book),
            "book_beta": 0.0 if cfg.strategy == "funding" else store.get_meta("book_beta"),
            "dispersion": store.get_meta("dispersion"),
            "book": book,
            "whales": whales,
            "whales_enabled": bool(cfg.whales.enabled),   # carry/champion don't use the whale overlay -> hide the tab
            "trade_stats": store.trade_stats(mode),   # tables are paginated via /api/trades & /api/fills
            "equity_series": live_series,
            # grid only: equity before this ts is BACKTEST-seeded (dashed on the chart), after is live paper
            "backtest_until_ts": int(store.get_meta("grid_backtest_until_ts", 0) or 0),
            "cycles": len(series),
            # money manager (Treasury): profit swept to the safe vault (auto-sweep, usually off)
            "treasury_enabled": bool(cfg.treasury.enabled),
            "sweep_frac": cfg.treasury.sweep_frac,
            "vault": _treasury.get("vault", 0.0),
            "total_swept": _treasury.get("total_swept", 0.0),
            "trading_equity": max(0.0, eq - _treasury.get("vault", 0.0)),
            # free / withdrawable: equity minus the margin the open positions are using
            # (positions open at per_name_leverage, so margin_used = gross / that leverage)
            "margin_used": (gross / cfg.portfolio.per_name_leverage) if cfg.portfolio.per_name_leverage else 0.0,
            "free_capital": max(0.0, eq - ((gross / cfg.portfolio.per_name_leverage)
                                           if cfg.portfolio.per_name_leverage else 0.0)
                                        - _treasury.get("vault", 0.0)),
            # SAFE to withdraw: leaves a buffer so post-withdrawal gross leverage stays <= safe cap
            # (so pulling cash never spikes leverage into liquidation risk before the next rebalance)
            "safe_withdraw": max(0.0, eq - (gross / (cfg.risk.safe_withdraw_leverage or 2.0))
                                        - _treasury.get("vault", 0.0)),
            "safe_withdraw_leverage": cfg.risk.safe_withdraw_leverage,
        }
    finally:
        store.close()


def _page_payload(path: str) -> bytes:
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(path)
    q = parse_qs(parsed.query)

    def _int(name, default):
        try:
            return int(q.get(name, [default])[0])
        except (ValueError, TypeError):
            return default
    offset = max(0, _int("offset", 0))
    limit = min(100, max(1, _int("limit", 25)))
    store = Store(_CFG.state.db_path, _CFG.state.equity_csv)
    try:
        if parsed.path.endswith("/fills"):
            rows, total = store.fills_page(_CFG.mode, offset, limit)
        else:
            rows, total = store.closed_trades_page(_CFG.mode, offset, limit)
        return json.dumps({"rows": rows, "total": total, "offset": offset, "limit": limit}).encode()
    except Exception as e:  # pragma: no cover
        return json.dumps({"rows": [], "total": 0, "error": str(e)}).encode()
    finally:
        store.close()


REPO_URL = "https://github.com/adensvaz/Sentinal_Hyperliquid"

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#ff8a5c"/><stop offset="1" stop-color="#9a7bff"/></linearGradient>'
    '<filter id="s"><feGaussianBlur stdDeviation="1.4"/></filter></defs>'
    '<rect width="64" height="64" rx="15" fill="url(#g)"/>'
    '<path d="M37 7 L17 37 H29 L27 57 L47 25 H34 Z" fill="#fff" filter="url(#s)" opacity=".5"/>'
    '<path d="M37 7 L17 37 H29 L27 57 L47 25 H34 Z" fill="#fff"/></svg>'
)

ROBOTS_TXT = (
    "# Sentinel Edge — live algorithmic crypto trading signals (Hyperliquid futures)\n"
    "# Four strategies: delta-neutral funding harvest, regime-gated momentum, funding-carry, cross-sectional trend (CTA)\n"
    "\n"
    "User-agent: *\n"
    "Allow: /\n"
    "Allow: /signals\n"
    "Allow: /signals.md\n"
    "Allow: /signals.json\n"
    "Allow: /llms.txt\n"
    "Allow: /.well-known/\n"
    "Disallow: /api/\n"
    "\n"
    "# Live signal feed (no JS, LLM-readable):\n"
    "# /signals       -> semantic HTML with current positions + regime\n"
    "# /signals.md    -> clean Markdown (best for AI agent consumption)\n"
    "# /signals.json  -> JSON-LD structured data (schema.org DataFeed)\n"
    "# /llms.txt      -> AI agent guide\n"
    f"# Source: {REPO_URL}\n"
)

def _build_llms_txt(cfg: Config) -> str:
    """Dynamic llms.txt — static guide + live regime/signal status so AI agents get fresh context."""
    try:
        state = gather_state(cfg)
        strat = state.get("strategy", "neutral")
        pnl_pct = state.get("pnl_pct", 0.0)
        eq = state.get("equity", 0.0)
        dd = state.get("drawdown_pct", 0.0)
        n_pos = state.get("n_positions", 0)
        regime = state.get("regime")
        r = state.get("regime") or {}
        regime_line = ""
        if regime:
            on = r.get("on", False)
            ma = r.get("ma_period", 100)
            dist = r.get("dist_pct", 0.0)
            regime_line = (
                f"\n## Live market regime (BTC vs {ma}-day MA)\n"
                f"- Status: {'RISK-ON — deployed, holding positions' if on else 'RISK-OFF — 100% cash, waiting for uptrend'}\n"
                f"- BTC is {abs(dist):.1f}% {'above' if on else 'below'} its {ma}-day line\n"
                f"- This is the champion strategy's entry gate\n"
            )
        live_line = (
            f"\n## Live strategy status (updated every 5 seconds)\n"
            f"- Strategy: {strat}\n"
            f"- Equity: ${eq:,.2f}\n"
            f"- All-time PnL: {pnl_pct:+.2f}%\n"
            f"- Drawdown from peak: {dd:.2f}%\n"
            f"- Open positions: {n_pos}\n"
            f"- Mode: paper (simulated — no real funds)\n"
        )
    except Exception:
        live_line = ""
        regime_line = ""
    return f"""# Sentinel Edge — Live Algorithmic Crypto Trading Signals

> Autonomous algorithmic crypto-futures trading running on Hyperliquid perpetuals.
> Four uncorrelated strategies updated daily — each with live positions, signals, and performance.
> Built as a copy-trading signal provider. Paper-traded (no real funds at risk).

## How to access live signals
- **/signals.md** — current positions, regime, and signals in clean Markdown (recommended for AI agents)
- **/signals.json** — JSON-LD structured data feed (schema.org DataFeed + FinancialProduct)
- **/signals** — semantic HTML signal page (no JavaScript required)
- **/api/state** — raw JSON snapshot with full position detail
{live_line}{regime_line}
## The three live strategies
1. **Momentum + Regime / Champion** (port 8788) — directional, long-only top-5 momentum.
   Gated by a BTC 100-day MA regime brake: fully invested when BTC is in an uptrend,
   100% cash in bear markets.

2. **Funding Carry** (port 8789) — market-neutral, harvests perpetual funding premium.
   Shorts the highest-funding coins (collects what crowded longs pay), longs the cheapest,
   momentum-tilted to avoid shorting a ripping coin.
   Live to date: 203 closed trades, profit factor 0.975 — the signal has traded roughly flat, and
   fees rather than selection are what put it behind. Gross was halved to 1.0x in response.

3. **Trend** (port 8790) — dollar-neutral cross-sectional trend-following (CTA).
   Longs the coins highest in their own 45-day price range, shorts those lowest in theirs.
   Range position replaced price-vs-moving-average because it survives realistic fees: the old
   scorer stopped earning at 10bps, this one still made +9.5% there on a survivorship-stripped
   backtest, and it turns over less.
   Honest caveat: still one-regime. Positive in 5 of 9 backtested quarters, and capable of losing
   heavily in a bad one.

**Retired — Funding Harvest** (was port 8787). Delta-neutral cash-and-carry needs a spot leg, and
Hyperliquid spot has only ~8 pairs trading over $1M/day, so a 15-name basket cannot be hedged.
Hedging a perp with a perp means paying funding to collect it: measured at 16.0%/yr collected
against 15.0%/yr paid, a 0.9%/yr spread. Shut down rather than left running as a paper curiosity.

## What "signal" means here
- A signal is a ranked coin with a target side (LONG/SHORT) and score.
- Signals are generated daily from live Hyperliquid price, funding, and volume data.
- No signal = the strategy is in cash (regime brake active or no qualifying names).
- All signals are paper-traded; treat them as informational, not financial advice.

## Signal update cadence
- Signals update every 24 hours at 14:00 UTC.
- The live endpoints (/signals.md, /api/state) reflect the current open book in real-time.
- Funding rates update every 8 hours (funding carry strategy).

## Source code
{REPO_URL}

## Disclaimer
This is software, not financial advice. Paper-traded simulated results — real money performance will
differ. Crypto futures trading involves substantial risk of loss.
"""


def _signals_md(cfg: Config) -> str:
    """Live signal snapshot as clean Markdown — what an AI agent browsing for signals reads."""
    import datetime as _dt
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        s = gather_state(cfg)
    except Exception:
        return f"# Sentinel Edge — Signals\n\nUnavailable at {now}.\n"
    strat = s.get("strategy", "neutral")
    strat_name = {"funding": "Funding Harvest (delta-neutral)", "champion": "Momentum + Regime (Champion)",
                  "carry": "Funding Carry", "trend": "Trend (CTA)", "consensus": "Consensus"}.get(strat, strat)
    pnl = s.get("pnl", 0.0); pnl_pct = s.get("pnl_pct", 0.0); eq = s.get("equity", 0.0)
    dd = s.get("drawdown_pct", 0.0); book = s.get("book", [])
    r = s.get("regime") or {}
    lines = [
        f"# Sentinel Edge — Live Trading Signals",
        f"**Strategy:** {strat_name}  |  **Updated:** {now}  |  **Mode:** paper (simulated)",
        f"**Equity:** ${eq:,.2f}  |  **All-time PnL:** {pnl_pct:+.2f}%  |  **Drawdown:** {dd:.2f}%",
        "",
    ]
    if r:
        on = r.get("on", False); ma = r.get("ma_period", 100); dist = r.get("dist_pct", 0.0)
        price = r.get("price", 0.0); ma_val = r.get("ma", 0.0)
        lines += [
            f"## Market Regime (BTC vs {ma}-day MA)",
            f"- **Status:** {'✅ RISK-ON — deployed' if on else '🔴 RISK-OFF — 100% cash'}",
            f"- BTC price: ${price:,.0f}  |  {ma}-day MA: ${ma_val:,.0f}",
            f"- BTC is **{abs(dist):.1f}%** {'above' if on else 'below'} its {ma}-day line",
            "",
        ]
    if book:
        longs = [p for p in book if p.get("side") == "LONG"]
        shorts = [p for p in book if p.get("side") == "SHORT"]
        lines.append(f"## Open Positions ({len(book)} total)")
        if longs:
            lines.append("\n### Long positions (bullish)")
            lines.append("| Asset | Mark price | Unrealised PnL | Score |")
            lines.append("|---|---|---|---|")
            for p in longs:
                sym = p["symbol"].replace("E-", "").replace("-USDT", "")
                sc = f"{p['score']:+.2f}" if p.get("score") is not None else "—"
                lines.append(f"| {sym} | ${p['mark']:,.4g} | ${p['upnl']:+,.2f} | {sc} |")
        if shorts:
            lines.append("\n### Short positions (bearish)")
            lines.append("| Asset | Mark price | Unrealised PnL | Score |")
            lines.append("|---|---|---|---|")
            for p in shorts:
                sym = p["symbol"].replace("E-", "").replace("-USDT", "")
                sc = f"{p['score']:+.2f}" if p.get("score") is not None else "—"
                lines.append(f"| {sym} | ${p['mark']:,.4g} | ${p['upnl']:+,.2f} | {sc} |")
    else:
        lines += [
            "## Open Positions",
            "_No open positions — strategy is currently in cash._",
            "",
            f"The {'regime brake is off (BTC below its ' + str(r.get('ma_period',100)) + '-day MA)' if r and not r.get('on') else 'book is flat for today'}.",
        ]
    lines += [
        "",
        "---",
        f"*These are paper-traded signals — simulated, not real money. Not financial advice.*",
        f"*Full data: /api/state (JSON) | Dashboard: / | Source: {REPO_URL}*",
    ]
    return "\n".join(lines) + "\n"


def _signals_json(cfg: Config) -> str:
    """JSON-LD structured data feed — schema.org DataFeed + FinancialProduct for LLM/search discovery."""
    import datetime as _dt
    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        s = gather_state(cfg)
        book = s.get("book", [])
        items = []
        for p in book:
            sym = p["symbol"].replace("E-", "").replace("-USDT", "")
            items.append({
                "@type": "TradeAction",
                "instrument": {"@type": "FinancialProduct", "name": sym + " perpetual futures",
                               "currency": "USDT", "category": "crypto-futures"},
                "price": p.get("mark"), "priceCurrency": "USDT",
                "actionStatus": "ActiveActionStatus",
                "description": f"{p.get('side','').title()} {sym} at ${p.get('mark'):,.4g}, "
                               f"unrealised PnL ${p.get('upnl',0):+,.2f}",
            })
        regime = s.get("regime") or {}
        strat = s.get("strategy", "neutral")
        feed = {
            "@context": "https://schema.org",
            "@type": "DataFeed",
            "name": "Sentinel Edge Live Trading Signals",
            "description": "Real-time algorithmic crypto trading signals from four uncorrelated strategies "
                           "(delta-neutral funding harvest, momentum+regime, funding-carry, cross-sectional trend) on Hyperliquid futures.",
            "url": REPO_URL,
            "dateModified": now_iso,
            "keywords": ["crypto signals", "trading signals", "algorithmic trading", "Hyperliquid",
                         "momentum", "funding carry", "market neutral", "perpetual futures"],
            "provider": {"@type": "SoftwareApplication", "name": "Sentinel Edge",
                         "applicationCategory": "FinanceApplication", "url": REPO_URL},
            "dataFeedElement": items,
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "strategy", "value": strat},
                {"@type": "PropertyValue", "name": "equity_usd", "value": s.get("equity")},
                {"@type": "PropertyValue", "name": "pnl_pct", "value": s.get("pnl_pct")},
                {"@type": "PropertyValue", "name": "drawdown_pct", "value": s.get("drawdown_pct")},
                {"@type": "PropertyValue", "name": "open_positions", "value": s.get("n_positions")},
                {"@type": "PropertyValue", "name": "market_regime",
                 "value": ("risk-on" if regime.get("on") else "risk-off") if regime else "n/a"},
                {"@type": "PropertyValue", "name": "mode", "value": "paper"},
            ],
        }
    except Exception as e:
        feed = {"@context": "https://schema.org", "@type": "DataFeed",
                "name": "Sentinel Edge Signals", "error": str(e)}
    return json.dumps(feed, indent=2)


def _signals_html(cfg: Config) -> str:
    """Semantic HTML signal page — no JS, fully crawlable, human + LLM readable."""
    md = _signals_md(cfg)
    # convert the markdown into clean semantic HTML (no external deps — manual conversion)
    import re, html as _html
    def row_to_html(line):
        cells = [c.strip() for c in line.strip("|").split("|")]
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    out = []
    in_table = False; in_code = False
    for line in md.splitlines():
        if line.startswith("# "):
            out.append(f"<h1>{_html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{_html.escape(line[4:])}</h3>")
        elif line.startswith("|---|"):
            if not in_table:
                out.append("<table><tbody>"); in_table = True
        elif line.startswith("|") and "|" in line[1:]:
            if not in_table:
                out.append("<table><thead>"); in_table = True
            out.append(row_to_html(_html.escape(line)))
        else:
            if in_table:
                out.append("</tbody></table>"); in_table = False
            if line.startswith("- **") or line.startswith("- "):
                txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _html.escape(line[2:]))
                out.append(f"<li>{txt}</li>")
            elif line.startswith("*") and line.endswith("*"):
                out.append(f"<p><em>{_html.escape(line.strip('*'))}</em></p>")
            elif line.startswith("---"):
                out.append("<hr>")
            elif line.strip():
                txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _html.escape(line))
                out.append(f"<p>{txt}</p>")
            else:
                out.append("")
    if in_table:
        out.append("</tbody></table>")
    body = "\n".join(out)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel Edge — Live Crypto Trading Signals</title>
<meta name="description" content="Live algorithmic crypto trading signals from Sentinel Edge: "
    "delta-neutral funding harvest, regime-gated momentum, funding-carry and cross-sectional trend on Hyperliquid futures. "
    "Updated every 24 hours. Machine-readable at /signals.md and /signals.json.">
<meta name="keywords" content="crypto trading signals, live trading signals, algorithmic trading, "
    "momentum signals, funding carry, market neutral, Hyperliquid signals, crypto futures signals, "
    "BTC regime, copy trading signals, perpetual futures, crypto quant">
<meta property="og:title" content="Sentinel Edge — Live Crypto Trading Signals">
<meta property="og:description" content="Live algorithmic signals: long/short positions across "
    "four uncorrelated strategies on Hyperliquid futures.">
<link rel="canonical" href="/signals">
<link rel="alternate" type="text/markdown" href="/signals.md" title="Signals as Markdown">
<link rel="alternate" type="application/ld+json" href="/signals.json" title="Signals as JSON-LD">
<script type="application/ld+json">
{{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "Sentinel Edge Live Trading Signals",
  "description": "Real-time crypto trading signals from four algorithmic strategies on Hyperliquid futures.",
  "url": "/signals",
  "isPartOf": {{"@type": "WebSite", "name": "Sentinel Edge", "url": "{REPO_URL}"}},
  "mainContentOfPage": {{"@type": "DataFeed", "url": "/signals.json"}}
}}
</script>
<style>
  body{{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:24px;
    background:#07090f;color:#e2e8f0;line-height:1.65}}
  h1{{font-size:1.9em;font-weight:800;margin-bottom:.2em;color:#fff}}
  h2{{font-size:1.25em;font-weight:700;margin-top:2em;color:#a5b4fc}}
  h3{{font-size:1em;font-weight:700;margin-top:1.5em;color:#94a3b8}}
  p{{margin:.6em 0}} strong{{color:#fff}} em{{color:#94a3b8;font-size:.9em}}
  li{{margin:.3em 0}} ul,ol{{padding-left:1.4em}} hr{{border:0;border-top:1px solid #1e2d3d;margin:2em 0}}
  table{{border-collapse:collapse;width:100%;margin:1em 0}}
  td,th{{padding:7px 12px;border:1px solid #1e2d3d;text-align:left;font-size:.92em}}
  tr:nth-child(even){{background:#0d1117}}
  a{{color:#ff8a5c}} a:hover{{color:#9ab0ff}}
  .nav{{display:flex;gap:16px;margin-bottom:2em;font-size:.88em}}
  .nav a{{color:#ff8a5c;text-decoration:none;padding:.3em .7em;border:1px solid #1e2d3d;border-radius:6px}}
  .nav a:hover{{background:#1e2d3d}}
</style>
</head>
<body>
<div class="nav">
  <a href="/">← Dashboard</a>
  <a href="/signals.md">Markdown</a>
  <a href="/signals.json">JSON-LD</a>
  <a href="/llms.txt">llms.txt</a>
</div>
{body}
</body>
</html>"""


AI_PLUGIN_JSON = json.dumps({
    "schema_version": "v1",
    "name_for_model": "sentinel_edge_signals",
    "name_for_human": "Sentinel Edge",
    "description_for_model": (
        "Fetch live algorithmic crypto trading signals from Sentinel Edge, an autonomous "
        "strategy running on Hyperliquid futures. Returns current positions, market regime "
        "(BTC trend status), unrealised PnL, and signal scores for four uncorrelated strategies: "
        "a delta-neutral funding harvest (port 8787), regime-gated momentum Champion (port 8788), "
        "funding-carry (port 8789), and cross-sectional trend / CTA (port 8790). "
        "Use /signals.md for a readable summary or /api/state for raw JSON."
    ),
    "description_for_human": "Get live crypto trading signals and positions from Sentinel Edge algorithmic strategies.",
    "auth": {"type": "none"},
    "api": {"type": "openapi", "url": "/.well-known/openapi.yaml"},
    "logo_url": "/favicon.svg",
    "contact_email": "noreply@sentineledge.app",
    "legal_info_url": REPO_URL,
}, indent=2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default stderr logging
        return

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                payload = json.dumps(gather_state(_CFG)).encode()
            except Exception as e:  # pragma: no cover
                payload = json.dumps({"error": str(e)}).encode()
            self._send(200, payload, "application/json")
        elif self.path.startswith("/api/trades") or self.path.startswith("/api/fills"):
            self._send(200, _page_payload(self.path), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/signals":
            self._send(200, _signals_html(_CFG).encode(), "text/html; charset=utf-8")
        elif self.path == "/signals.md":
            self._send(200, _signals_md(_CFG).encode(), "text/markdown; charset=utf-8")
        elif self.path == "/signals.json":
            self._send(200, _signals_json(_CFG).encode(), "application/ld+json; charset=utf-8")
        elif self.path == "/llms.txt":
            self._send(200, _build_llms_txt(_CFG).encode(), "text/markdown; charset=utf-8")
        elif self.path in ("/favicon.svg", "/favicon.ico"):
            self._send(200, FAVICON_SVG.encode(), "image/svg+xml")
        elif self.path == "/robots.txt":
            self._send(200, ROBOTS_TXT.encode(), "text/plain; charset=utf-8")
        elif self.path == "/.well-known/ai-plugin.json":
            self._send(200, AI_PLUGIN_JSON.encode(), "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")


def serve(cfg: Config, port: int = 8787, open_browser: bool = True) -> None:
    global _CFG
    _CFG = cfg
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    url = f"http://127.0.0.1:{port}"
    log.info("dashboard live at %s  (Ctrl-C to stop)", url)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("dashboard stopped")
    finally:
        httpd.server_close()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sentinel Edge — Algorithmic Crypto Trading Bot (3 strategies)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<meta name="description" content="Sentinel Edge — four uncorrelated algorithmic crypto-futures strategies on Hyperliquid sharing one engine: a delta-neutral funding harvest, regime-gated momentum (champion), funding-carry, and a cross-sectional trend (CTA) book. Live paper-trading dashboards with real PnL, equity curves and backtests."/>
<meta name="theme-color" content="#ff8a5c"/>
<meta property="og:title" content="Sentinel Edge — Algorithmic Crypto Trading Bot"/>
<meta property="og:description" content="Four uncorrelated crypto strategies on Hyperliquid — a delta-neutral funding harvest, momentum+regime, funding-carry, and cross-sectional trend (CTA) — with live PnL dashboards."/>
<meta property="og:type" content="website"/>
<meta name="twitter:card" content="summary"/>
<meta name="keywords" content="crypto trading signals, live trading signals, algorithmic trading bot,
  momentum signals, funding carry signals, market neutral crypto, Hyperliquid signals, BTC regime,
  copy trading signals, perpetual futures signals, crypto quant, automated trading"/>
<link rel="alternate" type="text/markdown" href="/signals.md" title="Live signals (Markdown)"/>
<link rel="alternate" type="application/ld+json" href="/signals.json" title="Live signals (JSON-LD)"/>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Sentinel Edge",
  "description": "Four uncorrelated algorithmic crypto-futures trading strategies on Hyperliquid: a delta-neutral funding harvest, regime-gated momentum, funding-carry, and cross-sectional trend (CTA). Live signals updated daily.",
  "applicationCategory": "FinanceApplication",
  "operatingSystem": "Web",
  "url": "https://github.com/adensvaz/Sentinal_Hyperliquid",
  "featureList": [
    "Live crypto trading signals",
    "Market-neutral momentum strategy",
    "BTC regime-gated momentum (Champion)",
    "Funding carry market-neutral strategy",
    "Real-time PnL tracking",
    "Copy-trading signal provider"
  ],
  "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
  "mainEntityOfPage": {
    "@type": "DataFeed",
    "name": "Live Trading Signals",
    "url": "/signals.json",
    "description": "Machine-readable live signals feed"
  }
}
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    /* Ghibli daylight — soft painterly sky, lush green hills, warm sun, washi-cream cards */
    --bg:#bfe3f2; --bg2:#d8ecd0;
    --surface:rgba(255,253,249,.93); --surface2:rgba(255,255,253,.95); --line:rgba(108,90,60,.20);
    --txt:#2d2922; --mut:#665d4c; --dim:#938974;
    --grn:#179a55; --grn-d:#dcefe0; --red:#e1512c; --red-d:#f8ded7;   /* forest green = up · persimmon = down */
    --accent:#f0652a; --accent2:#3f80ba; --gold:#e0a52e;             /* persimmon · indigo · yamabuki gold */
    --teal:#36a39c; --sakura:#e389ab;
    --r:18px;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{background:
      radial-gradient(420px 300px at 82% 9%, rgba(255,246,206,.96), rgba(255,226,150,.40) 42%, transparent 70%),  /* warm sun */
      radial-gradient(340px 130px at 20% 16%, rgba(255,255,255,.92), transparent 72%),    /* cloud */
      radial-gradient(300px 120px at 36% 26%, rgba(255,255,255,.78), transparent 72%),    /* cloud */
      radial-gradient(420px 150px at 62% 12%, rgba(255,255,255,.85), transparent 72%),    /* cloud */
      radial-gradient(260px 110px at 88% 30%, rgba(255,255,255,.66), transparent 72%),    /* cloud */
      linear-gradient(180deg, #79bdec 0%, #9fd3ea 28%, #c4e6d8 55%, #e2eecb 78%, #f4e7c6 100%); /* blue sky -> aqua -> green -> warm cream */
    background-attachment:fixed;
    color:var(--txt); font:14px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; min-height:100vh}
  /* frosted-glass surfaces over the dawn gradient */
  .card,.panel,.regime,.modal-box,.cd,.seg,.pill,.tblwrap{
    backdrop-filter:blur(16px) saturate(1.3); -webkit-backdrop-filter:blur(16px) saturate(1.3)}
  /* Japanese scene: Mt Fuji + skyline + rising sun (fixed backdrop) + drifting sakura */
  #skyline{position:fixed;left:0;right:0;bottom:0;height:min(42vh,380px);width:100vw;z-index:-1;pointer-events:none;opacity:.92}
  .sakura{position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden}
  .sakura i{position:absolute;top:-5%;width:12px;height:12px;
    background:radial-gradient(circle at 32% 28%,#ffd9e6,#ff9ec4);
    border-radius:92% 8% 92% 8%/8% 92% 8% 92%;opacity:.7;animation:fall linear infinite;
    filter:drop-shadow(0 0 4px rgba(255,158,196,.35))}
  @keyframes fall{0%{transform:translateY(-30px) translateX(0) rotate(0)}8%{opacity:.7}
    92%{opacity:.55}100%{transform:translateY(104vh) translateX(70px) rotate(420deg);opacity:0}}
  @media(prefers-reduced-motion:reduce){.sakura{display:none}}
  .num{font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1}
  .grn{color:var(--grn)} .red{color:var(--red)} .mut{color:var(--mut)}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}

  header{display:flex;align-items:center;gap:14px;padding:11px 22px;
    position:sticky;top:12px;z-index:9;max-width:1340px;width:calc(100% - 28px);margin:14px auto 2px;
    border:1px solid var(--line);border-radius:22px;
    background:linear-gradient(180deg,rgba(255,254,250,.97),rgba(255,253,247,.85));
    box-shadow:0 10px 34px rgba(80,60,30,.14);
    backdrop-filter:blur(16px) saturate(1.3);-webkit-backdrop-filter:blur(16px) saturate(1.3)}
  .brand{display:flex;align-items:center;gap:11px}
  .brand .mark{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;font-size:15px;flex:none;
    background:linear-gradient(135deg,var(--accent),var(--gold));box-shadow:0 4px 16px rgba(240,101,42,.4)}
  .brand-tx{display:flex;flex-direction:column;line-height:1}
  .brand-name{font-weight:850;font-size:16.5px;letter-spacing:.1px;color:var(--txt)}
  .brand-sub{font-size:8.5px;font-weight:800;letter-spacing:2.4px;color:var(--accent);text-transform:uppercase;margin-top:3px}
  .pill{font-size:11px;font-weight:750;padding:4px 11px;border-radius:999px;letter-spacing:.6px;text-transform:uppercase}
  .pill.paper{background:rgba(240,101,42,.16);color:#bf441a;border:1px solid rgba(240,101,42,.38)}
  .pill.live{background:var(--red-d);color:var(--red);border:1px solid rgba(255,122,138,.45)}
  .status{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--mut);white-space:nowrap;
    background:var(--surface);border:1px solid var(--line);padding:5px 11px;border-radius:999px}
  .dot{width:8px;height:8px;border-radius:50%}
  .dot.on{background:var(--grn);box-shadow:0 0 0 0 rgba(75,224,176,.6);animation:pulse 1.8s infinite}
  .dot.off{background:#4b5563}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(75,224,176,.5)}70%{box-shadow:0 0 0 7px rgba(75,224,176,0)}100%{box-shadow:0 0 0 0 rgba(75,224,176,0)}}
  .meta{font-size:12px;color:var(--dim)}
  #mode{margin-left:auto}
  .cd{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--mut);
    background:linear-gradient(135deg,rgba(240,101,42,.1),rgba(63,127,190,.06));
    border:1px solid rgba(240,101,42,.28);padding:5px 12px;border-radius:999px;white-space:nowrap}
  .cd b{color:var(--txt);font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:.4px;min-width:62px;text-align:right}
  .cd-ic{color:var(--accent);font-size:13px;animation:spin 7s linear infinite}
  .cd.soon{border-color:rgba(245,196,81,.5);background:rgba(245,196,81,.08)} .cd.soon b{color:var(--gold)}
  .cd.paused{border-color:rgba(255,122,138,.4);background:rgba(255,122,138,.07)} .cd.paused b{color:var(--red)}
  @keyframes spin{to{transform:rotate(360deg)}}
  .meta{margin-left:14px}

  main{padding:22px 26px;max-width:1280px;margin:0 auto}
  /* champion market-regime module — cinematic */
  #regimeWrap{margin-bottom:18px}
  .regime{--rgc:var(--gold);--prox:0;position:relative;border-radius:var(--r);padding:20px 22px;overflow:hidden;
    border:1px solid var(--line);background:linear-gradient(180deg,var(--surface2),var(--surface));isolation:isolate}
  .regime.on{--rgc:var(--grn)}
  .regime::before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:linear-gradient(180deg,var(--rgc),transparent)}
  /* breathing ambient glow, brighter as BTC nears the line (--prox) */
  .regime::after{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;
    background:radial-gradient(120% 90% at 8% 0%,color-mix(in srgb,var(--rgc) 14%,transparent),transparent 60%);
    opacity:calc(.35 + .55*var(--prox));animation:rgBreath 6s ease-in-out infinite}
  @keyframes rgBreath{0%,100%{opacity:calc(.30 + .45*var(--prox))}50%{opacity:calc(.55 + .45*var(--prox))}}
  /* radar scan sweep across the card */
  .rg-scan{position:absolute;top:0;bottom:0;width:42%;left:-42%;z-index:-1;pointer-events:none;
    background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--rgc) 9%,transparent),transparent);
    animation:rgScan 7s linear infinite}
  @keyframes rgScan{0%{left:-42%}100%{left:100%}}
  .rg-top{display:flex;align-items:center;gap:18px}
  .rg-head{flex:1;min-width:0}
  .rg-state{display:flex;align-items:center;gap:9px;font-size:21px;font-weight:850;letter-spacing:.4px;color:var(--rgc)}
  .rg-dot{width:9px;height:9px;border-radius:50%;background:currentColor;animation:rgpulse 2s infinite}
  @keyframes rgpulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--rgc) 55%,transparent)}70%{box-shadow:0 0 0 8px transparent}100%{box-shadow:0 0 0 0 transparent}}
  .rg-sub{font-size:12.5px;color:var(--dim);margin-top:3px;text-transform:uppercase;letter-spacing:1.2px}
  .rg-spark-wrap{width:240px;height:46px;flex:none}
  .rg-spk{width:240px;height:46px;display:block;overflow:visible}
  @media(max-width:560px){.rg-spark-wrap{display:none}}
  .rg-spk-line{stroke-dasharray:1;stroke-dashoffset:1;animation:rgDraw 1.6s cubic-bezier(.4,0,.2,1) forwards}
  @keyframes rgDraw{to{stroke-dashoffset:0}}
  .rg-spk-tip{filter:drop-shadow(0 0 4px currentColor)}
  .rg-spk-ping{transform-box:fill-box;transform-origin:center;animation:rgPing 2.4s ease-out infinite;opacity:0}
  @keyframes rgPing{0%{transform:scale(1);opacity:.8}100%{transform:scale(4);opacity:0}}
  .rg-gauge{margin:16px 0 10px}
  .rg-track{position:relative;height:10px;border-radius:6px;background:rgba(60,45,30,.09);overflow:visible}
  .rg-fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px;width:0;
    background:linear-gradient(90deg,color-mix(in srgb,var(--rgc) 22%,transparent),var(--rgc));opacity:.5;
    transition:width 1.3s cubic-bezier(.22,.9,.3,1)}
  .rg-trig{position:absolute;left:50%;top:-5px;bottom:-5px;width:2px;background:rgba(80,60,35,.45);transform:translateX(-1px);
    box-shadow:0 0 6px rgba(80,60,35,.25);animation:rgTrig 3s ease-in-out infinite}
  @keyframes rgTrig{0%,100%{opacity:.45}50%{opacity:.9}}
  .rg-mk{position:absolute;top:50%;left:0;width:16px;height:16px;border-radius:50%;background:var(--rgc);
    transform:translate(-50%,-50%);border:2.5px solid var(--bg);
    box-shadow:0 0 0 0 var(--rgc),0 0 12px color-mix(in srgb,var(--rgc) 70%,transparent);
    transition:left 1.3s cubic-bezier(.22,.9,.3,1);animation:rgMk 2.2s ease-out infinite}
  @keyframes rgMk{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--rgc) 55%,transparent),0 0 12px color-mix(in srgb,var(--rgc) 60%,transparent)}
    70%{box-shadow:0 0 0 calc(7px + 7px*var(--prox)) transparent,0 0 12px color-mix(in srgb,var(--rgc) 60%,transparent)}
    100%{box-shadow:0 0 0 0 transparent,0 0 12px color-mix(in srgb,var(--rgc) 60%,transparent)}}
  .rg-scale{display:flex;justify-content:space-between;font-size:10.5px;color:var(--mut);margin-top:9px;letter-spacing:.4px}
  .rg-trig-lbl{color:var(--txt);font-weight:650}
  .rg-read{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;font-size:14px;color:var(--dim);margin-top:6px}
  .rg-read b{color:var(--txt);font-variant-numeric:tabular-nums}
  .rg-vs{font-weight:750} .rg-vs.grn{color:var(--grn)} .rg-vs.red{color:#ff7a8a}
  .rg-need{color:var(--gold)} .rg-need b{color:var(--gold)}
  .rg-tip{margin-top:13px;padding-top:13px;border-top:1px solid rgba(60,45,30,.09);font-size:13px;
    color:var(--txt);line-height:1.5}
  @media(prefers-reduced-motion:reduce){.rg-scan,.regime::after,.rg-mk,.rg-trig,.rg-dot,.rg-spk-ping,.rg-spk-line{animation:none}
    .rg-spk-line{stroke-dashoffset:0}}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:14px;margin-bottom:18px}
  .card{position:relative;background:linear-gradient(180deg,var(--surface2),var(--surface));border:1px solid var(--line);
    border-radius:var(--r);padding:16px 18px;
    transition:transform .22s cubic-bezier(.22,.9,.3,1),border-color .22s ease,box-shadow .22s ease,z-index 0s}
  .card:hover{transform:translateY(-3px);border-color:rgba(255,138,92,.4);z-index:40;
    box-shadow:0 16px 44px rgba(0,0,0,.42),0 0 26px rgba(255,138,92,.13)}
  /* inset, rounded, glowing accent line — soft fade at the ends */
  .card::before{content:"";position:absolute;left:16px;right:16px;top:0;height:2px;border-radius:2px;
    background:linear-gradient(90deg,transparent,var(--accent) 22%,var(--accent2) 78%,transparent);
    opacity:0;transform:scaleX(.6);transform-origin:center;
    transition:opacity .28s ease,transform .35s cubic-bezier(.22,.9,.3,1)}
  .card:hover::before{opacity:1;transform:scaleX(1);box-shadow:0 0 14px rgba(122,150,255,.6)}
  .card .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.9px;color:var(--mut);margin-bottom:9px}
  .info{display:inline-grid;place-items:center;width:14px;height:14px;border-radius:50%;border:1px solid var(--line);
    color:var(--dim);font-size:9px;font-weight:700;font-style:italic;cursor:help;vertical-align:middle;position:relative;text-transform:none;
    transition:color .15s,border-color .15s,box-shadow .15s,transform .15s}
  /* fill the dot on hover — a white glyph alone is invisible on this light background */
  .info:hover{color:#fff;background:var(--accent);border-color:var(--accent);
    box-shadow:0 0 0 3px rgba(240,101,42,.22);transform:scale(1.15)}
  /* caret */
  .info::before{content:"";position:absolute;left:50%;top:calc(100% + 5px);width:10px;height:10px;
    background:#2a1857;border-left:1px solid rgba(255,138,92,.55);border-top:1px solid rgba(255,138,92,.55);
    transform:translateX(-50%) rotate(45deg) scale(.4);transform-origin:center;opacity:0;pointer-events:none;
    transition:opacity .14s ease,transform .3s cubic-bezier(.34,1.7,.5,1);z-index:41}
  .info::after{content:attr(data-tip);position:absolute;left:50%;top:calc(100% + 10px);
    transform:translateX(-50%) translateY(8px) scale(.9);transform-origin:top center;
    width:288px;background:linear-gradient(180deg,#2a1857,#190f3a);border:1px solid rgba(255,138,92,.4);border-radius:12px;
    padding:11px 13px;font-size:11.5px;font-weight:500;font-style:normal;letter-spacing:0;line-height:1.55;color:#c4cdde;
    /* th/td set white-space:nowrap, which the tooltip inherits and which defeats the fixed width. pre-line both
       restores wrapping AND honours the \n in the tip text, so threshold lists read as separate lines. */
    white-space:pre-line;overflow-wrap:break-word;
    text-transform:none;text-align:left;backdrop-filter:blur(14px);
    box-shadow:0 20px 50px rgba(0,0,0,.7),0 0 30px rgba(255,138,92,.16),inset 0 1px 0 rgba(255,255,255,.05);
    opacity:0;pointer-events:none;transition:opacity .16s ease,transform .34s cubic-bezier(.34,1.56,.5,1);z-index:40}
  .info:hover::after{opacity:1;transform:translateX(-50%) translateY(0) scale(1)}
  .info:hover::before{opacity:1;transform:translateX(-50%) rotate(45deg) scale(1)}
  /* .tipL: anchor the panel to the icon's RIGHT edge, for columns near the right of the table where a
     centred 288px panel would run off screen (that's what clipped the Score tooltip) */
  .info.tipL::after{left:auto;right:-7px;transform:translateX(0) translateY(8px) scale(.9);transform-origin:top right}
  .info.tipL:hover::after{transform:translateX(0) translateY(0) scale(1)}
  .info.tipL::before{left:auto;right:0}
  .info.tipL:hover::before{transform:rotate(45deg) scale(1)}
  .lev{display:inline-block;font-size:9.5px;font-weight:700;color:var(--dim);background:rgba(60,45,30,.09);border:1px solid var(--line);
    border-radius:4px;padding:1px 4px;margin-left:5px;vertical-align:middle;font-variant-numeric:tabular-nums}
  .card .v{font-size:25px;font-weight:760;letter-spacing:-.3px}
  .card .s{font-size:12px;color:var(--mut);margin-top:6px}
  .bar{height:7px;border-radius:5px;background:rgba(70,55,35,.1);overflow:hidden;margin-top:11px;display:flex;border:1px solid rgba(90,70,45,.22)}
  .bar>span{height:100%;display:block;transition:width .4s ease}

  .panel{background:linear-gradient(180deg,var(--surface2),var(--surface));border:1px solid var(--line);
    border-radius:var(--r);padding:18px 20px;margin-bottom:18px}
  .panel h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--mut);margin:0 0 14px;font-weight:700}
  /* Capital & Profit panel */
  .cappanel:empty{display:none}
  .caphead{font-size:10.5px;text-transform:uppercase;letter-spacing:.9px;color:var(--mut);font-weight:700;margin-bottom:13px}
  .capgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}
  .capb{background:var(--surface);border:1px solid var(--line);border-radius:13px;padding:13px 15px}
  .capk{font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--mut);font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:6px}
  .capv{font-size:21px;font-weight:780;letter-spacing:-.3px;font-variant-numeric:tabular-nums}
  .caps{font-size:11.5px;color:var(--dim);margin-top:4px;line-height:1.4}
  @media(max-width:720px){.capgrid{grid-template-columns:repeat(2,1fr);gap:10px}}
  @media(max-width:430px){.capgrid{grid-template-columns:1fr}}
  .ph{display:flex;align-items:flex-end;gap:14px;margin-bottom:12px}
  .ph .lab{font-size:10.5px;text-transform:uppercase;letter-spacing:.9px;color:var(--mut)}
  .ph .big{font-size:30px;font-weight:780;letter-spacing:-.5px;margin-top:2px}
  .seg{margin-left:auto;display:inline-flex;background:rgba(70,55,35,.1);border:1px solid var(--line);border-radius:10px;padding:3px;gap:2px}
  .seg .t{font-size:11.5px;font-weight:600;color:var(--mut);border:0;background:transparent;border-radius:7px;padding:5px 11px;cursor:pointer;transition:.12s}
  .seg .t:hover{color:var(--txt)} .seg .t.on{color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 2px 10px rgba(255,138,92,.35)}
  .segs{display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap}

  .cols{display:grid;grid-template-columns:1.3fr 1fr;gap:18px}
  @media(max-width:920px){.cols{grid-template-columns:1fr}}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:left;padding:13px 17px;white-space:nowrap;vertical-align:middle}
  thead tr{position:sticky;top:0;z-index:2}
  thead th{
    color:var(--mut);font-weight:750;font-size:11px;text-transform:uppercase;
    letter-spacing:.9px;background:rgba(255,253,248,.96);backdrop-filter:blur(10px);
    border-bottom:1px solid var(--line);border-top:0;
  }
  thead th.r{text-align:right}
  tbody tr{border-bottom:1px solid var(--line);transition:background .1s}
  tbody tr:nth-child(even){background:rgba(122,96,54,.06)}
  tbody tr:hover{background:rgba(240,101,42,.1);cursor:default}
  tbody tr:last-child{border-bottom:0}
  td.r{text-align:right}
  td.num{font-variant-numeric:tabular-nums;font-family:'SF Mono','Fira Code','Courier New',monospace;font-size:13.5px;font-weight:650}
  td.mut{color:var(--mut)}
  .fmut{color:var(--dim);font-size:10.5px;margin-left:1px}
  td.asset{font-weight:750;font-size:14.5px;letter-spacing:.4px}
  .chip{display:inline-block;font-size:11px;font-weight:800;padding:4px 10px;border-radius:6px;letter-spacing:.6px;text-transform:uppercase}
  .chip.LONG{background:rgba(75,224,176,.15);color:var(--grn);border:1px solid rgba(75,224,176,.25)}
  .chip.SHORT{background:rgba(255,122,138,.13);color:var(--red);border:1px solid rgba(255,122,138,.22)}
  .sc{font-size:12px;font-weight:750;padding:4px 9px;border-radius:6px;background:rgba(60,45,30,.09);border:1px solid var(--line);font-variant-numeric:tabular-nums}

  .wrow{display:grid;grid-template-columns:58px 1fr 64px;align-items:center;gap:10px;padding:6px 0}
  .wrow .cn{font-weight:600;font-size:12.5px}
  .div{position:relative;height:16px;background:rgba(70,55,35,.1);border:1px solid rgba(90,70,45,.22);border-radius:6px;overflow:hidden}
  .div .c{position:absolute;left:50%;top:1px;bottom:1px;width:1px;background:#2a3445}
  .div .f{position:absolute;top:2px;bottom:2px;border-radius:4px}
  .div .f.l{background:linear-gradient(90deg,rgba(75,224,176,.5),var(--grn))}
  .div .f.s{background:linear-gradient(90deg,var(--red),rgba(255,122,138,.5))}
  .wpct{text-align:right;font-size:12px;font-weight:700}
  .wallets{margin-top:14px;border-top:1px solid var(--line);padding-top:12px;font-size:12px}
  .wallets .w{display:flex;justify-content:space-between;align-items:center;padding:4px 0;color:var(--mut)}
  #chart{max-height:340px}
  .chartwrap{position:relative}
  .chart-note{display:none;margin-top:8px;font-size:11.5px;color:rgba(52,46,36,.55);letter-spacing:.02em;text-align:center}
  .chart-empty{display:none;position:absolute;inset:0;flex-direction:column;align-items:center;justify-content:center;
    text-align:center;color:var(--dim);font-size:14px;font-weight:650;pointer-events:none;padding:0 30px}
  .chart-empty.on{display:flex}
  .chart-empty .ce-sub{display:block;margin-top:6px;font-size:11.5px;font-weight:500;color:var(--mut);opacity:.9;max-width:440px;line-height:1.55}
  .charttip{position:absolute;left:0;top:0;pointer-events:none;opacity:0;z-index:6;will-change:transform;
    transform:translate(0,0);transition:opacity .16s ease,transform .12s cubic-bezier(.22,.9,.3,1);
    background:linear-gradient(180deg,rgba(18,26,40,.97),rgba(10,14,21,.97));border:1px solid rgba(255,138,92,.42);
    border-radius:12px;padding:9px 13px;backdrop-filter:blur(12px);white-space:nowrap;
    box-shadow:0 16px 44px rgba(0,0,0,.62),0 0 26px rgba(255,138,92,.2),inset 0 1px 0 rgba(60,45,30,.09)}
  .charttip .ct-v{font-size:19px;font-weight:820;letter-spacing:-.3px;font-variant-numeric:tabular-nums;line-height:1.05}
  .charttip .ct-t{font-size:10.5px;color:var(--dim);margin-top:4px;letter-spacing:.2px}
  .empty{color:var(--dim);padding:14px 2px;font-size:13px}

  .share{background:rgba(60,45,30,.09);border:1px solid var(--line);color:var(--mut);border-radius:7px;
    padding:3px 9px;cursor:pointer;font-size:13px;line-height:1;transition:.12s}
  .share:hover{color:var(--accent);border-color:var(--accent);box-shadow:0 0 0 2px rgba(240,101,42,.15)}
  @keyframes modalIn{from{transform:scale(.84) translateY(28px);opacity:0}to{transform:scale(1) translateY(0);opacity:1}}
  @keyframes overlayIn{from{opacity:0}to{opacity:1}}
  .modal{position:fixed;inset:0;background:rgba(38,30,20,.55);backdrop-filter:blur(14px);
    display:none;align-items:center;justify-content:center;z-index:50;padding:22px;animation:overlayIn .25s ease}
  .modal-box{background:var(--surface);border:1px solid var(--line);border-radius:18px;max-width:780px;
    width:100%;overflow:hidden;box-shadow:0 40px 100px rgba(0,0,0,.7);
    animation:modalIn .45s cubic-bezier(.34,1.46,.64,1)}
  .modal-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;
    border-bottom:1px solid var(--line);font-size:14px;font-weight:600}
  .modal-head .x{background:transparent;border:0;color:var(--mut);font-size:17px;cursor:pointer}
  .modal-head .x:hover{color:var(--txt)}
  #cardCanvas{display:block;width:100%;height:auto;background:#cfe6da}
  .modal-foot{display:flex;align-items:center;gap:12px;padding:14px 18px;flex-wrap:wrap}
  .swatches{display:flex;gap:8px;margin-right:auto}
  .sw{width:32px;height:32px;border-radius:9px;border:2px solid transparent;cursor:pointer;padding:0}
  .sw.on{border-color:var(--accent);box-shadow:0 0 0 2px rgba(240,101,42,.22)}
  .btn{background:rgba(60,45,30,.11);border:1px solid var(--line);color:var(--txt);border-radius:10px;
    padding:9px 18px;font-weight:600;cursor:pointer;font-size:13px}
  .btn:hover{filter:brightness(1.12)}
  .btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;color:#fff}
  .tstrip{display:flex;flex-wrap:wrap;gap:22px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}
  .tstrip .it{display:flex;flex-direction:column;gap:3px}
  .tstrip .it b{font-size:17px;font-weight:750;font-variant-numeric:tabular-nums}
  .tstrip .it span{font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--mut)}
  .tabwrap{padding-top:6px}
  .tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin:-6px -6px 16px;padding:0 6px;flex-wrap:wrap}
  .tab{background:transparent;border:0;border-bottom:2px solid transparent;color:var(--mut);font-size:13px;
    font-weight:650;padding:11px 15px;cursor:pointer;display:flex;align-items:center;gap:8px;transition:.12s;margin-bottom:-1px}
  .tab:hover{color:var(--txt)}
  .tab.on{color:var(--txt);border-bottom-color:var(--accent)}
  .tab b{font-size:10.5px;font-weight:700;background:rgba(60,45,30,.09);border:1px solid var(--line);border-radius:999px;padding:1px 8px;color:var(--mut)}
  .tab.on b{color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));border:0}
  .pane{display:none} .pane.on{display:block;animation:fade .18s ease}
  @keyframes fade{from{opacity:0}to{opacity:1}}
  .pager{display:flex;align-items:center;gap:10px;margin-top:14px;padding-top:13px;border-top:1px solid var(--line);flex-wrap:wrap}
  .pgcount{font-size:12px;color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}
  .pager .info{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
  .pager .ctrls{margin-left:auto;display:flex;gap:4px;align-items:center}
  .pgb{background:rgba(60,45,30,.09);border:1px solid var(--line);color:var(--mut);min-width:32px;height:32px;
    padding:0 10px;border-radius:9px;font-size:12.5px;font-weight:650;cursor:pointer;transition:.12s;font-variant-numeric:tabular-nums}
  .pgb:hover:not(:disabled){color:var(--accent);border-color:rgba(255,138,92,.5);box-shadow:0 0 0 2px rgba(255,138,92,.12)}
  .pgb.on{color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;box-shadow:0 2px 10px rgba(255,138,92,.35)}
  .pgb:disabled{opacity:.32;cursor:default}
  .pager .dots{color:var(--dim);padding:0 3px;font-weight:700}

  /* ===================== HEADER NAV ===================== */
  .nav{display:flex;gap:4px;background:rgba(255,254,250,.94);border:1px solid var(--line);border-radius:11px;padding:3px}
  .navbtn{background:transparent;border:0;color:#4a4233;font-size:12.5px;font-weight:750;white-space:nowrap;
    padding:6px 14px;border-radius:8px;cursor:pointer;transition:.16s;letter-spacing:.2px}
  .navbtn:hover{color:var(--accent)}
  .navbtn.on{color:#fff;background:linear-gradient(135deg,var(--accent),var(--gold));box-shadow:0 3px 13px rgba(240,101,42,.42)}

  /* ===================== ALIEN STRATEGY PAGE ===================== */
  #alienBg{display:none}
  .spage{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:10px 26px 80px}
  @keyframes revUp{from{opacity:0;transform:translateY(34px)}to{opacity:1;transform:translateY(0)}}
  .reveal{opacity:0}
  .reveal.in{animation:revUp .8s cubic-bezier(.22,.9,.3,1) forwards}

  /* hero */
  .hero{position:relative;text-align:center;padding:70px 20px 56px;overflow:hidden}
  .orb{position:absolute;top:-160px;left:50%;transform:translateX(-50%);width:620px;height:620px;border-radius:50%;
    background:radial-gradient(circle,rgba(154,123,255,.34),rgba(75,224,176,.14) 42%,transparent 68%);
    filter:blur(28px);animation:orbFloat 9s ease-in-out infinite;z-index:-1}
  @keyframes orbFloat{0%,100%{transform:translateX(-50%) translateY(0) scale(1)}50%{transform:translateX(-50%) translateY(26px) scale(1.07)}}
  .hero-tag{font-size:11.5px;letter-spacing:3.4px;font-weight:800;
    background:linear-gradient(90deg,#d8521f,#c2851a,#3f6ea8,#0f8a4a);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:18px}
  .hero-title{font-size:clamp(46px,8vw,92px);font-weight:850;letter-spacing:-2px;line-height:1.1;margin:0 0 14px;padding-bottom:.12em;
    background:linear-gradient(118deg,#2d2922,#574326);-webkit-background-clip:text;background-clip:text;color:transparent;
    filter:drop-shadow(0 2px 9px rgba(70,50,30,.13));position:relative}
  .hero-sub{font-size:clamp(15px,2.4vw,20px);color:var(--mut);max-width:620px;margin:0 auto;line-height:1.55}
  .hero-sub b{color:var(--txt);font-weight:700}
  .hero-stats{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-top:40px}
  .hstat{flex:1;min-width:130px;max-width:200px;background:rgba(255,255,253,.95);border:1px solid var(--line);
    border-radius:16px;padding:18px 14px;backdrop-filter:blur(8px);transition:.25s}
  .hstat:hover{border-color:rgba(255,138,92,.5);transform:translateY(-3px);box-shadow:0 14px 40px rgba(75,224,176,.16)}
  .hstat .hv{font-size:31px;font-weight:850;letter-spacing:-.5px;font-variant-numeric:tabular-nums;color:var(--accent2)}
  .hero-stats .hstat:nth-child(1) .hv{color:var(--grn)}
  .hero-stats .hstat:nth-child(2) .hv{color:var(--accent2)}
  .hero-stats .hstat:nth-child(3) .hv{color:var(--red)}
  .hero-stats .hstat:nth-child(4) .hv{color:var(--teal)}
  .hstat .hl{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--mut);margin-top:6px;font-weight:600}

  /* section blocks */
  .sblock{margin-top:78px}
  .seye{font-size:11px;letter-spacing:2.6px;font-weight:700;color:var(--accent);margin-bottom:12px}
  .sh2{font-size:clamp(24px,4vw,38px);font-weight:800;letter-spacing:-.8px;margin:0 0 30px;line-height:1.12;
    background:linear-gradient(120deg,#f0652a,#e0a52e 45%,#3f80ba);-webkit-background-clip:text;background-clip:text;color:transparent}

  .steps{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
  @media(max-width:780px){.steps{grid-template-columns:1fr}}
  .step{position:relative;background:linear-gradient(180deg,rgba(255,253,249,.93),rgba(255,255,253,.95));
    border:1px solid var(--line);border-radius:20px;padding:26px 22px;overflow:hidden;transition:.3s}
  .step::after{content:"";position:absolute;inset:0;border-radius:20px;padding:1px;background:linear-gradient(135deg,rgba(75,224,176,.5),transparent 45%);
    -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;opacity:0;transition:.3s}
  .step:hover{transform:translateY(-5px);box-shadow:0 22px 60px rgba(0,0,0,.5)}
  .step:hover::after{opacity:1}
  .snum{position:absolute;top:14px;right:20px;font-size:58px;font-weight:850;line-height:1;
    background:linear-gradient(135deg,#f0652a,#e0a52e);-webkit-background-clip:text;background-clip:text;color:transparent;opacity:.42}
  .sicon{font-size:30px;margin-bottom:14px}
  .step h3{font-size:19px;font-weight:760;margin:0 0 9px}
  .step p{font-size:13.5px;color:var(--mut);line-height:1.6;margin:0}

  .why{margin-top:22px;background:rgba(255,253,249,.93);border:1px solid var(--line);border-radius:18px;padding:8px 22px;
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
  .whyrow{display:grid;grid-template-columns:200px 1fr;gap:18px;padding:16px 0;border-bottom:1px solid rgba(28,36,51,.5)}
  .whyrow:last-child{border-bottom:0}
  @media(max-width:640px){.whyrow{grid-template-columns:1fr;gap:5px}}
  .wk{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--accent)}
  .wv{font-size:14px;color:var(--mut);line-height:1.55} .wv b{color:var(--txt)} .wv i{color:#9a7bff;font-style:normal;font-weight:600}

  /* flow svg */
  .flowwrap{background:radial-gradient(circle at 50% 35%,rgba(255,255,253,.95),rgba(250,245,232,.9));
    border:1px solid var(--line);border-radius:22px;padding:14px;overflow:hidden;
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
  #flowSvg{width:100%;height:auto;display:block}

  /* architecture */
  .arch{display:flex;flex-direction:column;align-items:center;gap:6px}
  .alayer{width:100%;max-width:680px;background:linear-gradient(135deg,rgba(255,253,249,.93),rgba(255,255,253,.95));
    border:1px solid var(--line);border-left:3px solid var(--ac);border-radius:14px;padding:18px 24px;transition:.28s}
  .alayer:hover{transform:scale(1.02);box-shadow:0 0 0 1px var(--ac),0 16px 44px rgba(0,0,0,.5)}
  .aname{font-size:16px;font-weight:760;color:var(--txt);margin-bottom:4px}
  .alayer .adesc{font-size:13px;color:var(--mut);line-height:1.5}
  .aflow{color:var(--accent);font-size:15px;opacity:.5;animation:aPulse 2s ease-in-out infinite}
  @keyframes aPulse{0%,100%{opacity:.3;transform:translateY(0)}50%{opacity:.8;transform:translateY(3px)}}

  /* performance */
  .sp-note{font-size:13px;color:var(--dim);margin:-18px 0 26px}
  .perfgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  @media(max-width:680px){.perfgrid{grid-template-columns:repeat(2,1fr)}}
  .pcard{background:linear-gradient(180deg,rgba(255,253,249,.93),rgba(255,255,253,.95));border:1px solid var(--line);
    border-radius:18px;padding:26px 18px;text-align:center;transition:.28s}
  .pcard:hover{transform:translateY(-4px);border-color:rgba(255,138,92,.45);box-shadow:0 18px 50px rgba(75,224,176,.14)}
  .pcard .pv{font-size:36px;font-weight:840;letter-spacing:-1px;font-variant-numeric:tabular-nums;color:var(--txt)}
  .pcard .pv.grn{color:var(--grn)} .pcard .pv.red{color:var(--red)}
  .pcard .pl{font-size:11.5px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-top:9px}
  .disclaim{margin-top:24px;font-size:12.5px;color:var(--dim);line-height:1.65;background:rgba(245,196,81,.05);
    border:1px solid rgba(245,196,81,.18);border-radius:14px;padding:16px 20px}
  .sfoot{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;
    margin:70px auto 0;padding:14px 22px;border:1px solid var(--line);border-radius:16px;
    background:rgba(255,253,247,.72);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
    font-size:13px;color:var(--mut);letter-spacing:.2px}
  .sfoot-mark{font-weight:800;color:var(--txt)} .sfoot-txt{color:var(--mut);font-weight:600}
  .sfoot .ghlink{margin-left:0}

  /* two-strategy comparison */
  .tchip{display:inline-block;vertical-align:middle;font-size:.42em;font-weight:800;letter-spacing:1.4px;
    text-transform:uppercase;padding:.45em .8em;border-radius:999px;margin-left:.5em;color:#0b0e14;
    background:linear-gradient(135deg,#ffd166,#ffb24a);transform:translateY(-.28em)}
  .tchip.cy{background:linear-gradient(135deg,#4be0b0,#4be0b0)}
  .vsgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .vsgrid.vs3{grid-template-columns:1fr 1fr 1fr}
  .vsgrid.vs3{grid-template-columns:repeat(3,1fr);gap:14px}
  @media(max-width:1200px){.vsgrid.vs3{grid-template-columns:1fr 1fr}}
  @media(max-width:980px){.vsgrid.vs3{grid-template-columns:1fr} .vscard.cur,.vscard.cur:hover{transform:none}}
  @media(max-width:640px){.vsgrid.vs3{grid-template-columns:1fr}}
  @media(max-width:720px){.vsgrid{grid-template-columns:1fr}}
  /* strategy cards */
  .vscard{position:relative;display:flex;flex-direction:column;background:linear-gradient(180deg,rgba(255,253,249,.93),rgba(255,255,253,.95));
    border:1px solid var(--line);border-radius:20px;padding:24px 22px 22px;transition:.3s}
  .vscard:hover{transform:translateY(-4px);box-shadow:0 18px 50px rgba(80,60,30,.18)}
  .vscard.cur{transform:scale(1.05);z-index:3;border-color:transparent;
    background:linear-gradient(180deg,rgba(255,250,239,.98),rgba(255,254,248,.98));
    box-shadow:0 0 0 2.5px var(--accent),0 30px 74px rgba(240,101,42,.26)}
  .vscard.cur:hover{transform:scale(1.05) translateY(-4px)}
  #vs-carry.cur{box-shadow:0 0 0 2.5px var(--grn),0 30px 74px rgba(23,154,85,.26);
    background:linear-gradient(180deg,rgba(236,250,241,.98),rgba(250,255,251,.98))}
  #vs-champion.cur{box-shadow:0 0 0 2.5px var(--gold),0 30px 74px rgba(224,165,46,.28);
    background:linear-gradient(180deg,rgba(255,249,233,.98),rgba(255,253,243,.98))}
  #vs-trend.cur{box-shadow:0 0 0 2.5px #3f80ba,0 30px 74px rgba(63,128,186,.26);
    background:linear-gradient(180deg,rgba(233,242,250,.98),rgba(249,252,255,.98))}
  /* header row: icon + name + badge(s) all on one line, no overlap */
  .vshead{font-size:17px;font-weight:800;letter-spacing:-.3px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-right:0}
  .vsicon{font-size:18px;flex:none} .vsicon.n{color:var(--accent)} .vsicon.c{color:var(--gold)} .vsicon.k{color:var(--grn)} .vsicon.f{color:#3f80ba}
  .vsbadge{font-size:9px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:#3a2c10;flex:none;
    background:linear-gradient(135deg,#f3bd44,#e0a52e);border-radius:999px;padding:.32em .65em}
  .vsbadge.cb{background:linear-gradient(135deg,#1fb866,#179a55);color:#fff}
  /* "you are here" — separate row, never overlaps the header */
  .vs-here{display:none;font-size:9.5px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;
    padding:.32em .8em;border-radius:999px;border:1px solid;margin-top:8px;align-self:flex-start}
  .vscard.cur .vs-here{display:inline-block}
  .vscard.cur .vs-here{color:var(--accent);background:rgba(255,138,92,.1);border-color:rgba(255,138,92,.3)}
  #vs-carry.cur .vs-here{color:var(--grn);background:rgba(75,224,176,.1);border-color:rgba(75,224,176,.3)}
  #vs-champion.cur .vs-here{color:#b9821c;background:rgba(224,165,46,.16);border-color:rgba(224,165,46,.42)}
  #vs-trend.cur .vs-here{color:#3f80ba;background:rgba(63,128,186,.1);border-color:rgba(63,128,186,.32)}
  .vsq{font-size:10.5px;font-weight:600;color:var(--dim);opacity:.75}
  .vstag{font-size:11px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;color:var(--dim);margin:10px 0 13px}
  .vscard p{font-size:13px;line-height:1.62;color:var(--txt);margin:0 0 14px}
  .vsstats{list-style:none;padding:0;margin:0 0 15px;display:flex;flex-direction:column;gap:8px}
  .vsstats li{font-size:13px;color:var(--dim);padding-left:18px;position:relative}
  .vsstats li::before{content:'';position:absolute;left:0;top:7px;width:7px;height:7px;border-radius:50%;
    background:var(--accent);opacity:.7}
  #vs-champion .vsstats li::before{background:var(--gold);opacity:.9}
  #vs-carry .vsstats li::before{background:var(--grn);opacity:.85}
  #vs-trend .vsstats li::before{background:#3f80ba;opacity:.85}
  .vsstats b{color:var(--txt)} .vsbest{font-size:12px;color:var(--mut);font-style:italic;margin:auto 0 16px}
  /* "switch to this" button — same-tab navigate */
  .vslink{display:inline-block;align-self:flex-start;font-size:13px;font-weight:700;color:var(--accent);text-decoration:none;
    padding:.42em 1em;border-radius:999px;border:1px solid rgba(255,138,92,.35);
    background:rgba(255,138,92,.08);transition:.2s;cursor:pointer}
  .vslink:hover{background:rgba(255,138,92,.18);border-color:rgba(255,138,92,.6)}
  #vs-champion .vslink{color:#b9821c;border-color:rgba(224,165,46,.4);background:rgba(224,165,46,.1)}
  #vs-champion .vslink:hover{background:rgba(224,165,46,.2);border-color:rgba(224,165,46,.65)}
  #vs-carry .vslink{color:var(--grn);border-color:rgba(23,154,85,.4);background:rgba(23,154,85,.1)}
  #vs-carry .vslink:hover{background:rgba(23,154,85,.2);border-color:rgba(23,154,85,.65)}
  #vs-trend .vslink{color:#2f6a9e;border-color:rgba(63,128,186,.4);background:rgba(63,128,186,.1)}
  #vs-trend .vslink:hover{background:rgba(63,128,186,.2);border-color:rgba(63,128,186,.65)}
  /* active card's "Currently active" pill — full-width filled button (standalone class, no .vslink hover) */
  .vslink-cur{display:block;width:100%;box-sizing:border-box;text-align:center;font-size:13px;font-weight:800;
    padding:.62em 1em;border-radius:999px;cursor:default;color:#fff;border:0;text-decoration:none;
    background:linear-gradient(135deg,var(--accent),#d8531f)}
  #vs-champion .vslink-cur{color:#3a2c10;background:linear-gradient(135deg,#f3bd44,#e0a52e)}
  #vs-carry .vslink-cur{background:linear-gradient(135deg,#1fb866,#179a55)}
  #vs-trend .vslink-cur{background:linear-gradient(135deg,#3f80ba,#2f6a9e)}
  /* strategy switcher in nav — single pill dropdown */
  .nav-switcher{position:relative;display:flex;align-items:center}
  .nav-switcher-btn{display:flex;align-items:center;gap:6px;background:rgba(255,138,92,.12);border:1px solid rgba(255,138,92,.3);
    color:var(--txt);font-size:12.5px;font-weight:700;padding:.38em .85em .38em .7em;border-radius:999px;cursor:pointer;transition:.2s;white-space:nowrap}
  .nav-switcher-btn:hover{background:rgba(255,138,92,.22);border-color:rgba(255,138,92,.5)}
  .nav-switcher-btn .sw-arrow{font-size:9px;opacity:.7;transition:transform .18s}
  .nav-switcher.open .sw-arrow{transform:rotate(180deg)}
  .nav-switcher-menu{display:none;position:absolute;top:calc(100% + 7px);left:0;min-width:200px;
    background:#fffaf0;border:1px solid rgba(108,90,60,.22);border-radius:14px;padding:6px;
    box-shadow:0 18px 50px rgba(80,60,30,.28);z-index:300}
  .nav-switcher.open .nav-switcher-menu{display:block}
  .sw-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:9px;cursor:pointer;
    font-size:13px;font-weight:600;color:var(--txt);transition:.15s;border:0;background:0;width:100%;text-align:left}
  .sw-item:hover{background:rgba(60,45,30,.09)}
  .sw-item.sw-cur{background:rgba(255,138,92,.14);color:var(--txt);cursor:default}
  .sw-item .sw-dot{width:8px;height:8px;border-radius:50%;flex:none}
  .sw-item .sw-cur-tag{font-size:9px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;
    margin-left:auto;color:var(--mut)}

  .dfoot{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;
    margin-top:26px;padding:14px 22px;border:1px solid var(--line);border-radius:16px;
    background:rgba(255,253,247,.72);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
    font-size:13px;color:var(--mut)}
  .dfoot-mark{font-weight:800;color:var(--txt)} .dfoot-dot{color:var(--dim)}
  .dfoot-txt{letter-spacing:.2px;color:var(--mut);font-weight:600}
  .ghlink{display:inline-flex;align-items:center;gap:7px;margin-left:auto;color:var(--mut);text-decoration:none;
    background:rgba(255,255,253,.95);border:1px solid var(--line);border-radius:10px;padding:6px 12px;font-weight:600;transition:.18s}
  .ghlink:hover{color:#fff;border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,138,92,.16);transform:translateY(-1px);text-decoration:none}
  @media(max-width:560px){.ghlink{margin-left:0}}

  /* horizontally-scrollable tables on small screens (keeps columns crisp, no squish) */
  .tblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:10px}
  .tblwrap::-webkit-scrollbar{height:6px}
  .tblwrap::-webkit-scrollbar-thumb{background:rgba(108,90,60,.35);border-radius:3px}
  .tblwrap::-webkit-scrollbar-track{background:transparent}

  /* ===================== MOBILE: 720px ===================== */
  @media(max-width:720px){
    main{padding:14px 12px}
    /* header: wrap into 2 rows; brand+mode on row 1, nav+status on row 2 */
    header{padding:11px 13px;gap:8px 10px;flex-wrap:wrap;border-radius:16px;width:calc(100% - 20px);top:6px}
    .brand{flex:1 1 auto} .brand-name{font-size:14.5px} .brand-sub{font-size:7.5px;letter-spacing:1.8px}
    .nav{order:5;width:100%;justify-content:center;gap:3px}
    .navbtn{font-size:11.5px;padding:6px 10px}
    .meta{display:none}
    #mode{margin-left:0}
    .pill{font-size:10px;padding:3px 9px}
    .cd{margin-left:0;font-size:11px;padding:4px 9px}
    .status{font-size:11px;padding:4px 8px}
    /* strategy switcher: prevent overflow */
    .nav-switcher-btn{font-size:12px;padding:6px 10px;max-width:160px}
    .nav-switcher-btn span{overflow:hidden;text-overflow:ellipsis}
    .sw-menu{right:0;left:auto;min-width:180px}
    /* stat cards: 2-up grid */
    .cards{grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
    .card{padding:13px 14px} .card .v{font-size:20px} .card .k{margin-bottom:6px;font-size:9px}
    .info::after{width:min(72vw,260px)}
    /* panels */
    .panel{padding:14px 13px;margin-bottom:14px}
    /* performance section: stack PnL + toggles vertically */
    .ph{flex-wrap:wrap;gap:10px}
    .ph .big{font-size:24px}
    .seg{margin-left:0;margin-top:4px;padding:2px}
    .seg .t{padding:5px 10px;font-size:11px}
    #chart{max-height:240px}
    .segs{gap:8px}
    /* chart tooltip: keep inside viewport */
    .charttip{max-width:calc(100vw - 40px)} .charttip .ct-v{font-size:16px}
    /* tables: horizontal scroll instead of clipping */
    .cols{grid-template-columns:1fr}
    table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch}
    th,td{padding:9px 11px;font-size:13px}
    /* tabs: scroll horizontally */
    .tabs{gap:1px;overflow-x:auto;-webkit-overflow-scrolling:touch;flex-wrap:nowrap}
    .tab{padding:10px 9px;font-size:12.5px;white-space:nowrap;flex-shrink:0}
    /* regime panel */
    .regime{padding:16px 14px}
    .rg-head .rg-title{font-size:13px}
    /* comparison cards: kill the scale transform (causes horizontal overflow) */
    .vscard.cur,.vscard.cur:hover{transform:none}
    /* strategy page */
    .spage{padding:4px 14px 60px} .hero{padding:46px 6px 38px}
    .sblock{margin-top:52px} .sh2{margin-bottom:22px}
    .hero-stats{gap:9px} .hstat{min-width:0;flex:1 1 44%}
    .perfgrid{grid-template-columns:repeat(2,1fr)}
    .whyrow{grid-template-columns:1fr;gap:4px}
    .dfoot{flex-direction:column;gap:8px;text-align:center} .ghlink{margin-left:0}
    /* pager */
    .pager{flex-wrap:wrap;gap:6px}
    /* modal: full-width on mobile */
    .modal-box{width:calc(100vw - 24px);max-height:88vh;overflow-y:auto}
  }
  /* ===================== MOBILE: 430px (small phones) ===================== */
  @media(max-width:430px){
    header{padding:9px 10px;gap:6px 8px;border-radius:13px;top:4px;width:calc(100% - 16px)}
    .brand-name{font-size:13px} .brand-sub{display:none}
    .brand .mark{width:26px;height:26px;font-size:13px}
    .cards{grid-template-columns:1fr}
    .card .v{font-size:20px} .card .k{font-size:8.5px}
    .hstat{flex:1 1 100%} .perfgrid{grid-template-columns:1fr}
    .ph{flex-direction:column;align-items:flex-start;gap:6px}
    .ph .big{font-size:21px}
    .seg{width:100%;justify-content:center}
    th,td{padding:7px 9px;font-size:12px}
    .nav-switcher-btn{max-width:130px;font-size:11px}
  }
</style>
</head>
<body>
<svg id="skyline" viewBox="0 0 1440 340" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
  <!-- soft Mt Fuji in the haze (peak lowered + base widened so the summit never slices off on wide/short windows) -->
  <polygon points="640,300 960,150 1280,300" fill="#a9c2e0" opacity=".55"/>
  <polygon points="928,188 960,150 992,188 981,196 969,189 957,197 945,189" fill="#fcfeff" opacity=".85"/>
  <!-- a couple of soft Tokyo silhouettes, warm and faint -->
  <g fill="#cdbf9e" opacity=".45">
    <rect x="250" y="196" width="30" height="120"/><rect x="292" y="172" width="22" height="144"/>
    <polygon points="320,316 334,150 348,316"/></g>
  <!-- lush rolling green hills, layered front-to-back (Ghibli) -->
  <path d="M0,236 Q300,196 620,232 T1440,226 V340 H0 Z" fill="#9fce93" opacity=".7"/>
  <path d="M0,272 Q360,236 780,272 T1440,266 V340 H0 Z" fill="#7cbd76" opacity=".82"/>
  <path d="M0,306 Q440,282 920,306 T1440,302 V340 H0 Z" fill="#5fa861" opacity=".92"/>
</svg>
<div class="sakura" aria-hidden="true">
  <i style="left:6%;animation-duration:14s;animation-delay:0s"></i>
  <i style="left:20%;animation-duration:18s;animation-delay:-4s"></i>
  <i style="left:38%;animation-duration:12s;animation-delay:-8s"></i>
  <i style="left:55%;animation-duration:20s;animation-delay:-2s"></i>
  <i style="left:71%;animation-duration:15s;animation-delay:-11s"></i>
  <i style="left:86%;animation-duration:17s;animation-delay:-6s"></i>
  <i style="left:94%;animation-duration:13s;animation-delay:-14s"></i>
</div>
<header>
  <div class="brand"><span class="mark">⚡</span><span class="brand-tx"><span class="brand-name">Sentinel&nbsp;Edge</span><span class="brand-sub">Autonomous&nbsp;Desk</span></span></div>
  <nav class="nav">
    <button class="navbtn on" data-page="dash" onclick="goPage('dash')">Dashboard</button>
    <button class="navbtn" data-page="strat" onclick="goPage('strat')">How it works</button>
    <div class="nav-switcher" id="navSwitcher">
      <button class="nav-switcher-btn" onclick="toggleSwitcher(event)">
        <span id="swCurIcon">—</span><span id="swCurLbl">Strategy</span><span class="sw-arrow">▾</span>
      </button>
      <div class="nav-switcher-menu" id="swMenu"></div>
    </div>
  </nav>
  <span id="mode" class="pill paper">paper</span>
  <span class="status"><span id="dot" class="dot off"></span><span id="run">checking…</span></span>
  <span class="cd" id="cdwrap" title="time until the next scheduled daily rebalance"><span class="cd-ic">⟳</span> <span id="cdlbl">next rebalance</span> <b id="cd">—</b></span>
  <span class="meta">updated <b id="upd">—</b></span>
</header>
<main id="pageDash">
  <section id="regimeWrap" style="display:none"></section>
  <div class="cards" id="cards"></div>

  <div class="panel cappanel" id="capitalPanel"></div>

  <div class="panel">
    <div class="ph">
      <div><div class="lab">Performance</div><div class="big num" id="chartval">—</div></div>
      <div class="seg" id="metricTog"><button class="t on" data-m="pnl">PnL</button><button class="t" data-m="value">Value</button></div>
    </div>
    <div class="segs"><div class="seg" id="rangeTog">
      <button class="t" data-r="24H">24H</button><button class="t" data-r="7D">7D</button>
      <button class="t" data-r="30D">30D</button><button class="t on" data-r="ALL">ALL</button>
    </div></div>
    <div class="chartwrap"><canvas id="chart"></canvas><div id="chartTip" class="charttip"></div><div id="chartEmpty" class="chart-empty"></div></div>
    <div id="chartNote" class="chart-note"></div>
  </div>

  <div class="panel tabwrap">
    <div class="tabs" id="tabs">
      <button class="tab on" data-t="positions">Positions <b id="np">0</b></button>
      <button class="tab" data-t="trades">Trades <b id="tcount">0</b></button>
      <button class="tab" data-t="fills">Fills</button>
      <button class="tab" data-t="smart">Smart Money <b id="nw">0</b></button>
    </div>

    <div class="pane on" id="pane-positions">
      <div class="tblwrap"><table><colgroup><col style="width:88px"><col style="width:108px"><col style="width:86px"><col style="width:108px"><col style="width:108px"><col style="width:110px"><col style="width:102px"><col style="width:72px"><col style="width:106px"><col style="width:152px"><col style="width:92px"><col style="width:34px"></colgroup>
      <thead><tr><th>Asset</th><th>Side</th><th class="r">Size</th><th class="r">Value</th>
      <th class="r">Entry</th><th class="r">Mark</th><th class="r">uPnL</th>
      <th class="r">% <span class="info" data-tip="Return on this position: uPnL as a % of what the position was worth at entry. Not a % of your account — a 10% move on a $2,000 position is $200, which is 2% of a $10,000 book.">i</span></th>
      <th class="r">Funding</th><th class="r">Opened</th>
      <th class="r">Score <span class="info tipL" id="scoreTip" data-tip="">i</span></th><th></th></tr></thead>
      <tbody id="pos"></tbody></table></div>
    </div>

    <div class="pane" id="pane-trades">
      <div class="tstrip" id="tstrip"></div>
      <div class="tblwrap"><table><thead><tr><th>Asset</th><th>Side</th><th class="r">Entry</th><th class="r">Exit</th>
      <th class="r">Net PnL</th>
      <th class="r">% <span class="info" data-tip="Price move on this trade, in the direction it was held: for a long, exit vs entry; for a short, the inverse. It is the move on the position itself, not a % of your account.">i</span></th>
      <th class="r">Opened</th><th class="r">Closed</th><th class="r">Held</th></tr></thead>
      <tbody id="ctrd"></tbody></table></div>

      <div class="pager" id="pgTrades"></div>
    </div>

    <div class="pane" id="pane-fills">
      <div class="tblwrap"><table><thead><tr><th>Asset</th><th>Side</th><th>Action</th><th class="r">Vol</th><th class="r">Price</th><th class="r">Status</th></tr></thead>
      <tbody id="fillrows"></tbody></table></div>
      <div class="pager" id="pgFills"></div>
    </div>

    <div class="pane" id="pane-smart">
      <div id="consensus"></div>
      <div class="wallets" id="wallets"></div>
    </div>
  </div>

  <footer class="dfoot">
    <span class="dfoot-mark">⚡ Sentinel&nbsp;Edge</span>
    <span class="dfoot-dot">·</span>
    <span class="dfoot-txt">market-making · mean-reversion · Hyperliquid futures</span>
    <a class="ghlink" href="https://github.com/adensvaz/Sentinal_Hyperliquid" target="_blank" rel="noopener" title="View source on GitHub">
      <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      <span>GitHub</span>
    </a>
  </footer>
</main>

<canvas id="alienBg"></canvas>
<main id="pageStrat" class="spage" style="display:none">

  <!-- HERO (populated per strategy by paintStrategy) -->
  <section class="hero reveal">
    <div class="orb"></div>
    <div class="hero-tag" id="heroTag">HYPERLIQUID FUTURES</div>
    <h1 class="hero-title" id="heroTitle">Sentinel&nbsp;Edge</h1>
    <p class="hero-sub" id="heroSub">Four strategies, one engine.</p>
    <div class="hero-stats" id="heroStats"></div>
  </section>

  <!-- THREE STRATEGIES -->
  <section class="sblock reveal">
    <div class="seye">THE SYSTEM</div>
    <h2 class="sh2">Four strategies, one engine</h2>
    <p class="sp-note">Same data, same execution, same risk rails — three uncorrelated ways to trade. You're viewing <b id="curStrat">—</b>.</p>
    <div class="vsgrid vs3">
      <div class="vscard" id="vs-champion">
        <div class="vshead"><span class="vsicon c">⚡</span> Momentum <span class="vsbadge">Champion</span></div>
        <span class="vs-here">You are here</span>
        <div class="vstag">Growth · rides bull trends</div>
        <p>Holds the 5 strongest coins <b>only while Bitcoin is above its 100-day line</b>, and sits fully in <b>cash</b> the rest of the time. All the upside of a bull run, zero exposure in bear or chop.</p>
        <ul class="vsstats"><li><b>In cash now</b> — BTC below its line</li><li>directional · <b>high variance</b> when live</li><li>backtest is survivorship-caveated</li></ul>
        <div class="vsbest">Best for — bull-market growth; deliberately does nothing otherwise</div>
        <a class="vslink" id="link-champion">Switch to this →</a>
      </div>
      <div class="vscard" id="vs-carry">
        <div class="vshead"><span class="vsicon k">💰</span> Funding Carry <span class="vsbadge cb">Live</span></div>
        <span class="vs-here">You are here</span>
        <div class="vstag">Market-neutral · momentum + funding</div>
        <p>Dollar-neutral: <b>longs the strongest coins, shorts the weakest</b>, tilted by the funding each pays. Profits from <i>which coins beat which</i> — not the market's direction.</p>
        <ul class="vsstats"><li><b>+8%</b> live · best book so far</li><li><b>~15–22%</b>/yr honest · Sharpe ~1.0</li><li><b>lumpy</b> — big years <i>and</i> dead ones</li></ul>
        <div class="vsbest">Best for — copyable market-neutral returns; a real diversifier</div>
        <a class="vslink" id="link-carry">Switch to this →</a>
      </div>
      <div class="vscard" id="vs-trend">
        <div class="vshead"><span class="vsicon f">📈</span> Trend <span class="vsbadge cb">Live</span></div>
        <span class="vs-here">You are here</span>
        <div class="vstag">Trend-following · CTA</div>
        <p>Dollar-neutral trend book: <b>longs the coins highest in their own 45-day range, shorts those lowest</b>. Range position replaced price-vs-moving-average because it survives realistic fees — the old scorer stopped earning at 10bps.</p>
        <ul class="vsstats"><li><b>0.87</b> Sharpe <span class="vsq">survivorship-stripped</span></li><li>positive in <b>5 of 9</b> quarters</li><li>one-regime — can lose heavily</li></ul>
        <div class="vsbest">Best for — the trend edge at a cost it can actually pay</div>
        <a class="vslink" id="link-trend">Switch to this →</a>
      </div>
    </div>
  </section>

  <!-- THE IDEA (strategy-aware, populated by paintStrategy) -->
  <section class="sblock reveal">
    <div class="seye">01 — THE IDEA</div>
    <h2 class="sh2" id="ideaTitle">—</h2>
    <div class="steps" id="ideaSteps"></div>
    <div class="why reveal" id="ideaWhy"></div>
  </section>

  <!-- FLOW DIAGRAM -->
  <section class="sblock reveal">
    <div class="seye">02 — THE FLOW</div>
    <h2 class="sh2">How a trade is born</h2>
    <div class="flowwrap"><svg id="flowSvg" viewBox="0 0 1000 360" preserveAspectRatio="xMidYMid meet"></svg></div>
  </section>

  <!-- ARCHITECTURE -->
  <section class="sblock reveal">
    <div class="seye">03 — THE MACHINE</div>
    <h2 class="sh2">Five layers, fully automated</h2>
    <div class="arch" id="archLayers"></div>
  </section>

  <!-- PERFORMANCE (strategy-aware, populated by paintStrategy) -->
  <section class="sblock reveal">
    <div class="seye">04 — THE NUMBERS</div>
    <h2 class="sh2" id="numTitle">Backtest performance</h2>
    <p class="sp-note" id="numNote">—</p>
    <div class="perfgrid" id="perfGrid"></div>
    <div class="disclaim reveal" id="numDisclaim">—</div>
  </section>

  <div class="sfoot reveal" id="sfoot">
    <span class="sfoot-mark">⚡ Sentinel&nbsp;Edge</span>
    <span class="sfoot-txt">· four strategies, one clean engine</span>
    <a class="ghlink" href="https://github.com/adensvaz/Sentinal_Hyperliquid" target="_blank" rel="noopener" title="View source on GitHub">
      <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      <span>GitHub</span>
    </a>
  </div>
</main>

<div id="cardModal" class="modal">
  <div class="modal-box">
    <div class="modal-head"><span>Share trade card</span><button class="x" onclick="closeCard()">✕</button></div>
    <canvas id="cardCanvas"></canvas>
    <div class="modal-foot">
      <div class="swatches" id="swatches"></div>
      <button class="btn" onclick="copyCard()">Copy</button>
      <button class="btn primary" onclick="downloadCard()">Download PNG</button>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const money=n=>'$'+(+n||0).toLocaleString(undefined,{maximumFractionDigits:2});
const sgn=n=>(n>=0?'+':'−')+'$'+Math.abs(+n||0).toLocaleString(undefined,{maximumFractionDigits:2});
const sig6=n=>n==null?'—':(+n).toLocaleString(undefined,{maximumSignificantDigits:6});
const short=s=>(s||'').replace(/^E-|-USDT$/g,'');
const dur=s=>{s=+s||0;const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?`${d}d ${h}h`:(h?`${h}h ${m}m`:`${m}m`);};
// exact local timestamp from epoch SECONDS — e.g. "Jun 26, 17:28:04"
const dt=s=>!s?'—':new Date(s*1000).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
const ago=s=>{if(!s)return '';const d=Math.max(0,Date.now()/1000-s);return dur(d)+' ago';};
let DATA=null, chart=null, metric='pnl', range='ALL';
const RANGES={'24H':864e5,'7D':6048e5,'30D':2592e6,'ALL':Infinity};

/* ============ STRATEGY CONTENT (the "How it works" page is shared by both books) ============ */
let CUR_STRAT=null, STRAT_PAINTED=false;
const stepHTML=(n,icon,h,p)=>`<div class="step reveal"><div class="snum">${n}</div><div class="sicon">${icon}</div><h3>${h}</h3><p>${p}</p></div>`;
const whyHTML=(k,v)=>`<div class="whyrow"><span class="wk">${k}</span><span class="wv">${v}</span></div>`;
const pcardHTML=(v,l,cls,o={})=>`<div class="pcard reveal"><div class="pv ${cls||''}" data-count="${v}"${o.dec?` data-dec="${o.dec}"`:''}${o.suf?` data-suffix="${o.suf}"`:''}${o.sign?` data-sign="${o.sign}"`:''}${o.int?' data-int="1"':''}>0</div><div class="pl">${l}</div></div>`;
const archHTML=layers=>layers.map(L=>`<div class="alayer reveal" style="--ac:${L.c}"><div class="aname">⬡ ${L.n}</div><div class="adesc">${L.d}</div></div>`).join('<div class="aflow">▼</div>');
const STRAT_INFO={
  funding:{
    tag:'FUNDING HARVEST · DELTA-NEUTRAL · HYPERLIQUID',
    title:'Sentinel&nbsp;Edge <span class="tchip cy">Funding Harvest</span>',
    sub:'The all-weather book. It <b>collects the funding</b> that crowded longs pay — shorting the perps with stable positive funding while holding matching spot — so <b>price cancels</b> and only the premium remains. Market-neutral by construction: it earns in bull, bear <b>and</b> chop.',
    stats:[{v:17,suf:'%',sign:'+',l:'est. annual'},{v:2.5,dec:1,l:'Sharpe (realistic)'},{v:3,suf:'%',l:'max drawdown'},{v:0,dec:2,l:'BTC exposure'}],
    ideaTitle:'Get paid to provide leverage. Stay delta-neutral.',
    steps:stepHTML(1,'🛰️','Read funding','Every coin&rsquo;s perp charges <i>funding</i> — a fee longs pay shorts (or vice-versa) to keep the perp pinned to spot. It reads the trailing rate for the whole HL universe.')
        +stepHTML(2,'🌾','Short the payers, hedge with spot','It <b>shorts</b> the perps paying stable positive funding (collecting that fee) and holds a matching <b>long spot</b> — the two price legs cancel, leaving pure funding income.')
        +stepHTML(3,'🎚️','Harvest the stable, scale to the premium','Picks by funding <i>consistency</i> (not the biggest one-off spike), and deploys more when the premium is fat, less when it&rsquo;s thin — so it preserves capital when there&rsquo;s nothing to harvest.'),
    why:whyHTML('Where the income comes from','Perp funding is a structural payment from leveraged longs to the shorts who take the other side. Being the delta-neutral short harvests it — a fee for providing leverage, not a market bet.')
       +whyHTML('Why it&rsquo;s all-weather','With price hedged, <b>market direction doesn&rsquo;t matter</b> — BTC-β ≈ 0. It earned in 2024, 2025 <b>and</b> 2026, the years momentum books stalled.')
       +whyHTML('Why we trust this one','It <b>survived survivorship-honest testing</b>: on point-in-time data (coins liquid at each date, not today&rsquo;s winners) momentum collapsed +37% → +9.5%, but this held +28% → +20%. Delta-neutral harvesting doesn&rsquo;t depend on picking winning coins, so it isn&rsquo;t corrupted by survivorship. <b>Replaces the retired Grid</b> (a fill illusion).'),
    arch:archHTML([
      {c:'#4a9eff',n:'Signal',d:'Trailing funding mean + consistency (t-stat), per coin, across the wide HL universe'},
      {c:'#a78bfa',n:'Portfolio',d:'Short stable-+funding perps / long matching spot · delta-neutral · opportunity-scaled'},
      {c:'#ff7a8a',n:'Risk',d:'Market-neutral (β≈0) · only tail is a basis blowout · drawdown kill-switch'},
      {c:'#4be0b0',n:'Execution',d:'POST-only maker both legs → paper (simulated hedge) or live Hyperliquid + this dashboard'},
      {c:'#ffd166',n:'State &amp; Dashboard',d:'SQLite · funding history · equity curve · copy-trade signal'},
    ]),
    numNote:'point-in-time HL backtest (survivorship-honest) · funding + real basis · maker + slippage',
    perf:pcardHTML(17,'Est. annual','grn',{suf:'%',sign:'+'})+pcardHTML(2.5,'Sharpe (realistic)','',{dec:1})
        +pcardHTML(3,'Max drawdown','red',{suf:'%'})+pcardHTML(0,'BTC correlation','',{dec:2})
        +pcardHTML(3,'Green years / 3','',{})+pcardHTML(15,'Coins harvested','',{}),
    disclaim:'⚠ Delta-neutral funding harvest — the survivorship-robust edge (~+15-20%/yr point-in-time, positive every year 2024-26, ~0-3% DD). PAPER simulates the spot hedge and books funding + the real mark-vs-oracle basis; the shown Sharpe assumes a good hedge — real is ~2-3 once basis noise + the crash-blowout tail are modeled. Fully-delisted coins are absent from any backtest, and the premium can compress as it gets crowded. Live is Phase 2 (real spot legs). Prove the forward number in paper first. Not a guarantee.'
  },
  champion:{
    tag:'DIRECTIONAL MOMENTUM · REGIME-FILTERED · HYPERLIQUID FUTURES',
    title:'Sentinel&nbsp;Edge <span class="tchip">Champion</span>',
    sub:'The growth book. It rides the <b>strongest coins</b> while the market trends up — and steps fully to cash the moment Bitcoin turns down.',
    stats:[{v:0,dec:0,l:'live PnL — in cash'},{v:39,suf:'%',sign:'+',l:'backtest CAGR*'},{v:32,suf:'%',l:'max drawdown'},{v:100,suf:'%',l:'cash when BTC weak'}],
    ideaTitle:'Ride the strongest coins. Sit in cash when the market turns.',
    steps:stepHTML(1,'🛰️','Scan','Every day it reads daily price action on 24 Hyperliquid coins and measures each one&rsquo;s 30-day momentum.')
        +stepHTML(2,'📈','Rank','It ranks all 24 by 30-day momentum and holds the 5 strongest — weighted toward the leaders, capped at 40% in any one name.')
        +stepHTML(3,'🚦','Ride or Cash','It holds them while Bitcoin is above its 100-day line. The moment BTC drops below it, the whole book goes to <b>cash</b>.'),
    why:whyHTML('Why it works','Crypto&rsquo;s strongest trends keep running for weeks. Owning the current leaders captures that drift.')
       +whyHTML('Why it survives crashes','The BTC-regime brake pulls everything to cash in bear markets — where momentum&rsquo;s worst drawdowns happen.')
       +whyHTML('The edge word','<b>Regime-gated momentum</b> — validated walk-forward across 5.5 years, ~3&times; buy-and-hold&rsquo;s risk-adjusted return.'),
    arch:archHTML([
      {c:'#4a9eff',n:'Signal',d:'30-day momentum across 24 coins → one strength score per name'},
      {c:'#ffd166',n:'Regime',d:'BTC above its 100-day line → risk-on; below → everything to cash'},
      {c:'#a78bfa',n:'Portfolio',d:'Top-5 by momentum · momentum-weighted (≤40%/name) · long-only'},
      {c:'#ff7a8a',n:'Risk',d:'Vol-target · drawdown throttle · crash guard · per-name catastrophe stop'},
      {c:'#4be0b0',n:'Execution',d:'Reconcile → maker orders → paper or live Hyperliquid + this dashboard'},
    ]),
    numNote:'directional long-only · regime-gated (in cash ~45% of the time) · backtest survivorship-caveated',
    perf:pcardHTML(0,'Live PnL (in cash)','',{})+pcardHTML(39,'Backtest CAGR*','grn',{suf:'%',sign:'+'})
        +pcardHTML(32,'Max drawdown','red',{suf:'%'})+pcardHTML(45,'Time in cash','',{suf:'%'})
        +pcardHTML(1.12,'Backtest Sharpe*','',{dec:2})+pcardHTML(3,'vs buy &amp; hold','',{suf:'×'}),
    disclaim:'⚠ Currently IDLE — its BTC-100-day-MA gate has it 100% in cash right now (protecting capital; BTC is below the line). It only trades in confirmed bull regimes, where it&rsquo;s high-variance directional. * The backtest is survivorship-caveated like every crypto backtest (delisted coins absent, which flatters it) — the real number is lower. Not a guarantee.'
  },
  carry:{
    tag:'FUNDING CARRY · MARKET-NEUTRAL · HYPERLIQUID FUTURES',
    title:'Sentinel&nbsp;Edge <span class="tchip cy">Carry</span>',
    sub:'The income book. It <b>collects the funding</b> that crowded longs pay — shorting the priciest coins, longing the cheapest — and stays market-neutral.',
    stats:[{v:22,suf:'%',sign:'+',l:'honest CAGR'},{v:1.0,dec:1,l:'Sharpe (real)'},{v:28,suf:'%',l:'max drawdown'},{v:8,suf:'%',sign:'+',l:'live so far'}],
    ideaTitle:'Get paid to hold the spread. Stay market-neutral.',
    steps:stepHTML(1,'🛰️','Read funding','Every day it reads each coin&rsquo;s <i>funding rate</i> — the fee perp traders pay to hold a position — averaged over recent days.')
        +stepHTML(2,'💰','Short pricey, long cheap','It <b>shorts</b> the 5 coins paying the highest funding (it collects that fee) and <b>longs</b> the 5 paying the lowest/negative.')
        +stepHTML(3,'⚖️','Balance &amp; tilt','Equal dollars long and short (market-neutral), tilted by momentum so it&rsquo;s never short a coin that&rsquo;s ripping.'),
    why:whyHTML('Where the income comes from','Perp funding is a structural payment from crowded longs to shorts. Being on the paid side harvests it.')
       +whyHTML('Why it&rsquo;s a diversifier','Its returns are <b>uncorrelated</b> to the other books (corr ~0) and to Bitcoin — it earns in regimes where they stall.')
       +whyHTML('Honest status','It&rsquo;s <b>mostly a momentum book</b> with a funding tilt (funding alone is a weak edge). The old &ldquo;+123%/yr&rdquo; headline was ~5&times; too rosy — 2&times; gross + survivorship + one monster month. Honest through-cycle: <b>~+15&ndash;22%/yr, Sharpe ~1.0, lumpy</b> (big years AND dead ones, like 2025 at +3%). Live so far <b>+8%</b> — the best of our books.'),
    arch:archHTML([
      {c:'#4a9eff',n:'Signal',d:'Trailing-average funding rate + 21-day momentum, per coin'},
      {c:'#a78bfa',n:'Portfolio',d:'Short top-5 funding / long bottom-5 · momentum-tilted · dollar-neutral'},
      {c:'#ff7a8a',n:'Risk',d:'Vol-target · drawdown throttle · per-name catastrophe stop'},
      {c:'#4be0b0',n:'Execution',d:'Reconcile → maker orders → paper or live Hyperliquid + this dashboard'},
      {c:'#ffd166',n:'State &amp; Dashboard',d:'SQLite · funding history · equity curve · trades · copy-trade signal'},
    ]),
    numNote:'honest reconciliation · realistic fills + ~25% survivorship haircut · vs the inflated +123% headline',
    perf:pcardHTML(22,'Honest CAGR','grn',{suf:'%',sign:'+'})+pcardHTML(1.0,'Sharpe (real)','',{dec:1})
        +pcardHTML(28,'Max drawdown','red',{suf:'%'})+pcardHTML(8,'Live PnL','grn',{suf:'%',sign:'+'})
        +pcardHTML(0,'BTC correlation','',{dec:2})+pcardHTML(46,'Live trades','',{}),
    disclaim:'⚠ The headline &ldquo;+123%&rdquo; was survivorship-flattered + 2&times; gross + one huge month; the honest through-cycle number is <b>~+15&ndash;22%/yr at Sharpe ~1.0</b>, and it&rsquo;s lumpy (dead years happen). Crypto backtests over-state in both directions — the <b>live paper (+8% so far) is the number to trust</b>. Not a guarantee.'
  },
  trend:{
    tag:'TREND · CTA · HYPERLIQUID',
    title:'Sentinel&nbsp;Edge <span class="tchip cy">Trend</span>',
    sub:'A <b>dollar-neutral trend-following (CTA) book</b> — longs the coins in the strongest uptrends, shorts the strongest downtrends. The most durable systematic edge in finance.',
    stats:[{v:8,suf:'%',l:'live: down (1 trade)'},{v:12,suf:'%',l:'max drawdown'},{v:50,suf:'%',l:'live win rate'},{v:0.5,dec:1,l:'profit factor'}],
    ideaTitle:'Ride the strong. Fade the weak. Follow the trend.',
    steps:stepHTML(1,'📈','Measure the trend','For every coin, measures how far price sits above or below its 30-day moving average — the direction and strength of its trend.')
        +stepHTML(2,'🎯','Long strong / short weak','Longs the 8 coins in the strongest uptrends, shorts the 8 in the strongest downtrends — dollar-neutral, so it never bets on the market&rsquo;s overall direction.')
        +stepHTML(3,'🛡️','Throttle the risk','Vol-target + drawdown-throttle scale the book down in violent regimes, taming trend-following&rsquo;s naturally deep drawdowns.'),
    why:whyHTML('Why it&rsquo;s durable','Trend-following (managed futures / CTA) is the most-validated systematic edge in finance — it has worked across markets and decades. On 5 years of crypto data it was profitable every single year.')
       +whyHTML('The unique angle','Not a funding or momentum-rank book — pure cross-sectional trend. Long/short and dollar-neutral, so it earns whether the leaders rip up or the laggards bleed down.')
       +whyHTML('Honest status','The old &ldquo;+75%/yr&rdquo; was <b>survivorship fantasy</b> (backtests only see coins that survived and trended). Live, it&rsquo;s <b>&minus;7.5%</b> — but almost all of that is <b>one trade</b>: the ACE short-squeeze (&minus;$1,141). Strip ACE and it&rsquo;s ~breakeven. The current book is actually up (open +$520). The real trend edge is <b>modest and lumpy</b>, and the ACE-type tail is now capped by a per-name stop.'),
    arch:archHTML([
      {c:'#4a9eff',n:'Signal',d:'Price vs its 30-day moving average — trend strength per coin'},
      {c:'#a78bfa',n:'Portfolio',d:'Long top-8 uptrends / short top-8 downtrends · dollar-neutral'},
      {c:'#ff7a8a',n:'Risk',d:'Vol-target · drawdown throttle (tames the raw ~44% drawdown)'},
      {c:'#4be0b0',n:'Execution',d:'Reconcile → maker orders → paper or live Hyperliquid + this dashboard'},
      {c:'#ffd166',n:'State &amp; Dashboard',d:'SQLite · equity curve · trades · copy-trade signal'},
    ]),
    numNote:'live Hyperliquid paper · the +75% backtest was survivorship-inflated — trust the live number',
    perf:pcardHTML(8,'Live: down (ACE)','red',{suf:'%'})+pcardHTML(12,'Max drawdown','red',{suf:'%'})
        +pcardHTML(50,'Live win rate','',{suf:'%'})+pcardHTML(0.5,'Profit factor','',{dec:1})
        +pcardHTML(1141,'ACE loss ($)','red',{int:1})+pcardHTML(52,'Live trades','',{}),
    disclaim:'⚠ Trend-following IS a durable edge, but the &ldquo;+75%/yr&rdquo; backtest was survivorship-flattered — on point-in-time data momentum-style books drop hard (+37% &rarr; +9.5%). Live it&rsquo;s &minus;7.5%, essentially all from one ACE short-squeeze now capped by a per-name stop; the current book is up. Real expectation: <b>modest, lumpy, positive over time</b> once it earns the ACE hit back. Trust the live number, not the backtest. Not a guarantee.'
  }
};
const STRAT_PORT={funding:8787,champion:8788,carry:8789,trend:8790,consensus:8791};
const STRAT_ICON={funding:'🌾',champion:'⚡',carry:'💰',trend:'📈'};
const STRAT_LABEL={funding:'🌾 Funding Harvest',champion:'⚡ Momentum (Champion)',carry:'💰 Funding Carry',trend:'📈 Trend'};
const STRAT_SHORT={funding:'Funding Harvest',champion:'Champion',carry:'Carry',trend:'Trend'};
const STRAT_DOT={funding:'#ff8a5c',champion:'#ffd166',carry:'#4be0b0',trend:'#3f80ba'};
// The books actually running. Funding Harvest is RETIRED: it needs a spot leg to be
// delta-neutral, HL spot has only ~8 pairs over $1M/day so a 15-name basket cannot be hedged,
// and hedging a perp with a perp means paying funding to collect funding (measured: 16.0%/yr
// collected against 15.0%/yr paid, a 0.9%/yr spread). Its definitions are kept above so
// re-enabling is one entry here, but it must not appear in the nav while it is not trading.
const STRAT_LIVE=['champion','carry','trend','consensus'];
function stratUrl(k){const h=location.hostname||'localhost';return 'http://'+h+':'+STRAT_PORT[k];}
function switchTo(k){if(k!==CUR_STRAT) location.href=stratUrl(k);}
function toggleSwitcher(e){e.stopPropagation();$('navSwitcher').classList.toggle('open');}
document.addEventListener('click',()=>{ const s=$('navSwitcher'); if(s) s.classList.remove('open'); });
function buildSwitcher(strat){
  const menu=$('swMenu'); if(!menu) return;
  // include the current book even if retired, so a stale tab still shows where it is
  const items=STRAT_LIVE.includes(strat)?STRAT_LIVE:STRAT_LIVE.concat([strat]);
  menu.innerHTML=items.map(k=>`<button class="sw-item${k===strat?' sw-cur':''}" onclick="${k!==strat?`switchTo('${k}')`:''}">`+
    `<span class="sw-dot" style="background:${STRAT_DOT[k]}"></span>${STRAT_SHORT[k]}`+
    (k===strat?'<span class="sw-cur-tag">here</span>':'')+`</button>`).join('');
  const btn=$('swCurIcon'); if(btn) btn.textContent=STRAT_ICON[strat];
  const lbl=$('swCurLbl'); if(lbl) lbl.textContent=STRAT_SHORT[strat];
}
function paintStrategy(strat){
  if(!STRAT_INFO[strat]) strat='funding';
  if(STRAT_PAINTED && CUR_STRAT===strat) return;
  CUR_STRAT=strat; STRAT_PAINTED=true;
  buildSwitcher(strat);
  const S=STRAT_INFO[strat];
  $('heroTag').textContent=S.tag;
  $('heroTitle').innerHTML=S.title;
  $('heroSub').innerHTML=S.sub;
  $('heroStats').innerHTML=S.stats.map(s=>`<div class="hstat"><div class="hv" data-count="${s.v}"${s.dec?` data-dec="${s.dec}"`:''}${s.suf?` data-suffix="${s.suf}"`:''}${s.sign?` data-sign="${s.sign}"`:''}>0</div><div class="hl">${s.l}</div></div>`).join('');
  $('ideaTitle').textContent=S.ideaTitle;
  $('ideaSteps').innerHTML=S.steps;
  $('ideaWhy').innerHTML=S.why;
  $('archLayers').innerHTML=S.arch;
  $('numNote').textContent=S.numNote;
  $('perfGrid').innerHTML=S.perf;
  $('numDisclaim').innerHTML=S.disclaim;
  // comparison block: label + highlight the one you're viewing
  $('curStrat').textContent=STRAT_LABEL[strat];
  // Drive this off STRAT_LIVE and null-guard the lookup. Retiring a book removes its card, and an
  // unguarded getElementById would return null here and throw, taking the whole page render with it.
  STRAT_LIVE.forEach(k=>{
    const card=document.getElementById('vs-'+k);
    if(card) card.classList.toggle('cur',strat===k);
    const ln=$('link-'+k);
    if(ln){
      if(k===strat){ ln.textContent='Currently active ✓'; ln.className='vslink-cur'; ln.onclick=null; ln.removeAttribute('href'); }
      else{ ln.textContent='Switch to this →'; ln.className='vslink'; ln.onclick=()=>switchTo(k); ln.href='javascript:void(0)'; }
    }
  });
  if($('pageStrat').style.display!=='none') runReveals();
  if(stratBuilt) buildStrat();
}

// SCORE is an annualised % in the funding book and a z-score in every other book. Say which, and say
// what counts as a GOOD number — an unexplained figure on screen is worse than no figure.
const SCORE_TIP={
  funding:'FUNDING RATE — annualised %/yr. We hold the perp short against a hedge, so we collect this.\n'
    +'\nabove +10   rich · the best of the book'
    +'\n+3 to +10   normal · worth holding'
    +'\n0 to +3     thin · barely clears fees'
    +'\nbelow 0     we would be PAYING · dropped'
    +'\n\nHigher is simply better. Nothing is forecast here — it is the rate being paid right now.',
  other:'SIGNAL STRENGTH — standard deviations from the average coin. Re-scored every rebalance, so it is '
    +'always relative to today’s field, never a price target.\n'
    +'\n0        an average coin · no edge'
    +'\n±1       clearly stronger than the field'
    +'\n±2       top or bottom ~2% · high conviction'
    +'\nrange    about −3 to +3'
    +'\n\nSign follows the sleeve: LONGS come from the top of the ranking (positive), SHORTS from the bottom '
    +'(negative). A LONG showing a negative score has weakened and is likely to rotate out. That rotation is '
    +'the exit rule — positions are never closed at a fixed profit or loss.'};
function paintScoreTip(strat){
  const el=document.getElementById('scoreTip'); if(!el) return;
  el.setAttribute('data-tip', strat==='funding'?SCORE_TIP.funding:SCORE_TIP.other);
}

function card(k,v,cls,s,bar,tip){
  const info=tip?` <span class="info" data-tip="${tip}">i</span>`:'';
  return `<div class="card"><div class="k">${k}${info}</div><div class="v num ${cls||''}">${v}</div>${bar||''}${s?`<div class="s">${s}</div>`:''}</div>`}

function render(d){
  if(d.error){$('run').textContent='error: '+d.error;return;}
  DATA=d;
  $('mode').textContent=d.mode; $('mode').className='pill '+d.mode;
  if(d.strategy){ paintStrategy(d.strategy); paintScoreTip(d.strategy); }
  $('dot').className='dot '+(d.running?'on':'off');
  $('run').textContent=d.running?'running':'stopped'; $('run').className=d.running?'grn':'mut';
  $('upd').textContent=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',hour12:false});
  NEXT_REBAL=d.next_rebalance_ts||null; PAUSED=d.paused_until||null; REGIME=d.regime||null;
  renderCountdown(); renderRegime(d.regime);

  const pc=d.pnl>=0?'grn':'red';
  const lev=d.leverage||0, levPct=Math.min(100,lev/(d.max_gross_leverage||10)*100);
  const ln=d.long_notional||0, sn=d.short_notional||0, tot=ln+sn||1, lp=ln/tot*100, sp=sn/tot*100;
  const imb=(ln-sn)/tot;   // dollar-neutral book -> ~0; beta-hedge may run a small intentional imbalance,
  const bias=Math.abs(imb)<0.06?'Neutral ·':(imb>0?'Bullish ↑':'Bearish ↓');  // so neutral band sits above the 5% net rail
  const bc=Math.abs(imb)<0.06?'mut':(imb>0?'grn':'red');
  $('cards').innerHTML=[
    card('All-time PnL',sgn(d.pnl),pc,`<span class="${pc}">${d.pnl_pct>=0?'+':''}${d.pnl_pct.toFixed(2)}%</span> · equity ${money(d.equity)}`,'',
      'Total profit/loss since the account started — realized trades + open positions + funding. The % is vs your starting capital.'),
    card('Leverage',lev.toFixed(2)+'×','',`${money(d.gross)} in positions · from ${money(d.equity)} of your money`,
      `<div class="bar"><span style="width:${levPct}%;background:linear-gradient(90deg,var(--accent),var(--accent2))"></span></div>`,
      'This is your REAL leverage — total position value ÷ your money. ~1.5× is moderate. It drifts a little every tick as prices move (mark-to-market), and the bot targets ~2× and trims it in risky conditions. (Each individual trade is margined at 3× on the exchange so it only ties up a third of its value as a deposit — that is a margin mechanic, not your book leverage.)'),
    card('Direction bias',`<span class="${bc}">${bias}</span>`,'',
      `${lp.toFixed(0)}% long · ${sp.toFixed(0)}% short`+(d.book_beta!=null?` · BTC-β ${(d.book_beta>=0?'+':'')}${d.book_beta.toFixed(2)}`:''),
      `<div class="bar"><span style="width:${lp}%;background:linear-gradient(90deg,rgba(75,224,176,.6),var(--grn))"></span><span style="width:${sp}%;background:linear-gradient(90deg,var(--red),rgba(255,122,138,.6))"></span></div>`,
      'Long $ vs short $ split. A market-neutral book sits near 50/50 (Carry); a grid tilts long on dips and short on rips. BTC-β is the books net Bitcoin exposure — near 0 means it is largely immune to the overall market direction.'),
    card('Drawdown',d.drawdown_pct.toFixed(2)+'%',d.drawdown_pct>0?'red':'',`peak ${money(d.peak)} · fees ${money(d.fees)} · funding ${sgn(d.funding||0)}`,'',
      'How far equity has fallen from its highest point. A kill-switch flattens everything and pauses if this exceeds 15%.'),
  ].join('');

  // ---- Capital & Profit panel: booked vs open PnL, exposure, and the Vault ----
  // Booked = locked-in PnL. The paper broker already nets fees INSIDE realized_pnl (paper_broker:
  // `realized_pnl -= fee`), so subtracting them again double-counted the fees and made every
  // broker-based book look worse than it was — booked + open no longer added up to all-time PnL.
  // The funding book is the exception: it stores basis in realized and fees separately, so it does subtract.
  const booked=(d.strategy==='funding')
    ? (d.realized||0)+(d.funding||0)-(d.fees||0)
    : (d.realized||0)+(d.funding||0);
  const open=d.unrealized||0;                                 // still moving in open positions
  const grossN=d.gross||0, longN=d.long_notional||0, shortN=d.short_notional||0;
  const vault=d.vault||0, tradingEq=d.trading_equity||d.equity||0;
  // no info-tooltips here — the sub-text under each value already explains it (tooltips were
  // dropping down and covering the Performance panel below)
  const capBlock=(label,val,cls,sub)=>
    `<div class="capb"><div class="capk">${label}</div>`+
    `<div class="capv ${cls||''}">${val}</div><div class="caps">${sub}</div></div>`;
  let blocks=[
    capBlock('Booked profit',sgn(booked),booked>=0?'grn':'red','realized trades + funding − fees · locked in',
      'Profit you have actually locked in from closed trades and funding, minus fees. This can not be lost.'),
    capBlock('Open profit',sgn(open),open>=0?'grn':'red',`${d.n_positions||0} open positions · still moving`,
      'Unrealized profit/loss on your open positions. It changes every tick and is not locked in until the trades close.'),
    capBlock('In positions',money(d.margin_used||0),'',`your money holding the trades · ${money(grossN)} exposure (${(d.leverage||0).toFixed(2)}×)`),
    capBlock('💵 Safe to withdraw',money(d.safe_withdraw||0),(d.safe_withdraw>0?'grn':''),
      `keeps a safety cushion · max ${money(d.free_capital||0)} (but that leaves no buffer)`),
  ];
  if(d.treasury_enabled){   // only if the auto-sweep money-manager is turned on
    blocks.push(capBlock('🏦 Vault (banked)',money(vault),'grn',
      `${((d.sweep_frac||0.4)*100).toFixed(0)}% of profit swept weekly · safe to withdraw`));
  }
  $('capitalPanel').innerHTML=`<div class="caphead">Capital &amp; Profit</div><div class="capgrid">${blocks.join('')}</div>`;

  const b=d.book||[];
  $('np').textContent=b.length;
  $('pos').innerHTML=b.map((p,i)=>{
    const up=p.upnl, uc=up==null?'mut':(up>=0?'grn':'red');
    // return on the POSITION (uPnL / its entry value) — not a % of the account
    const pct=(up!=null&&p.entry_notional)?100*up/p.entry_notional:null;
    const scc=p.score>=0?'grn':'red';
    // funding: + = this position EARNS funding this interval, − = it PAYS
    let fcell='<span class="mut">—</span>';
    if(p.funding_pay!=null){
      const fp=p.funding_pay, fc=fp>=0?'grn':'red', amt=Math.abs(fp);
      const amtS=amt<0.01?amt.toFixed(4):amt.toFixed(2);
      fcell=`<span class="${fc}" title="${fp>=0?'earns':'pays'} ${(p.funding_rate>=0?'+':'')}${p.funding_rate}% per 8h">`
        +`${fp>=0?'+$':'−$'}${amtS}</span><span class="fmut">/8h</span>`;
    }
    return `<tr><td class="asset">${short(p.symbol)}</td>
      <td><span class="chip ${p.side}">${p.side}</span></td>
      <td class="r num mut">${(+p.contracts).toLocaleString()}</td>
      <td class="r num">${money(p.notional)}</td>
      <td class="r num mut" style="font-size:11.5px">${sig6(p.entry)}</td>
      <td class="r num" style="font-size:11.5px">${sig6(p.mark)}</td>
      <td class="r num ${uc}" style="font-weight:700">${up!=null?sgn(up):'—'}</td>
      <td class="r num ${uc}" style="font-weight:600">${pct!=null?(pct>=0?'+':'')+pct.toFixed(1)+'%':'—'}</td>
      <td class="r num" style="font-size:11.5px">${fcell}</td>
      <td class="r num mut" style="font-size:11px" title="${p.opened_ts?'held '+ago(p.opened_ts):''}">${dt(p.opened_ts)}</td>
      <td class="r"><span class="sc ${p.score!=null?scc:''}">${p.score!=null?(p.score>=0?'+':'')+(+p.score).toFixed(2):'—'}</span></td>
      <td class="r"><button class="share" title="Share" onclick="openCard(${i})">↗</button></td></tr>`;
  }).join('')||'<tr><td colspan="12" class="empty">no open positions</td></tr>';

  const w=d.whales||{}, cons=w.consensus||[];
  setWhaleTab(d.whales_enabled!==false);   // hide Smart Money on books that don't use whales (carry/champion)
  $('nw').textContent=w.n_wallets||0;
  $('consensus').innerHTML=cons.slice(0,12).map(c=>{
    const v=Math.max(-1,Math.min(1,c.bias)), mag=Math.abs(v)*50;
    const seg=v>=0?`<span class="f l" style="left:50%;width:${mag}%"></span>`:`<span class="f s" style="right:50%;width:${mag}%"></span>`;
    return `<div class="wrow"><span class="cn">${c.coin}</span>
      <span class="div"><span class="c"></span>${seg}</span>
      <span class="wpct ${v>=0?'grn':'red'}">${v>=0?'L':'S'} ${(Math.abs(v)*100).toFixed(0)}%</span></div>`;
  }).join('')||'<div class="empty">no whale data yet (populates next cycle)</div>';
  $('wallets').innerHTML=(w.wallets||[]).map(x=>{const net=x.net||0;
    return `<div class="w"><a href="https://hyperdash.com/address/${x.address}" target="_blank">${x.address.slice(0,6)}…${x.address.slice(-4)}</a>
      <span>${money(x.equity)} · <span class="${net>=0?'grn':'red'}">${net>=0?'net long':'net short'}</span></span></div>`;
  }).join('');

  const ts=d.trade_stats||{};
  $('tcount').textContent=ts.n||0;
  const pf=ts.profit_factor||0;
  $('tstrip').innerHTML=[
    ['Win rate',(ts.win_rate||0).toFixed(0)+'%',(ts.win_rate||0)>=50?'grn':'red'],
    ['W / L',`${ts.wins||0} / ${ts.losses||0}`,''],
    ['Avg win',sgn(ts.avg_win||0),'grn'],
    ['Avg loss',sgn(ts.avg_loss||0),'red'],
    ['Profit factor',pf?pf.toFixed(2):'—',pf>=1?'grn':'red'],
    ['Best',sgn(ts.max_win||0),'grn'],
    ['Worst',sgn(ts.max_loss||0),'red'],
    ['Total',sgn(ts.total_pnl||0),(ts.total_pnl||0)>=0?'grn':'red'],
  ].map(([k,v,c])=>`<div class="it"><b class="${c}">${v}</b><span>${k}</span></div>`).join('');
  const _at=document.querySelector('#tabs .tab.on'), _atid=_at?_at.dataset.t:'';
  if(_atid==='trades') pagerTrades.tick(); else if(_atid==='fills') pagerFills.tick();

  drawChart();
}

function drawChart(){
  if(!DATA) return;
  const now=Date.now(), span=RANGES[range];
  let s=(DATA.equity_series||[]).filter(r=>span===Infinity||r.ts*1000>=now-span);
  // long ranges track DAILY: collapse intraday ticks to one point/day (the day's closing equity)
  const daily=(range==='30D'||range==='ALL');
  if(daily){
    const byDay=new Map();
    for(const r of s){const d=new Date(r.ts*1000); byDay.set(d.getFullYear()+'-'+d.getMonth()+'-'+d.getDate(), r);}
    s=[...byDay.values()].sort((a,b)=>a.ts-b.ts);
  }
  const fmt=daily?{month:'short',day:'numeric'}:{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'};
  const labels=s.map(r=>new Date(r.ts*1000).toLocaleString([], fmt));
  const vals=s.map(r=>metric==='value'?r.equity:(r.equity-DATA.starting_capital));
  const last=vals.length?vals[vals.length-1]:0;
  $('chartval').innerHTML=metric==='value'?money(last):`<span class="${last>=0?'grn':'red'}">${sgn(last)}</span>`;
  const up=metric==='value'?true:last>=0;
  const line=up?'#4be0b0':'#ff7a8a';
  // graceful empty-state: a window needs >=2 points to draw a line. A single point (e.g. right after a
  // recording gap / outage, or a brand-new book) would render as a confusing blank grid.
  const ce=$('chartEmpty');
  if(s.length<2){
    const anyData=(DATA.equity_series||[]).length>0;
    ce.innerHTML=anyData
      ? '📈 Building history for the '+range+' view…<span class="ce-sub">Only one data point in this window so far — the equity curve had a gap (the loop was interrupted). It fills back in as new ticks record. Try a longer range for now.</span>'
      : '📈 No history yet<span class="ce-sub">This book hasn&rsquo;t recorded its first rebalance — check back shortly.</span>';
    ce.classList.add('on');
    if(chart){chart.data.labels=[];chart.data.datasets[0].data=[];chart.update('none');}
    return;
  }
  ce.classList.remove('on');
  const ctx=$('chart').getContext('2d');
  const grad=ctx.createLinearGradient(0,0,0,340);
  grad.addColorStop(0, up?'rgba(75,224,176,.30)':'rgba(255,122,138,.30)');
  grad.addColorStop(1,'rgba(0,0,0,0)');
  // grid: draw the backtest-seeded portion DASHED + greyed so it's never mistaken for live paper
  const BT=(DATA.backtest_until_ts||0), isBT=i=>s[i]&&s[i].ts<=BT;
  const seg=BT>0?{borderDash:c=>isBT(c.p1DataIndex)?[5,4]:undefined,
                  borderColor:c=>isBT(c.p1DataIndex)?'rgba(140,140,150,.75)':line}:undefined;
  const ds={data:vals,borderColor:line,backgroundColor:grad,fill:true,tension:.32,pointRadius:0,pointHoverRadius:0,borderWidth:2.4,segment:seg};
  if(!chart){
    chart=new Chart(ctx,{type:'line',data:{labels,datasets:[ds]},
      plugins:[crosshairPlugin],
      options:{responsive:true,animation:{duration:300},
        interaction:{mode:'index',intersect:false},
        plugins:{legend:{display:false},
          tooltip:{enabled:false,external:externalTip}},
        scales:{x:{ticks:{color:'rgba(52,46,36,.5)',maxTicksLimit:8,font:{size:11}},grid:{color:'rgba(60,50,30,.1)'}},
                y:{ticks:{color:'rgba(52,46,36,.5)',font:{size:11}},grid:{color:'rgba(60,50,30,.1)'}}}}});
  }else{chart.data.labels=labels;chart.data.datasets[0]=ds;chart.update('none');}
  const note=$('chartNote');
  if(note){const anyBT=BT>0&&s.some(r=>r.ts<=BT), anyLive=s.some(r=>r.ts>BT);
    note.innerHTML=(anyBT&&anyLive)?'&#9622; dashed = backtested &nbsp;·&nbsp; &#9644; solid = live paper'
                  :(anyBT?'&#9622; backtested history — live paper begins as new ticks record':'');
    note.style.display=anyBT?'':'none';}
}

// cinematic chart hover: glassmorphic floating value card that glides to each point
function externalTip(ctx){
  const tip=$('chartTip'); if(!tip) return;
  const tt=ctx.tooltip;
  if(!tt||tt.opacity===0||!tt.dataPoints||!tt.dataPoints.length){ tip.style.opacity=0; return; }
  const dp=tt.dataPoints[0], y=dp.parsed.y, isVal=metric==='value';
  const col=isVal?(y>=0?'#4be0b0':'#ff7a8a'):(y>=0?'#4be0b0':'#ff7a8a');
  tip.innerHTML=`<div class="ct-v" style="color:${col}">${isVal?money(y):sgn(y)}</div><div class="ct-t">${dp.label}</div>`;
  tip.style.opacity=1;
  const cw=ctx.chart.canvas, w=tip.offsetWidth, h=tip.offsetHeight;
  let left=Math.max(2,Math.min(tt.caretX - w/2, cw.clientWidth - w - 2));
  let top=tt.caretY - h - 18; if(top<2) top=tt.caretY + 18;
  tip.style.transform=`translate(${left}px,${top}px)`;
}
// glowing vertical crosshair + pulsing point at the hovered x
const crosshairPlugin={id:'crosshair', afterDatasetsDraw(chart){
  const act=chart.getActiveElements?chart.getActiveElements():[];
  if(!act||!act.length) return;
  const c=chart.ctx, el=act[0].element, x=el.x, py=el.y;
  const {top,bottom}=chart.chartArea, col=chart.data.datasets[0].borderColor;
  c.save();
  const g=c.createLinearGradient(0,top,0,bottom);
  g.addColorStop(0,'rgba(255,138,92,0)');g.addColorStop(.5,'rgba(255,138,92,.5)');g.addColorStop(1,'rgba(255,138,92,0)');
  c.strokeStyle=g;c.lineWidth=1;c.beginPath();c.moveTo(x,top);c.lineTo(x,bottom);c.stroke();
  c.shadowColor=col;c.shadowBlur=18;c.fillStyle=col;c.beginPath();c.arc(x,py,5,0,7);c.fill();
  c.shadowBlur=0;c.fillStyle='#190f3a';c.beginPath();c.arc(x,py,2.4,0,7);c.fill();
  c.fillStyle='#fff';c.beginPath();c.arc(x,py,1.3,0,7);c.fill();
  c.restore();
}};

document.querySelectorAll('#metricTog .t').forEach(t=>t.onclick=()=>{
  metric=t.dataset.m;document.querySelectorAll('#metricTog .t').forEach(x=>x.classList.toggle('on',x===t));drawChart();});
document.querySelectorAll('#rangeTog .t').forEach(t=>t.onclick=()=>{
  range=t.dataset.r;document.querySelectorAll('#rangeTog .t').forEach(x=>x.classList.toggle('on',x===t));drawChart();});
// ---- server-side paginated tables (Trades / Fills) ----
function pageList(page,pages){
  if(pages<=7) return Array.from({length:pages},(_,i)=>i+1);
  const out=[1];
  if(page>4) out.push('…');
  for(let i=Math.max(2,page-1);i<=Math.min(pages-1,page+1);i++) out.push(i);
  if(page<pages-3) out.push('…');
  out.push(pages); return out;
}
function tradeRow(t){const pc=t.pnl>=0?'grn':'red';
  // Return on the position, always measured against the ENTRY (what was actually at risk).
  // Shorts are the long return negated — dividing by the exit instead understated losing shorts:
  // a short from 0.049015 to 0.064537 is -31.7%, but entry/exit-1 reported it as only -24.1%.
  const raw=(t.entry>0&&t.exit>0)?(t.exit/t.entry-1):null;
  const mv=raw==null?null:100*(t.side==='SHORT'?-raw:raw);
  return `<tr><td class="asset">${short(t.symbol)}</td><td><span class="chip ${t.side}">${t.side}</span></td>
  <td class="r num mut">${sig6(t.entry)}</td><td class="r num">${sig6(t.exit)}</td>
  <td class="r num ${pc}">${sgn(t.pnl)}</td>
  <td class="r num ${mv==null?'mut':(mv>=0?'grn':'red')}" style="font-weight:600">${mv!=null?(mv>=0?'+':'')+mv.toFixed(1)+'%':'—'}</td>
  <td class="r num mut" style="font-size:11px">${dt(t.opened_ts)}</td>
  <td class="r num mut" style="font-size:11px">${dt(t.ts)}</td>
  <td class="r num mut">${dur(t.duration_s)}</td></tr>`;}
function fillRow(t){const c=t.side==='BUY'?'grn':'red';
  return `<tr><td class="asset">${short(t.symbol)}</td><td class="${c}">${t.side}</td><td class="mut">${t.open_close}</td>
  <td class="r num">${(+t.volume).toLocaleString()}</td><td class="r num">${t.price}</td><td class="r mut">${t.status}</td></tr>`;}
function makePager(endpoint,tbodyId,pagerId,renderRow,gname,cols){
  let offset=0,total=0,loaded=false; const size=25;
  async function load(){
    let r; try{ r=await (await fetch(endpoint+`?offset=${offset}&limit=${size}`)).json(); }catch(e){ return; }
    total=r.total||0; loaded=true;
    $(tbodyId).innerHTML=(r.rows&&r.rows.length)?r.rows.map(renderRow).join(''):`<tr><td colspan="${cols}" class="empty">nothing yet</td></tr>`;
    draw();
  }
  function draw(){
    const pages=Math.max(1,Math.ceil(total/size)),page=Math.floor(offset/size)+1;
    const from=total?offset+1:0,to=Math.min(offset+size,total);
    const b=(lbl,p,on,dis)=>`<button class="pgb ${on?'on':''}" ${dis?'disabled':''} onclick="${gname}.go(${p})">${lbl}</button>`;
    let h=`<span class="pgcount">${from.toLocaleString()}–${to.toLocaleString()} of ${total.toLocaleString()}</span><div class="ctrls">`;
    h+=b('«',1,false,page<=1)+b('‹',Math.max(1,page-1),false,page<=1);
    for(const n of pageList(page,pages)) h+= n==='…'?'<span class="dots">…</span>':b(n,n,n===page,false);
    h+=b('›',page+1,false,page>=pages)+b('»',pages,false,page>=pages);
    $(pagerId).innerHTML=h+'</div>';
  }
  return { go(p){const pages=Math.max(1,Math.ceil(total/size));offset=Math.max(0,Math.min(pages-1,p-1))*size;load();},
           ensure(){if(!loaded)load();}, tick(){if(offset===0)load();} };
}
window.pagerTrades=makePager('/api/trades','ctrd','pgTrades',tradeRow,'pagerTrades',9);
window.pagerFills =makePager('/api/fills','fillrows','pgFills',fillRow,'pagerFills',6);

document.querySelectorAll('#tabs .tab').forEach(t=>t.onclick=()=>{
  const id=t.dataset.t;
  document.querySelectorAll('#tabs .tab').forEach(x=>x.classList.toggle('on',x===t));
  document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='pane-'+id));
  if(id==='trades') pagerTrades.ensure(); else if(id==='fills') pagerFills.ensure();});

// Smart Money tab is only meaningful on books that run the whale overlay (neutral).
// Hide it where whales are disabled (carry / champion); fall back to Positions if it was active.
function setWhaleTab(enabled){
  const tab=document.querySelector('#tabs .tab[data-t="smart"]'), pane=$('pane-smart');
  if(!tab||!pane) return;
  tab.style.display=enabled?'':'none';
  if(!enabled && tab.classList.contains('on')){
    const pos=document.querySelector('#tabs .tab[data-t="positions"]'); if(pos) pos.click();
  }
}

// ---- shareable trade card — cinematic animated canvas ----
let cardIdx=0,cardVar=0,cardRaf=null,cardT0=null,cardStars=[];

function genStars(W,H,n){
  cardStars=[];
  for(let i=0;i<n;i++) cardStars.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.4+.2,
    speed:Math.random()*.18+.04,phase:Math.random()*Math.PI*2,drift:Math.random()*.3-.15});
}
function drawStars(c,W,H,t){   // soft daylight motes / bokeh (was night stars)
  c.save();
  cardStars.forEach(s=>{
    const tw=.16+.4*(.5+.5*Math.sin(t*1.3+s.phase));
    c.globalAlpha=tw; c.fillStyle='rgba(255,255,255,.85)';
    c.beginPath(); c.arc((s.x+s.drift*t*10)%W,(s.y+s.speed*t*8)%H,s.r*1.5,0,7); c.fill();
  });
  c.globalAlpha=1; c.restore();
}

const VARS=[
 {n:'Daylight',sw:'linear-gradient(180deg,#7cbfee,#bfe3d6,#f4e7c6)',d:(c,W,H,t)=>{
   let g=c.createLinearGradient(0,0,0,H);
   g.addColorStop(0,'#79bdec');g.addColorStop(.46,'#a9d9e6');g.addColorStop(.73,'#cbe8d3');g.addColorStop(1,'#f2e7ca');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   const sx=W*.82+Math.sin(t*.25)*8,sy=H*.22,sun=c.createRadialGradient(sx,sy,12,sx,sy,W*.55);
   sun.addColorStop(0,'rgba(255,241,205,.9)');sun.addColorStop(.45,'rgba(255,228,172,.2)');sun.addColorStop(1,'rgba(255,228,172,0)');
   c.fillStyle=sun;c.fillRect(0,0,W,H); drawStars(c,W,H,t);}},
 {n:'Hills',sw:'linear-gradient(180deg,#9fd3ea,#6fb86c)',d:(c,W,H,t)=>{
   let g=c.createLinearGradient(0,0,0,H);
   g.addColorStop(0,'#8fc9ec');g.addColorStop(.5,'#c2e2dc');g.addColorStop(1,'#e9efcb');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   const hill=(yb,amp,ph,col)=>{c.fillStyle=col;c.beginPath();c.moveTo(0,H);
     for(let x=0;x<=W;x+=18)c.lineTo(x,yb+Math.sin(x/W*4+ph+t*.05)*amp);
     c.lineTo(W,H);c.closePath();c.fill();};
   hill(H*.6,24,0,'rgba(159,206,147,.9)');hill(H*.71,30,1.7,'rgba(124,189,118,.92)');hill(H*.82,22,3.1,'rgba(95,168,97,.95)');
   drawStars(c,W,H,t);}},
 {n:'Sakura',sw:'linear-gradient(135deg,#f6cfe0,#fdeef4)',d:(c,W,H,t)=>{
   let g=c.createLinearGradient(0,0,W,H);
   g.addColorStop(0,'#f7d6e3');g.addColorStop(.5,'#fce7ef');g.addColorStop(1,'#fff3e9');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   c.save();cardStars.forEach((s,i)=>{const px=(s.x+s.drift*t*26+t*14)%W,py=(s.y+s.speed*t*38)%H;
     c.globalAlpha=.45;c.fillStyle=i%2?'#f1a6c2':'#f8c4d6';
     c.beginPath();c.ellipse(px,py,s.r*2.4,s.r*1.3,(px+py)/50,0,7);c.fill();});c.globalAlpha=1;c.restore();}},
 {n:'Sunrise',sw:'linear-gradient(135deg,#f0652a,#e9a93a,#f6ead0)',d:(c,W,H,t)=>{
   let g=c.createLinearGradient(0,0,W*.55,H);
   g.addColorStop(0,'#ef6a2c');g.addColorStop(.4,'#ef9a3a');g.addColorStop(.78,'#edc56a');g.addColorStop(1,'#f5e7cc');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   const sx=W*.3,sy=H*.46+Math.sin(t*.3)*6,sun=c.createRadialGradient(sx,sy,10,sx,sy,W*.45);
   sun.addColorStop(0,'rgba(255,247,224,.92)');sun.addColorStop(.5,'rgba(255,216,150,.22)');sun.addColorStop(1,'rgba(255,216,150,0)');
   c.fillStyle=sun;c.fillRect(0,0,W,H); drawStars(c,W,H,t);}},
 {n:'Washi',sw:'linear-gradient(135deg,#fffaf0,#efe2cc)',d:(c,W,H,t)=>{
   let g=c.createLinearGradient(0,0,W,H);g.addColorStop(0,'#fffaf0');g.addColorStop(1,'#efe3ce');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   c.strokeStyle='rgba(120,95,55,.05)';c.lineWidth=1;
   for(let x=0;x<W;x+=46){c.beginPath();c.moveTo(x,0);c.lineTo(x,H);c.stroke();}
   for(let y=0;y<H;y+=46){c.beginPath();c.moveTo(0,y);c.lineTo(W,y);c.stroke();}
   const sx=W*.8,sy=H*.25,sun=c.createRadialGradient(sx,sy,10,sx,sy,W*.5);
   sun.addColorStop(0,'rgba(240,101,42,.08)');sun.addColorStop(1,'rgba(240,101,42,0)');
   c.fillStyle=sun;c.fillRect(0,0,W,H); drawStars(c,W,H,t);}},
];

function roiPct(p){if(p.entry_notional&&p.leverage){const m=p.entry_notional/p.leverage;if(m>0)return p.upnl/m*100;}return 0;}
function sizeStr(p){const u=p.base_size!=null?p.base_size:Math.abs(p.contracts);
  return u.toLocaleString(undefined,{maximumFractionDigits:4})+' '+short(p.symbol);}

function easeOut(t){return 1-Math.pow(1-t,3);}
function easeSpring(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}

function drawCardFrame(ts){
  if(!cardT0) cardT0=ts;
  const elapsed=(ts-cardT0)/1000;           // seconds since open
  const p=DATA.book[cardIdx]; if(!p){cardRaf=requestAnimationFrame(drawCardFrame);return;}
  const cv=$('cardCanvas'),W=1200,H=675,c=cv.getContext('2d');
  if(cv.width!==W){cv.width=W;cv.height=H;}

  // ---- background (animated) ----
  VARS[cardVar].d(c,W,H,elapsed);

  // ---- cinematic scan-line reveal (first 0.7s) ----
  const scanProgress=Math.min(1,elapsed/.65);
  if(scanProgress<1){
    const sx=W*easeOut(scanProgress);
    // wash unseen area (light)
    c.fillStyle=`rgba(255,253,247,${.6*(1-scanProgress)})`;c.fillRect(sx,0,W-sx,H);
    // glowing scan beam
    const sg=c.createLinearGradient(sx-90,0,sx+6,0);
    sg.addColorStop(0,'rgba(255,255,255,0)');sg.addColorStop(.7,'rgba(255,247,235,.3)');
    sg.addColorStop(1,'rgba(255,255,255,.6)');
    c.fillStyle=sg;c.fillRect(0,0,W,H);
    c.strokeStyle='rgba(240,101,42,.6)';c.lineWidth=1.5;
    c.beginPath();c.moveTo(sx,0);c.lineTo(sx,H);c.stroke();
  }

  // ---- light scrim so dark text stays legible on any variant (left + bottom) ----
  let ov=c.createLinearGradient(0,0,W,0);
  ov.addColorStop(0,'rgba(255,253,247,.62)');ov.addColorStop(.55,'rgba(255,253,247,.14)');ov.addColorStop(1,'rgba(255,253,247,.34)');
  c.fillStyle=ov;c.fillRect(0,0,W,H);
  c.fillStyle='rgba(255,253,247,.5)';c.fillRect(0,H-176,W,176);

  c.textBaseline='alphabetic';c.textAlign='left';

  // ---- header (fades in 0-0.25s) ----
  const hA=Math.min(1,elapsed/.25);
  c.globalAlpha=hA;
  c.fillStyle='#2d2922';c.font='800 27px ui-sans-serif,system-ui,Arial';c.fillText('⚡ SENTINEL EDGE',58,74);
  c.fillStyle='rgba(45,41,34,.58)';c.font='600 17px ui-sans-serif,system-ui,Arial';c.textAlign='right';
  const _ts=p.opened_ts?('OPENED '+new Date(p.opened_ts*1000).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})):new Date().toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  c.fillText(_ts+' · '+(DATA.mode||'paper').toUpperCase(),W-58,72);
  c.textAlign='left';c.globalAlpha=1;

  // ---- asset + side (slides in 0.15-0.5s) ----
  const assA=Math.min(1,Math.max(0,(elapsed-.15)/.35));
  const assOff=(1-easeOut(assA))*40;
  c.globalAlpha=assA;
  const col=p.side==='LONG'?'#179a55':'#e1512c';
  c.fillStyle='#2d2922';c.font='800 58px ui-sans-serif,system-ui,Arial';
  const asset=short(p.symbol);c.fillText(asset,58,216-assOff);
  const aw=c.measureText(asset).width;
  c.fillStyle=col;c.font='800 30px ui-sans-serif,system-ui,Arial';
  c.fillText(`${p.side} ${p.leverage||1}X`,58+aw+24,216-assOff);
  c.globalAlpha=1;

  // ---- PnL counter + glow (0.3-0.9s counts up; then pulses) ----
  const pnl=p.upnl||0,roi=roiPct(p),pc=pnl>=0?'#179a55':'#e1512c';
  const pnlA=Math.min(1,Math.max(0,(elapsed-.25)/.65));
  const pnlVal=pnl*easeSpring(pnlA);  // count up with spring
  const pnlS=(pnlVal>=0?'+$':'−$')+Math.abs(pnlVal).toLocaleString(undefined,{maximumFractionDigits:2,minimumFractionDigits:2});
  // subtle soft drop-shadow for depth (classy — not a neon glow)
  c.shadowColor='rgba(55,42,28,.2)';c.shadowBlur=5;c.shadowOffsetY=2;
  c.fillStyle=pc;c.globalAlpha=Math.min(1,pnlA+.05);
  c.font='800 80px ui-sans-serif,system-ui,Arial';c.fillText(pnlS,58,322);
  const bigW=c.measureText(pnlS).width;
  c.font='800 36px ui-sans-serif,system-ui,Arial';
  const roiS=`(${roi>=0?'+':''}${roi.toFixed(2)}%)`;c.fillText(roiS,58+bigW+20,322);
  c.shadowBlur=0;c.shadowOffsetY=0;c.globalAlpha=1;

  // ---- bottom rows (fade in 0.55-0.85s) ----
  const rowA=Math.min(1,Math.max(0,(elapsed-.5)/.35));
  c.globalAlpha=rowA;
  const rows=[['SIZE',sizeStr(p)],['ENTRY','$'+sig6(p.entry)],['MARK','$'+sig6(p.mark)]];
  c.font='600 22px ui-sans-serif,system-ui,Arial';let y=H-120;
  rows.forEach(([k,v])=>{
    c.fillStyle='rgba(45,41,34,.55)';c.textAlign='left';c.fillText(k,58,y);
    c.fillStyle='#2d2922';c.textAlign='right';c.fillText(v,W-58,y);
    c.strokeStyle='rgba(45,41,34,.14)';c.beginPath();c.moveTo(58,y+15);c.lineTo(W-58,y+15);c.stroke();y+=44;
  });
  c.textAlign='left';c.globalAlpha=1;

  cardRaf=requestAnimationFrame(drawCardFrame);
}

function stopCardAnim(){if(cardRaf){cancelAnimationFrame(cardRaf);cardRaf=null;}}
function renderSwatches(){$('swatches').innerHTML=VARS.map((v,i)=>`<button class="sw ${i===cardVar?'on':''}" style="background:${v.sw}" title="${v.n}" onclick="pickVar(${i})"></button>`).join('');}
function openCard(i){
  stopCardAnim();cardIdx=i;cardVar=0;cardT0=null;
  genStars(1200,675,220);
  $('cardModal').style.display='flex';renderSwatches();
  cardRaf=requestAnimationFrame(drawCardFrame);
}
function closeCard(){stopCardAnim();$('cardModal').style.display='none';}
function pickVar(i){cardVar=i;cardT0=null;renderSwatches();}   // restart reveal on theme switch
function downloadCard(){
  stopCardAnim();
  const p=DATA.book[cardIdx],cv=$('cardCanvas');
  // draw one clean static frame for export
  const W=1200,H=675,c=cv.getContext('2d');cv.width=W;cv.height=H;
  VARS[cardVar].d(c,W,H,2.0);
  let ov=c.createLinearGradient(0,0,W,0);ov.addColorStop(0,'rgba(255,253,247,.62)');ov.addColorStop(.55,'rgba(255,253,247,.14)');ov.addColorStop(1,'rgba(255,253,247,.34)');
  c.fillStyle=ov;c.fillRect(0,0,W,H);c.fillStyle='rgba(255,253,247,.5)';c.fillRect(0,H-176,W,176);
  c.textBaseline='alphabetic';c.textAlign='left';
  c.fillStyle='#2d2922';c.font='800 27px ui-sans-serif,system-ui,Arial';c.fillText('⚡ SENTINEL EDGE',58,74);
  c.fillStyle='rgba(45,41,34,.58)';c.font='600 17px ui-sans-serif,system-ui,Arial';c.textAlign='right';
  const _ts=p.opened_ts?('OPENED '+new Date(p.opened_ts*1000).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})):new Date().toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  c.fillText(_ts+' · '+(DATA.mode||'paper').toUpperCase(),W-58,72);
  c.textAlign='left';
  const col=p.side==='LONG'?'#179a55':'#e1512c';
  c.fillStyle='#2d2922';c.font='800 58px ui-sans-serif,system-ui,Arial';const asset=short(p.symbol);c.fillText(asset,58,216);
  const aw=c.measureText(asset).width;c.fillStyle=col;c.font='800 30px ui-sans-serif,system-ui,Arial';
  c.fillText(`${p.side} ${p.leverage||1}X`,58+aw+24,216);
  const pnl=p.upnl||0,roi=roiPct(p),pc=pnl>=0?'#179a55':'#e1512c';
  const pnlS=(pnl>=0?'+$':'−$')+Math.abs(pnl).toLocaleString(undefined,{maximumFractionDigits:2,minimumFractionDigits:2});
  c.shadowColor='rgba(55,42,28,.2)';c.shadowBlur=5;c.shadowOffsetY=2;c.fillStyle=pc;c.font='800 80px ui-sans-serif,system-ui,Arial';c.fillText(pnlS,58,322);
  const bigW=c.measureText(pnlS).width;c.font='800 36px ui-sans-serif,system-ui,Arial';
  c.fillText(`(${roi>=0?'+':''}${roi.toFixed(2)}%)`,58+bigW+20,322);
  c.shadowBlur=0;c.shadowOffsetY=0;
  const rows=[['SIZE',sizeStr(p)],['ENTRY','$'+sig6(p.entry)],['MARK','$'+sig6(p.mark)]];
  c.font='600 22px ui-sans-serif,system-ui,Arial';let y=H-120;
  rows.forEach(([k,v])=>{c.fillStyle='rgba(45,41,34,.55)';c.textAlign='left';c.fillText(k,58,y);
    c.fillStyle='#2d2922';c.textAlign='right';c.fillText(v,W-58,y);
    c.strokeStyle='rgba(45,41,34,.14)';c.beginPath();c.moveTo(58,y+15);c.lineTo(W-58,y+15);c.stroke();y+=44;});
  cv.toBlob(b=>{const a=document.createElement('a');a.href=URL.createObjectURL(b);
    a.download=`sentinel-${short(p.symbol)}-${p.side}.png`;a.click();
    // restart live animation
    cardT0=null;cardRaf=requestAnimationFrame(drawCardFrame);});
}
function copyCard(){try{$('cardCanvas').toBlob(async b=>{try{await navigator.clipboard.write([new ClipboardItem({'image/png':b})]);}catch(e){downloadCard();}});}catch(e){downloadCard();}}
$('cardModal').addEventListener('click',e=>{if(e.target.id==='cardModal')closeCard();});

async function tick(){try{render(await (await fetch('/api/state')).json());}catch(e){$('run').textContent='offline';}}
tick();setInterval(tick,5000);

// ---- live rebalance countdown (ticks every second) ----
let NEXT_REBAL=null, PAUSED=null, REGIME=null;
function renderCountdown(){
  const el=$('cd'), wrap=$('cdwrap'), lbl=$('cdlbl'); if(!el) return;
  const now=Date.now()/1000;
  wrap.classList.remove('soon','paused');
  // champion re-checks daily but only TRADES when risk-on — relabel so the timer isn't misleading
  if(lbl){
    if(REGIME) { lbl.textContent = REGIME.on ? 'next rebal' : 'next check';
      wrap.title = REGIME.on ? 'time until the next daily rebalance'
        : 'Re-checks daily, but stays in cash until BTC reclaims its '+REGIME.ma_period+'-day line (see Regime below)'; }
    else lbl.textContent='next rebal';
  }
  if(PAUSED && PAUSED>now){ el.textContent='paused'; wrap.classList.add('paused'); return; }
  if(!NEXT_REBAL){ el.textContent='—'; return; }
  let s=Math.max(0,Math.floor(NEXT_REBAL-now));
  if(s===0){ el.textContent='now…'; wrap.classList.add('soon'); return; }
  const h=Math.floor(s/3600), m=Math.floor(s%3600/60), sec=s%60;
  const p=n=>String(n).padStart(2,'0');
  el.textContent=(h>0?h+':':'')+p(m)+':'+p(sec);
  if(s<3600) wrap.classList.add('soon');   // amber in the final hour
}
setInterval(renderCountdown,1000);

/* ---- CHAMPION: market-regime gauge (why it's in cash / how close to flipping) ---- */
function rgMoney(n){return '$'+Number(n).toLocaleString(undefined,{maximumFractionDigits:n<100?2:0});}
function regimeSpark(series){
  if(!series||series.length<2) return '';
  const W=240,H=46,n=series.length;
  const all=series.flatMap(p=>[p.c,p.ma]); const lo=Math.min(...all),hi=Math.max(...all),rng=(hi-lo)||1;
  const x=i=>i/(n-1)*W, y=v=>H-3-((v-lo)/rng)*(H-6);
  const path=arr=>arr.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
  const last=series[n-1], above=last.c>=last.ma, col=above?'#4be0b0':'#ff7a8a';
  const cx=x(n-1).toFixed(1), cy=y(last.c).toFixed(1);
  return `<svg class="rg-spk" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <defs><filter id="rgglow"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <path d="${path(series.map(p=>p.ma))}" fill="none" stroke="#ffd166" stroke-width="1.1" stroke-dasharray="3 3" opacity=".7"/>
    <path class="rg-spk-line" pathLength="1" d="${path(series.map(p=>p.c))}" fill="none" stroke="${col}" stroke-width="1.8" filter="url(#rgglow)"/>
    <circle class="rg-spk-tip" cx="${cx}" cy="${cy}" r="2.8" fill="${col}"/>
    <circle class="rg-spk-ping" cx="${cx}" cy="${cy}" r="2.8" fill="none" stroke="${col}"/></svg>`;
}
let RG_BUILT=false, RG_SIG=null;
function buildRegimeSkeleton(wrap){
  wrap.innerHTML=`<div class="regime" id="rgCard">
    <div class="rg-scan"></div>
    <div class="rg-top">
      <div class="rg-head"><div class="rg-state" id="rgState"></div><div class="rg-sub" id="rgSub"></div></div>
      <div class="rg-spark-wrap" id="rgSpark"></div>
    </div>
    <div class="rg-gauge">
      <div class="rg-track"><div class="rg-fill" id="rgFill"></div>
        <div class="rg-trig"></div><div class="rg-mk" id="rgMk"><span class="rg-mk-core"></span></div></div>
      <div class="rg-scale"><span>← cash</span><span class="rg-trig-lbl" id="rgTrigLbl"></span><span>invested →</span></div>
    </div>
    <div class="rg-read"><span>BTC <b id="rgPrice">—</b></span>
      <span class="rg-vs" id="rgVs"></span><span class="rg-need" id="rgNeed"></span></div>
    <div class="rg-tip" id="rgTip"></div>
  </div>`;
}
function renderRegime(r){
  const wrap=$('regimeWrap'); if(!wrap) return;
  if(!r){ wrap.style.display='none'; RG_BUILT=false; return; }
  wrap.style.display='';
  if(!RG_BUILT){ buildRegimeSkeleton(wrap); RG_BUILT=true; RG_SIG=null; }
  const on=r.on, dist=r.dist_pct, cross=r.to_cross_pct, P=r.ma_period, cls=on?'on':'off';
  const span=0.40, pos=Math.max(3,Math.min(97,((r.price/r.ma-1)/span*0.5+0.5)*100));
  const prox=Math.max(0,Math.min(1,1-Math.abs(cross)/20));   // 0 far → 1 at the line: drives pulse urgency
  const card=$('rgCard'); card.className='regime '+cls;
  card.style.setProperty('--prox', prox.toFixed(3));
  // glide the marker + fill (CSS transitions animate left/width → smooth slide as BTC moves)
  const mk=$('rgMk'); mk.className='rg-mk '+cls; mk.style.left=pos+'%';
  mk.style.animationDuration=(2.6-1.9*prox).toFixed(2)+'s';   // pulses faster the closer BTC gets
  const fill=$('rgFill'); fill.className='rg-fill '+cls; fill.style.width=pos+'%';
  $('rgState').innerHTML='<span class="rg-dot"></span>'+(on?'RISK-ON':'RISK-OFF');
  $('rgSub').textContent=on?'deployed · holding the strongest names':'holding 100% cash';
  $('rgPrice').textContent=rgMoney(r.price);
  const vs=$('rgVs'); vs.className='rg-vs '+(on?'grn':'red'); vs.textContent=(on?'+':'')+dist.toFixed(1)+'% vs line';
  const need=$('rgNeed'); need.style.display=on?'none':''; need.innerHTML=on?'':('needs <b>+'+cross.toFixed(1)+'%</b> to flip risk-on');
  $('rgTrigLbl').textContent=P+'-day line · '+rgMoney(r.ma);
  let tip;
  if(on) tip=`✅ BTC is ${Math.abs(dist).toFixed(1)}% above its ${P}-day line — trend intact, riding the leaders.`;
  else if(cross<=3) tip=`👀 Knocking on the door — BTC is just ${cross.toFixed(1)}% under its ${P}-day line. A flip to risk-on could be near.`;
  else if(cross<=10) tip=`Climbing back — BTC ${cross.toFixed(1)}% under its ${P}-day line. Getting warmer.`;
  else tip=`Deep risk-off — BTC ${cross.toFixed(1)}% under its ${P}-day line. The cash wait is the edge; patience pays.`;
  $('rgTip').textContent=tip;
  // sparkline only redraws when the daily series actually changes (~5min) → its draw-in animation replays then
  const sig=(r.series&&r.series.length)?(r.series.length+':'+r.series[r.series.length-1].c+':'+r.series[0].c):'';
  if(sig!==RG_SIG){ $('rgSpark').innerHTML=regimeSpark(r.series); RG_SIG=sig; }
}

/* ============ STRATEGY PAGE: nav, particles, reveals, counters, flow ============ */
let stratBuilt=false, alienRaf=null, alienParts=[];
function goPage(p){
  const dash=p==='dash';
  $('pageDash').style.display=dash?'':'none';
  $('pageStrat').style.display=dash?'none':'block';
  document.querySelectorAll('.navbtn').forEach(b=>b.classList.toggle('on',b.dataset.page===p));
  $('alienBg').classList.toggle('on',!dash);
  window.scrollTo({top:0,behavior:'instant'});
  if(!dash){ if(!STRAT_PAINTED) paintStrategy((DATA&&DATA.strategy)||'funding');
    startAlien(); if(!stratBuilt){buildStrat();stratBuilt=true;} runReveals(); }
  else stopAlien();
}

/* --- alien particle field (canvas) --- */
function startAlien(){
  return;   // particle field removed by request
  const cv=$('alienBg'); function size(){cv.width=innerWidth;cv.height=innerHeight;}
  size(); if(!alienParts.length){for(let i=0;i<70;i++)alienParts.push({
    x:Math.random()*cv.width,y:Math.random()*cv.height,vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25,r:Math.random()*1.6+.4});}
  cv._sz=size; addEventListener('resize',size);
  const c=cv.getContext('2d');
  function frame(){
    c.clearRect(0,0,cv.width,cv.height);
    for(const p of alienParts){
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=cv.width;if(p.x>cv.width)p.x=0;if(p.y<0)p.y=cv.height;if(p.y>cv.height)p.y=0;
    }
    // links
    for(let i=0;i<alienParts.length;i++)for(let j=i+1;j<alienParts.length;j++){
      const a=alienParts[i],b=alienParts[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.hypot(dx,dy);
      if(d<140){c.strokeStyle=`rgba(255,138,92,${.10*(1-d/140)})`;c.lineWidth=1;
        c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.stroke();}
    }
    for(const p of alienParts){c.fillStyle='rgba(154,123,255,.55)';c.beginPath();c.arc(p.x,p.y,p.r,0,7);c.fill();}
    alienRaf=requestAnimationFrame(frame);
  }
  if(!alienRaf)frame();
}
function stopAlien(){if(alienRaf){cancelAnimationFrame(alienRaf);alienRaf=null;}}

/* --- scroll reveal + number count-up --- */
const revObserver=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){
  e.target.classList.add('in');
  e.target.querySelectorAll('[data-count]').forEach(countUp);
  if(e.target.matches('[data-count]'))countUp(e.target);
  revObserver.unobserve(e.target);
}});},{threshold:.18});
function runReveals(){document.querySelectorAll('#pageStrat .reveal,#pageStrat [data-count]').forEach(el=>revObserver.observe(el));}
function countUp(el){
  if(el._done)return; el._done=true;
  const target=parseFloat(el.dataset.count),dec=+(el.dataset.dec||0),isInt=el.dataset.int,
    suf=el.dataset.suffix||'',sign=el.dataset.sign||'';
  const dur=1100,t0=performance.now();
  function step(t){const k=Math.min(1,(t-t0)/dur),e=1-Math.pow(1-k,3),v=target*e;
    el.textContent=sign+(isInt?Math.round(v).toLocaleString():v.toFixed(dec))+suf;
    if(k<1)requestAnimationFrame(step);}
  requestAnimationFrame(step);
}

/* --- animated flow diagram (SVG) --- */
function buildStrat(){
  const svg=$('flowSvg'),NS='http://www.w3.org/2000/svg';
  let nodes,links;
  if(CUR_STRAT==='champion'){
    nodes=[
      {x:90,y:110,t:'Hyperliquid',s:'daily data',c:'#4a9eff'},
      {x:90,y:270,t:'BTC Regime',s:'above 100d MA?',c:'#ffd166'},
      {x:360,y:110,t:'Momentum',s:'rank 30d',c:'#9a7bff'},
      {x:620,y:180,t:'Top-5 or Cash',s:'dual-confirmed',c:'#a78bfa'},
      {x:860,y:180,t:'Execute',s:'maker orders',c:'#4be0b0'},
    ];
    links=[[0,2],[2,3],[1,3],[3,4]];
  }else if(CUR_STRAT==='carry'){
    nodes=[
      {x:90,y:110,t:'Funding',s:'trailing avg',c:'#4a9eff'},
      {x:90,y:270,t:'Momentum',s:'21-day',c:'#4a9eff'},
      {x:360,y:180,t:'Combined Rank',s:'funding + trend',c:'#9a7bff'},
      {x:610,y:90,t:'Short high-fund',s:'collect funding',c:'#ff7a8a'},
      {x:610,y:270,t:'Long low-fund',s:'cheap to hold',c:'#4be0b0'},
      {x:860,y:180,t:'Execute',s:'dollar-neutral',c:'#4be0b0'},
    ];
    links=[[0,2],[1,2],[2,3],[2,4],[3,5],[4,5]];
  }else{
    nodes=[
      {x:90,y:60,t:'Hyperliquid',s:'live data',c:'#4a9eff'},
      {x:90,y:180,t:'Whales',s:'10 wallets',c:'#4a9eff'},
      {x:90,y:300,t:'Funding',s:'crowding',c:'#4a9eff'},
      {x:340,y:180,t:'Residual Score',s:'beta-stripped',c:'#9a7bff'},
      {x:600,y:180,t:'Rank & Build',s:'long 5 / short 5',c:'#a78bfa'},
      {x:840,y:90,t:'Risk Gate',s:'guards + stops',c:'#ff7a8a'},
      {x:840,y:270,t:'Execute',s:'maker orders',c:'#4be0b0'},
    ];
    links=[[0,3],[1,3],[2,3],[3,4],[4,5],[4,6]];
  }
  let defs=`<defs><filter id="glow"><feGaussianBlur stdDeviation="3.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  let paths='',dots='',boxes='';
  links.forEach(([a,b],i)=>{const A=nodes[a],B=nodes[b];
    const mx=(A.x+B.x)/2;
    const d=`M ${A.x+58} ${A.y} C ${mx} ${A.y}, ${mx} ${B.y}, ${B.x-58} ${B.y}`;
    paths+=`<path d="${d}" fill="none" stroke="rgba(255,138,92,.28)" stroke-width="1.6"/>
      <path d="${d}" fill="none" stroke="${B.c}" stroke-width="1.8" stroke-dasharray="7 220" filter="url(#glow)">
        <animate attributeName="stroke-dashoffset" from="227" to="0" dur="${1.6+i*.12}s" repeatCount="indefinite"/></path>`;
  });
  nodes.forEach((n)=>{
    boxes+=`<g transform="translate(${n.x-58},${n.y-26})">
      <rect width="116" height="52" rx="12" fill="rgba(255,255,253,.95)" stroke="${n.c}" stroke-width="1.3"/>
      <rect width="116" height="52" rx="12" fill="${n.c}" opacity=".08"/>
      <text x="58" y="22" text-anchor="middle" fill="#2d2922" font-size="13" font-weight="700" font-family="ui-sans-serif,system-ui">${n.t}</text>
      <text x="58" y="38" text-anchor="middle" fill="#665d4c" font-size="10" font-family="ui-sans-serif,system-ui">${n.s}</text>
      <circle cx="58" cy="-1" r="2.5" fill="${n.c}"><animate attributeName="opacity" values=".3;1;.3" dur="2.4s" repeatCount="indefinite"/></circle>
    </g>`;
  });
  svg.innerHTML=defs+paths+boxes+dots;
}
</script>
</body>
</html>"""
