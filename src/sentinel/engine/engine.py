"""Engine: wires the components together and runs one rebalance cycle.

Pipeline per cycle:
  universe -> market data -> funding filter -> scores -> equity -> drawdown guard
  -> dollar-neutral target book -> leverage safety scale -> reconcile vs live positions
  -> (live: per-contract setup) -> execute -> persist equity/trades/state.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..exchange.client import KoinbayClient
from ..exchange.contracts import ContractRegistry
from ..exchange.futures import KoinbayFutures
from ..exchange.factory import make_futures
from ..execution.book_probe import post_outcome_bps, probe
from ..execution.exec_policy import ExecutionPolicy
from ..execution.live_broker import LiveBroker
from ..execution.paper_broker import PaperBroker, PaperPosition
from ..execution.reconciler import reconcile
from ..risk.limits import (crash_guard, dispersion_scale, drawdown_pct, filter_by_funding,
                           leverage_scale, net_exposure_ratio, position_loss_breached,
                           position_stop_breached, risk_scale)
from ..signal.market_proxy import MarketProxySignal
from ..signal.technicals import ta_consensus
from ..signal.whale_tracker import WhaleTracker, map_bias_to_symbols
from ..state.store import Store
from ..strategy import grid_book
from ..strategy.funding_carry import FundingCarryStrategy
from ..strategy.consensus import ConsensusStrategy
from ..strategy.trend import TrendStrategy
from ..portfolio.treasury import Treasury
from ..strategy.momentum_regime import MomentumRegimeStrategy, regime_on
from ..strategy.portfolio import beta_scales, book_beta, compute_betas, demean_by_beta, dispersion
from ..strategy.sentiment_edge import SentimentEdgeStrategy
from ..strategy.sizing import adv_caps, net_scales
from ..util.concurrency import pmap
from ..util.mathx import parse_duration_to_hours, realized_vol
from .marketdata import build_universe, filter_by_age, load_symbol_data, mark_price, pick_base_interval

log = logging.getLogger("sentinel")

# How often the intraday regime brake may be re-read. The monitor tick is every 60s, but the gate
# moves on daily closes, so polling it that hard is wasted API calls for no extra resolution.
REGIME_POLL_SECONDS = 600

# How long a resting order is given to fill, and the horizon of the post-fill mark used to charge
# it for adverse selection. Measured, not chosen: across 100,551 real Hyperliquid fills sampled
# from 300 leaderboard addresses, passive fills underperform aggressive ones by 3.5bps at 1 minute,
# 6.3bps at 5 and 12.4bps at 15 — and by 60 minutes the effect has inverted into noise. Settling at
# the next daily rebalance, as this did originally, measured market drift rather than execution.
EXEC_SETTLE_MINUTES = 15


@dataclass
class RebalanceReport:
    mode: str
    equity: float
    gross: float
    net: float
    drawdown_pct: float
    paused: bool
    universe_size: int
    n_long: int
    n_short: int
    n_orders: int
    positions: list[dict] = field(default_factory=list)
    fills: list = field(default_factory=list)
    top_scores: list = field(default_factory=list)


class Engine:
    def __init__(self, cfg: Config, live: bool = False):
        self.cfg = cfg
        self.live = live
        self.mode = "live" if live else "paper"
        self.fx = make_futures(cfg)                      # KoinbayFutures | HyperliquidFutures per cfg.exchange.venue
        self.client = getattr(self.fx, "c", None)        # underlying KoinBay client (None on Hyperliquid)
        self._funding_interval_h = getattr(self.fx, "funding_interval_hours", 8.0)  # HL=1h, KoinBay=8h
        self.btc_sym = "BTC" if cfg.exchange.venue == "hyperliquid" else "E-BTC-USDT"  # venue-aware BTC reference
        self.strategy = SentimentEdgeStrategy(cfg.portfolio)
        self.champion = MomentumRegimeStrategy(cfg)   # directional momentum+regime (used when cfg.strategy=='champion')
        self.carry = FundingCarryStrategy(cfg)        # funding-carry+momentum, market-neutral (cfg.strategy=='carry')
        self.trend = TrendStrategy(cfg)               # dollar-neutral cross-sectional trend/CTA (cfg.strategy=='trend')
        self.consensus = ConsensusStrategy(cfg)       # trade only what the other three agree on (cfg.strategy=='consensus')
        self.treasury = Treasury(cfg)                 # money manager: weekly profit sweep into a safe vault
        self.store = Store(cfg.state.db_path, cfg.state.equity_csv)
        self.registry: Optional[ContractRegistry] = None
        self.signal: Optional[MarketProxySignal] = None
        self.base_interval = "4h"   # refined by a live probe in _ensure_market()
        self.history_bars = 120
        self.coin_to_symbol: dict[str, str] = {}
        self.whales = self._build_whale_tracker()
        self.exec_policy: Optional[ExecutionPolicy] = None   # set by _build_broker
        self.broker = self._build_broker()

    # -- construction helpers -------------------------------------------------
    def _build_whale_tracker(self) -> Optional[WhaleTracker]:
        cfg = self.cfg
        if not cfg.whales.enabled:
            return None
        addresses = list(cfg.whales.addresses)
        path = cfg.whales.source_file
        if path and os.path.exists(path):   # prefer the refreshed basket
            try:
                data = json.loads(open(path).read())
                if data.get("addresses"):
                    addresses = data["addresses"]
            except Exception as e:
                log.warning("could not read %s: %s", path, e)
        if not addresses:
            return None
        log.info("tracking %d whale wallets", len(addresses))
        return WhaleTracker(addresses, cfg.whales.info_url, cfg.whales.weighting)

    def _maybe_refresh_whales(self) -> None:
        """Weekly re-screen: if the basket file is missing or older than refresh_days, rebuild it."""
        cfg = self.cfg
        if not cfg.whales.enabled or self.registry is None:
            return
        path = cfg.whales.source_file
        if path and os.path.exists(path):
            try:
                age = time.time() - json.loads(open(path).read()).get("selected_ts", 0)
                if age < cfg.whales.refresh_days * 86_400:
                    return
            except Exception:
                pass
        from ..signal.whale_select import select_top_whales, write_basket
        kb_coins = {s.multiplier_coin.upper() for s in self.registry.active_usdt()}
        log.info("whale basket missing/stale -> weekly re-screen (~1-2 min)...")
        try:
            picks = select_top_whales(kb_coins, info_url=cfg.whales.info_url,
                                      max_inactive_days=cfg.whales.max_inactive_days, log=log)
            if picks:
                write_basket(path, picks)
                self.whales = self._build_whale_tracker()
                log.info("whale basket refreshed: %d wallets", len(picks))
        except Exception as e:
            log.warning("whale refresh failed: %s", e)

    def _load_exec_policy(self):
        """Restore the execution policy's learned cells, or start from the physics prior.

        It is loaded here rather than constructed fresh because its whole value is accumulated
        evidence: a policy that forgets on every restart never leaves its prior and is worse than
        useless — it pays the cost of exploring and never banks the result."""
        try:
            return ExecutionPolicy.from_json(self.store.get_meta("exec_policy", "") or "")
        except Exception as e:                                # noqa: BLE001
            log.warning("exec policy load failed, starting from prior: %s", e)
            return ExecutionPolicy()

    def _save_exec_policy(self) -> None:
        if self.exec_policy is None:
            return
        try:
            self.store.set_meta("exec_policy", self.exec_policy.to_json())
        except Exception as e:                                # noqa: BLE001
            log.warning("exec policy save failed: %s", e)

    def _settle_exec(self, max_rows: int = 60) -> None:
        """Score POST decisions now that the market has had time to answer them.

        A resting order's outcome is not knowable when it is sent, so record_exec leaves cost_bps
        NULL and this settles it from real subsequent price action: did the market trade through
        our touch? If it did we earned the spread; if it did not, the cost is what crossing NOW
        actually costs — never zero. Rows that cannot be scored are LEFT unresolved rather than
        closed at zero, because a fabricated zero is exactly what would teach the policy that
        posting is free.

        Never raises: an execution learner must not be able to stop the book trading.
        """
        if self.exec_policy is None:
            return
        try:
            rows = self.store.unresolved_exec(self.mode,
                                              older_than_s=EXEC_SETTLE_MINUTES * 60 + 120)
        except Exception as e:                                # noqa: BLE001
            log.warning("exec settle skipped: %s", e)
            return
        settled = 0
        for r in rows[:max_rows]:
            try:
                mid0, touch = float(r.get("arrival_mid") or 0), float(r.get("ref_price") or 0)
                if mid0 <= 0 or touch <= 0:
                    continue                                  # pre-migration row: cannot be scored
                # ORDER LIFETIME, not "since the decision". A resting order that gets hit six hours
                # later is not the thing being modelled, and 1h bars over eight hours answer a
                # different question than "did my order fill". Measured on 100,551 real Hyperliquid
                # fills, the adverse-selection signal lives at 1-15 minutes (3.5bps at 1min rising
                # to 12.4bps at 15min) and has vanished by 60 minutes, where market noise dominates.
                # So both the fill test and the post-fill mark use a short window.
                t0 = float(r["ts"])
                bars = self.fx.klines(r["symbol"], "1m", 60) or []
                win = [b for b in bars
                       if t0 <= int(b["idx"]) / 1000.0 <= t0 + EXEC_SETTLE_MINUTES * 60]
                if len(win) < 3:
                    continue                                  # not enough price action yet
                lo = min(float(b["low"]) for b in win)
                hi = max(float(b["high"]) for b in win)
                post_mid = float(win[-1]["close"])             # the mark ~15 minutes after the fill
                bs = probe(self.fx, r["symbol"], r["side"], abs(float(r.get("notional") or 0)))
                completion = bs.cross_bps if bs else float(self.cfg.execution.slippage_bps)
                cost, filled = post_outcome_bps(r["side"], touch, mid0, lo, hi, completion,
                                                post_mid=post_mid)
                self.exec_policy.record(r["context"], r.get("shadow") or r["arm"], cost)
                self.store.resolve_exec(r["id"], cost_bps=cost, filled=filled,
                                        fill_price=touch if filled else (bs.mid if bs else mid0))
                settled += 1
            except Exception:                                 # noqa: BLE001
                continue
        if settled:
            log.info("execution policy: settled %d pending decisions", settled)
            self._save_exec_policy()

    def _build_broker(self):
        self.exec_policy = self._load_exec_policy()
        if self.live:
            # SHADOW: the policy observes and records every order and changes nothing. It takes
            # over only when store.exec_stats() shows it beating the fixed default on realised
            # cost — see live_broker.policy_live.
            return LiveBroker(self.fx, capital_override=self.cfg.capital_usdt,
                              stop_loss_pct=self.cfg.risk.stop_loss_pct,
                              exec_policy=self.exec_policy, store=self.store)
        capital = self.cfg.capital_usdt or 10_000.0
        # Paper trains on the REAL book (fx), never on its own synthetic fill.
        broker = PaperBroker(starting_capital=capital, exec_policy=self.exec_policy,
                             store=self.store, fx=self.fx)
        acct = self.store.load_account(self.mode)
        if acct:
            broker.starting_capital = acct.get("starting_capital") or capital
            broker.realized_pnl = acct.get("realized_pnl") or 0.0
            broker.fees_paid = acct.get("fees_paid") or 0.0
            broker.funding_pnl = acct.get("funding_pnl") or 0.0
            broker.last_funding_ts = acct.get("last_funding_ts") or 0.0
        for sym, p in self.store.load_positions(self.mode).items():
            broker._positions[sym] = PaperPosition(
                contracts=int(p["contracts"]), avg_price=float(p["avg_price"]),
                multiplier=float(p.get("multiplier", 1.0)), opened_ts=float(p.get("opened_ts", 0.0)))
        return broker

    def _ensure_market(self) -> None:
        if self.registry is not None:
            return
        self.registry = ContractRegistry.from_futures(self.fx)
        active = self.registry.active_usdt()
        self.coin_to_symbol = {s.multiplier_coin.upper(): s.name for s in active}
        ref = self.btc_sym if self.registry.has(self.btc_sym) else (active[0].name if active else self.btc_sym)
        horizons = [parse_duration_to_hours(h) for h in self.cfg.signal.momentum_horizons]
        if self.cfg.strategy == "champion":
            # the champion's signals are DAILY (50/100-day MAs, 30-day momentum) — force daily bars
            # with enough history to compute the regime/trend MAs.
            ch = self.cfg.champion
            self.base_interval = "1day"
            self.history_bars = min(300, max(ch.regime_ma, ch.trend_ma, ch.lookback, ch.breadth_ma) + 30)
        elif self.cfg.strategy == "consensus":
            # every vote needs its own window; the regime brake's 100-day MA is the deepest
            c = self.cfg
            self.base_interval = "1day"
            self.history_bars = min(300, max(c.champion.regime_ma, c.champion.lookback, c.carry.lookback,
                                             getattr(c.trend, "donchian_period", 45),
                                             c.universe.min_listing_days + 5) + 30)
        elif self.cfg.strategy in ("carry", "trend"):
            # carry / trend rank on DAILY bars — force daily with enough history.
            self.base_interval = "1day"
            if self.cfg.strategy == "carry":
                self.history_bars = min(300, self.cfg.carry.lookback + 30)
            else:
                # Size the window to whichever score is actually in use. Reading only ma_period
                # here would silently starve score_mode="breakout" with a longer donchian_period:
                # _breakout returns -1e9, every name is dropped as "insufficient history", and the
                # book comes back empty with nothing in the logs to say why. Same failure shape as
                # the funding drawdown limit that was declared and never compared against.
                tr = self.cfg.trend
                need = tr.ma_period
                if getattr(tr, "score_mode", "ma") == "breakout":
                    need = max(need, getattr(tr, "donchian_period", 45))
                self.history_bars = min(300, need + 35)
            # ...and enough to PROVE the listing age from these same bars. Asking for a longer window
            # costs no extra requests (one call per symbol either way) — which is the whole point, since
            # a separate per-symbol age probe is what blew the rate budget and emptied the book.
            self.history_bars = min(300, max(self.history_bars, self.cfg.universe.min_listing_days + 5))
        elif self.cfg.strategy == "grid":
            # grid trades intraday mean-reversion — force HOURLY bars with enough history for the
            # ranging/trend MA filter (the grid re-evaluates every rebalance_minutes, e.g. every 15m).
            self.base_interval = "1h"
            self.history_bars = min(400, self.cfg.grid.ma_period + 20)
        elif self.cfg.strategy == "funding":
            # funding harvest uses index (mark/oracle/funding) + fundingHistory, not klines — minimal setup
            self.base_interval = "1h"
            self.history_bars = 30
        else:
            self.base_interval = pick_base_interval(self.fx, ref)
            itv_h = parse_duration_to_hours(self.base_interval)
            self.history_bars = max(30, min(300, int(max(horizons) / itv_h) + 5))
        w = self.cfg.signal.weights
        self.signal = MarketProxySignal(
            horizons_hours=horizons,
            weights={"momentum": w.momentum, "funding": w.funding, "funding_vel": w.funding_vel,
                     "volume": w.volume, "whale": w.whale, "reversal": w.reversal},
            funding_sign=self.cfg.signal.funding_sign,
            base_interval=self.base_interval,
            history_bars=self.history_bars,
            reversal_hours=self.cfg.signal.reversal_hours,
        )
        log.info("market ready: %d active USDT perps; base interval=%s; history=%d bars",
                 len(active), self.base_interval, self.history_bars)

    # -- core cycle -----------------------------------------------------------
    def run_once(self, flatten: bool = False) -> RebalanceReport:
        cfg = self.cfg
        if cfg.strategy == "grid":
            return self.run_grid_once(flatten)      # resting-limit market-maker — its own tick-driven path
        if cfg.strategy == "funding":
            return self.run_funding_once(flatten)   # delta-neutral funding harvest — its own tick-driven path
        if self.live:
            self.broker.preflight()
        self._ensure_market()
        assert self.registry is not None and self.signal is not None
        self._maybe_refresh_whales()
        self._settle_exec()          # score last cycle's resting orders before placing new ones

        universe = build_universe(self.fx, self.registry, cfg.universe)
        data, prices, funding = load_symbol_data(self.fx, self.registry, universe,
                                                 self.base_interval, self.history_bars)
        allowed = set(filter_by_funding(funding, cfg.risk.funding_abs_max))
        data = {s: d for s, d in data.items() if s in allowed}
        data = filter_by_age(data, cfg.universe.min_listing_days)
        # DEGRADED-FETCH GUARD. pmap swallows a failed symbol fetch and load_symbol_data drops it, so a
        # rate-limited cycle silently returns a fraction of the universe. Sizing a book off that shrunken
        # set produced an empty TargetBook, and the reconciler dutifully sold everything — the book went
        # to 0 positions on nothing but throttling. If most of the universe failed to load, keep what we
        # hold and wait for the next cycle instead.
        if universe and len(data) < 0.5 * len(universe):
            log.warning("only %d/%d symbols loaded (API degraded) — holding the current book this cycle",
                        len(data), len(universe))
            return RebalanceReport(mode=self.mode, equity=self.broker.equity(), gross=0.0, net=0.0,
                                   drawdown_pct=0.0, paused=False, universe_size=len(universe),
                                   n_long=0, n_short=0, positions=[], fills=[], top_scores=[])
        prices = {s: p for s, p in prices.items() if s in allowed}

        whale_bias: dict[str, float] = {}
        if self.whales:
            try:
                snaps = self.whales.fetch()
                coin_bias = self.whales.blended_bias(snaps)
                whale_bias = map_bias_to_symbols(coin_bias, self.coin_to_symbol)
                self._persist_whales(snaps, coin_bias)
                log.info("whales: %d/%d wallets, consensus on %d coins", len(snaps),
                         len(self.cfg.whales.addresses), len(coin_bias))
            except Exception as e:
                log.warning("whale tracker failed: %s", e)

        # BTC-betas (cross-section) computed once and reused for whale-demean + book beta-hedge
        betas_cs = compute_betas({s: d.closes for s, d in data.items()},
                                 self.btc_sym, cfg.portfolio.beta_lookback) if data else {}
        if cfg.portfolio.beta_neutralize and whale_bias and betas_cs:
            # kill the whale overlay's directional BTC tilt at SOURCE (orthogonalize vs beta)
            whale_bias = demean_by_beta(whale_bias, betas_cs)

        scores = self.signal.scores(data, whale_bias=whale_bias)
        # beta-aware selection: rank by RESIDUAL score (strip the part explained by BTC-beta) so the book
        # is born market-neutral — residual momentum instead of raw, which carries a hidden market bet
        if cfg.portfolio.beta_neutral_scores and betas_cs:
            scores = demean_by_beta(scores, betas_cs)
        # snapshot funding so funding-level / funding-velocity become backtestable from real history later
        self.store.record_funding(self.mode, funding, {s: d.next_funding for s, d in data.items()})
        top_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        if not self.live:
            self.broker.set_marks(prices)
        raw_equity = self.broker.equity() or (cfg.capital_usdt or 0.0)
        # Money manager: sweep 40% of new profit into the Vault weekly (safe, not traded).
        # The at-risk trading pool = equity - vault, so the book de-risks as it wins.
        vault = self._run_treasury(raw_equity)
        trading_equity = max(0.0, raw_equity - vault)
        # Portfolio-level allocation weight: risk-parity allocator sets capital_weight
        # (0-1) so each book sizes to its share of the total portfolio capital.
        equity = trading_equity * cfg.capital_weight

        peak = max(self.store.peak_equity(self.mode, equity), equity)
        dd = drawdown_pct(peak, equity)
        breached = dd > cfg.risk.max_drawdown_pct

        paused = self.is_paused()
        book = None
        if flatten or breached or paused:
            if breached:
                log.warning("drawdown %.2f%% exceeds limit %.2f%% -> flattening and pausing %.0fh",
                            dd, cfg.risk.max_drawdown_pct, cfg.risk.pause_hours)
                self.store.set_paused_until(self.mode, time.time() + cfg.risk.pause_hours * 3600)
            elif paused:
                log.info("paused (drawdown cooldown) -> staying flat this cycle")
            target: dict[str, int] = {}
        else:
            gscale = 1.0
            if cfg.risk.dynamic_risk:
                rets = self.store.daily_returns(self.mode, cfg.risk.vol_lookback)
                ddf = drawdown_pct(peak, equity) / 100.0
                gscale = risk_scale(rets, ddf, cfg.risk.vol_target_daily, cfg.risk.min_risk_scale,
                                    cfg.risk.dd_throttle_start, cfg.risk.dd_throttle_full)
                if gscale < 0.999:
                    log.info("dynamic de-risk: gross x%.2f (drawdown=%.1f%%)", gscale, ddf * 100)
            if cfg.risk.crash_guard:
                itv_h = parse_duration_to_hours(self.base_interval)
                ref_c = (data.get(self.btc_sym) or next(iter(data.values()), None))
                cg = crash_guard(ref_c.closes if ref_c else [],
                                 int(cfg.risk.crash_dd_hours / itv_h), int(cfg.risk.crash_bounce_hours / itv_h),
                                 cfg.risk.crash_dd_trigger, cfg.risk.crash_bounce_trigger,
                                 cfg.risk.crash_scale_floor)
                if cg < 0.999:
                    log.warning("MOMENTUM-CRASH guard: gross x%.2f (bear + violent bounce regime)", cg)
                    gscale *= cg
            if cfg.risk.dispersion_gate:
                disp_now = dispersion({s: d.closes for s, d in data.items()})
                hist = self.store.get_meta("disp_hist", []) or []
                ref = sorted(hist)[len(hist) // 2] if hist else 0.0   # trailing median
                dg = dispersion_scale(disp_now, ref, cfg.risk.dispersion_min_scale)
                if dg < 0.999:
                    log.info("dispersion gate: gross x%.2f (low edge: disp %.4f vs ref %.4f)", dg, disp_now, ref)
                    gscale *= dg
                self.store.set_meta("disp_hist", (hist + [round(disp_now, 6)])[-cfg.risk.dispersion_ref_window:])
            if cfg.risk.daily_loss_limit_pct > 0:
                gscale *= float(self.store.get_meta("daily_loss_scale", 1.0) or 1.0)
            self.store.set_meta("crash_scale", round(gscale, 3))
            if cfg.strategy == "champion":
                # DIRECTIONAL champion: long-only top-K momentum, BTC-regime gated. Beta is intentional —
                # NO beta-neutralize / net-clamp / dollar-neutrality (that's the whole point of this book).
                book = self.champion.build_book({s: d.closes for s, d in data.items()}, prices,
                                                self.registry, equity, gross_scale=gscale)
            elif cfg.strategy == "carry":
                # CARRY: funding-carry + momentum, dollar-neutral long/short by construction. Rank on the
                # TRAILING-AVERAGE funding (persistent, cleaner than a single rate); funding is kept
                # unfiltered via risk.funding_abs_max so high-funding shorts are retained. No extra
                # beta-hedge — the long/short legs are already balanced.
                favg = self.store.recent_funding_avg(self.mode, cfg.carry.funding_avg_cycles)
                fsig = {s: favg.get(s, funding.get(s)) for s in funding}   # trailing avg, fallback to current
                # pass what we already hold so the keep_buffer band can keep a name that is merely
                # hovering at the cut-off rather than selling and re-buying it on ranking noise
                book = self.carry.build_book({s: d.closes for s, d in data.items()}, fsig, prices,
                                             self.registry, equity, gross_scale=gscale,
                                             held=self.broker.positions())
                # carry is dollar-neutral by construction but NOT beta-neutral: it shorts crowded
                # (high-funding) names, which run hotter than what it goes long. Hedge that out, then
                # clamp back inside the dollar rail — the hedge shrinks one sleeve, so it necessarily
                # opens a dollar imbalance and must be followed by the same clamp the neutral book uses.
                self._beta_neutralize(book, betas_cs, equity)
                if book and book.positions:
                    self._apply_scales(book, net_scales({s: p.target_notional for s, p in book.positions.items()},
                                                        equity, cfg.risk.max_net_exposure))
            elif cfg.strategy == "trend":
                # TREND (CTA): dollar-neutral cross-sectional trend — long the strongest uptrends, short
                # the strongest downtrends. Dollar-neutral by construction (equal-weight sleeves); the
                # vol-target / drawdown-throttle overlay (gross_scale) tames the raw drawdown.
                book = self.trend.build_book({s: d.closes for s, d in data.items()}, prices,
                                             self.registry, equity, gross_scale=gscale)
                # trend is dollar-neutral long/short exactly like carry, so it carries the same hidden
                # exposure: the sleeves balance in DOLLARS while their betas do not. Same hedge, same
                # clamp after it. Missed when carry was fixed — the engine harness caught it.
                self._beta_neutralize(book, betas_cs, equity)
                if book and book.positions:
                    self._apply_scales(book, net_scales({s: p.target_notional for s, p in book.positions.items()},
                                                        equity, cfg.risk.max_net_exposure))
            elif cfg.strategy == "consensus":
                # CONSENSUS: only names that >=2 of {champion, carry, trend} agree on. Dollar-neutral
                # when both sleeves exist, long-only on the rare day nothing qualifies as a short —
                # so no beta-hedge and no net clamp here (max_net_exposure is 1.0 in its config), and
                # the votes are computed from the same functions the three books run.
                favg = self.store.recent_funding_avg(self.mode, cfg.carry.funding_avg_cycles)
                fsig = {s: favg.get(s, funding.get(s)) for s in funding}
                ref = data.get(self.btc_sym)
                book = self.consensus.build_book({s: d.closes for s, d in data.items()}, fsig, prices,
                                                 self.registry, equity, gross_scale=gscale,
                                                 held=self.broker.positions(),
                                                 btc_closes=(ref.closes if ref else None))
            else:
                ta_scores = ({s: ta_consensus(d.closes) for s, d in data.items()}
                             if cfg.signal.ta_veto > 0 else None)
                adv_quote = self._adv_quote(data) if cfg.portfolio.adv_cap_frac > 0 else {}
                vol_by_symbol = ({s: realized_vol(d.closes) for s, d in data.items()}
                                 if cfg.portfolio.risk_parity else None)
                book = self.strategy.build_book(scores, prices, self.registry, equity, gross_scale=gscale,
                                                ta=ta_scores, ta_veto=cfg.signal.ta_veto,
                                                vol_by_symbol=vol_by_symbol)
                # ---- portfolio robustness, in order: beta-hedge -> ADV caps -> dollar-neutral clamp ----
                closes_by = {s: d.closes for s, d in data.items()}
                betas = betas_cs if book.positions else {}   # computed once above (also used for whale-demean)
                self._beta_neutralize(book, betas, equity)
                if cfg.portfolio.adv_cap_frac > 0 and book.positions:
                    acaps = adv_caps({s: p.target_notional for s, p in book.positions.items()},
                                     adv_quote, cfg.portfolio.adv_cap_frac)
                    hit = [s for s, f in acaps.items() if f < 0.999]
                    self._apply_scales(book, acaps)
                    if hit:
                        log.info("ADV cap: shrank %d name(s) to <=%.0f%% of ADV: %s",
                                 len(hit), cfg.portfolio.adv_cap_frac * 100, ", ".join(hit))
                self.store.set_meta("dispersion", round(dispersion(closes_by), 5))
                # HARD dollar-neutral clamp, after the hedge + ADV caps + integer rounding
                self._apply_scales(book, net_scales({s: p.target_notional for s, p in book.positions.items()},
                                                    equity, cfg.risk.max_net_exposure))
            target = book.signed_contracts()
            scale = leverage_scale(book.realized_gross, equity, cfg.risk.max_gross_leverage)
            if scale < 0.999:
                target = {s: int(c * scale) for s, c in target.items()}
            nr = net_exposure_ratio(book.realized_net, equity)
            self.store.set_meta("net_ratio", round(nr, 4))
            if nr > cfg.risk.max_net_exposure + 0.01:   # residual past the clamp = pure rounding
                log.warning("net exposure %.1f%% above rail %.1f%% after clamp (integer rounding)",
                            nr * 100, cfg.risk.max_net_exposure * 100)

        current = self.broker.positions()
        self._accrue_funding(current)   # book funding earned/paid since last cycle
        for sym in current:
            if sym not in prices and self.registry.has(sym):
                mp = mark_price(self.fx, sym)
                if mp > 0:
                    prices[sym] = mp

        orders = reconcile(target, current, prices, self.registry, cfg.execution.min_trade_notional,
                           cfg.execution.min_rebalance_frac)
        if orders and self.live:
            self.broker.prepare(sorted({o.symbol for o in orders}), self.registry,
                                cfg.portfolio, cfg.execution)
        fills = self.broker.execute(orders, self.registry, cfg.execution)
        if orders:
            self._save_exec_policy()     # bank what this cycle's orders taught it

        if not self.live:
            self.broker.set_marks(prices)
            self._persist_paper()

        new_equity = self.broker.equity() or equity
        peak = self.store.update_peak_equity(self.mode, new_equity)
        gross = book.realized_gross if book else 0.0
        net = book.realized_net if book else 0.0
        self.store.record_equity(self.mode, new_equity, gross, net, drawdown_pct(peak, new_equity))
        self.store.record_fills(self.mode, fills)
        self.store.save_scores(self.mode, scores)
        self._record_closed()

        positions = []
        if book:
            for p in book.positions.values():
                positions.append({"symbol": p.symbol, "side": p.side, "contracts": p.target_contracts,
                                  "notional": p.target_notional, "score": round(p.score, 3)})
        n_long = sum(1 for c in target.values() if c > 0)
        n_short = sum(1 for c in target.values() if c < 0)
        return RebalanceReport(
            mode=self.mode, equity=new_equity, gross=gross, net=net,
            drawdown_pct=drawdown_pct(peak, new_equity), paused=breached,
            universe_size=len(universe), n_long=n_long, n_short=n_short, n_orders=len(orders),
            positions=positions, fills=fills, top_scores=top_scores[:10],
        )

    # -- GRID (resting-limit market-maker) -----------------------------------
    def run_grid_once(self, flatten: bool = False) -> RebalanceReport:
        """Advance the resting-limit grid book by every new hourly bar since the last tick. Each coin runs
        an independent order book (grid_book): resting buy limits below the anchor fill on the bar's low,
        take-profits on the high — capturing the spacing the snapshot approach could not. State + the
        last-processed timestamp persist, so each bar is processed exactly once (restart-safe, no churn)."""
        cfg = self.cfg
        g = cfg.grid
        self._ensure_market()                                   # base_interval='1h'
        assert self.registry is not None
        universe = build_universe(self.fx, self.registry, cfg.universe)
        cost = max(0.0, cfg.execution.slippage_bps / 1e4)       # per-fill cost (slippage-dominated; HL maker ~0)
        sp = g.spacing_pct / 100.0
        stop = g.stop_pct / 100.0

        bars: dict[str, list] = {}                              # coin -> [(ts_ms, high, low, close)] ascending
        for s in universe:
            rows = []
            for k in (self.fx.klines(s, "1h", self.history_bars) or []):
                try:
                    c = float(k["close"])
                    rows.append((int(k["idx"]), float(k.get("high", c)), float(k.get("low", c)), c))
                except (KeyError, TypeError, ValueError):
                    continue
            rows.sort()
            if len(rows) > g.ma_period + 2:
                bars[s] = rows
        coins = list(bars)

        gstate = self.store.get_meta("grid_state", {}) or {}
        last_ts = int(self.store.get_meta("grid_last_ts", 0) or 0)
        acct = self.store.load_account(self.mode) or {}
        cap = float(cfg.capital_usdt or 10000.0)
        booked = float(acct.get("realized_pnl", 0.0)) - float(acct.get("fees_paid", 0.0))
        fees_cum = float(acct.get("fees_paid", 0.0))
        paused_until = float(acct.get("paused_until", 0.0) or 0.0)
        now = time.time()
        latest_bar = max((rows[-1][0] for rows in bars.values()), default=last_ts)

        equity0 = self.store.last_equity(self.mode) or (cap + booked)
        unit = max(0.0, g.target_gross * equity0) / (max(1, len(coins)) * max(1, g.levels))

        newest = last_ts
        new_realized = 0.0
        new_fees = 0.0
        closed_trades: list[dict] = []
        active = (now >= paused_until) and not flatten
        if active:
            for s in coins:
                rows = bars[s]
                closes = [r[3] for r in rows]
                st = gstate.get(s) or grid_book.new_state(rows[0][3])
                for j, (ts, hi, lo, cl) in enumerate(rows):
                    if ts <= last_ts:
                        continue
                    lo0 = max(0, j - g.ma_period)
                    ma = sum(closes[lo0:j + 1]) / (j + 1 - lo0)
                    ranging = ma > 0 and abs(cl / ma - 1) < g.range_band
                    r = grid_book.tick(st, hi, lo, cl, levels=g.levels, spacing=sp, stop=stop,
                                       ranging=ranging, cost=cost, unit_notional=unit,
                                       fill_delta=g.fill_delta_pct / 100.0)
                    new_realized += r["realized"]
                    new_fees += r["fees"]
                    for tr in r["closed"]:
                        closed_trades.append({"symbol": s, "side": tr["side"], "entry": tr["entry"],
                                              "exit": tr["exit"], "contracts": 0, "pnl": tr["pnl"],
                                              "opened_ts": ts / 1000.0, "closed_ts": ts / 1000.0})
                    if ts > newest:
                        newest = ts
                gstate[s] = st
        else:
            newest = latest_bar                                 # paused/flatten: skip these bars, don't retro-trade

        booked += new_realized
        fees_cum += new_fees

        # mark open lots + build the position book at each coin's latest close
        positions_store: dict[str, dict] = {}
        report_positions: list[dict] = []
        open_pnl = 0.0
        gross_notional = 0.0
        net_notional = 0.0
        n_long = 0
        n_short = 0
        for s in coins:
            st = gstate.get(s) or {}
            longs = st.get("longs", [])
            shorts = st.get("shorts", [])
            last_close = bars[s][-1][3]
            open_pnl += (sum((last_close / bp - 1) * unit for bp in longs)
                         + sum((bp / last_close - 1) * unit for bp in shorts))
            gross_notional += (len(longs) + len(shorts)) * unit
            net_units = len(longs) - len(shorts)
            net_notional += net_units * unit
            if net_units != 0 and self.registry.has(s):
                spec = self.registry.get(s)
                contracts = spec.notional_to_contracts(abs(net_units) * unit, last_close)
                if contracts > 0:
                    signed = contracts if net_units > 0 else -contracts
                    avg = (sum(longs) + sum(shorts)) / max(1, len(longs) + len(shorts))
                    positions_store[s] = {"contracts": signed, "avg_price": avg,
                                          "multiplier": spec.multiplier, "opened_ts": 0.0}
                    report_positions.append({"symbol": s, "side": "LONG" if net_units > 0 else "SHORT",
                                             "contracts": signed, "notional": net_units * unit,
                                             "score": float(net_units)})
                    n_long += 1 if net_units > 0 else 0
                    n_short += 1 if net_units < 0 else 0

        equity = cap + booked + open_pnl
        peak = self.store.update_peak_equity(self.mode, equity)
        dd = drawdown_pct(peak, equity)

        # portfolio drawdown kill-switch: flatten every book (book the open PnL) + pause
        breached = False
        if flatten or (cfg.risk.max_drawdown_pct and dd >= cfg.risk.max_drawdown_pct):
            for s in coins:
                st = gstate.get(s) or {}
                last_close = bars[s][-1][3]
                for bp in st.get("longs", []):
                    booked += ((last_close / bp - 1) - 2 * cost) * unit
                for bp in st.get("shorts", []):
                    booked += ((bp / last_close - 1) - 2 * cost) * unit
                st["longs"] = []
                st["shorts"] = []
                gstate[s] = st
            open_pnl = 0.0
            positions_store = {}
            report_positions = []
            gross_notional = net_notional = 0.0
            n_long = n_short = 0
            equity = cap + booked
            if not flatten:
                paused_until = now + cfg.risk.pause_hours * 3600
                breached = True
                log.warning("grid drawdown %.1f%% >= %.1f%% — flattened + paused %.0fh",
                            dd, cfg.risk.max_drawdown_pct, cfg.risk.pause_hours)

        realized_gross = booked + fees_cum
        self.store.set_meta("grid_state", gstate)
        self.store.set_meta("grid_last_ts", int(newest))
        self.store.save_account(self.mode, starting_capital=cap, realized_pnl=realized_gross,
                                fees_paid=fees_cum, funding_pnl=0.0, last_funding_ts=0.0)
        if breached:
            self.store.conn.execute("UPDATE account SET paused_until=? WHERE mode=?", (paused_until, self.mode))
            self.store.conn.commit()
        self.store.save_positions(self.mode, positions_store)
        if closed_trades:
            self.store.record_closed_trades(self.mode, closed_trades)
        self.store.record_equity(self.mode, equity, gross_notional, net_notional, dd)
        self.store.save_scores(self.mode, {p["symbol"]: p["score"] for p in report_positions})
        log.info("GRID | equity=$%.2f | %dL/%dS | %d fills | dd=%.2f%% | %s", equity, n_long, n_short,
                 len(closed_trades), dd, "PAUSED" if breached else "ok")
        return RebalanceReport(mode=self.mode, equity=equity, gross=gross_notional, net=net_notional,
                               drawdown_pct=dd, paused=breached, universe_size=len(coins),
                               n_long=n_long, n_short=n_short, n_orders=len(closed_trades),
                               positions=report_positions, fills=[], top_scores=[])

    def _grid_risk_check(self) -> dict:
        """The grid runs its own per-coin stops + portfolio drawdown kill-switch inside run_grid_once; the
        between-tick safety poll is a no-op (nothing to do until the next hourly bar closes)."""
        return {"action": None}

    # -- FUNDING HARVEST (delta-neutral cash-and-carry) -----------------------
    def run_funding_once(self, flatten: bool = False) -> RebalanceReport:
        """Short the perps paying STABLE positive funding + hold a matching spot long (simulated in paper), so price
        cancels and we collect the funding premium — market beta ~ 0, works in every regime. EVERY tick books funding
        accrued + the real mark-vs-oracle basis move - fees on the held book. But we only RE-PICK / RE-SIZE the basket
        every fh.rebalance_days (a sticky basket held via keep_top_n + min_hold_days, resized only past no_trade_band),
        because funding accrues whether or not we trade — rebalancing hourly just bleeds fees (~0.08%/day of gross,
        enough to flip a backtested +14%/yr harvest to -26%/yr). Fresh slots are filled by funding MAGNITUDE (fat
        premium) subject to a consistency floor; gross is opportunity-scaled. Live (Phase 2) uses a real spot leg."""
        cfg = self.cfg
        fh = cfg.funding
        self._ensure_market()
        assert self.registry is not None
        universe = build_universe(self.fx, self.registry, cfg.universe)
        now = time.time()
        cost = fh.cost_per_side

        gstate = self.store.get_meta("funding_state", {}) or {}
        last_ts = float(self.store.get_meta("funding_last_ts", now) or now)
        last_rebal = float(self.store.get_meta("funding_last_rebal", 0.0) or 0.0)
        dt_h = min(24.0, max(0.0, (now - last_ts) / 3600.0))     # hours since last tick (cap the first accrual)
        acct = self.store.load_account(self.mode) or {}
        cap = float(cfg.capital_usdt or 10000.0)
        # keep funding income and basis PnL in SEPARATE buckets so the dashboard can show real funding earned
        # (they used to be merged into realized_pnl, which is why the funding card always read $0)
        basis_cum = float(acct.get("realized_pnl", 0.0))
        fund_cum = float(acct.get("funding_pnl", 0.0))
        fees_cum = float(acct.get("fees_paid", 0.0))
        booked = basis_cum + fund_cum - fees_cum
        equity0 = self.store.last_equity(self.mode) or (cap + booked)

        # ---- decide the tick TYPE *before* fetching, so we only pull what this tick actually needs ----
        # A hold tick just accrues on legs we already own: it needs a fresh mark/oracle/funding for those (~15 calls),
        # NOT a full-universe trailing-funding scan (~80 calls). Scanning every tick was pure waste and the likely
        # cause of the rate-limit stalls that froze the loop (alive but writing nothing) for ~30h.
        held_syms = list(gstate)
        do_rebal = bool(flatten or not gstate or (now - last_rebal) >= fh.rebalance_days * 86400.0)

        def _fetch(syms: list[str], with_hist: bool) -> dict[str, dict]:
            out: dict[str, dict] = {}
            for s in syms:
                try:
                    idx = self.fx.index(s)
                    mark = float(idx.get("tagPrice") or 0.0)
                    oracle = float(idx.get("indexPrice") or mark)
                    if mark <= 0 or oracle <= 0:
                        continue
                    d = {"mark": mark, "oracle": oracle, "funding": float(idx.get("currentFundRate") or 0.0),
                         "basis": (mark - oracle) / oracle}
                    if with_hist:
                        d["hist"] = self.fx.funding_history(s, fh.funding_window_days)
                    out[s] = d
                except Exception:
                    continue
            return out

        info = _fetch(universe if do_rebal else held_syms, do_rebal)

        # Per-leg run-length of NEGATIVE funding. A single negative hourly print is ordinary noise (across ~15 legs
        # at least one is negative almost every hour), so reacting to one would force a full re-pick every tick and
        # undo the whole turnover fix. Only a sustained run means the premium has really flipped and we should exit.
        neg_run: dict[str, int] = {}
        for s, st in gstate.items():
            prev_run = int(st.get("neg", 0))
            d = info.get(s)
            if d is None:
                neg_run[s] = prev_run                      # no quote -> carry the counter, don't reset it
            else:
                neg_run[s] = prev_run + 1 if float(d.get("funding", 0.0)) < 0.0 else 0
        if not do_rebal and any(v >= fh.exit_neg_ticks for v in neg_run.values()):
            flipped = [s for s, v in neg_run.items() if v >= fh.exit_neg_ticks]
            log.info("FUNDING | %s paying negative for %d+ ticks — re-picking early", ",".join(flipped),
                     fh.exit_neg_ticks)
            do_rebal = True
            info = _fetch(universe, True)

        # book PnL on the CURRENTLY HELD book: funding accrued (short collects +funding) + basis move, per unit held
        new_funding = new_basis = 0.0
        for s, st in gstate.items():
            d = info.get(s)
            if not d:
                continue
            notional = float(st.get("notional", 0.0))
            new_funding += notional * d["funding"] * dt_h                                  # funding accrual
            new_basis += -(d["basis"] - float(st.get("basis", d["basis"]))) * notional     # long-spot/short-perp basis
        fund_cum += new_funding
        basis_cum += new_basis
        booked += new_funding + new_basis

        # ---- CATASTROPHE BRAKE ----------------------------------------------------------------
        # This book used to enforce nothing: risk_check skipped it entirely and max_drawdown_pct was
        # configured but never acted on — a limit that reads as protection while doing none. It is
        # delta-neutral, so in paper (where the spot hedge is simulated as perfect) that never showed.
        # Live it matters: if the hedge breaks, the perp legs are naked and nothing was watching.
        equity_now = cap + booked
        peak_now = max(float(self.store.peak_equity(self.mode, equity_now) or equity_now), equity_now)
        dd_now = drawdown_pct(peak_now, equity_now)
        if not flatten and cfg.risk.max_drawdown_pct and dd_now >= cfg.risk.max_drawdown_pct:
            log.warning("FUNDING | drawdown %.2f%% exceeds limit %.2f%% -> unwinding and pausing %.0fh",
                        dd_now, cfg.risk.max_drawdown_pct, cfg.risk.pause_hours)
            self.store.set_paused_until(self.mode, now + cfg.risk.pause_hours * 3600)
            flatten = True
        elif not flatten and self.is_paused():
            flatten = True                                  # still in the cooldown -> stay flat

        # ---- HEDGE-BREAKDOWN GUARD -------------------------------------------------------------
        # The whole book rests on perp ~ spot. A leg whose mark has torn away from the oracle is a leg
        # where that assumption has stopped holding, so harvesting its funding is no longer neutral.
        # Drop those names rather than keep collecting a premium against an un-hedged position.
        blown = {s for s, d in info.items()
                 if s in gstate and abs(float(d.get("basis", 0.0))) > fh.max_basis}
        if blown:
            log.warning("FUNDING | basis blown out on %s (>%.2f%%) -> dropping",
                        ",".join(sorted(blown)), fh.max_basis * 100)

        # score every candidate: funding MAGNITUDE (mu) + CONSISTENCY (t-stat = mu/sd). Only a rebalance tick
        # fetches `hist`, so this is empty on hold ticks (nothing to re-pick anyway).
        scored: dict[str, tuple[float, float]] = {}
        for s, d in info.items():
            h = d.get("hist") or []
            if len(h) < fh.funding_window_days - 3:
                continue
            mu = sum(h) / len(h)
            sd = (sum((x - mu) ** 2 for x in h) / len(h)) ** 0.5 or 1e-9
            scored[s] = (mu, mu / sd)
        qualify = {s: v for s, v in scored.items()
                   if v[0] > fh.min_funding_daily and v[1] > fh.tstat_min and s not in blown}

        # degraded-scan guard: never tear up a live book on partial data. If an API wobble priced only some of the
        # universe, the "best" names are an artifact of what happened to respond — hold instead of churning on it.
        if do_rebal and gstate and not flatten and len(scored) < max(fh.max_coins, int(0.5 * len(universe))):
            log.warning("FUNDING | degraded scan (%d/%d coins scored) — holding this tick instead of rebalancing",
                        len(scored), len(universe))
            do_rebal = False

        if flatten:
            picks = []
        elif do_rebal:
            # STICKY: keep held names that still qualify + sit in the wide keep-band + are past min-hold (or have
            # stopped paying -> dropped by the qualify filter). Fill open slots with the richest-FUNDING fresh names.
            keepset = set(sorted(scored, key=lambda s: -scored[s][1])[: fh.keep_top_n])
            keep = [s for s in gstate
                    if s in qualify and s in keepset
                    and ((now - float(gstate[s].get("entry_ts", now))) >= fh.min_hold_days * 86400.0
                         or qualify[s][0] > 0)]
            keep = keep[: fh.max_coins]
            pool = sorted([s for s in qualify if s not in keep], key=lambda s: -qualify[s][0])
            picks = keep + pool[: max(0, fh.max_coins - len(keep))]
        else:
            # HOLD every leg we own — a missing quote must NOT drop a position — except one whose hedge
            # has broken, which we shed immediately rather than waiting for the next rebalance.
            picks = [s for s in held_syms if s not in blown]

        # opportunity-scaled gross: fewer qualifying coins (thin premium) -> deploy less -> preserve capital
        g = fh.target_gross * (min(1.0, len(picks) / fh.max_coins) if fh.max_coins else 0.0)
        unit = (g * equity0) / max(1, len(picks)) if picks else 0.0

        new_state: dict[str, dict] = {}
        positions_store: dict[str, dict] = {}
        report_positions: list[dict] = []
        prev_positions = self.store.load_positions(self.mode)
        turnover = 0.0
        pickset = set(picks)
        for s in picks:
            d = info.get(s)
            cur = float(gstate.get(s, {}).get("notional", 0.0))
            if d is None:
                # no fresh quote this tick (transient API failure) — carry the leg forward EXACTLY as-is.
                # Without this, a momentary outage would silently liquidate the position and charge a fee for it.
                new_state[s] = dict(gstate.get(s, {}))
                p = prev_positions.get(s)
                if p:
                    positions_store[s] = p
                    report_positions.append({"symbol": s, "side": "SHORT", "contracts": int(p["contracts"]),
                                             "notional": -cur, "score": None})
                continue
            # no-trade band: on a rebalance, only resize an existing leg if the target drifts beyond the band;
            # between rebalances (do_rebal False) leave every leg exactly as-is -> zero turnover while holding.
            if not do_rebal:
                target = cur
            elif cur > 0 and abs(unit - cur) / cur < fh.no_trade_band:
                target = cur
            else:
                target = unit
            turnover += abs(target - cur)
            entry_ts = float(gstate.get(s, {}).get("entry_ts", now)) if cur > 0 else now
            new_state[s] = {"notional": target, "basis": d["basis"], "entry_ts": entry_ts,
                            "neg": neg_run.get(s, 0)}
            if self.registry.has(s):
                spec = self.registry.get(s)
                contracts = spec.notional_to_contracts(target, d["mark"])
                if contracts > 0:
                    # surface the real age of the leg (entry_ts is what min_hold_days already tracks) — writing 0.0
                    # here made the dashboard's OPENED column read "—" for every position
                    positions_store[s] = {"contracts": -contracts, "avg_price": d["mark"],
                                          "multiplier": spec.multiplier, "opened_ts": entry_ts}
                    report_positions.append({"symbol": s, "side": "SHORT", "contracts": -contracts,
                                             "notional": -target, "score": round(d["funding"] * 24 * 365 * 100, 1)})
        for s, st in gstate.items():
            if s not in pickset:
                turnover += abs(float(st.get("notional", 0.0)))

        fee = turnover * 2 * cost           # two legs (spot + perp) per unit of turnover
        fees_cum += fee
        booked -= fee
        equity = cap + booked               # delta-neutral -> PnL booked each tick, no open MtM
        gross = sum(abs(float(st.get("notional", 0.0))) for st in new_state.values())
        peak = self.store.update_peak_equity(self.mode, equity)
        dd = drawdown_pct(peak, equity)

        self.store.set_meta("funding_state", new_state)
        self.store.set_meta("funding_last_ts", now)
        if do_rebal:
            self.store.set_meta("funding_last_rebal", now)
        # realized_pnl = basis PnL, funding_pnl = funding earned (kept apart so the dashboard shows real funding)
        self.store.save_account(self.mode, starting_capital=cap, realized_pnl=basis_cum,
                                fees_paid=fees_cum, funding_pnl=fund_cum, last_funding_ts=now)
        self.store.save_positions(self.mode, positions_store)
        self.store.record_equity(self.mode, equity, gross, 0.0, dd)   # net = 0 (delta-neutral by construction)
        self.store.save_scores(self.mode, {p["symbol"]: p["score"] for p in report_positions
                                           if p["score"] is not None})
        log.info("FUNDING | %s | equity=$%.2f | %d coins | gross=$%.0f | dd=%.2f%% | funding %+.2f fee %.2f "
                 "| api %d calls", "REBAL" if do_rebal else "hold", equity, len(picks), gross, dd,
                 new_funding, fee, len(info) * (2 if do_rebal else 1))
        return RebalanceReport(mode=self.mode, equity=equity, gross=gross, net=0.0, drawdown_pct=dd,
                               paused=False, universe_size=len(universe), n_long=0, n_short=len(report_positions),
                               n_orders=int(turnover > 0), positions=report_positions, fills=[], top_scores=[])

    def flatten(self) -> RebalanceReport:
        return self.run_once(flatten=True)

    def _persist_paper(self) -> None:
        if self.live:
            return
        b = self.broker
        self.store.save_account(self.mode, starting_capital=b.starting_capital, realized_pnl=b.realized_pnl,
                                fees_paid=b.fees_paid, funding_pnl=b.funding_pnl, last_funding_ts=b.last_funding_ts)
        self.store.save_positions(self.mode, {
            s: {"contracts": p.contracts, "avg_price": p.avg_price, "multiplier": p.multiplier,
                "opened_ts": p.opened_ts} for s, p in b._positions.items()})

    def _beta_neutralize(self, book, betas: dict, equity: float) -> None:
        """Shrink the heavier-beta sleeve until the book's net BTC-beta is ~0.

        Dollar-neutral is not market-neutral. Carry SHORTS high-funding coins, and high funding means
        crowded longs — which are the hot, high-beta names. So its short sleeve carries systematically
        more beta than its long sleeve, and the book ends up quietly net-SHORT the market. Measured live:
        longs averaged beta 1.14 against shorts at 1.38, a $348 dollar-net book sitting on -$1,169 of
        beta — about 12% of equity short, which is why a market rally showed up as losses on a book
        that is supposed to be indifferent to direction.

        beta_neutralize was already True in every config; it simply was never applied outside the
        sentiment book. The shrink is bounded by beta_max_imbalance and by max_net_exposure so the
        hedge cannot itself breach the dollar-neutral rail.
        """
        cfg = self.cfg
        if not (cfg.portfolio.beta_neutralize and betas and book and book.positions):
            return
        sn = {s: p.target_notional for s, p in book.positions.items()}
        b0 = book_beta(sn, betas, equity)
        heavier = max(sum(n for n in sn.values() if n > 0), sum(-n for n in sn.values() if n < 0))
        net_cap = (cfg.risk.max_net_exposure * equity / heavier) if heavier > 0 else 0.0
        self._apply_scales(book, beta_scales(sn, betas, min(cfg.portfolio.beta_max_imbalance, net_cap)))
        b1 = book_beta({s: p.target_notional for s, p in book.positions.items()}, betas, equity)
        self.store.set_meta("book_beta", round(b1, 4))
        if abs(b0) >= 0.05:
            log.info("beta-neutralize: book BTC-beta %+.2f -> %+.2f", b0, b1)

    def _apply_scales(self, book, scales: dict) -> None:
        """Apply per-symbol scale factors (<=1) to a target book in place, re-rounding to contracts.
        Shared by the beta-hedge, ADV-cap, and net-clamp robustness stages."""
        for s, p in book.positions.items():
            f = scales.get(s, 1.0)
            if f < 0.999:
                spec = self.registry.get(s)
                p.target_contracts = int(p.target_contracts * f)
                rn = spec.contracts_to_notional(p.target_contracts, p.price)
                p.target_notional = rn if p.target_contracts > 0 else -rn

    def _adv_quote(self, data: dict) -> dict:
        """Estimate each name's average daily quote volume (USDT) from recent kline volume.
        per-bar quote = contracts * multiplier * price; scaled to a day by the bar count."""
        per_day = max(1.0, 24.0 / parse_duration_to_hours(self.base_interval))
        out = {}
        for s, d in data.items():
            vv = [v for v in d.volumes[-30:] if v and v > 0]
            if vv and self.registry.has(s):
                avg_bar = self.registry.get(s).contracts_to_notional(sum(vv) / len(vv), d.last_price)
                out[s] = avg_bar * per_day
        return out

    # -- safety rails (run frequently by the loop between rebalances) ---------
    def regime_gate_changed(self) -> bool:
        """True when champion's BTC regime brake has flipped since it was last observed.

        The brake is a binary function of an observable — is BTC above its 100-day average — but it
        was only ever SAMPLED at the daily rebalance. On 2026-08-19 that cost the book a ~9% move:
        at the 14:00 cycle BTC sat 0.6% UNDER its 100-day MA, so the brake was correctly off and the
        book stayed flat; the market crossed 58 minutes later and the next observation was 23 hours
        after that, by which point BTC had run from 65,892 to 71,781.

        Nothing malfunctioned — but "invest when BTC is above its 100-day average", which is what
        the config says, and "invest if BTC was above its average at 14:00 yesterday", which is what
        a daily cron delivers, are different specifications. This closes that gap by re-reading the
        gate on the existing monitor tick and asking for a rebalance when it moves.

        HONESTY ABOUT EVIDENCE: this is NOT backed by a backtest. Hyperliquid serves at most 5,000
        hourly bars (208 days), champion needs 100 of those days just to warm up its MA, and the
        remaining window contains THREE regime flips. A test on three events cannot distinguish this
        from luck, and the one it would lean on is the incident that prompted it. The justification
        is that the gate should do what it is specified to do, not that a backtest liked it.

        Never raises, and self-throttles — the poll is cheap but the tick is every 60s.
        """
        c = self.cfg
        if c.strategy not in ("champion", "consensus") \
                or not getattr(c.champion, "intraday_regime_check", False):
            return False
        now = time.time()
        try:
            last = float(self.store.get_meta("regime_poll_ts", 0) or 0)
            if now - last < REGIME_POLL_SECONDS:
                return False
            self.store.set_meta("regime_poll_ts", int(now))
            rows = self.fx.klines(self.btc_sym, "1day", c.champion.regime_ma + 5) or []
            closes = [float(r["close"]) for r in rows if r.get("close")]
            if len(closes) < c.champion.regime_ma + 1:
                return False
            on = regime_on(closes, c.champion.regime_ma)
            prev = self.store.get_meta("regime_state", None)
            self.store.set_meta("regime_state", bool(on))
            if prev is None:
                return False                      # first observation sets the baseline only
            if bool(on) == bool(prev):
                return False
            log.info("regime brake flipped %s -> %s intraday; requesting an early rebalance",
                     "ON" if prev else "OFF", "ON" if on else "OFF")
            return True
        except Exception as e:                    # noqa: BLE001
            log.warning("intraday regime check failed: %s", e)
            return False

    def is_paused(self) -> bool:
        return time.time() < self.store.paused_until(self.mode)

    def _marks_for(self, symbols) -> dict:
        out = {}
        for s in symbols:
            if self.registry and self.registry.has(s):
                mp = mark_price(self.fx, s)
                if mp > 0:
                    out[s] = mp
        return out

    def _index_data(self, symbols):
        """One index call per symbol -> (marks, funding_rates). Fetched concurrently."""
        def _one(s):
            if not (self.registry and self.registry.has(s)):
                return None
            idx = self.fx.index(s)
            p = float(idx.get("tagPrice") or 0)
            return (s, p, float(idx.get("currentFundRate") or 0)) if p > 0 else None

        marks, funding = {}, {}
        for r in pmap(_one, symbols, workers=12):
            if r:
                marks[r[0]], funding[r[0]] = r[1], r[2]
        return marks, funding

    def _accrue_funding(self, positions) -> None:
        if self.live:
            return  # live funding is settled by the exchange itself
        marks, funding = self._index_data(positions) if positions else ({}, {})
        self.broker.accrue_funding(funding, marks, interval_hours=self._funding_interval_h)

    def _run_treasury(self, raw_equity: float) -> float:
        """Money-manager weekly profit sweep. Returns the current vault balance (0 if disabled)."""
        t = self.store.load_treasury(self.mode)
        trading_eq = max(0.0, raw_equity - t["vault"])
        res = self.treasury.maybe_sweep(trading_eq, t["vault"], t["hwm"],
                                        t["last_sweep_ts"], time.time())
        if res.swept > 0:
            log.info("TREASURY: swept $%.2f (%.0f%% of new profit) to vault -> vault=$%.2f",
                     res.swept, self.cfg.treasury.sweep_frac * 100, res.vault)
            self.store.save_treasury(self.mode, res.vault, res.hwm, res.last_sweep_ts,
                                     t["total_swept"] + res.swept)
        elif res.checkpoint or abs(res.hwm - t["hwm"]) > 1e-9:
            self.store.save_treasury(self.mode, res.vault, res.hwm, res.last_sweep_ts, t["total_swept"])
        return res.vault

    def _live_equity(self, marks: dict) -> float:
        if not self.live:
            self.broker.set_marks(marks)
            return self.broker.equity()
        return self.broker.equity() or (self.cfg.capital_usdt or 0.0)

    def _exposure(self, positions: dict, marks: dict):
        gross = net = 0.0
        for s, c in positions.items():
            if not self.registry.has(s) or s not in marks:
                continue
            ntl = abs(c) * marks[s] * self.registry.get(s).multiplier
            gross += ntl
            net += ntl if c > 0 else -ntl
        return gross, net

    def close_all(self, reason: str = "manual") -> list:
        self._ensure_market()
        cur = self.broker.positions()
        marks = self._marks_for(cur)
        orders = reconcile({}, cur, marks, self.registry, 0.0)
        if self.live and orders:
            self.broker.prepare(sorted({o.symbol for o in orders}), self.registry,
                                self.cfg.portfolio, self.cfg.execution)
        fills = self.broker.execute(orders, self.registry, self.cfg.execution)
        if not self.live:
            self.broker.set_marks(marks)
            self._persist_paper()
        eq = self.broker.equity() or 0.0
        peak = self.store.update_peak_equity(self.mode, eq)
        self.store.record_equity(self.mode, eq, 0.0, 0.0, drawdown_pct(peak, eq))
        self.store.record_fills(self.mode, fills)
        self.store.save_scores(self.mode, {})
        self._record_closed()
        log.warning("FLATTEN ALL (%s): closed %d positions, equity=%.2f", reason, len(cur), eq)
        return fills

    def close_symbol(self, sym: str, marks: dict) -> None:
        cur = self.broker.positions()
        if sym not in cur:
            return
        orders = reconcile({}, {sym: cur[sym]}, marks, self.registry, 0.0)
        fills = self.broker.execute(orders, self.registry, self.cfg.execution)
        if not self.live:
            self.broker.set_marks(marks)
            self._persist_paper()
        self.store.record_fills(self.mode, fills)
        self._record_closed()
        log.warning("catastrophe stop: closed %s", sym)

    def risk_check(self) -> dict:
        """Lightweight between-rebalance safety check: drawdown kill-switch + per-position stop.
        Records an equity tick so the curve/dashboard stay live."""
        if self.cfg.strategy == "grid":
            return self._grid_risk_check()
        if self.cfg.strategy == "funding":
            # The funding book books its PnL and enforces its own drawdown brake on every tick, so there is
            # no between-tick price poll here (that scan is what used to stall the loop). Report the state
            # honestly instead of a bare None, so a paused book is visible rather than silently idle.
            eq = self.store.last_equity(self.mode)
            peak = self.store.peak_equity(self.mode, eq or 0.0) if eq is not None else None
            dd = drawdown_pct(peak, eq) if eq is not None and peak else 0.0
            if self.is_paused():
                return {"action": "paused", "until": self.store.paused_until(self.mode),
                        "drawdown_pct": round(dd, 2)}
            return {"action": "none", "equity": round(eq, 2) if eq is not None else None,
                    "drawdown_pct": round(dd, 2)}
        self._ensure_market()
        cur = self.broker.positions()
        marks, funding = self._index_data(cur) if cur else ({}, {})
        if not self.live:
            self.broker.accrue_funding(funding, marks, interval_hours=self._funding_interval_h)
        eq = self._live_equity(marks)
        peak = self.store.update_peak_equity(self.mode, eq)
        dd = drawdown_pct(peak, eq)
        gross, net = self._exposure(cur, marks)
        # check the kill-switch every monitor tick, but only PERSIST an equity point every ~15 min so
        # the curve stays daily-clean (the dashboard adds a live "now" point regardless of this)
        now = time.time()
        if now - getattr(self, "_last_eq_persist", 0.0) >= 900:
            self.store.record_equity(self.mode, eq, gross, net, dd)
            self._last_eq_persist = now
        # daily-loss circuit-breaker: at each day rollover, score the day that just ended; a big down-day
        # shrinks the NEXT day's gross (read in run_once). Matches the backtest's next-bar cut.
        if self.cfg.risk.daily_loss_limit_pct > 0:
            today = time.strftime("%Y-%m-%d", time.localtime(now))
            if self.store.get_meta("day_start_date") != today:
                ds_eq = self.store.get_meta("day_start_eq")
                if ds_eq and float(ds_eq) > 0:
                    day_ret = eq / float(ds_eq) - 1.0
                    self.store.set_meta("daily_loss_scale",
                                        self.cfg.risk.daily_loss_scale
                                        if day_ret <= -self.cfg.risk.daily_loss_limit_pct / 100.0 else 1.0)
                self.store.set_meta("day_start_date", today)
                self.store.set_meta("day_start_eq", round(eq, 4))
        if dd > self.cfg.risk.max_drawdown_pct:
            self.close_all("drawdown kill-switch %.2f%% > %.2f%%" % (dd, self.cfg.risk.max_drawdown_pct))
            self.store.set_paused_until(self.mode, time.time() + self.cfg.risk.pause_hours * 3600)
            return {"action": "flatten_drawdown", "drawdown_pct": round(dd, 2)}
        paper_pos = getattr(self.broker, "_positions", {})
        for sym, contracts in list(cur.items()):
            pp = paper_pos.get(sym)
            if pp is None or sym not in marks:
                continue
            pnl = (marks[sym] - pp.avg_price) * contracts * pp.multiplier
            entry_notional = abs(pp.avg_price * contracts * pp.multiplier)
            # per-name stop (caps short-squeeze / single-name blow-ups like the ACE short) OR the
            # catastrophe backstop (a name melting a big % of the whole book) — whichever trips first.
            if (position_stop_breached(pnl, entry_notional, self.cfg.risk.position_stop_pct)
                    or position_loss_breached(pnl, eq, self.cfg.risk.max_position_loss_pct)):
                self.close_symbol(sym, marks)
                return {"action": "close_position", "symbol": sym, "loss": round(pnl, 2)}
        return {"action": "none", "equity": round(eq, 2), "drawdown_pct": round(dd, 2)}

    def _record_closed(self) -> None:
        """Drain the broker's closed round-trips into the store (the Trades ledger)."""
        ct = getattr(self.broker, "closed_trades", None)
        if ct:
            self.store.record_closed_trades(self.mode, ct)
            ct.clear()

    # -- snapshots for the dashboard -----------------------------------------
    def _persist_whales(self, snaps, coin_bias: dict) -> None:
        wallets = [{"address": s.address, "equity": s.equity, "gross": s.gross, "net": s.net}
                   for s in snaps]
        consensus = [{"coin": c, "bias": round(v, 4)}
                     for c, v in sorted(coin_bias.items(), key=lambda x: -abs(x[1]))
                     if abs(v) >= 0.005][:20]
        self.store.save_whales(consensus, wallets)

    # -- read-only diagnostics ------------------------------------------------
    def selftest(self) -> None:
        log.info("== Sentinel Edge selftest (read-only) ==")
        log.info("futures ping: %s | serverTime=%s", self.fx.ping(), self.fx.server_time())
        self._ensure_market()
        assert self.registry is not None and self.signal is not None
        universe = build_universe(self.fx, self.registry, self.cfg.universe)
        log.info("universe (%d): %s", len(universe), ", ".join(universe[:12]) + (" ..." if len(universe) > 12 else ""))
        data, prices, funding = load_symbol_data(self.fx, self.registry, universe,
                                                 self.base_interval, self.history_bars)
        scores = self.signal.scores(data)
        equity = self.cfg.capital_usdt or 10_000.0
        book = self.strategy.build_book(scores, prices, self.registry, equity)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        log.info("scored %d symbols. Top: %s", len(scores),
                 ", ".join(f"{s}={v:+.2f}" for s, v in top[:5]))
        log.info("bottom: %s", ", ".join(f"{s}={v:+.2f}" for s, v in top[-5:]))
        log.info("proposed book @ equity=%.0f gross=%.0f net=%.2f | %d long / %d short",
                 equity, book.realized_gross, book.realized_net,
                 sum(1 for p in book.positions.values() if p.target_contracts > 0),
                 sum(1 for p in book.positions.values() if p.target_contracts < 0))
        orders = reconcile(book.signed_contracts(), {}, prices, self.registry,
                           self.cfg.execution.min_trade_notional)
        log.info("orders it WOULD place (%d):", len(orders))
        for o in orders:
            log.info("  %-14s %-4s %-5s vol=%-8d @~%.6g  ($%.0f)", o.symbol, o.side, o.open,
                     o.volume, o.ref_price, o.notional)
        if self.cfg.api_key and self.cfg.api_secret:
            try:
                acct = self.fx.account()
                coins = [a.get("marginCoin") for a in (acct.get("account") or [])]
                log.info("signed account OK (signing works). margin coins: %s", coins)
            except Exception as e:
                log.error("signed account call FAILED: %s", e)
        else:
            log.info("no API keys set -> skipped signed account check (public data only).")

    def close(self) -> None:
        if self.whales:
            self.whales.close()
        self.store.close()
        if self.client is not None:
            self.client.close()
        elif hasattr(self.fx, "close"):
            self.fx.close()
