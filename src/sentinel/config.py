"""Typed configuration: loads config.yaml + secrets from .env, validates with pydantic."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class ExchangeCfg(BaseModel):
    venue: Literal["koinbay", "hyperliquid"] = "koinbay"   # which exchange to read/trade
    futures_host: str = "https://futuresopenapi.koinbay.com"
    spot_host: str = "https://openapi.koinbay.com"
    hyperliquid_info_url: str = "https://api.hyperliquid.xyz/info"   # HL public data (Phase 1, no keys)
    recv_window_ms: int = 5000
    timeout_s: float = 15.0


class UniverseCfg(BaseModel):
    top_n: int = 24
    allowlist: list[str] = Field(default_factory=list)
    denylist: list[str] = Field(default_factory=list)
    min_quote_volume_usdt: float = 500_000
    min_listing_days: int = 90       # a coin must have traded this long before we will hold it.
                                     # Volume alone does not keep you safe: CASHCAT was rank-8 by volume
                                     # ($14M/day) yet 24 days old, and its worst day swung 181% — against
                                     # 7-35% for every other name in the book. It cost Carry $533 on one
                                     # short, more than the book's whole profit over 110 trades. Age is the
                                     # tell that volume misses, and unlike a volatility screen it blocks the
                                     # NEXT one before it has shown anyone what it can do.


class SignalWeights(BaseModel):
    momentum: float = 0.5
    funding: float = 0.3
    funding_vel: float = 0.0  # funding velocity (next - current): positioning acceleration (live-only)
    volume: float = 0.2
    whale: float = 0.0      # blended smart-money positioning (0 = off)
    reversal: float = 0.0   # short-horizon mean-reversion: fade recent movers (anti-correlated to momentum)


class SignalCfg(BaseModel):
    momentum_horizons: list[str] = Field(default_factory=lambda: ["4h", "24h", "7d"])
    weights: SignalWeights = Field(default_factory=SignalWeights)
    funding_sign: int = -1  # -1 contrarian (fade crowded funding), +1 trend-follow
    reversal_hours: float = 24.0  # horizon to fade for the reversal factor
    ta_veto: float = 0.0    # TA-confirmation veto (0=off; e.g. 0.3 blocks longs with TA<-0.3 / shorts with TA>+0.3)


class WhalesCfg(BaseModel):
    """Track a basket of top Hyperliquid traders and blend their net positioning into the score.
    The live basket is read from `source_file` (written by `refresh-whales`); `addresses` is the
    fallback seed if that file is absent."""
    enabled: bool = False
    addresses: list[str] = Field(default_factory=list)
    source_file: str = "data/whales.json"
    weighting: Literal["equal", "equity"] = "equal"
    info_url: str = "https://api.hyperliquid.xyz/info"
    refresh_days: int = 7          # auto re-screen the basket when it's older than this
    max_inactive_days: int = 5     # drop wallets with no on-chain fill in this many days (recency)


class PortfolioCfg(BaseModel):
    longs: int = 5
    shorts: int = 5
    weighting: Literal["score", "equal"] = "score"
    conviction: float = 1.0   # weight ~ |score|^conviction; >1 concentrates into the strongest signals
    risk_parity: bool = False # size by inverse volatility (equal-risk): wild coins get smaller allocations
    beta_neutral_scores: bool = False  # rank by RESIDUAL (beta-adjusted) score so the book is born
                                       # market-neutral at selection — residual momentum, not raw
    target_gross_leverage: float = 3.0
    per_name_leverage: int = 3
    max_name_weight: float = 0.25
    beta_neutralize: bool = True       # neutralize the book's BTC-beta (dollar-neutral != market-neutral)
    beta_max_imbalance: float = 0.15   # cap the dollar imbalance introduced when neutralizing beta
    beta_lookback: int = 30            # bars of returns used to estimate each coin's BTC-beta
    adv_cap_frac: float = 0.0          # cap each name's notional at this fraction of its ADV (0 = off);
                                       # capacity guardrail for copy-trading — only binds at scale


class ExecutionCfg(BaseModel):
    order_type: Literal["limit", "market"] = "limit"
    maker: bool = False          # rest POST_ONLY at the touch (maker fee ~1/3 of taker; fills not guaranteed)
    maker_fill_ratio: float = Field(default=1.0, ge=0.0, le=1.0)   # fraction of POST_ONLY orders that
                                 # actually rest-and-fill as maker; the rest chase across the spread as
                                 # taker (higher fee + slippage). 1.0 = optimistic, ~0.5 = realistic.
    slippage_bps: float = 8.0
    time_in_force: Literal["IOC", "FOK", "POST_ONLY"] = "IOC"
    min_trade_notional: float = 25.0
    min_rebalance_frac: float = 0.10   # skip same-side adjustments smaller than this fraction of the
                                       # current position (anti-churn / fee guard)
    asym_rebalance: bool = False       # let winners run / don't feed losers: widen the no-trade band for
                                       # trimming a winning position or adding to a losing one
    asym_band_mult: float = 3.0        # band multiplier applied in those two cases
    take_profit_pct: float = 0.0       # close a held name once its gain since entry exceeds this % (0 = off)
    scale_out_pct: float = 0.0         # once a held name is up this %, permanently trim it (partial profit)
    scale_out_frac: float = 0.0        # ... by this fraction, keeping the rest running (0 = off)
    margin_mode: Literal["isolated", "cross"] = "isolated"
    position_mode: Literal["net", "two_way"] = "net"

    @property
    def margin_model_code(self) -> int:
        return 2 if self.margin_mode == "isolated" else 1

    @property
    def position_model_code(self) -> int:
        return 1 if self.position_mode == "net" else 2


class RiskCfg(BaseModel):
    max_gross_leverage: float = 10.0
    max_net_exposure: float = 0.05
    max_drawdown_pct: float = 15.0
    funding_abs_max: float = 0.003
    monitor_seconds: int = 60          # how often the safety rail checks risk between rebalances
    max_position_loss_pct: float = 6.0  # close a single position if its loss exceeds this % of equity
    position_stop_pct: float = 30.0     # per-name stop: close a position if it loses this % of its OWN
                                        # entry value. Caps short-squeeze / blow-up risk (e.g. the ACE
                                        # short that squeezed +89%). Measured vs the position's notional,
                                        # not equity, so it fires on one name before it wrecks the book.
                                        # Backtested: caps tail disasters without hurting the core trend
                                        # edge. 0 = off.
    pause_hours: float = 12.0          # after a drawdown kill-switch, stay flat this long
    stop_loss_pct: float = 0.0         # LIVE only: rest a protective stop this % from entry (0 = off)
    safe_withdraw_leverage: float = 2.0  # the dashboard's "Safe to withdraw" leaves enough buffer that
                                         # post-withdrawal gross leverage stays at/below this, so pulling
                                         # cash out never spikes you into liquidation risk before the bot
                                         # rebalances positions back down. (Max withdrawable is higher but
                                         # leaves no cushion — shown as a caveat.)
    # dynamic de-risking (smooths drawdowns / negative months by cutting gross in bad regimes)
    dynamic_risk: bool = True
    vol_target_daily: float = 0.015    # de-risk when recent daily book-return vol exceeds this
    vol_lookback: int = 21             # days of returns for the vol estimate
    dd_throttle_start: float = 0.04    # start cutting gross at this drawdown fraction
    dd_throttle_full: float = 0.12     # gross fully throttled (to min) at this drawdown fraction
    min_risk_scale: float = 0.25       # never cut gross below this fraction of base
    # momentum-crash guard (cuts gross in the bear-then-violent-bounce regime that runs over momentum)
    crash_guard: bool = True
    crash_dd_hours: float = 720.0      # bear context: drawdown measured over this window (~30d)
    crash_bounce_hours: float = 48.0   # snapback measured over this short window (~2d)
    crash_dd_trigger: float = 0.15     # require the market this far below its recent high
    crash_bounce_trigger: float = 0.05 # AND a snapback of at least this much over the short window
    crash_scale_floor: float = 0.5     # cut gross no lower than this fraction in a full-blown crash regime
    # dispersion gate (shrink the book on no-edge days when coins move together)
    dispersion_gate: bool = False
    dispersion_ref_window: int = 20    # trailing window (rebalances) for the dispersion reference
    dispersion_ref_abs: float = 0.0    # if >0, use this ABSOLUTE healthy-dispersion level as the reference
                                       # (catches sustained low-dispersion regimes a trailing median misses)
    dispersion_min_scale: float = 0.5  # floor when dispersion collapses
    # daily loss circuit-breaker (cap the worst single days)
    daily_loss_limit_pct: float = 0.0  # if a day loses >= this %, shrink next day's gross (0 = off)
    daily_loss_scale: float = 0.5      # gross multiplier applied the day after a daily-loss breach


class ChampionCfg(BaseModel):
    """Directional momentum + regime strategy (the backtest champion). Long-only, keeps market beta."""
    top_k: int = 5                 # hold the top-K strongest names
    lookback: int = 30             # momentum lookback (days)
    trend_ma: int = 50             # a name must be above this MA to qualify (absolute uptrend)
    regime_ma: int = 100           # only invest when BTC is above this MA (risk-on brake)
    dual_confirm: bool = True      # require both relative strength AND absolute uptrend
    target_gross: float = 1.0      # gross leverage when fully risk-on (vol-target may scale below)
    # --- improvements validated on 5.5yr walk-forward (scripts/champion_research.py) ---
    breadth_scale: bool = False    # scale gross by market breadth (fraction of universe above breadth_ma):
                                   # a parallel market-health check that de-risks thin/narrow rallies.
                                   # Backtest: maxDD 35%->24% with equal-or-better Sharpe & robustness.
    breadth_ma: int = 100          # MA each coin is measured against for the breadth count
    breadth_lo: float = 0.30       # gross scales to 0 at/below this breadth fraction
    breadth_hi: float = 0.80       # gross scales to full at/above this fraction (linear in between)
    mom_weighted: bool = False     # size by momentum strength instead of equal-weight (lean into leaders)
    max_name_weight: float = 0.40  # cap any single name at this fraction of gross (copy-trade-safe)
    intraday_regime_check: bool = False   # re-read the BTC regime brake on the monitor tick and
                                          # rebalance when it flips, instead of sampling it once a
                                          # day. See Engine.regime_gate_changed.


class CarryCfg(BaseModel):
    """Funding-carry + momentum combo (market-neutral). Long hi-momentum/lo-funding, short lo-momentum/
    hi-funding, dollar-neutral. Harvests the structural perp-funding premium; orthogonal to price."""
    top_k: int = 5                 # names per sleeve (long K / short K)
    lookback: int = 21             # momentum lookback (days)
    target_gross: float = 2.0      # total gross (~1x per side); vol-target/DD-throttle may scale below
    funding_avg_cycles: int = 5    # rank on the trailing average of this many funding snapshots
                                   # (funding is persistent — the average is a cleaner signal than a
                                   #  single instantaneous rate; backtest Sharpe ~1.6 -> ~2.0 with it)
    keep_buffer: int = 4           # HYSTERESIS. Enter at rank K, but keep an existing leg until it falls
                                   # out of rank K+buffer, so a name hovering at the cut-off is not sold
                                   # and re-bought on rank noise. Trade-level evidence: every dollar of
                                   # loss came from positions held <=2 days (339 trades, -$4,391) while
                                   # everything held longer made money (+$6,926). Those short holds are
                                   # names flickering across the boundary, paying a round trip each time.
                                   # A top-12 keep-band on a top-8 entry cut trades 648 -> 346, churn
                                   # trades 368 -> 106, and moved CAGR/Sharpe/maxDD 8.5%/0.44/22.8%
                                   # -> 33.1%/1.19/12.3%. Same fix that ended the funding book's bleed.
                                   # 0 restores the old drop-on-exit behaviour.


class TrendCfg(BaseModel):
    """Trend (CTA) book — dollar-neutral cross-sectional trend-following. Long the K coins in the
    strongest uptrends (price furthest above its MA), short the K in the strongest downtrends. The most
    durable systematic edge across markets; validated on 5yr Binance data at Sharpe ~1.35, +75-82% CAGR,
    positive every full year (raw DD tamed by the vol-target/DD-throttle overlay). Replaces the retired
    Funding-Alpha book, whose pure-funding-harvest edge decayed to a 5yr Sharpe of 0.38."""
    top_k: int = 8                 # names per sleeve (long K strongest up / short K strongest down)
    ma_period: int = 30            # trend strength = price vs this-day moving average
    target_gross: float = 2.0      # total gross (~1x per side); vol-target/DD-throttle may scale below
    score_mode: str = "ma"         # "ma" = price vs moving average | "breakout" = Donchian range
                                   # position. Default stays "ma" so existing configs are untouched.
    donchian_period: int = 45      # lookback for score_mode="breakout". 20/30/45 all clear zero at
                                   # every universe width on the survivorship-stripped backtest;
                                   # 45 is chosen for lowest turnover and best cost tolerance.


class ConsensusCfg(BaseModel):
    """Trade only what champion, carry and trend agree on. See strategy/consensus.py."""
    enter: int = 2                 # a name needs this many agreeing votes to be OPENED
    hold: int = 1                  # ...and is KEPT while it still has this many (hysteresis)
    max_per_side: int = 8          # cap each sleeve; strongest agreement kept if it binds
    target_gross: float = 1.0      # split across sleeves when both exist; all-long on a no-short day


class GridCfg(BaseModel):
    """Grid / market-making book — a RESTING-LIMIT order book per coin (strategy/grid_book.py, driven by
    engine.run_grid_once). Resting buy limits below a per-coin anchor fill on the bar low (buy the dip),
    rest a take-profit one spacing above, and book the spacing on the bounce; symmetric shorts above.
    1x/K sizing (NO leverage), a trend filter that stands aside in strong trends, and a hard stop.

    Validated (replaying grid_book over 4.5yr hourly, 7 coins, resting-limit fills + maker/slippage):
    portfolio Sharpe ~1.4 (last year ~1.9), +288% / +47% (1yr), ~+3%/month, maxDD ~12%, ~87% win,
    positive every year 2022-2025. Fee-sensitive (maker fills essential); best in chop."""
    levels: int = 6                # K: ladder depth per side. Each step = 1/K of the coin's budget.
    spacing_pct: float = 1.2       # % between grid levels (one step's profit target)
    stop_pct: float = 10.0         # recenter + flatten if price runs this far from the anchor (tail cap)
    range_band: float = 0.08       # only grid when |price/MA - 1| < this; stand aside in trends
    ma_period: int = 100           # MA (in base-interval bars) for the ranging/trend filter
    target_gross: float = 1.0      # total gross at full extension across ALL coins (1.0 = no leverage)
    fill_delta_pct: float = 0.05   # REALISM: a resting limit fills only if price trades this % THROUGH it
                                   # (models queue position / adverse selection). 0 = fill-on-touch (rosy,
                                   # +288%); 0.05% = realistic-conservative (~+7-25%/yr); breakeven ~0.07%.
                                   # We run PAPER at this conservative value and measure the real one live.


class FundingCfg(BaseModel):
    """Funding Harvest — DELTA-NEUTRAL cash-and-carry (engine.run_funding_once). Short the perps paying
    STABLE positive funding and hold a matching spot long, so price cancels and you collect the funding
    premium (the structural cost of leverage that longs pay). Market beta ~ 0 -> works in bull, bear, chop.

    The ONE edge that survived survivorship-honest, point-in-time backtesting: ~+15-20%/yr, positive every
    year (2024-26), ~0-3% drawdown, because delta-neutral harvesting doesn't depend on picking winning coins.
    Selection keeps a funding CONSISTENCY (t-stat) floor but fills fresh slots by funding MAGNITUDE (where the fat
    premium is). PAPER simulates the spot hedge (books funding + real mark-vs-oracle basis - fees); live = real spot.

    TURNOVER CONTROL is the whole game: funding accrues every tick regardless, so the book only re-picks/re-sizes
    every `rebalance_days` (not every tick), keeps names via a wide `keep_top_n` band + `min_hold_days`, and only
    resizes a leg past a `no_trade_band`. Rebalancing hourly with no band bleeds ~0.08%/day of gross in fees, which
    turns a backtested +14%/yr harvest into a ~-26%/yr loser — the fees, not the market, are the enemy here."""
    funding_window_days: int = 14    # trailing window to measure each coin's funding mean + consistency
    tstat_min: float = 0.7           # only harvest coins whose funding is persistently one-signed (mean/std)
    min_funding_daily: float = 0.0001  # ...and meaningfully positive (>1bp/day) so it clears fees
    max_coins: int = 15              # cap the book breadth (diversify across the richest-stable-funding names)
    target_gross: float = 3.0        # delta-neutral -> ~0 DD, so it safely carries higher gross than a directional book
    cost_per_side: float = 0.0004    # maker + slippage on each leg, charged on turnover (entry/exit/rotation)
    rebalance_days: float = 3.0      # re-pick/re-size the basket only this often (funding still accrues every tick)
    no_trade_band: float = 0.10      # only resize an existing leg when its target notional drifts beyond this
    min_hold_days: float = 3.0       # don't drop a name held less than this (unless its funding has gone negative)
    keep_top_n: int = 25             # a held name stays as long as it's still within this wide consistency band
    max_basis: float = 0.02          # drop a leg whose |mark - oracle| / oracle exceeds this (2%). The book only
                                     # works because perp ~ spot; a leg that has torn away from its oracle is one
                                     # where that has stopped being true, so its funding is no longer being
                                     # harvested against a hedge. Normal basis is well under 0.5%.
    exit_neg_ticks: int = 8          # consecutive NEGATIVE funding ticks on a held leg before re-picking early.
                                     # Individual coins print a negative hourly rate all the time — across ~15 legs
                                     # at least one is negative almost every hour — so reacting to a single print
                                     # forces a full re-pick every tick and undoes the turnover fix. Only a sustained
                                     # run (default ~8h) means the premium has genuinely flipped.


class ScheduleCfg(BaseModel):
    rebalance_minutes: int = 240
    rebalance_hour_utc: Optional[int] = None  # if 0-23, rebalance daily at this fixed UTC hour
                                              # (predictable + restart-proof); else use the interval


class StateCfg(BaseModel):
    db_path: str = "data/sentinel.db"
    equity_csv: str = "data/equity.csv"
    log_path: str = "data/sentinel.log"


class TreasuryCfg(BaseModel):
    """Money manager: sweep a fraction of NEW profit into a separate Vault on a schedule.

    Every `sweep_interval_days`, if the trading pool is at a new high, move `sweep_frac` of the
    new profit into the Vault — safe cash held aside for withdrawal or reinvestment (never traded).
    The at-risk trading pool = equity - vault, so the book de-risks as it wins. High-water-mark
    based, so it never sweeps a down/flat period (no over-sweeping in chop).

    Backtested trade-off (scripts/weekly_sweep_bt.py): banks steady income + cuts at-risk drawdown,
    at the cost of some compounding (cheap on Trend/Neutral, ~pricey on strong compounders like Carry)."""
    enabled: bool = False              # OFF: no auto-sweep — withdraw manually instead (the dashboard
                                       # shows "Free to withdraw"). Set true in a config to auto-bank profit.
    sweep_frac: float = 0.40           # take 40% of new profit each sweep (only if enabled)
    sweep_interval_days: float = 7.0   # weekly


class Config(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    # neutral = market-neutral momentum book; champion = directional momentum+regime; carry = funding-carry+momentum
    # trend = dollar-neutral CTA; grid = market-making (retired); funding = delta-neutral funding harvest
    strategy: Literal["neutral", "champion", "carry", "trend", "grid", "funding", "consensus"] = "neutral"
    exchange: ExchangeCfg = Field(default_factory=ExchangeCfg)
    capital_usdt: Optional[float] = 10_000.0
    # Portfolio-level allocation weight (0.0-1.0). When a risk-parity allocator runs
    # above the books, it writes this into each book's config to control what fraction
    # of the total portfolio capital this book trades with. 1.0 = use full capital (default).
    capital_weight: float = 1.0
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    signal: SignalCfg = Field(default_factory=SignalCfg)
    champion: ChampionCfg = Field(default_factory=ChampionCfg)
    carry: CarryCfg = Field(default_factory=CarryCfg)
    trend: TrendCfg = Field(default_factory=TrendCfg)
    consensus: ConsensusCfg = Field(default_factory=ConsensusCfg)
    grid: GridCfg = Field(default_factory=GridCfg)
    funding: FundingCfg = Field(default_factory=FundingCfg)
    whales: WhalesCfg = Field(default_factory=WhalesCfg)
    portfolio: PortfolioCfg = Field(default_factory=PortfolioCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    schedule: ScheduleCfg = Field(default_factory=ScheduleCfg)
    state: StateCfg = Field(default_factory=StateCfg)
    treasury: TreasuryCfg = Field(default_factory=TreasuryCfg)

    # secrets (from .env, never config.yaml)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    # Hyperliquid execution (Phase 2) — use an AGENT/API wallet (trade-only, cannot withdraw).
    # Read-only data + PAPER need neither of these.
    hl_account_address: Optional[str] = None   # your main account address (public)
    hl_agent_key: Optional[str] = None         # agent wallet private key — .env ONLY, never commit

    @model_validator(mode="after")
    def _check(self) -> "Config":
        if self.portfolio.target_gross_leverage > self.risk.max_gross_leverage:
            raise ValueError(
                f"portfolio.target_gross_leverage ({self.portfolio.target_gross_leverage}) "
                f"exceeds risk.max_gross_leverage ({self.risk.max_gross_leverage})"
            )
        if self.portfolio.per_name_leverage > self.risk.max_gross_leverage:
            raise ValueError("portfolio.per_name_leverage exceeds risk.max_gross_leverage")
        if self.portfolio.longs < 1 or self.portfolio.shorts < 1:
            raise ValueError("portfolio.longs and portfolio.shorts must each be >= 1")
        w = self.signal.weights
        if (w.momentum + w.funding + w.funding_vel + w.volume + w.whale + w.reversal) <= 0:
            raise ValueError("signal.weights must sum to a positive number")
        return self

    def require_keys(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "KOINBAY_API_KEY / KOINBAY_API_SECRET not set. Copy .env.example to .env and fill them in."
            )


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config.yaml, overlay secrets from the environment (.env), and validate."""
    load_dotenv()
    p = Path(path)
    data: dict = {}
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    cfg = Config(**data)
    cfg.api_key = os.getenv("KOINBAY_API_KEY") or None
    cfg.api_secret = os.getenv("KOINBAY_API_SECRET") or None
    cfg.hl_account_address = os.getenv("HL_ACCOUNT_ADDRESS") or None
    cfg.hl_agent_key = os.getenv("HL_AGENT_KEY") or None
    return cfg
