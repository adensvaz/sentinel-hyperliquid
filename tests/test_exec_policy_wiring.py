"""The learned policy must be able to observe live orders without being able to break them.

Shadow mode is the whole safety argument: the policy watches every order, records what it would
have done, and has no authority until store.exec_stats() proves it beats the fixed default. These
tests assert that separation actually holds in code, because "it only logs" is exactly the kind of
claim that quietly stops being true.
"""
import tempfile

from sentinel.execution.broker import Order
from sentinel.execution.exec_policy import ARMS, ExecutionPolicy
from sentinel.execution.live_broker import LiveBroker
from sentinel.exchange.contracts import ContractRegistry
from sentinel.state.store import Store

SYM = "BTC"


def _registry():
    return ContractRegistry([{"symbol": SYM, "type": "E", "marginCoin": "USDT", "multiplier": 0.001,
                              "multiplierCoin": SYM, "pricePrecision": 2, "minOrderVolume": 1,
                              "maxMarketVolume": 100_000_000, "minLever": 0, "maxLever": 100,
                              "status": 1}])


class _Cfg:
    maker = True
    order_type = "limit"
    slippage_bps = 5
    time_in_force = "IOC"
    margin_model_code = 1


class _Fx:
    """Records every order the broker actually sends."""
    def __init__(self):
        self.sent = []

    def create_order(self, **kw):
        self.sent.append(kw)
        return f"oid{len(self.sent)}"


def _order():
    return Order(symbol=SYM, side="BUY", open="OPEN", volume=10, ref_price=100.0, notional=1000.0)


def _store():
    return Store(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)


def test_shadow_mode_does_not_change_what_is_sent():
    """THE safety test. With policy_live=False the wire behaviour must be byte-identical to
    having no policy at all, whatever the policy happens to think."""
    fx_a, fx_b = _Fx(), _Fx()
    plain = LiveBroker(fx_a)
    shadow = LiveBroker(fx_b, exec_policy=ExecutionPolicy(seed=1), store=_store())
    reg, cfg = _registry(), _Cfg()
    plain.execute([_order()], reg, cfg)
    shadow.execute([_order()], reg, cfg)
    assert fx_a.sent == fx_b.sent, "shadow policy altered a live order"
    assert fx_b.sent[0]["time_in_force"] == "POST_ONLY"


def test_shadow_mode_still_records_what_it_would_have_done():
    st = _store()
    b = LiveBroker(_Fx(), exec_policy=ExecutionPolicy(seed=1), store=st)
    b.execute([_order()], _registry(), _Cfg())
    rows = st.unresolved_exec("live")
    assert len(rows) == 1
    assert rows[0]["shadow"] in ARMS
    assert rows[0]["arrival_mid"] == 100.0
    assert rows[0]["cost_bps"] is None, "outcome must stay unknown until it is actually observed"


def test_policy_failure_never_blocks_an_order():
    """An execution learner that can halt trading is a liability, not an asset."""
    class Broken:
        def context(self, **kw):
            raise RuntimeError("boom")

        def choose(self, *a, **k):
            raise RuntimeError("boom")

    fx = _Fx()
    b = LiveBroker(fx, exec_policy=Broken(), store=_store())
    fills = b.execute([_order()], _registry(), _Cfg())
    assert len(fx.sent) == 1 and len(fills) == 1


def test_broken_store_never_blocks_an_order():
    class BadStore:
        def record_exec(self, *a, **k):
            raise RuntimeError("disk full")

    fx = _Fx()
    b = LiveBroker(fx, exec_policy=ExecutionPolicy(seed=1), store=BadStore())
    b.execute([_order()], _registry(), _Cfg())
    assert len(fx.sent) == 1


def test_policy_live_can_override_but_only_when_enabled():
    """When it HAS earned authority, choosing CROSS must actually stop the POST_ONLY."""
    class AlwaysCross(ExecutionPolicy):
        def choose(self, *a, **k):
            return "CROSS"

    fx = _Fx()
    b = LiveBroker(fx, exec_policy=AlwaysCross(seed=1), store=_store(), policy_live=True)
    b.execute([_order()], _registry(), _Cfg())
    assert fx.sent[0]["time_in_force"] != "POST_ONLY", "policy_live had no effect"


def test_exec_stats_scores_shadow_against_reality():
    st = _store()
    rid = st.record_exec("live", symbol=SYM, side="BUY", context="us_open|s0|v0|N",
                         arm="POST", shadow="CROSS", arrival_mid=100.0, notional=1000.0)
    st.resolve_exec(rid, fill_price=100.05, cost_bps=5.0)
    s = st.exec_stats("live")
    assert s["n"] == 1
    assert s["by_arm"]["POST"]["mean_bps"] == 5.0
    assert s["agreement"] == 0.0          # shadow disagreed on this order


def test_exec_stats_empty_is_safe():
    assert _store().exec_stats("live")["n"] == 0


def test_regime_gate_changed_only_fires_on_a_flip():
    """The intraday regime brake must trigger a rebalance when BTC crosses its MA, stay silent
    otherwise, set a baseline on first observation, and never raise. Guards the 2026-08-19 failure
    where a daily-sampled gate missed a crossing by 58 minutes."""
    import types
    import sentinel.engine.engine as E

    class Store:
        def __init__(self):
            self.m = {}

        def get_meta(self, k, d=None):
            return self.m.get(k, d)

        def set_meta(self, k, v):
            self.m[k] = v

    def mk(closes):
        eng = object.__new__(E.Engine)
        eng.cfg = types.SimpleNamespace(
            strategy="champion",
            champion=types.SimpleNamespace(regime_ma=3, intraday_regime_check=True))
        eng.store = Store()
        eng.btc_sym = "BTC"
        eng.fx = types.SimpleNamespace(
            klines=lambda s, i, n: [{"close": c} for c in closes])
        return eng

    eng = mk([10, 10, 10, 9])                       # below the 3-bar MA -> OFF
    assert eng.regime_gate_changed() is False       # first look only sets the baseline
    assert eng.store.get_meta("regime_state") is False

    eng.store.set_meta("regime_poll_ts", 0)         # allow an immediate re-poll
    eng.fx.klines = lambda s, i, n: [{"close": c} for c in [10, 10, 10, 20]]   # now ABOVE -> flip
    assert eng.regime_gate_changed() is True

    eng.store.set_meta("regime_poll_ts", 0)
    assert eng.regime_gate_changed() is False       # same state twice -> silent

    # throttle: a second call inside REGIME_POLL_SECONDS must not re-read
    eng.fx.klines = lambda s, i, n: [{"close": c} for c in [10, 10, 10, 1]]
    assert eng.regime_gate_changed() is False

    # disabled for the other books
    eng2 = mk([10, 10, 10, 9]); eng2.cfg.strategy = "carry"
    assert eng2.regime_gate_changed() is False

    # a broken exchange must not stop the loop
    eng3 = mk([10, 10, 10, 9])
    eng3.fx.klines = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down"))
    assert eng3.regime_gate_changed() is False


def test_trend_stays_invested_when_every_coin_is_in_an_uptrend():
    """A cross-sectional book must short the WEAKEST name even when nothing is falling.

    Guards the 2026-08-21 incident: all 30 universe coins sat in the upper half of their 45-day
    range, the old `if t < 0` filter emptied the short sleeve, and the live book closed all 10
    positions and held nothing for four days through a rally.
    """
    from sentinel.config import load_config
    from sentinel.strategy.trend import TrendStrategy

    cfg = load_config("config.hl-trend.yaml")
    st = TrendStrategy(cfg)
    K = cfg.trend.top_k
    don = getattr(cfg.trend, "donchian_period", 45)

    class Spec:
        def notional_to_contracts(self, n, p): return max(int(n / p), 1)
        def contracts_to_notional(self, c, p): return c * p

    class Reg:
        def has(self, s): return True
        def get(self, s): return Spec()

    # Every coin rallies, then retraces by a DIFFERENT amount. So every breakout score is
    # positive (nothing is in an absolute downtrend) but they still differ, which is exactly the
    # market state that emptied the live book. A purely monotonic rise would put every coin at its
    # range high and score them all +0.5 — no spread, nothing to rank.
    n = 2 * K + 4
    closes, prices = {}, {}
    for i in range(n):
        rise = [100.0 * 1.01 ** k for k in range(don + 5)]
        top = rise[-1]
        retrace = 0.02 * i                       # 0% .. ~40% back off the high
        series = rise + [top * (1.0 - retrace)]
        closes[f"C{i}"] = series
        prices[f"C{i}"] = series[-1]

    book = st.build_book(closes, prices, Reg(), equity=10_000.0)
    longs = [p for p in book.positions.values() if p.target_contracts > 0]
    shorts = [p for p in book.positions.values() if p.target_contracts < 0]

    assert longs, "long sleeve must be populated"
    assert shorts, "short sleeve must be populated even with nothing in an absolute downtrend"
    assert len(longs) == K and len(shorts) == K, "book must be a balanced top-K / bottom-K"
    # and it must short the WEAKEST names, not the strongest
    assert min(p.score for p in longs) > max(p.score for p in shorts)


def test_equity_series_keeps_the_newest_data_however_long_the_book_runs(tmp_path):
    """Reproduces the champion 7D outage: 74 days of 15-minute ticks (~7,100 rows) used to come
    back as the OLDEST 5,000 via `ORDER BY ts ASC LIMIT 5000`, silently dropping the last three
    weeks. The 7D chart then held one point while the table held 673 for that week."""
    import time
    from sentinel.state.store import Store

    st = Store(str(tmp_path / "t.db"))
    now = int(time.time())
    step = 900                                            # one tick every 15 min
    n = 74 * 96                                           # 74 days -> 7,104 rows, past the old cap
    st.conn.executemany(
        "INSERT INTO equity_curve(ts,mode,equity,gross,net,drawdown_pct) VALUES(?,?,?,?,?,?)",
        [(now - (n - i) * step, "paper", 10_000 + i, 0.0, 0.0, 0.0) for i in range(n)])
    st.conn.commit()

    s = st.equity_series("paper")
    ts = [r["ts"] for r in s]
    assert ts == sorted(ts), "must be chronological"

    week = [r for r in s if r["ts"] >= now - 7 * 86400]
    assert len(week) >= 7 * 96 - 2, f"last 7 days must be at full resolution, got {len(week)}"
    assert s[-1]["equity"] == 10_000 + n - 1, "the NEWEST row must be present"

    older = [r for r in s if r["ts"] < now - 8 * 86400]
    assert 60 <= len(older) <= 70, f"older history collapses to ~1/day, got {len(older)}"
    assert s[0]["ts"] <= now - 73 * 86400, "the ALL view must still reach back to day one"
    assert len(s) < 1200, "payload stays small no matter how long the book runs"
    assert set(s[0].keys()) == {"ts", "equity", "gross", "net", "drawdown_pct"}
