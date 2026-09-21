"""Consensus strategy: votes, hysteresis, sleeve construction, engine wiring."""
import types

from sentinel.config import load_config
from sentinel.strategy.consensus import ConsensusStrategy, consensus_votes


class _Spec:
    # continuous, like Hyperliquid's fractional sizes — integer rounding would make a $625 slice of a
    # $380 coin into one contract and break dollar-neutrality for reasons unrelated to the strategy
    def notional_to_contracts(self, n, p): return n / p
    def contracts_to_notional(self, c, p): return c * p


class _Reg:
    def has(self, s): return True
    def get(self, s): return _Spec()


def _cfg():
    return load_config("config.consensus.yaml")


def _series(start, daily_ret, n=140):
    return [start * (1 + daily_ret) ** k for k in range(n)]


def _world(n_coins=24):
    """Coins with monotonically graded strength: C0 strongest ... C23 weakest, plus a rising BTC
    so the regime brake is ON. Funding graded the other way so carry's rank agrees with momentum."""
    closes, funding, prices = {}, {}, {}
    for i in range(n_coins):
        r = 0.010 - 0.0009 * i                       # +1.0%/day down to about -1.1%/day
        s = _series(100.0, r); closes[f"C{i}"] = s; prices[f"C{i}"] = s[-1]
        funding[f"C{i}"] = -0.0002 + 0.00002 * i     # strongest names have the LOWEST funding -> best longs
    btc = _series(50_000.0, 0.004)
    return closes, funding, prices, btc


def test_votes_agree_on_the_extremes_and_champion_never_shorts():
    c = _cfg(); closes, funding, prices, btc = _world()
    v = consensus_votes(closes, funding, btc, c.champion, c.carry, c.trend)
    assert v["C0"] == 3, "strongest name: champion + carry + trend all long"
    assert v["C23"] == -2, "weakest name: carry + trend short, champion abstains"
    assert min(v.values()) >= -2 and max(v.values()) <= 3


def test_champion_abstains_when_regime_is_off():
    c = _cfg(); closes, funding, prices, btc = _world()
    falling_btc = _series(50_000.0, -0.004)
    v = consensus_votes(closes, funding, falling_btc, c.champion, c.carry, c.trend)
    assert max(v.values()) <= 2, "with the brake off nobody can reach +3"


def test_book_is_dollar_neutral_and_capped():
    c = _cfg(); st = ConsensusStrategy(c); closes, funding, prices, btc = _world()
    bk = st.build_book(closes, funding, prices, _Reg(), 10_000.0, btc_closes=btc)
    L = [p for p in bk.positions.values() if p.target_contracts > 0]
    S = [p for p in bk.positions.values() if p.target_contracts < 0]
    assert L and S
    assert len(L) <= c.consensus.max_per_side and len(S) <= c.consensus.max_per_side
    ln = sum(p.target_notional for p in L); sn = sum(p.target_notional for p in S)
    assert abs(ln + sn) < 0.05 * abs(ln), "dollar-neutral when both sleeves exist"
    assert all(p.score >= c.consensus.enter for p in L)
    assert all(p.score <= -c.consensus.enter for p in S)


def test_hysteresis_keeps_a_held_name_at_one_vote_but_does_not_open_one(monkeypatch):
    """Book construction in isolation: votes are injected so every level is present."""
    import sentinel.strategy.consensus as M
    c = _cfg(); st = ConsensusStrategy(c)
    syms = [f"C{i}" for i in range(8)]
    fixed = {"C0": 3, "C1": 2, "C2": 1, "C3": 0, "C4": -1, "C5": -2, "C6": -2, "C7": 1}
    monkeypatch.setattr(M, "consensus_votes", lambda *a, **k: dict(fixed))
    closes = {s_: [100.0] * 140 for s_ in syms}; prices = {s_: 100.0 for s_ in syms}; fund = {s_: 0.0 for s_ in syms}

    fresh = st.build_book(closes, fund, prices, _Reg(), 10_000.0, btc_closes=[1.0] * 140)
    assert set(fresh.positions) == {"C0", "C1", "C5", "C6"}, "open only at |vote| >= 2"
    assert "C2" not in fresh.positions, "a +1 name is NOT opened"

    held = st.build_book(closes, fund, prices, _Reg(), 10_000.0, btc_closes=[1.0] * 140,
                         held={"C2": 5, "C4": -5, "C3": 5})
    assert held.positions["C2"].target_contracts > 0, "a held +1 long is KEPT (hold=1)"
    assert held.positions["C4"].target_contracts < 0, "a held -1 short is KEPT"
    assert "C3" not in held.positions, "a held name at 0 votes is dropped"

    flipped = st.build_book(closes, fund, prices, _Reg(), 10_000.0, btc_closes=[1.0] * 140,
                            held={"C5": 5})                       # we are LONG a name now voting -2
    assert flipped.positions["C5"].target_contracts < 0, "hysteresis never protects the wrong side"


def test_long_only_when_no_short_qualifies():
    c = _cfg(); st = ConsensusStrategy(c)
    # every coin rising strongly: the neutral books still rank bottom-K short, so shorts exist;
    # remove them by making carry and trend unable to vote (too few names) but champion able.
    closes, funding, prices, btc = _world(n_coins=6)     # < 2*top_k -> carry/trend abstain
    bk = st.build_book(closes, funding, prices, _Reg(), 10_000.0, btc_closes=btc)
    # only champion votes (+1 each) -> nothing reaches enter=2 -> empty book, and that is correct
    assert len(bk.positions) == 0


def test_engine_constructs_and_dispatches_consensus(tmp_path):
    import sentinel.engine.engine as E
    cfg = _cfg()
    cfg.state.db_path = str(tmp_path / "c.db"); cfg.state.equity_csv = str(tmp_path / "c.csv")
    cfg.whales.enabled = False
    E.make_futures = lambda c: types.SimpleNamespace(funding_interval_hours=1.0, klines=lambda *a, **k: [])
    eng = E.Engine(cfg, live=False)
    assert isinstance(eng.consensus, ConsensusStrategy)
    assert eng.regime_gate_changed() is False          # first look sets the baseline only
    eng._ensure_market = lambda: None
    eng.base_interval = "1day"
