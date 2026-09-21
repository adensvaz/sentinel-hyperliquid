"""CONSENSUS — trade only the names the other three books agree on.

WHERE IT COMES FROM
-------------------
Seventy-three days of live ledgers across champion, carry and trend showed the same thing three
times: each book makes money on the names it shares with the others and loses it on the names it
picks alone. Measured on 440 days of Hyperliquid data, forward return by how many books agree:

    +2 / +3  (two or three long)     +15 bps/day
    +1       (one book alone)        -10 bps/day     <- the solo picks are the losers
    -2       (both neutral short)    -28 bps/day     t = -2.0

So this is not a fourth signal. It is a FILTER: three horizons of momentum (30d, 14d + funding,
45d range position) vote, and only names with agreement get traded. Everything else sits in cash.
It is built from the exact functions the live books use — _momentum, carry_rank, _breakout,
regime_on — so a vote here is what that book would actually have done.

WHAT THE TEST SAID (research harness, same panel, live 30% stop, live costs)
    equal-weight of the three books    Sharpe 0.88   @2x fees 0.44   @3x fees -0.04
    consensus (this config)            Sharpe 1.35   @2x fees 0.98   @3x fees  0.63
    split-half 1.35 / 1.35.  Permutation p = 0.000 (shuffled votes score -1.50).
    Deflated Sharpe 0.863 over 7 variants — marginal, the ceiling everything hits on 440 days.
Correlation to carry +0.65, to trend +0.73: it REPLACES those two rather than sitting beside them.
Correlation to champion +0.09: champion stays as a genuine diversifier.

HYSTERESIS is what makes it deployable. Without it turnover was 0.87/day and the Sharpe at 3x fees
was 0.36; entering at >=2 votes and holding while >=1 halves turnover, cuts maxDD 19% -> 13.5%, and
nearly doubles the 3x-fee Sharpe. Cost is this system's binding constraint, so that trade — a
little peak Sharpe for a lot of fee robustness — is the right one.

Dollar-neutral whenever both sleeves have names; long-only on the rare day nothing qualifies as a
short (1 of 440 in the test). That asymmetry is deliberate and is why max_net_exposure is 1.0.
"""
from __future__ import annotations

from ..exchange.contracts import ContractRegistry
from .funding_carry import carry_rank
from .momentum_regime import _momentum, regime_on
from .sentiment_edge import TargetBook, TargetPosition
from .trend import _breakout


def consensus_votes(closes_by_symbol: dict, funding_by_symbol: dict, btc_closes: list | None,
                    champion_cfg, carry_cfg, trend_cfg, symbols: list | None = None) -> dict:
    """Per-symbol vote in {-2..+3}: champion +1 (never shorts), carry +/-1, trend +/-1.

    Pure and separable from book construction so it can be tested, logged and, later, used as a
    veto by the other books if that ever proves worthwhile."""
    syms = list(symbols if symbols is not None else closes_by_symbol)
    vote = {s: 0 for s in syms}

    # champion: top_k by lookback momentum, only when the regime brake is on, only positive momentum
    if btc_closes is not None and regime_on(btc_closes, champion_cfg.regime_ma):
        cand = []
        for s in syms:
            m = _momentum(closes_by_symbol.get(s, []), champion_cfg.lookback)
            if m > -1e8:
                cand.append((s, m))
        cand.sort(key=lambda x: -x[1])
        for s, m in cand[:champion_cfg.top_k]:
            if m > 0:
                vote[s] += 1

    # carry: momentum + funding rank, top_k long / bottom_k short
    cand = []
    for s in syms:
        f = funding_by_symbol.get(s)
        if f is None:
            continue
        m = _momentum(closes_by_symbol.get(s, []), carry_cfg.lookback)
        if m > -1e8:
            cand.append((s, m, float(f)))
    if len(cand) >= 2 * carry_cfg.top_k:
        ranked = carry_rank(cand)
        for s in ranked[:carry_cfg.top_k]:
            vote[s] += 1
        for s in ranked[-carry_cfg.top_k:]:
            vote[s] -= 1

    # trend: Donchian range position, top_k long / bottom_k short (cross-sectional, no sign filter)
    sc = []
    for s in syms:
        t = _breakout(closes_by_symbol.get(s, []), getattr(trend_cfg, "donchian_period", 45))
        if t > -1e8:
            sc.append((s, t))
    if len(sc) >= 2 * trend_cfg.top_k:
        sc.sort(key=lambda x: -x[1])
        for s, _ in sc[:trend_cfg.top_k]:
            vote[s] += 1
        for s, _ in sc[-trend_cfg.top_k:]:
            vote[s] -= 1
    return vote


class ConsensusStrategy:
    def __init__(self, cfg):
        self.c = cfg.consensus
        self.champion = cfg.champion
        self.carry = cfg.carry
        self.trend = cfg.trend

    def build_book(self, closes_by_symbol: dict, funding_by_symbol: dict, prices: dict,
                   registry: ContractRegistry, equity: float, gross_scale: float = 1.0,
                   held: dict | None = None, btc_closes: list | None = None) -> TargetBook:
        c, held = self.c, (held or {})
        syms = [s for s in closes_by_symbol if s in prices and prices[s] > 0 and registry.has(s)]
        vote = consensus_votes(closes_by_symbol, funding_by_symbol, btc_closes,
                               self.champion, self.carry, self.trend, syms)

        # enter at |vote| >= enter; a held name survives while |vote| >= hold on the SAME side
        longs = [s for s in syms if vote[s] >= c.enter
                 or (held.get(s, 0) > 0 and vote[s] >= c.hold)]
        shorts = [s for s in syms if vote[s] <= -c.enter
                  or (held.get(s, 0) < 0 and vote[s] <= -c.hold)]
        longs.sort(key=lambda s: -vote[s])            # strongest agreement first if we must cut
        shorts.sort(key=lambda s: vote[s])
        longs, shorts = longs[:c.max_per_side], shorts[:c.max_per_side]
        if not longs and not shorts:
            return TargetBook({}, equity, 0.0)

        gross = c.target_gross * equity * gross_scale
        side = gross / 2.0 if (longs and shorts) else gross   # long-only on a no-short day
        positions: dict[str, TargetPosition] = {}
        for s in longs:
            spec = registry.get(s)
            contracts = spec.notional_to_contracts(side / len(longs), prices[s])
            if contracts > 0:
                positions[s] = TargetPosition(s, float(vote[s]), "LONG", contracts,
                                              spec.contracts_to_notional(contracts, prices[s]), prices[s])
        for s in shorts:
            spec = registry.get(s)
            contracts = spec.notional_to_contracts(side / len(shorts), prices[s])
            if contracts > 0:
                positions[s] = TargetPosition(s, float(vote[s]), "SHORT", -contracts,
                                              -spec.contracts_to_notional(contracts, prices[s]), prices[s])
        return TargetBook(positions, equity, gross)
