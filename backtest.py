"""Strategy check: hold GC=F, halve the position on predicted high-vol days.

Timing (the main place look-ahead could creep into a backtest):
    signal row t   = features at day t's close  -> probability p_t
    position       = 0.5 if p_t >= cutoff else 1.0, held from close t to close t+1
    P&L            = position_t * simple return on t+1
So the position that earns day t+1's return is fixed with information up to
day t. Returns are reported on the date they are earned (t+1).

Costs: 0.05% of portfolio value every time the position changes (1.0 <-> 0.5),
charged on the day the new position starts. This is a flat per-change charge
and errs on the high side. Buy-and-hold pays it once, on entry.

Run `python backtest.py` after `python model.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST = 0.0005
DAYS = 252


def positions(p: pd.Series, cutoff: float, low: float = 0.5) -> pd.Series:
    return pd.Series(np.where(p >= cutoff, low, 1.0), index=p.index)


def run_strategy(pos: pd.Series, ret_next: pd.Series, next_dates: pd.Series,
                 cost: float = COST) -> pd.Series:
    """Daily net returns, indexed by the date they are earned (t+1)."""
    changes = pos.diff().fillna(pos.iloc[0]).ne(0)     # first day = entry
    net = pos * ret_next - cost * changes.astype(float)
    net.index = pd.DatetimeIndex(next_dates.loc[pos.index].values)
    return net


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    return dict(ann_return=float(eq.iloc[-1] ** (1 / years) - 1),
                ann_vol=float(r.std() * np.sqrt(DAYS)),
                sharpe=float(r.mean() / r.std() * np.sqrt(DAYS)),
                max_dd=float(dd.min()),
                total_return=float(eq.iloc[-1] - 1))


def backtest(res: dict, family: str | None = None, variant: str = "main",
             n_random: int = 1000, seed: int = 42) -> dict:
    """Backtest one model on the test period; defaults to the strategy model."""
    d = res["variants"][variant]["data"]
    fam, fs = (family, res["chosen"][family]) if family else res["strategy_model"]
    exp = res["variants"][variant]["exps"][(fam, fs)]

    test = d["test"]
    idx = d["feat"].index
    next_dates = pd.Series(idx[1:].append(pd.DatetimeIndex([pd.NaT])), index=idx)
    ret = test["simple_ret_next"]

    pos = positions(exp["p_test"], exp["cutoff"])
    strat = run_strategy(pos, ret, next_dates)
    bh = run_strategy(pd.Series(1.0, index=test.index), ret, next_dates)

    # Control: halve the same number of days, picked at random. Compared
    # before costs, because random flags change position ~3x as often as the
    # model and would otherwise lose mostly to costs. If the model has timing
    # skill, its gross Sharpe should beat most of these.
    gross_sharpe = stats(run_strategy(pos, ret, next_dates, cost=0.0))["sharpe"]
    rng = np.random.default_rng(seed)
    n_low = int((pos < 1).sum())
    rand_sharpes, rand_dds = [], []
    for _ in range(n_random):
        rp = pd.Series(1.0, index=test.index)
        rp.iloc[rng.choice(len(rp), n_low, replace=False)] = 0.5
        s = stats(run_strategy(rp, ret, next_dates, cost=0.0))
        rand_sharpes.append(s["sharpe"])
        rand_dds.append(s["max_dd"])

    s_strat, s_bh = stats(strat), stats(bh)
    s_strat.update(avg_exposure=float(pos.mean()),
                   n_changes=int(pos.diff().fillna(0).ne(0).sum()))
    s_bh.update(avg_exposure=1.0, n_changes=0)
    rand_sharpes = np.array(rand_sharpes)

    # Did the flagged days really carry more risk? (realised next-day |return|)
    flagged = pos < 1
    return dict(
        model=(fam, fs),
        returns=pd.DataFrame({"strategy": strat, "buy_hold": bh}),
        positions=pos,
        stats=pd.DataFrame({"Halve on predicted high-vol": s_strat, "Buy & hold": s_bh}),
        random=dict(sharpe_median=float(np.median(rand_sharpes)),
                    sharpe_p5=float(np.percentile(rand_sharpes, 5)),
                    sharpe_p95=float(np.percentile(rand_sharpes, 95)),
                    strategy_gross_sharpe=float(gross_sharpe),
                    pct_beaten=float((rand_sharpes < gross_sharpe).mean()),
                    max_dd_median=float(np.median(rand_dds))),
        flag_check=dict(mean_abs_ret_flagged=float(ret[flagged].abs().mean()),
                        mean_abs_ret_other=float(ret[~flagged].abs().mean()),
                        mean_ret_flagged=float(ret[flagged].mean()),
                        mean_ret_other=float(ret[~flagged].mean())),
    )


def equity_and_drawdown(rets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eq = (1 + rets).cumprod()
    return eq, eq / eq.cummax() - 1


if __name__ == "__main__":
    from model import load

    res = load()
    for fam in ("logreg", "lgbm"):
        bt = backtest(res, family=fam)
        print(f"\n=== {fam} ({bt['model'][1]} features) ===")
        print(bt["stats"].round(4).to_string())
        print("random-halving control:", {k: round(v, 3) for k, v in bt["random"].items()})
        print("flag check:", {k: round(v, 5) for k, v in bt["flag_check"].items()})
