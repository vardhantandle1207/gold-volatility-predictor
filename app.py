"""Streamlit dashboard: streamlit run app.py

Loads results/run.pkl (written by `python model.py`). If it's missing, the
app runs the full pipeline once, which takes about 40 seconds.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import backtest, equity_and_drawdown
from model import RUN_PKL, importance, load, run, save

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
NAMES = {"logreg": "Logistic regression", "lgbm": "LightGBM"}

st.set_page_config(page_title="Gold high-vol predictor", layout="wide")


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_results() -> dict:
    if RUN_PKL.exists():
        return load()
    res = run(verbose=False)
    save(res)
    return res


@st.cache_data(show_spinner=False)
def get_backtest(family: str, variant: str) -> dict:
    return backtest(get_results(), family=family, variant=variant, n_random=500)


@st.cache_data(show_spinner=False)
def get_importance(family: str, fs: str, variant: str) -> pd.DataFrame:
    res = get_results()
    return importance(res["variants"][variant]["exps"][(family, fs)],
                      res["variants"][variant]["data"]["test"])


res = get_results()

# --- sidebar -----------------------------------------------------------------
st.sidebar.header("Settings")
family = st.sidebar.radio("Model", ["logreg", "lgbm"], format_func=NAMES.get,
                          index=["logreg", "lgbm"].index(res["strategy_model"][0]))
fs = st.sidebar.radio("Features", ["fomc", "base"],
                      format_func={"fomc": "Base + FOMC flag", "base": "Base only"}.get,
                      index=["fomc", "base"].index(res["chosen"][family]))
variant = st.sidebar.radio(
    "Timing", ["main", "strict_timing"],
    format_func={"main": "Main (all inputs at day-t close)",
                 "strict_timing": "Strict (macro inputs lagged 1 day)"}.get,
    help="Gold settles at 1:30pm ET; GVZ, DXY and the 10Y yield close later. "
         "Strict timing uses only the previous day's macro data, so every input "
         "is known at gold's settlement.")
st.sidebar.caption(f"Train 2010-2020 · test 2021 onward. Model picked by training CV: "
                   f"**{NAMES[res['strategy_model'][0]]}** with "
                   f"**{'FOMC flag' if res['strategy_model'][1] == 'fomc' else 'base features'}**.")

d = res["variants"][variant]["data"]
exp = res["variants"][variant]["exps"][(family, fs)]
test = d["test"]
flags = exp["p_test"] >= exp["cutoff"]

# --- header --------------------------------------------------------------------
st.title("Gold futures: predicting high-volatility days")
st.caption(f"Label: next-day |log return| above {d['threshold']:.2%}, the 80th percentile of "
           f"2010-2020. Test-period base rate: {test['y'].mean():.1%}.")

last_date = d["latest"].index[0].date()
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"P(high-vol) for day after {last_date}", f"{exp['p_latest']:.1%}")
c2.metric("Signal", "HALVE position" if exp["p_latest"] >= exp["cutoff"] else "Full position",
          help=f"Flag when probability ≥ {exp['cutoff']:.1%} (top 20% of out-of-fold "
               "training predictions)")
c3.metric("Test ROC-AUC", f"{exp['test']['auc']:.3f}",
          f"{exp['test']['auc'] - res['gvz_only']['test']['auc']:+.3f} vs GVZ-only",
          delta_color="normal")
c4.metric("Training CV AUC", f"{exp['cv_auc']:.3f}")

# --- price chart -------------------------------------------------------------
st.subheader("Gold price with predicted high-vol days")
fig = go.Figure()
fig.add_scatter(x=test.index, y=test["close"], mode="lines", name="GC=F close",
                line=dict(color=GRAY, width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>")
ft = test[flags]
fig.add_scatter(x=ft.index, y=ft["close"], mode="markers", name="Predicted high-vol next day",
                marker=dict(color=ORANGE, size=6),
                customdata=exp["p_test"][flags].values,
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<br>P=%{customdata:.1%}<extra></extra>")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                  legend=dict(orientation="h", y=1.08), yaxis_title="USD / oz")
st.plotly_chart(fig, width="stretch")

# --- metrics -------------------------------------------------------------------
st.subheader("Model metrics (test period)")
rows = []
for (fam, f), e in res["variants"][variant]["exps"].items():
    t = e["test"]
    rows.append({"Model": NAMES[fam], "Features": "base + FOMC" if f == "fomc" else "base",
                 "CV AUC (train)": e["cv_auc"], "Test AUC": t["auc"],
                 "Precision": t["precision"], "Recall": t["recall"], "Flagged": t["flag_rate"]})
g = res["gvz_only"]["test"]
rows.append({"Model": "GVZ level only", "Features": "gvz", "CV AUC (train)": res["gvz_only"]["cv_auc"],
             "Test AUC": g["auc"], "Precision": g["precision"], "Recall": g["recall"],
             "Flagged": g["flag_rate"]})
mt = pd.DataFrame(rows)
st.dataframe(mt.style.format({"CV AUC (train)": "{:.3f}", "Test AUC": "{:.3f}",
                              "Precision": "{:.1%}", "Recall": "{:.1%}", "Flagged": "{:.1%}"}),
             hide_index=True, width="stretch")

m1, m2 = st.columns(2)
with m1:
    c = exp["test"]["confusion"]
    st.markdown(f"**Confusion matrix: {NAMES[family]}, {fs}**")
    st.dataframe(pd.DataFrame([[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                              index=["Actual normal", "Actual high-vol"],
                              columns=["Pred normal", "Pred high-vol"]),
                 width="stretch")
with m2:
    st.markdown("**Does the FOMC flag help?** (test AUC with − without, 95% block bootstrap)")
    if variant == "main":
        for fam, v in res["fomc_test"].items():
            st.write(f"{NAMES[fam]}: **{v['diff']:+.4f}** [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    else:
        for fam in ("logreg", "lgbm"):
            e = res["variants"][variant]["exps"]
            st.write(f"{NAMES[fam]}: **{e[(fam, 'fomc')]['test']['auc'] - e[(fam, 'base')]['test']['auc']:+.4f}**")
    fr = res["fomc_rates"]
    st.caption(f"P(high-vol on day after FOMC): train {fr['train']['p_high_after_fomc']:.0%} "
               f"vs {fr['train']['p_high_other']:.0%} other days; "
               f"test {fr['test']['p_high_after_fomc']:.0%} vs {fr['test']['p_high_other']:.0%}.")

# --- feature importance ------------------------------------------------------
st.subheader("Feature importance")
imp = get_importance(family, fs, variant)["importance"].iloc[::-1]
fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE,
                       hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis_title="mean |SHAP| on test set" if family == "lgbm"
                  else "|standardised coefficient|")
st.plotly_chart(fig, width="stretch")

# --- strategy ------------------------------------------------------------------
st.subheader("Strategy: halve the position on predicted high-vol days")
bt = get_backtest(family, variant)
st.caption("The position for day t+1 is set from day t's prediction. Costs: 0.05% per position change.")
eq, dd = equity_and_drawdown(bt["returns"])
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                    vertical_spacing=0.06, subplot_titles=("Equity (growth of 1)", "Drawdown"))
for col, name, color in (("strategy", "Halve on predicted high-vol", BLUE),
                         ("buy_hold", "Buy & hold", ORANGE)):
    fig.add_scatter(x=eq.index, y=eq[col], name=name, line=dict(color=color, width=2),
                    row=1, col=1, hovertemplate="%{y:.2f}")
    fig.add_scatter(x=dd.index, y=dd[col], name=name, line=dict(color=color, width=2),
                    row=2, col=1, showlegend=False, hovertemplate="%{y:.1%}")
fig.update_yaxes(tickformat=".0%", row=2, col=1)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

s = bt["stats"].T
st.dataframe(s[["ann_return", "ann_vol", "sharpe", "max_dd", "avg_exposure", "n_changes"]]
             .rename(columns={"ann_return": "Ann. return", "ann_vol": "Ann. vol", "sharpe": "Sharpe",
                              "max_dd": "Max drawdown", "avg_exposure": "Avg exposure",
                              "n_changes": "Position changes"})
             .style.format({"Ann. return": "{:.1%}", "Ann. vol": "{:.1%}", "Sharpe": "{:.2f}",
                            "Max drawdown": "{:.1%}", "Avg exposure": "{:.2f}",
                            "Position changes": "{:.0f}"}),
             width="stretch")
r = bt["random"]
st.caption(f"Random-halving control (same number of halved days, before costs): median Sharpe "
           f"{r['sharpe_median']:.2f}, 5-95% range {r['sharpe_p5']:.2f}-{r['sharpe_p95']:.2f}. "
           f"Strategy before costs: {r['strategy_gross_sharpe']:.2f}, better than "
           f"{r['pct_beaten']:.0%} of random runs.")

yr = bt["positions"].lt(1).groupby(bt["positions"].index.year).mean()
st.caption("Share of days at half position, by year: " +
           ", ".join(f"{y}: {v:.0%}" for y, v in yr.items()) +
           ". The label threshold is fixed at the 2010-2020 level, so in the high-vol 2025-26 "
           "regime the model flags almost every day.")
