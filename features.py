"""Feature and label construction.

Row t holds features known at the close of day t and the label for day t+1.

    label_t = 1 if |log return on t+1| > threshold, else 0

The threshold is the 80th percentile of |r_{t+1}| over **training rows only**
(see `make_labels`). Using the full sample would let the test period's
volatility leak into the label definition.

Where look-ahead could creep in, and what this module does about it:
  * Rolling windows: every rolling stat ends at t (pandas `rolling` is
    trailing by default). Nothing uses `center=True` and nothing uses a
    negative shift other than the label itself.
  * The label is the only `shift(-1)`. It is kept out of the feature list.
  * Label threshold: computed on training rows only (above).
  * Train/test boundary: the last training row's label is the first test
    day's return, so that row is purged in `model.py`.
  * Intraday timing: gold settles at 1:30pm ET, while GVZ (4:15pm), DXY
    (~5pm) and the 10Y yield close later. All are known by the evening of
    day t, but **not** at gold's settle. `lag_macro=True` shifts the macro
    features back by 1 day so they are strictly known by 1:30pm. model.py
    reports both versions.
  * FOMC: the flag on day t means "the Fed released a decision on day t".
    Scheduled dates are public years ahead. The flag is matched on the exact
    date only, so the 2020-03-15 (Sunday) emergency cut tags no row instead
    of being moved back onto Friday 2020-03-13, which would be look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GOLD_FEATURES = ["ret_1", "ret_5", "ret_21", "vol_5", "vol_21", "hl_range"]
MACRO_FEATURES = ["gvz", "gvz_chg_1", "dxy_ret_1", "dxy_ret_5", "tnx_chg_1"]
CALENDAR_FEATURES = ["dow"]
FOMC_FEATURE = "fomc_next"

BASE_FEATURES = GOLD_FEATURES + MACRO_FEATURES + CALENDAR_FEATURES
ALL_FEATURES = BASE_FEATURES + [FOMC_FEATURE]

TOP_QUANTILE = 0.80


def build_features(df: pd.DataFrame, fomc_dates: pd.DatetimeIndex | None = None,
                   lag_macro: bool = False) -> pd.DataFrame:
    """Return a frame with feature columns, next-day returns and helpers.

    `df` is the aligned output of data.load_data().
    """
    out = pd.DataFrame(index=df.index)
    logp = np.log(df["close"])
    r = logp.diff()

    # --- gold ---------------------------------------------------------------
    out["ret_1"] = r
    out["ret_5"] = logp.diff(5)
    out["ret_21"] = logp.diff(21)
    out["vol_5"] = r.rolling(5).std()
    out["vol_21"] = r.rolling(21).std()
    hi = df[["high", "open", "close"]].max(axis=1)
    lo = df[["low", "open", "close"]].min(axis=1)
    hl = (hi - lo) / df["close"]
    # high == low means Yahoo only has a settle price that day (149 rows),
    # not a real 0% range. Mark it missing; model.py imputes for logistic regression.
    hl[df["high"] == df["low"]] = np.nan
    out["hl_range"] = hl

    # --- macro --------------------------------------------------------------
    macro = pd.DataFrame(index=df.index)
    macro["gvz"] = df["gvz"]
    macro["gvz_chg_1"] = df["gvz"].diff()
    macro["dxy_ret_1"] = np.log(df["dxy"]).diff()
    macro["dxy_ret_5"] = np.log(df["dxy"]).diff(5)
    macro["tnx_chg_1"] = df["tnx"].diff()          # in percentage points
    if lag_macro:
        macro = macro.shift(1)
    out[MACRO_FEATURES] = macro

    # --- calendar -----------------------------------------------------------
    out["dow"] = df.index.day_of_week                # 0 = Monday

    # --- FOMC: a decision on day t moves gold on t+1 (settle is before 2pm) --
    if fomc_dates is not None:
        out[FOMC_FEATURE] = df.index.isin(fomc_dates).astype(int)
    else:
        out[FOMC_FEATURE] = 0

    # --- targets (never used as features) ------------------------------------
    out["ret_next"] = r.shift(-1)                               # log return t+1
    out["simple_ret_next"] = df["close"].pct_change().shift(-1)  # for backtest
    out["close"] = df["close"]
    return out


def make_labels(feat: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.Series, float]:
    """Binary label: next-day |return| above the training-period 80th percentile."""
    abs_next = feat["ret_next"].abs()
    threshold = float(abs_next[train_mask].quantile(TOP_QUANTILE))
    y = (abs_next > threshold).astype(float)
    y[abs_next.isna()] = np.nan      # last row: tomorrow is unknown
    return y, threshold
