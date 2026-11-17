"""Data layer: download, cache and align daily market data.

Run `python data.py` to (re)download and print data-quality checks.

Series (yfinance, daily, since 2010):
    GC=F      gold futures, continuous front-month contract (OHLC)
    ^GVZ      CBOE gold volatility index
    DX-Y.NYB  US dollar index
    ^TNX      10-year Treasury yield, in percent

GC=F is the trading calendar. The other series are put on gold's dates and
forward-filled by at most 1 day; rows that still have gaps are dropped.
A forward fill only copies an older value forward, so it can't leak future data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw_prices.csv"
FOMC_CSV = DATA_DIR / "fomc_dates.csv"

START = "2010-01-01"
GOLD = "GC=F"
MACRO = {"^GVZ": "gvz", "DX-Y.NYB": "dxy", "^TNX": "tnx"}
MAX_FFILL = 1


def download_raw(refresh: bool = False) -> pd.DataFrame:
    """Download every ticker separately and cache them as one long CSV."""
    if RAW_CSV.exists() and not refresh:
        return pd.read_csv(RAW_CSV, parse_dates=["date"])

    import yfinance as yf

    frames = []
    for ticker in [GOLD, *MACRO]:
        df = yf.download(ticker, start=START, auto_adjust=False,
                         progress=False, threads=False, multi_level_index=False)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df["ticker"] = ticker
        frames.append(df.reset_index())
    raw = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(RAW_CSV, index=False)
    return raw


def align(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """One row per gold trading day with gold OHLC and the 3 macro closes."""
    gold = (raw[raw.ticker == GOLD].set_index("date")
            [["Open", "High", "Low", "Close"]]
            .rename(columns=str.lower).sort_index())
    gold = gold[~gold.index.duplicated(keep="last")]
    n0 = len(gold)
    # Gold itself is never filled: a filled close would create a fake 0% return.
    gold = gold.dropna(subset=["close"])
    gold = gold[gold.close > 0]

    out = gold.copy()
    stats = {}
    for ticker, name in MACRO.items():
        s = (raw[raw.ticker == ticker].set_index("date")["Close"]
             .sort_index())
        s = s[~s.index.duplicated(keep="last")]
        # Put the series on the union of dates, fill forward by at most 1 day,
        # then keep only gold's dates. Filling only looks backward.
        idx = s.index.union(out.index)
        s_full = s.reindex(idx)
        missing_before = s_full.reindex(out.index).isna().sum()
        s_full = s_full.ffill(limit=MAX_FFILL)
        out[name] = s_full.reindex(out.index)
        stats[name] = (missing_before, out[name].isna().sum())

    before = len(out)
    out = out.dropna()
    if verbose:
        print(f"GC=F rows: {n0}, after removing bad closes: {len(gold)}")
        for name, (m0, m1) in stats.items():
            print(f"  {name}: {m0} gaps on gold dates, {m0 - m1} filled, {m1} still missing")
        print(f"Dropped {before - len(out)} rows with unfilled gaps -> {len(out)} rows "
              f"({out.index[0].date()} to {out.index[-1].date()})")
    return out


def load_fomc(path: Path = FOMC_CSV) -> pd.DatetimeIndex:
    """FOMC decision dates (scheduled and unscheduled)."""
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def load_data(refresh: bool = False, verbose: bool = False) -> pd.DataFrame:
    return align(download_raw(refresh=refresh), verbose=verbose)


if __name__ == "__main__":
    import sys

    df = load_data(refresh="--refresh" in sys.argv or not RAW_CSV.exists(), verbose=True)
    print(df.tail())
    print(df.describe().T[["min", "max", "mean"]])
