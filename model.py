"""Models: logistic regression baseline vs LightGBM.

Run `python model.py` to train, evaluate and save results/run.pkl.

Protocol (no shuffling anywhere):
  * Train = signal dates 2010-2020, Test = 2021 onward.
  * Every choice is made on the training period only, using expanding-window
    cross-validation by year (validate on 2015, 2016, ... 2020, each time
    training on all earlier years). That covers LightGBM hyperparameters,
    the probability cutoff for "predicted high-vol" and whether the FOMC
    flag goes into the main model.
  * Purging: the last row before each split has its label on the first day
    after the split, so that row is dropped from the fitting set.
  * The test set is scored once, with the final models fit on all of 2010-2020.
"""
from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_data, load_fomc
from features import ALL_FEATURES, BASE_FEATURES, FOMC_FEATURE, build_features, make_labels

RESULTS_DIR = Path(__file__).parent / "results"
RUN_PKL = RESULTS_DIR / "run.pkl"

TRAIN_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
CV_YEARS = list(range(2015, 2021))
FLAG_RATE = 0.20          # flag the top 20% of predicted probabilities
SEED = 42

LGBM_GRID = {
    "num_leaves": [7, 15],
    "n_estimators": [200, 500],
    "min_child_samples": [50, 100],
}
LGBM_FIXED = dict(learning_rate=0.02, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                  verbose=-1, n_jobs=4)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare(lag_macro: bool = False) -> dict:
    raw = load_data()
    feat = build_features(raw, load_fomc(), lag_macro=lag_macro)
    feat = feat.dropna(subset=[c for c in ALL_FEATURES if c != "hl_range"])

    train_mask = feat.index <= TRAIN_END
    y, threshold = make_labels(feat, pd.Series(train_mask, index=feat.index))
    feat["y"] = y

    train = feat[train_mask].iloc[:-1]                   # purge boundary row
    test_all = feat[feat.index >= TEST_START]
    test = test_all.dropna(subset=["y"])
    latest = test_all.iloc[[-1]]                         # today's row, label unknown
    return dict(feat=feat, train=train, test=test, latest=latest,
                threshold=threshold, lag_macro=lag_macro)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_lr(features: list[str]) -> Pipeline:
    num = [f for f in features if f != "dow"]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("dow", OneHotEncoder(handle_unknown="ignore", drop="first"), ["dow"]),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def make_lgbm(**params) -> LGBMClassifier:
    return LGBMClassifier(**LGBM_FIXED, **params)


def grid() -> list[dict]:
    keys = list(LGBM_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*LGBM_GRID.values())]


# --------------------------------------------------------------------------- #
# Walk-forward CV on the training period
# --------------------------------------------------------------------------- #
def cv_oof(train: pd.DataFrame, features: list[str], factory) -> tuple[pd.Series, list[float]]:
    """Expanding-window CV by year. Returns out-of-fold probabilities and per-fold AUC."""
    oof = pd.Series(np.nan, index=train.index)
    aucs = []
    for yr in CV_YEARS:
        fit = train[train.index.year < yr].iloc[:-1]     # purge boundary row
        val = train[train.index.year == yr]
        model = factory().fit(fit[features], fit["y"])
        p = model.predict_proba(val[features])[:, 1]
        oof[val.index] = p
        aucs.append(roc_auc_score(val["y"], p))
    return oof, aucs


def evaluate(y: pd.Series, p: np.ndarray, cutoff: float) -> dict:
    pred = (p >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                flag_rate=pred.mean(), base_rate=float(y.mean()),
                confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))


def block_bootstrap_auc_diff(y, p_a, p_b, n=2000, block=20, seed=SEED) -> dict:
    """Paired circular block bootstrap of AUC(a) - AUC(b) on the test set.

    Blocks of 20 days keep volatility clustering intact. Resampling single
    days would understate the uncertainty.
    """
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    T = len(y)
    nb = int(np.ceil(T / block))
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return dict(diff=float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
                lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                p_le_0=float((diffs <= 0).mean()))


# --------------------------------------------------------------------------- #
# One full experiment = one feature set x one model family
# --------------------------------------------------------------------------- #
def run_experiment(d: dict, features: list[str], family: str) -> dict:
    train, test = d["train"], d["test"]
    if family == "logreg":
        factory = lambda: make_lr(features)
        oof, aucs = cv_oof(train, features, factory)
        params = {}
    else:
        best = None
        for g in grid():
            factory_g = lambda g=g: make_lgbm(**g)
            oof_g, aucs_g = cv_oof(train, features, factory_g)
            if best is None or np.mean(aucs_g) > np.mean(best[2]):
                best = (g, oof_g, aucs_g)
        params, oof, aucs = best
        factory = lambda: make_lgbm(**params)

    # Cutoff = the 80th percentile of out-of-fold probabilities, so ~20% of
    # days are flagged. It's chosen without touching the test set.
    cutoff = float(np.nanquantile(oof, 1 - FLAG_RATE))
    model = factory().fit(train[features], train["y"])
    p_test = model.predict_proba(test[features])[:, 1]
    p_latest = model.predict_proba(d["latest"][features])[:, 1]
    return dict(family=family, features=features, params=params, model=model,
                cv_auc=float(np.mean(aucs)), cv_auc_folds=aucs, cutoff=cutoff,
                oof=oof, p_test=pd.Series(p_test, index=test.index),
                p_latest=float(p_latest[0]),
                test=evaluate(test["y"], p_test, cutoff))


def importance(exp: dict, test: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| for LightGBM, |standardised coefficient| for logistic regression."""
    feats = exp["features"]
    if exp["family"] == "lgbm":
        import warnings

        import shap
        with warnings.catch_warnings():   # shap notes its LightGBM output format changed
            warnings.simplefilter("ignore", UserWarning)
            sv = shap.TreeExplainer(exp["model"]).shap_values(test[feats])
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(axis=0), index=feats)
    else:
        pre, clf = exp["model"].named_steps["pre"], exp["model"].named_steps["clf"]
        names = pre.get_feature_names_out()
        coef = pd.Series(np.abs(clf.coef_[0]), index=names)
        imp = pd.Series({f: coef[[n for n in names if n.split("__")[1].startswith(f)]].sum()
                         if f == "dow" else coef[f"num__{f}"] for f in feats})
    return imp.sort_values(ascending=False).rename("importance").to_frame()


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    out = {"variants": {}}
    for lag in (False, True):
        d = prepare(lag_macro=lag)
        key = "strict_timing" if lag else "main"
        exps = {}
        for fs_name, fs in (("base", BASE_FEATURES), ("fomc", ALL_FEATURES)):
            for fam in ("logreg", "lgbm"):
                exps[(fam, fs_name)] = run_experiment(d, fs, fam)
                if verbose:
                    e = exps[(fam, fs_name)]
                    print(f"[{key:13s}] {fam:6s} {fs_name:4s}  CV AUC {e['cv_auc']:.3f}  "
                          f"test AUC {e['test']['auc']:.3f}  params {e['params']}")
        out["variants"][key] = dict(data=d, exps=exps)

    main = out["variants"]["main"]
    d, exps = main["data"], main["exps"]

    # Naive benchmark: rank days by today's GVZ level. No fitting, so its
    # "CV AUC" is just the AUC within each CV year.
    tr = d["train"]
    out["gvz_only"] = dict(
        cv_auc=float(np.mean([roc_auc_score(tr.y[tr.index.year == yr], tr.gvz[tr.index.year == yr])
                              for yr in CV_YEARS])),
        test=evaluate(d["test"]["y"], d["test"]["gvz"].values,
                      float(tr.gvz.quantile(1 - FLAG_RATE))))

    # FOMC flag: does it help? Test-set paired block bootstrap, both families.
    out["fomc_test"] = {fam: block_bootstrap_auc_diff(
        d["test"]["y"], exps[(fam, "fomc")]["p_test"], exps[(fam, "base")]["p_test"])
        for fam in ("logreg", "lgbm")}
    out["fomc_rates"] = {
        name: dict(n_events=int(df[FOMC_FEATURE].sum()),
                   p_high_after_fomc=float(df.loc[df[FOMC_FEATURE] == 1, "y"].mean()),
                   p_high_other=float(df.loc[df[FOMC_FEATURE] == 0, "y"].mean()))
        for name, df in (("train", d["train"]), ("test", d["test"]))}

    # Feature set for the main model is picked on training CV AUC, never on test.
    out["chosen"] = {}
    for fam in ("logreg", "lgbm"):
        fs = "fomc" if exps[(fam, "fomc")]["cv_auc"] > exps[(fam, "base")]["cv_auc"] else "base"
        out["chosen"][fam] = fs
    # The strategy uses the model family with the higher training CV AUC.
    best_fam = max(("logreg", "lgbm"), key=lambda f: exps[(f, out["chosen"][f])]["cv_auc"])
    out["strategy_model"] = (best_fam, out["chosen"][best_fam])

    out["importance"] = {fam: importance(exps[(fam, out["chosen"][fam])], d["test"])
                         for fam in ("logreg", "lgbm")}

    if verbose:
        print(f"\nLabel threshold |r| > {d['threshold']:.4%} (train 80th pct); "
              f"test base rate {d['test']['y'].mean():.1%}")
        g = out["gvz_only"]
        print(f"GVZ-only benchmark: CV AUC {g['cv_auc']:.3f}  test AUC {g['test']['auc']:.3f}")
        print("FOMC flag effect (test AUC with - without):",
              {k: f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
               for k, v in out["fomc_test"].items()})
        print("FOMC rates:", out["fomc_rates"])
        print("Chosen feature sets:", out["chosen"], "| strategy model:", out["strategy_model"])
    return out


def save(out: dict, path: Path = RUN_PKL) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)


def load(path: Path = RUN_PKL) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    res = run()
    save(res)
    print(f"saved {RUN_PKL}")
