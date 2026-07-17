#!/usr/bin/env python3
"""
optimize_weights.py
───────────────────
Data-driven grid search for optimal scan.py rank_score weights + entry thresholds.

Objective: maximize WR first, avg_ret_d5 second.

Sections:
  1. Load + clean scan_history.csv
  2. Per-feature correlation with outcome (quick signal quality check)
  3. Strategy-level WR + optimal strategy weights
  4. Logistic regression → data-driven feature weights (replaces hand-tuned pts)
  5. Threshold grid search (min_vol × RSI_cap × min_strategies × score_cap)
  6. Walk-forward validation (train on older dates, test on newer)
  7. Print recommended settings

Usage: python3 optimize_weights.py
"""

import csv, math, itertools, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
HISTORY = HERE / "scan_history.csv"

PROVEN_EDGE = {"pocket_pivot", "ema_ribbon", "cup_handle", "signal_velocity", "connors_rsi2"}
DISABLED    = {"stage4_short", "momentum", "bb_squeeze", "high_tight_flag"}

W = 80

def _hr(char="─"): print(char * W)
def _hdr(title): _hr("═"); print(f"  {title}"); _hr("═")
def _sub(title): print(f"\n  ── {title} {'─'*(W-6-len(title))}")


# ── 1. Load + clean ───────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(HISTORY)
    df = df[~df["strategy"].isin(DISABLED)]
    for col in ["ret_d5", "adx", "rsi", "vol_ratio", "score", "wr",
                "strategies_count", "r_multiple_d5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ret_d5"].notna() & (df["ret_d5"].abs() <= 15)]
    df["win"] = (df["ret_d5"] > 0).astype(int)
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    df = df.sort_values("scan_date")
    return df.reset_index(drop=True)


# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["is_proven"]      = d["strategy"].isin(PROVEN_EDGE).astype(int)
    d["adx_zone2"]      = ((d["adx"] >= 20) & (d["adx"] <= 35)).astype(int)
    d["adx_zone1"]      = (((d["adx"] >= 16) & (d["adx"] < 20)) |
                           ((d["adx"] > 35) & (d["adx"] <= 45))).astype(int)
    d["rsi_zone2"]      = ((d["rsi"] >= 50) & (d["rsi"] <= 65)).astype(int)
    d["rsi_zone1"]      = ((d["rsi"] > 65)  & (d["rsi"] <= 70)).astype(int)
    d["vol_2x"]         = (d["vol_ratio"] >= 2.0).astype(int)
    d["vol_1p5x"]       = ((d["vol_ratio"] >= 1.5) & (d["vol_ratio"] < 2.0)).astype(int)
    d["multi3"]         = (d["strategies_count"] >= 3).astype(int)
    d["multi2"]         = (d["strategies_count"] == 2).astype(int)
    d["score_low"]      = (d["score"] <= 3).astype(int)
    d["hist_wr_good"]   = (d["wr"] >= 60).astype(int)
    return d

FEATURES = [
    "is_proven", "adx_zone2", "adx_zone1", "rsi_zone2", "rsi_zone1",
    "vol_2x", "vol_1p5x", "multi3", "multi2", "score_low", "hist_wr_good",
]

FEATURE_LABELS = {
    "is_proven":    "PROVEN edge strategy",
    "adx_zone2":    "ADX 20-35 (sweet zone)",
    "adx_zone1":    "ADX 16-20 or 35-45",
    "rsi_zone2":    "RSI 50-65",
    "rsi_zone1":    "RSI 65-70",
    "vol_2x":       "Vol ≥2x avg",
    "vol_1p5x":     "Vol 1.5-2x avg",
    "multi3":       "3+ strategies fired",
    "multi2":       "2 strategies fired",
    "score_low":    "Score ≤3",
    "hist_wr_good": "Strategy hist WR ≥60%",
}

CURRENT_WEIGHTS = {
    "is_proven": 3, "adx_zone2": 2, "adx_zone1": 1,
    "rsi_zone2": 2, "rsi_zone1": 1,
    "vol_2x": 2, "vol_1p5x": 1,
    "multi3": 2, "multi2": 1,
    "score_low": 1, "hist_wr_good": 1,
}


# ── 3. Per-feature analysis ───────────────────────────────────────────────────

def feature_analysis(df: pd.DataFrame):
    _sub("Per-Feature WR + Avg Return")
    print(f"  {'Feature':<28} {'N(yes)':<8} {'WR(yes)':>8} {'WR(no)':>8} {'Δ WR':>8} {'Avg(yes)':>9} {'Avg(no)':>9} {'Δ Avg':>8}")
    _hr()
    results = []
    for feat in FEATURES:
        yes = df[df[feat] == 1]
        no  = df[df[feat] == 0]
        if len(yes) < 5: continue
        wr_yes  = yes["win"].mean() * 100
        wr_no   = no["win"].mean()  * 100 if len(no) > 0 else float("nan")
        avg_yes = yes["ret_d5"].mean()
        avg_no  = no["ret_d5"].mean()  if len(no) > 0 else float("nan")
        delta_wr  = wr_yes - wr_no
        delta_avg = avg_yes - avg_no
        results.append((feat, len(yes), wr_yes, wr_no, delta_wr, avg_yes, avg_no, delta_avg))
        flag = "  ★" if delta_wr >= 5 else ("  ↓" if delta_wr <= -5 else "")
        print(f"  {FEATURE_LABELS[feat]:<28} {len(yes):<8} {wr_yes:>7.1f}% {wr_no:>7.1f}% {delta_wr:>+7.1f}% {avg_yes:>+8.2f}% {avg_no:>+8.2f}% {delta_avg:>+7.2f}%{flag}")
    return results


# ── 4. Strategy-level WR ──────────────────────────────────────────────────────

def strategy_analysis(df: pd.DataFrame) -> dict:
    _sub("Strategy-Level WR (from actual outcomes)")
    strat_stats = {}
    rows = []
    for strat, g in df.groupby("strategy"):
        if len(g) < 5: continue
        wr  = g["win"].mean() * 100
        avg = g["ret_d5"].mean()
        n   = len(g)
        proven = "✦" if strat in PROVEN_EDGE else " "
        strat_stats[strat] = {"wr": wr, "avg": avg, "n": n}
        rows.append((strat, n, wr, avg, proven))
    rows.sort(key=lambda x: -x[2])
    print(f"  {'Strategy':<25} {'N':>5} {'WR':>7} {'Avg':>8} {'Proven':>8}")
    _hr()
    for strat, n, wr, avg, proven in rows:
        bar = "█" * int(wr / 5)
        print(f"  {strat:<25} {n:>5} {wr:>6.1f}% {avg:>+7.2f}%  {proven}  {bar}")
    return strat_stats


# ── 5. Logistic regression → optimal weights ──────────────────────────────────

def logistic_weights(df: pd.DataFrame) -> dict:
    _sub("Logistic Regression → Optimal Feature Weights")
    X = df[FEATURES].fillna(0).values
    y = df["win"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(Xs, y)

    # AUC on full set (in-sample — just for reference)
    proba = model.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, proba)

    # Coefficients: positive = predictive of win
    coefs = dict(zip(FEATURES, model.coef_[0]))

    # Scale to integer weights (1-5 range, preserve relative ordering)
    max_coef = max(abs(v) for v in coefs.values())
    scaled = {k: v / max_coef for k, v in coefs.items()}

    # Propose integer weights (round to nearest 0.5, cap at 5)
    proposed = {}
    for k, v in scaled.items():
        raw = v * 3  # scale so max maps to ~3
        proposed[k] = max(0, round(raw * 2) / 2)  # half-point resolution

    print(f"  In-sample AUC: {auc:.3f}  (0.5=random, 1.0=perfect)")
    print()
    print(f"  {'Feature':<28} {'Coef':>8} {'Current':>9} {'Proposed':>10}  {'Change'}")
    _hr()
    for feat in FEATURES:
        curr = CURRENT_WEIGHTS.get(feat, 0)
        prop = proposed[feat]
        coef = coefs[feat]
        change = f"+{prop-curr:.1f}" if prop > curr else (f"{prop-curr:.1f}" if prop < curr else "  →")
        flag = " ◄ INCREASE" if prop > curr + 0.4 else (" ◄ decrease" if prop < curr - 0.4 else "")
        print(f"  {FEATURE_LABELS[feat]:<28} {coef:>+8.3f} {curr:>9} {prop:>10.1f}  {change}{flag}")

    return proposed


# ── 6. Strategy weight optimization ──────────────────────────────────────────

def strategy_weights(df: pd.DataFrame, strat_stats: dict):
    _sub("Strategy Weight Optimization")
    print("  Method: WR-weighted vote (use actual WR as weight, not flat +1 per strategy)")
    print()

    # For each ticker×scan_date, compute: WR-weighted score vs current flat count
    # Then see which predicts the outcome better

    # Build ticker-date level: aggregate across strategies
    grp_cols = ["scan_date", "ticker"]
    agg = df.groupby(grp_cols).agg(
        win=("win", "first"),
        ret_d5=("ret_d5", "first"),
        strategies=("strategy", list),
        n_strats=("strategy", "count"),
    ).reset_index()

    def wr_score(strats):
        return sum(strat_stats.get(s, {}).get("wr", 50) / 100 for s in strats)

    agg["wr_score"] = agg["strategies"].apply(wr_score)
    agg["flat_score"] = agg["n_strats"]

    # Correlation with win
    corr_wr   = agg["wr_score"].corr(agg["win"])
    corr_flat = agg["flat_score"].corr(agg["win"])

    print(f"  Flat count correlation with win:    {corr_flat:+.4f}")
    print(f"  WR-weighted score correlation:      {corr_wr:+.4f}")
    better = "WR-weighted" if corr_wr > corr_flat else "flat count"
    print(f"  → {better} is more predictive of outcome")

    # Show top strategies to stack
    _sub("Best Multi-Strategy Combos (n≥5 occurrences)")
    combo_stats = defaultdict(lambda: {"wins": 0, "total": 0, "rets": []})
    for _, row in agg.iterrows():
        key = "+".join(sorted(row["strategies"]))
        combo_stats[key]["total"] += 1
        combo_stats[key]["wins"]  += row["win"]
        combo_stats[key]["rets"].append(row["ret_d5"])

    multi = [(k, v) for k, v in combo_stats.items() if v["total"] >= 3 and "+" in k]
    multi.sort(key=lambda x: -(x[1]["wins"] / x[1]["total"]))
    print(f"  {'Combo':<45} {'N':>4} {'WR':>7} {'Avg':>8}")
    _hr()
    for combo, v in multi[:15]:
        n   = v["total"]
        wr  = v["wins"] / n * 100
        avg = np.mean(v["rets"])
        print(f"  {combo[:45]:<45} {n:>4} {wr:>6.1f}% {avg:>+7.2f}%")


# ── 7. Threshold grid search ─────────────────────────────────────────────────

def threshold_grid_search(df: pd.DataFrame) -> dict:
    _sub("Threshold Grid Search (entry filters)")
    print("  Testing all combos of: min_strategies × min_vol_ratio × rsi_max × score_cap")
    print()

    MIN_STRATS   = [1, 2, 3]
    MIN_VOL      = [1.0, 1.5, 2.0]
    RSI_MAX      = [65, 70, 75, 100]
    SCORE_CAP    = [3, 4, 5, 99]

    results = []
    for ms, mv, rmax, sc in itertools.product(MIN_STRATS, MIN_VOL, RSI_MAX, SCORE_CAP):
        subset = df[
            (df["strategies_count"] >= ms) &
            (df["vol_ratio"].fillna(0) >= mv) &
            (df["rsi"].fillna(50) <= rmax) &
            (df["score"].fillna(99) <= sc)
        ]
        n = len(subset)
        if n < 10: continue
        wr  = subset["win"].mean() * 100
        avg = subset["ret_d5"].mean()
        results.append({
            "min_strats": ms, "min_vol": mv, "rsi_max": rmax, "score_cap": sc,
            "n": n, "wr": wr, "avg": avg,
            "score": wr * 0.6 + avg * 10 * 0.4,  # composite: 60% WR, 40% avg
        })

    results.sort(key=lambda x: -x["score"])

    print(f"  {'min_strats':>10} {'min_vol':>8} {'rsi_max':>8} {'score_cap':>10} {'N':>5} {'WR':>7} {'Avg':>8}  Composite")
    _hr()
    for r in results[:20]:
        flag = "  ◄ BEST" if r == results[0] else ""
        print(f"  {r['min_strats']:>10} {r['min_vol']:>8.1f} {r['rsi_max']:>8} {r['score_cap']:>10} "
              f"{r['n']:>5} {r['wr']:>6.1f}% {r['avg']:>+7.2f}%  {r['score']:>6.1f}{flag}")

    print(f"\n  Baseline (no filters): N={len(df)}, WR={df['win'].mean()*100:.1f}%, avg={df['ret_d5'].mean():+.2f}%")
    return results[0] if results else {}


# ── 8. Walk-forward validation ────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, best_thresh: dict, proposed_weights: dict):
    _sub("Walk-Forward Validation (train on first 7 dates, test on last 3)")

    dates = sorted(df["scan_date"].unique())
    if len(dates) < 4:
        print("  Not enough scan dates for walk-forward (need ≥4)")
        return

    split = max(4, len(dates) - 3)
    train_dates = dates[:split]
    test_dates  = dates[split:]

    train = df[df["scan_date"].isin(train_dates)]
    test  = df[df["scan_date"].isin(test_dates)]

    print(f"  Train: {[str(pd.Timestamp(d).date()) for d in train_dates]}")
    print(f"  Test:  {[str(pd.Timestamp(d).date()) for d in test_dates]}")
    print()

    def eval_subset(subset, label):
        n   = len(subset)
        wr  = subset["win"].mean() * 100 if n else float("nan")
        avg = subset["ret_d5"].mean()     if n else float("nan")
        print(f"    {label:<35} N={n:>4}  WR={wr:>5.1f}%  avg={avg:>+5.2f}%")
        return wr, avg

    print("  [NO FILTER — baseline]")
    eval_subset(train, "Train baseline")
    eval_subset(test,  "Test  baseline")

    # Apply best thresholds found
    def apply_thresh(d, t):
        return d[
            (d["strategies_count"] >= t.get("min_strats", 1)) &
            (d["vol_ratio"].fillna(0) >= t.get("min_vol", 1.0)) &
            (d["rsi"].fillna(50) <= t.get("rsi_max", 100)) &
            (d["score"].fillna(99) <= t.get("score_cap", 99))
        ]

    print("\n  [BEST THRESHOLDS from grid search]")
    eval_subset(apply_thresh(train, best_thresh), "Train filtered")
    tr_wr, tr_avg = eval_subset(apply_thresh(test, best_thresh), "Test  filtered")

    # Check for overfitting signal
    if not math.isnan(tr_wr):
        print()
        if tr_wr >= 55:
            print(f"  ✅  Test WR={tr_wr:.1f}% — thresholds generalise")
        else:
            print(f"  ⚠  Test WR={tr_wr:.1f}% — may be overfit to train dates; use with caution")


# ── 9. Final recommendations ─────────────────────────────────────────────────

def print_recommendations(proposed_weights: dict, best_thresh: dict, strat_stats: dict):
    _hdr("RECOMMENDATIONS")

    print("\n  ① OPTIMAL THRESHOLD FILTERS (apply before rank_score)")
    print(f"    min_strategies : {best_thresh.get('min_strats', 1)}")
    print(f"    min_vol_ratio  : {best_thresh.get('min_vol', 1.0):.1f}x")
    print(f"    rsi_max        : {best_thresh.get('rsi_max', 100)}")
    print(f"    score_cap      : {best_thresh.get('score_cap', 99)}")

    print("\n  ② DATA-DRIVEN RANK_SCORE WEIGHTS (from logistic regression)")
    print(f"    {'Feature':<28} {'Current':>8} {'Proposed':>9}")
    for feat in FEATURES:
        curr = CURRENT_WEIGHTS.get(feat, 0)
        prop = proposed_weights.get(feat, curr)
        marker = " ←" if abs(prop - curr) > 0.4 else ""
        print(f"    {FEATURE_LABELS[feat]:<28} {curr:>8} {prop:>9.1f}{marker}")

    print("\n  ③ STRATEGY WEIGHTS (use WR as multiplier instead of flat +1)")
    strats_sorted = sorted(strat_stats.items(), key=lambda x: -x[1]["wr"])
    for s, v in strats_sorted[:10]:
        proven = "✦" if s in PROVEN_EDGE else " "
        print(f"    {s:<25} actual WR={v['wr']:>5.1f}%  n={v['n']:>3}  {proven}")

    print("\n  ④ NEXT STEPS")
    print("    a. Add hard threshold filters at start of main() in scan.py")
    print("       before rank scoring — drop signals that fail them entirely")
    print("    b. Replace integer _rank_score pts with proposed weights above")
    print("       (only if walk-forward test WR ≥ 55%)")
    print("    c. In next 4 weeks, re-run optimize_weights.py to validate")
    print("       whether proposed weights held out-of-sample")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _hdr("SWING SCANNER WEIGHT OPTIMIZER")
    print(f"  Source: {HISTORY}")

    df = load_data()
    df = engineer_features(df)
    print(f"  Clean rows with ret_d5: {len(df)}  |  Scan dates: {df['scan_date'].nunique()}")
    print(f"  Overall WR: {df['win'].mean()*100:.1f}%  |  avg ret_d5: {df['ret_d5'].mean():+.2f}%")

    feature_analysis(df)
    strat_stats = strategy_analysis(df)
    proposed_weights = logistic_weights(df)
    strategy_weights(df, strat_stats)
    best_thresh = threshold_grid_search(df)
    walk_forward(df, best_thresh, proposed_weights)
    print_recommendations(proposed_weights, best_thresh, strat_stats)


if __name__ == "__main__":
    main()
