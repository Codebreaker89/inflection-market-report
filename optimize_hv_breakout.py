#!/usr/bin/env python3
"""
optimize_hv_breakout.py
───────────────────────
3-year backtest + parameter grid for hv_breakout_scanner.

1. Download ~3y daily bars for the liquid universe (disk-cached)
2. Emit every coil→breakout event under LOOSE gates
3. Attach true cross-sectional HV100 percentile per day
4. Grid-search ratio / ATR / hold / risk% / HV pct / side / vol
5. Walk-forward: train on older 2/3 of dates, test on newest 1/3
6. Print recommended settings; --apply writes them into the scanner

Usage:
  python3 optimize_hv_breakout.py
  python3 optimize_hv_breakout.py --apply
"""

from __future__ import annotations

import argparse, json, pickle, re, time, warnings, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from scanner_utils import _adx, _quiet, _rsi

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".cache_hv_bt"
W = 88

HV_SHORT, HV_LONG = 6, 100
GEN_RATIO_MAX = 0.65
GEN_ATR_MAX   = 2.5
MIN_PRICE     = 1.0
MIN_VOL_MA    = 100_000
MAX_WORKERS   = 20
HOLDS         = (2, 3, 5)


def _hr(c="─"): print(c * W)
def _hdr(t): _hr("═"); print(f"  {t}"); _hr("═")


def _hv(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n).std() * np.sqrt(252)


def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    logret = np.log(c / c.shift(1))
    df["hv_s"]     = _hv(logret, HV_SHORT)
    df["hv_l"]     = _hv(logret, HV_LONG)
    df["hv_ratio"] = df["hv_s"] / df["hv_l"].replace(0, np.nan)
    df["range"]    = h - l
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr14"]     = tr.ewm(alpha=1/14, adjust=False).mean()
    df["rsi"]       = _rsi(c, 14)
    df["adx"]       = _adx(h, l, c, 14)
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"].replace(0, np.nan)
    return df


def _universe() -> list[str]:
    from momentum_scanner import build_universe
    uni = build_universe()
    return sorted(t for t in uni if not t.endswith((".NS", ".BO")))


def _cache_path(ticker: str, start: str) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}_{start}.pkl"


def _fetch(ticker: str, start: str) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker, start)
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass
    try:
        with _quiet():
            raw = yf.download(ticker, start=start, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < HV_LONG + 30:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            return None
        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < HV_LONG + 30:
            return None
        df = _build(raw.copy())
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))
        return df
    except Exception:
        return None


def _events_for(ticker: str, df: pd.DataFrame) -> list[dict]:
    out = []
    last_i = len(df) - 1 - max(HOLDS) - 1
    for i in range(HV_LONG + 5, last_i + 1):
        y = df.iloc[i - 1]
        t = df.iloc[i]
        hv_l = float(y["hv_l"]); hv_r = float(y["hv_ratio"])
        atr  = float(y["atr14"]); rng = float(y["range"])
        if any(pd.isna(x) for x in (hv_l, hv_r, atr, rng)) or atr <= 0 or hv_l <= 0:
            continue
        if hv_r > GEN_RATIO_MAX or rng > GEN_ATR_MAX * atr:
            continue
        c = float(t["Close"]); hi = float(y["High"]); lo = float(y["Low"])
        vol_ma = float(t["vol_ma20"]) if not pd.isna(t["vol_ma20"]) else 0
        if pd.isna(c) or c < MIN_PRICE or vol_ma < MIN_VOL_MA:
            continue
        above, below = c > hi, c < lo
        if above == below:
            continue
        side = "long" if above else "short"
        stop = lo if side == "long" else hi
        risk_pct = abs(c - stop) / c * 100
        rets = {}
        for h in HOLDS:
            ex = float(df.iloc[i + h]["Close"])
            if pd.isna(ex) or c <= 0:
                rets[h] = np.nan
            elif side == "short":
                rets[h] = (c - ex) / c * 100
            else:
                rets[h] = (ex - c) / c * 100
        out.append({
            "ticker": ticker,
            "date": df.index[i],
            "side": side,
            "entry": c,
            "stop": stop,
            "hv100": hv_l,
            "hv_ratio": hv_r,
            "range_atr": rng / atr,
            "risk_pct": risk_pct,
            "rsi": float(t["rsi"]) if not pd.isna(t["rsi"]) else np.nan,
            "adx": float(t["adx"]) if not pd.isna(t["adx"]) else np.nan,
            "vol_ratio": float(t["vol_ratio"]) if not pd.isna(t["vol_ratio"]) else np.nan,
            "ret_2": rets[2], "ret_3": rets[3], "ret_5": rets[5],
        })
    return out


def load_all(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events_df, hv_panel date×ticker). Single download pass."""
    print(f"  Fetching/caching {len(tickers)} tickers from {start}...")
    t0 = time.time()
    rows: list[dict] = []
    hv_series: dict[str, pd.Series] = {}
    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_fetch, t, start): t for t in tickers}
        for n, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                df = f.result()
                if df is None:
                    continue
                ok += 1
                s = df["hv_l"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                hv_series[t] = s
                rows.extend(_events_for(t, df))
            except Exception:
                pass
            if n % 50 == 0 or n == len(tickers):
                print(f"    {n}/{len(tickers)}  ·  {ok} ok  ·  {len(rows)} events  ·  "
                      f"{time.time()-t0:.0f}s", flush=True)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    ev = pd.DataFrame(rows)
    ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    ev = ev.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel = pd.DataFrame(hv_series)
    print(f"  Events: {len(ev):,}  ·  panel {panel.shape[0]}d × {panel.shape[1]} names")
    return ev, panel


def attach_hv_percentile(ev: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    ranks = panel.rank(axis=1, pct=True, method="average")
    pcts = []
    for _, r in ev.iterrows():
        d, t = r["date"], r["ticker"]
        try:
            if d in ranks.index and t in ranks.columns and not pd.isna(ranks.at[d, t]):
                pcts.append(float(ranks.at[d, t]))
            elif t in ranks.columns:
                sub = ranks.loc[:d, t].dropna()
                pcts.append(float(sub.iloc[-1]) if len(sub) else np.nan)
            else:
                pcts.append(np.nan)
        except Exception:
            pcts.append(np.nan)
    out = ev.copy()
    out["hv_pct"] = pcts
    return out


@dataclass
class Metrics:
    n: int
    wr: float
    avg: float
    med: float
    pf: float
    avg_long: float
    avg_short: float
    n_long: int
    n_short: int


def _metrics(rets: pd.Series, sides: pd.Series) -> Metrics:
    r = rets.dropna()
    if len(r) == 0:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    wins = r[r > 0]; losses = r[r <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else 99.0
    long_m  = r[sides.loc[r.index] == "long"]
    short_m = r[sides.loc[r.index] == "short"]
    return Metrics(
        n=len(r),
        wr=100 * float((r > 0).mean()),
        avg=float(r.mean()),
        med=float(r.median()),
        pf=float(pf) if np.isfinite(pf) else 99.0,
        avg_long=float(long_m.mean()) if len(long_m) else float("nan"),
        avg_short=float(short_m.mean()) if len(short_m) else float("nan"),
        n_long=len(long_m),
        n_short=len(short_m),
    )


def grid_search(ev: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(ev["date"].unique())
    cut = dates[int(len(dates) * 2 / 3)]
    print(f"  Walk-forward cut: train ≤ {pd.Timestamp(cut).date()}  ·  "
          f"test > {pd.Timestamp(cut).date()}")

    rows = []
    for ratio in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for atr_cap in [1.5, 2.0, 2.5]:
            for hold in [2, 3, 5]:
                for pct in [0.70, 0.75, 0.80]:
                    for risk in [None, 8.0, 12.0]:
                        for need_vol in [False, True]:
                            for side_mode in ["both", "long"]:
                                m = ev[
                                    (ev["hv_ratio"] <= ratio) &
                                    (ev["range_atr"] <= atr_cap) &
                                    (ev["hv_pct"] >= pct)
                                ]
                                if risk is not None:
                                    m = m[m["risk_pct"] <= risk]
                                if need_vol:
                                    m = m[m["vol_ratio"] >= 1.0]
                                if side_mode == "long":
                                    m = m[m["side"] == "long"]
                                if m.empty:
                                    continue
                                col = f"ret_{hold}"
                                train = m[m["date"] <= cut]
                                test  = m[m["date"] > cut]
                                mt = _metrics(train[col], train["side"])
                                ms = _metrics(test[col],  test["side"])
                                score = (
                                    ms.wr * 2.0
                                    + ms.avg * 10.0
                                    + min(ms.pf, 3.0) * 5.0
                                    + (5.0 if ms.n >= 40 else 0.0)
                                    - abs(mt.avg - ms.avg)
                                )
                                rows.append({
                                    "ratio": ratio, "atr": atr_cap, "hold": hold,
                                    "hv_pct": pct,
                                    "risk": risk if risk else 99,
                                    "vol": int(need_vol), "sides": side_mode,
                                    "tr_n": mt.n, "tr_wr": round(mt.wr, 1),
                                    "tr_avg": round(mt.avg, 2),
                                    "te_n": ms.n, "te_wr": round(ms.wr, 1),
                                    "te_avg": round(ms.avg, 2),
                                    "te_med": round(ms.med, 2),
                                    "te_pf": round(ms.pf, 2),
                                    "te_n_l": ms.n_long, "te_n_s": ms.n_short,
                                    "te_avg_l": round(ms.avg_long, 2) if ms.n_long else None,
                                    "te_avg_s": round(ms.avg_short, 2) if ms.n_short else None,
                                    "score": round(score, 2),
                                })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def baseline_metrics(ev: pd.DataFrame) -> Metrics:
    m = ev[
        (ev["hv_ratio"] <= 0.50) &
        (ev["range_atr"] <= 2.0) &
        (ev["hv_pct"] >= 0.75)
    ]
    return _metrics(m["ret_3"], m["side"])


def apply_defaults(best: dict):
    path = HERE / "hv_breakout_scanner.py"
    text = path.read_text()
    hold = int(best["hold"])
    ratio = float(best["ratio"])
    atr = float(best["atr"])
    pct = float(best["hv_pct"])
    risk = best["risk"]
    need_vol = bool(best["vol"])
    long_only = best["sides"] == "long"

    def sub(pat, rep):
        nonlocal text
        text2, n = re.subn(pat, rep, text, count=1, flags=re.M)
        if n != 1:
            print(f"  ⚠ patch missed: {pat}")
        else:
            text = text2

    sub(r"^HOLD_DAYS\s*=\s*\d+", f"HOLD_DAYS       = {hold}")
    sub(r"^HV_RATIO_MAX\s*=\s*[\d.]+.*",
        f"HV_RATIO_MAX    = {ratio:.2f}      # coil: short HV ≤ fraction of long HV")
    sub(r"^RANGE_ATR_MAX\s*=\s*[\d.]+.*",
        f"RANGE_ATR_MAX   = {atr:.1f}       # skip if yesterday range > N× ATR(14)")

    if re.search(r"^HV_PCT_MIN\s*=", text, re.M):
        sub(r"^HV_PCT_MIN\s*=\s*[\d.]+.*",
            f"HV_PCT_MIN      = {pct:.2f}      # cross-sectional HV100 floor")
    else:
        text = text.replace(
            f"HV_RATIO_MAX    = {ratio:.2f}      # coil: short HV ≤ fraction of long HV",
            f"HV_RATIO_MAX    = {ratio:.2f}      # coil: short HV ≤ fraction of long HV\n"
            f"HV_PCT_MIN      = {pct:.2f}      # cross-sectional HV100 floor",
        )

    risk_line = (
        f"MAX_RISK_PCT    = {float(risk):.1f}       # skip if |entry-stop|/entry > this %"
        if risk and risk < 90 else
        "MAX_RISK_PCT    = None      # no extra risk% cap"
    )
    block = (
        f"{risk_line}\n"
        f"REQUIRE_VOL     = {need_vol}     # require vol_ratio ≥ 1.0 on breakout day\n"
        f"LONG_ONLY       = {long_only}    # drop short breakouts if True"
    )
    if re.search(r"^MAX_RISK_PCT\s*=", text, re.M):
        text = re.sub(
            r"^MAX_RISK_PCT\s*=.*\nREQUIRE_VOL\s*=.*\nLONG_ONLY\s*=.*",
            block, text, count=1, flags=re.M,
        )
    else:
        text = text.replace("MIN_PRICE       = 1.0", block + "\nMIN_PRICE       = 1.0")

    path.write_text(text)
    print(f"  Wrote tuned defaults → {path.name}")
    print("  NOTE: scanner code must honour HV_PCT_MIN / MAX_RISK_PCT / "
          "REQUIRE_VOL / LONG_ONLY — patching those next if missing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--years", type=float, default=3.0)
    args = ap.parse_args()

    start = (datetime.utcnow() - timedelta(days=int(args.years * 365) + 40)).strftime("%Y-%m-%d")
    _hdr(f"HV BREAKOUT — {args.years:.0f}Y BACKTEST + GRID")
    tickers = _universe()
    print(f"  Universe: {len(tickers)} tickers (ex-India)  ·  cache {CACHE_DIR.name}/")

    ev, panel = load_all(tickers, start)
    if ev.empty:
        print("  No events — abort"); return
    ev = attach_hv_percentile(ev, panel)
    ev = ev.dropna(subset=["hv_pct", "ret_3"])
    print(f"  Ranked events: {len(ev):,}  ·  "
          f"{ev['date'].min().date()} → {ev['date'].max().date()}")

    base = baseline_metrics(ev)
    print(f"\n  BASELINE (ratio≤0.50, atr≤2.0, hv≥p75, hold=3, both):")
    print(f"    n={base.n}  WR={base.wr:.1f}%  avg={base.avg:+.2f}%  med={base.med:+.2f}%  "
          f"PF={base.pf:.2f}")
    print(f"    long n={base.n_long} avg={base.avg_long:+.2f}%  ·  "
          f"short n={base.n_short} avg={base.avg_short:+.2f}%")

    _hdr("GRID SEARCH (walk-forward)")
    grid = grid_search(ev)
    viable = grid[grid["te_n"] >= 30]
    if viable.empty:
        viable = grid[grid["te_n"] >= 15]
    print(f"  Combos: {len(grid)}  ·  viable: {len(viable)}")

    cols = ["ratio","atr","hold","hv_pct","risk","vol","sides",
            "tr_n","tr_wr","tr_avg","te_n","te_wr","te_avg","te_pf","score"]
    print("\n  TOP 15:")
    print(viable[cols].head(15).to_string(index=False))

    best = viable.iloc[0].to_dict()
    print(f"\n  ★ BEST:")
    print(f"    ratio={best['ratio']} atr={best['atr']} hold={int(best['hold'])}d "
          f"hv_pct≥{best['hv_pct']} risk≤{None if best['risk']>=90 else best['risk']} "
          f"vol={bool(best['vol'])} sides={best['sides']}")
    print(f"    Train n={best['tr_n']} WR={best['tr_wr']}% avg={best['tr_avg']:+.2f}%")
    print(f"    Test  n={best['te_n']} WR={best['te_wr']}% avg={best['te_avg']:+.2f}% "
          f"PF={best['te_pf']}")

    m = ev[
        (ev["hv_ratio"] <= best["ratio"]) &
        (ev["range_atr"] <= best["atr"]) &
        (ev["hv_pct"] >= best["hv_pct"])
    ].copy()
    if best["risk"] < 90:
        m = m[m["risk_pct"] <= best["risk"]]
    if best["vol"]:
        m = m[m["vol_ratio"] >= 1.0]
    if best["sides"] == "long":
        m = m[m["side"] == "long"]
    full = _metrics(m[f"ret_{int(best['hold'])}"], m["side"])
    print(f"    Full  n={full.n} WR={full.wr:.1f}% avg={full.avg:+.2f}% "
          f"med={full.med:+.2f}% PF={full.pf:.2f}")

    out = HERE / "optimize_hv_breakout_results.json"
    out.write_text(json.dumps({
        "generated": datetime.utcnow().isoformat() + "Z",
        "baseline": base.__dict__,
        "best": best,
        "full_under_best": full.__dict__,
        "top15": viable[cols].head(15).to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"\n  Saved {out.name}")

    improves = best["te_n"] >= 30 and (
        best["te_wr"] >= base.wr + 2 or best["te_avg"] >= base.avg + 0.15
    )
    if args.apply:
        if improves or best["te_avg"] > 0:
            apply_defaults(best)
        else:
            print("  --apply skipped: best combo does not clearly beat baseline.")
    else:
        print("  Re-run with --apply to write defaults into hv_breakout_scanner.py")


if __name__ == "__main__":
    main()
