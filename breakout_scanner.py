#!/usr/bin/env python3
"""
Breakout Scanner  |  US + Europe + Canada
──────────────────────────────────────────
Catches stocks BEFORE they move — finds setups coiling for a breakout.
This is the companion to momentum_scanner.py:

  momentum_scanner.py  →  stock is ALREADY in momentum (crossovers happened)
  breakout_scanner.py  →  stock is ABOUT TO move (building energy, not yet run)

Why AMAT-type moves get missed by momentum_scanner:
  The ADX is low while the stock coils, then explodes. By the time ADX > 22
  and crossovers fire, you're already 5-10% late. This scanner catches the
  coiling phase by looking for: tightening price range + volume dry-up +
  price near pivot high + ADX starting to curl up from low base.

Two signal categories per stock:
  COIL   — classic VCP setup, stock building energy (enter with tight stop)
  BREAK  — coil + today's volume/price confirming the actual breakout started

python3 breakout_scanner.py                # full scan + backtest
python3 breakout_scanner.py --no-backtest  # signals only (~5-10 min)
python3 breakout_scanner.py --fast         # 30 US stocks smoke-test
python3 breakout_scanner.py --legend       # column & signal definitions

First run / if you see 401 errors:
  pip3 install --upgrade yfinance

Cron (weekdays 6:30am, BEFORE market open):
  00 6 * * 1-5 python3 /path/to/breakout_scanner.py --no-backtest >> ~/breakout.log 2>&1
"""

import io, os, sys, time, random, warnings, logging, contextlib
import requests
import numpy  as np
import pandas as pd
import yfinance as yf
from datetime           import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Optional
from scanner_utils import _adx, _ema, _fetch_html, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


# ── CONFIG ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 400
HOLD_DAYS       = 5
MAX_WORKERS     = 25
MIN_PRICE       = 1.0
MIN_AVG_VOL     = 100_000
EXCLUDE_SECTORS = {"Utilities", "Real Estate"}
TOP_N           = 50
HEADERS         = {"User-Agent": "Mozilla/5.0 (compatible; breakout-scanner/1.0)"}

# Breakout-specific thresholds
MIN_MINERVINI   = 5      # relaxed vs momentum_scanner (catching earlier in base)
ATR_CONTRACT    = 0.72   # recent ATR must be < 72% of prior ATR
VOL_DRYUP       = 0.65   # recent vol must be < 65% of 20d avg
PIVOT_PROXIMITY = 0.04   # price within 4% of 10-day high
ADX_COIL_MAX    = 25     # ADX must be below this (not already trending)
ADX_COIL_MIN    = 15     # ADX must be above this (not dead flat; raised from 10 to cut weak-trend FPs)
MIN_COIL_SCORE  = 4      # min to qualify as a coil setup
MIN_BREAK_SCORE = 7      # min to qualify as active breakout

# ── MARKET / BENCHMARK MAP (same as momentum_scanner) ────────────────────────
SUFFIX_MAP = {
    ".L":  ("UK", "^FTSE"),   ".DE": ("DE", "^GDAXI"),
    ".PA": ("FR", "^FCHI"),   ".AS": ("NL", "^AEX"),
    ".MC": ("ES", "^IBEX"),   ".MI": ("IT", "FTSEMIB.MI"),
    ".SW": ("CH", "^SSMI"),   ".HE": ("FI", "^OMXH25"),
    ".ST": ("SE", "^OMX"),    ".CO": ("DK", "^OMXC20"),
    ".OL": ("NO", "^OBX"),    ".TO": ("CA", "^GSPTSE"),
    ".BR": ("BE", "^BFX"),    ".LS": ("PT", "PSI20.LS"),
}
DEFAULT_MKT   = "US"
DEFAULT_BENCH = "^GSPC"

def get_mkt(t):
    for sfx, (mkt, _) in SUFFIX_MAP.items():
        if t.endswith(sfx): return mkt
    return DEFAULT_MKT

def get_bench(t):
    for sfx, (_, b) in SUFFIX_MAP.items():
        if t.endswith(sfx): return b
    return DEFAULT_BENCH


# ── ANSI COLORS ───────────────────────────────────────────────────────────────
_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t);  RED  = lambda t: _c("31", t)
YLW  = lambda t: _c("33", t);  CYN  = lambda t: _c("36", t)
MAG  = lambda t: _c("35", t);  BOLD = lambda t: _c("1",  t)
DIM  = lambda t: _c("2",  t)

def ret_fmt(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.2f}%") if v >= 0 else RED(f"{v:.2f}%")
def wr_fmt(v):
    if v is None: return DIM("  ─ ")
    return GRN(f"{v:.0f}%") if v >= 60 else (YLW(f"{v:.0f}%") if v >= 45 else RED(f"{v:.0f}%"))


# ── UNIVERSE (shared logic with momentum_scanner) ─────────────────────────────
_KNOWN_SFX = {".L",".DE",".PA",".AS",".MC",".MI",".SW",
              ".TO",".HE",".ST",".CO",".OL",".BR",".LS"}

def _clean(tickers, suffix="", max_len=8):
    out = []
    for t in tickers:
        if not isinstance(t, str): continue
        t = t.strip().upper()
        if not t or t == "-": continue
        if any(t.endswith(s) for s in _KNOWN_SFX):
            out.append(t); continue
        base  = t.replace(".", "-")
        final = base + suffix if suffix else base
        if len(final) <= max_len + 4: out.append(final)
    return out

def _ticker_col(tbl):
    for n in ["Ticker","ticker","Symbol","symbol","EPIC","Epic","Code"]:
        if n in tbl.columns: return n
    return None

def get_sp500_with_sectors():
    tbls = _fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    t = tbls[0]; t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    excl = set(t.loc[t["GICS Sector"].isin(EXCLUDE_SECTORS), "Symbol"])
    return t["Symbol"].tolist(), excl

def get_russell1000():
    try:
        r  = requests.get("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
                          "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
                          headers=HEADERS, timeout=20)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist())
    except Exception: return []

def get_ftse100():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/FTSE_100"):
            col = _ticker_col(t)
            if col: return _clean(t[col].dropna().tolist(), ".L", 6)
    except Exception: return []
    return []

def get_dax40():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/DAX"):
            col = _ticker_col(t)
            if col and len(t) >= 20: return _clean(t[col].dropna().tolist(), ".DE")
    except Exception: return []
    return []

def get_cac40():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/CAC_40"):
            col = _ticker_col(t)
            if col and len(t) >= 15: return _clean(t[col].dropna().tolist(), ".PA")
    except Exception: return []
    return []

def get_stoxx600():
    try:
        r  = requests.get("https://www.ishares.com/uk/individual/en/products/251813/"
                          "ishares-stoxx-europe-600-ucits-etf/1478372549652.ajax"
                          "?fileType=csv&fileName=EXW1_holdings&dataType=fund",
                          headers=HEADERS, timeout=25)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return [t.strip() for t in df["Ticker"].dropna()
                if isinstance(t, str) and t.strip() and " " not in t.strip()
                and t.strip() != "-" and len(t.strip()) <= 12]
    except Exception: return []

def get_tsx():
    try:
        r  = requests.get("https://www.ishares.com/ca/individual/en/products/239837/"
                          "ISHARES-SPTSX-CAPPED-COMPOSITE-INDEX-ETF/1490978163731.ajax"
                          "?fileType=csv&fileName=XIC_holdings&dataType=fund",
                          headers=HEADERS, timeout=25)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist(), ".TO")
    except Exception: return []

def get_universe(fast_mode=False) -> dict:
    print(DIM("  Loading tickers..."), flush=True)
    sp500, excl = get_sp500_with_sectors()
    if fast_mode:
        s = [t for t in random.sample(sp500, 50) if t not in excl][:30]
        print(DIM(f"  FAST MODE — {len(s)} US tickers")); return {t: DEFAULT_BENCH for t in s}
    r1000 = get_russell1000(); ftse = get_ftse100(); dax = get_dax40()
    cac   = get_cac40();       stoxx= get_stoxx600(); tsx = get_tsx()
    uni   = {}
    def _add(lst, ex=set()):
        for t in lst:
            if t not in ex and t not in uni: uni[t] = get_bench(t)
    _add([t for t in list(dict.fromkeys(sp500+r1000)) if t not in excl])
    _add(ftse); _add(dax); _add(cac); _add(stoxx); _add(tsx)
    counts = {k: sum(1 for t in uni if get_mkt(t)==k)
              for k in ["US","UK","DE","FR","CA"]}
    other  = sum(1 for t in uni if get_mkt(t) not in counts)
    cs     = "  ".join(f"{k}:{v}" for k,v in counts.items() if v) + f"  Other:{other}"
    print(DIM(f"  {len(uni)} tickers  ·  {cs}"))
    return uni

def fetch_benchmark_returns(syms: set) -> dict:
    results = {}
    for sym in syms:
        try:
            with _quiet():
                df = yf.download(sym, period="120d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            c = df["Close"].dropna()
            if len(c) >= 64:
                results[sym] = float((c.iloc[-1]-c.iloc[-63])/c.iloc[-63]*100)
        except Exception: pass
    return results


# ── INDICATORS ────────────────────────────────────────────────────────────────
def _atr(h,l,c,n=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c,h,l,v = df["Close"],df["High"],df["Low"],df["Volume"]
    df["sma50"]   = _sma(c,50);  df["sma150"] = _sma(c,150); df["sma200"] = _sma(c,200)
    df["ema9"]    = _ema(c,9);   df["ema21"]  = _ema(c,21)
    df["rsi"]     = _rsi(c,14)
    df["adx"]     = _adx(h,l,c,14)
    df["atr"]     = _atr(h,l,c,14)
    df["vol_ma20"]= v.rolling(20).mean()
    df["52w_high"]= c.rolling(252).max(); df["52w_low"] = c.rolling(252).min()
    # Range as % of price (normalised volatility)
    df["hl_pct"]  = (h-l)/c
    return df


# ── COIL + BREAKOUT SCORING ───────────────────────────────────────────────────

def score_row(df: pd.DataFrame, idx: int,
              bench_ret63: Optional[float] = None) -> Optional[dict]:
    if idx < 230: return None
    row  = df.iloc[idx]
    prev = df.iloc[idx-1]

    # Basic filters
    if float(row["Close"]) < MIN_PRICE: return None
    if float(row["vol_ma20"]) < MIN_AVG_VOL: return None

    # Minervini Trend Template — relaxed to ≥5 (earlier in base formation)
    m = sum([
        row["Close"]  > row["sma150"],
        row["Close"]  > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        row["Close"]  > row["sma50"],
        row["Close"]  >= 1.20 * row["52w_low"],   # relaxed: 20% off low vs 30%
        row["Close"]  >= 0.70 * row["52w_high"],   # relaxed: within 30% of high vs 25%
        row["sma200"] > df.iloc[idx-20]["sma200"],
    ])
    if m < MIN_MINERVINI: return None

    adx_val = float(row["adx"])

    # ADX floor: no trend = no setup; cap: overextended = chasing
    if adx_val < 16: return None
    if adx_val > 35: return None

    # ── COIL SIGNALS ─────────────────────────────────────────────────────────

    # 1. ATR contraction: recent 5-bar ATR < ATR_CONTRACT × prior 15-bar ATR
    atr_recent = float(df["atr"].iloc[idx-5:idx].mean())
    atr_prior  = float(df["atr"].iloc[idx-20:idx-5].mean())
    atr_ok     = atr_prior > 0 and (atr_recent / atr_prior) < ATR_CONTRACT
    atr_ratio  = round(atr_recent / atr_prior, 2) if atr_prior > 0 else None

    # 2. Volume dry-up: avg vol last 5 bars < VOL_DRYUP × 20d avg
    vol_recent = float(df["Volume"].iloc[idx-5:idx].mean())
    vol_avg20  = float(row["vol_ma20"])
    vol_quiet  = vol_avg20 > 0 and (vol_recent / vol_avg20) < VOL_DRYUP
    vol_ratio_5d = round(vol_recent / vol_avg20, 2) if vol_avg20 > 0 else None

    # 3. Price within PIVOT_PROXIMITY of 10-day high (coiling near resistance)
    pivot_high  = float(df["Close"].iloc[idx-10:idx+1].max())
    near_pivot  = (float(row["Close"]) / pivot_high) >= (1 - PIVOT_PROXIMITY)

    # 4. ADX coiling: low-to-mid range AND rising from recent low
    adx_low_5  = float(df["adx"].iloc[idx-5:idx].min())
    adx_coil   = (ADX_COIL_MIN < adx_val < ADX_COIL_MAX) and (adx_val > adx_low_5)

    # 5. RSI base: RSI between 50–62 (floor raised 45→50: RSI 45-50 = avg -0.25% in backtest)
    rsi_base   = 50 < float(row["rsi"]) < 62

    # 6. Relative strength improving: stock RS trend vs benchmark curling up
    rs_improving = False
    if bench_ret63 is not None and idx >= 63:
        p63 = float(df.iloc[idx-63]["Close"])
        if p63 > 0:
            stock_ret63 = (float(row["Close"]) - p63) / p63 * 100
            # RS improving even if not yet > benchmark (within 5% is OK for coil)
            rs_improving = stock_ret63 >= bench_ret63 - 5.0

    # 7. HL% range tightening: today's range < 0.7× avg of last 10 days
    hl_today = float(row["hl_pct"])
    hl_avg10  = float(df["hl_pct"].iloc[idx-10:idx].mean())
    range_tight = hl_avg10 > 0 and hl_today < hl_avg10 * 0.70

    coil_signals = {
        "ATR<":    atr_ok,
        "VOLdry":  vol_quiet,
        "NearPiv": near_pivot,
        "ADXcurl": adx_coil,
        "RSIbase": rsi_base,
        "RS~ok":   rs_improving,
        "RngTght": range_tight,
    }
    coil_score = sum(coil_signals.values())
    if coil_score < MIN_COIL_SCORE: return None

    # ── BREAKOUT CONFIRMATION SIGNALS ────────────────────────────────────────
    # These fire ON TOP of the coil to signal the actual breakout day

    # B1. Volume surge TODAY (the flush confirming the breakout)
    today_vol  = float(row["Volume"])
    vol_surge  = vol_avg20 > 0 and (today_vol / vol_avg20) > 1.8

    # B2. Price broke above 10-day high today
    prev_pivot = float(df["Close"].iloc[idx-11:idx].max())
    price_break= float(row["Close"]) > prev_pivot * 1.005

    # B3. ADX starting to accelerate (rising >15% vs 3 bars ago)
    adx_accel  = adx_val > float(df["adx"].iloc[idx-3]) * 1.15

    # B4. EMA 9 crossed above EMA 21 today or yesterday
    ema_cross  = False
    for k in range(max(1, idx-2), idx+1):
        if (df["ema9"].iloc[k] > df["ema21"].iloc[k] and
                df["ema9"].iloc[k-1] <= df["ema21"].iloc[k-1]):
            ema_cross = True; break

    break_signals = {
        "VolSurge": vol_surge,
        "PriceBrk": price_break,
        "ADXaccel": adx_accel,
        "EMAxover": ema_cross,
    }
    break_score = sum(break_signals.values())

    total_score = coil_score + break_score * 2  # break signals weighted double
    phase       = "BREAK" if (break_score >= 2 and total_score >= MIN_BREAK_SCORE) else "COIL"

    # Score cap: COIL signals score≥6 = WR drops to 49%, avg -0.27% (scan_history backtest)
    # Score 2-5 = sweet spot (WR 60-66%). BREAK phase exempt — confirmation overrides.
    if phase == "COIL" and total_score >= 6: return None

    coil_list  = [k for k, v in coil_signals.items() if v]
    break_list = [k for k, v in break_signals.items() if v]
    return {
        "score":        total_score,
        "phase":        phase,
        "coil_sigs":    coil_list,
        "break_sigs":   break_list,
        # unified scan.py display keys
        "fresh":        ([phase] + break_list) if break_list else [phase],
        "conf":         coil_list,
        "minervini":    m,
        "rsi":          round(float(row["rsi"]),  1),
        "adx":          round(adx_val,            1),
        "price":        round(float(row["Close"]),2),
        "vol_ratio":    round(float(row["Volume"]) / float(row["vol_ma20"]), 2) if float(row["vol_ma20"]) > 0 else 0,
        "atr_ratio":    atr_ratio,
        "vol_5d_ratio": vol_ratio_5d,
        "pivot_high":   round(pivot_high,         2),
        "pct_from_piv": round((float(row["Close"])/pivot_high - 1)*100, 1),
    }


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(230, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        sig = score_row(df, i)
        if not sig: continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n":   len(a),
            "wr":  round(100*(a>0).mean(), 1),
            "avg": round(float(a.mean()),  2),
            "med": round(float(np.median(a)), 2)}


# ── PER-TICKER PIPELINE ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret63: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period=f"{LOOKBACK_DAYS}d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 235: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        df  = build_indicators(raw.copy())
        sig = score_row(df, len(df)-1, bench_ret63)
        if not sig: return None
        result = {"ticker": ticker, "mkt": get_mkt(ticker), **sig}
        if with_backtest: result.update(run_backtest(df))
        return result
    except Exception:
        return None


# ── DISPLAY ───────────────────────────────────────────────────────────────────

W = 108

def print_results(results: list, with_backtest: bool, bench_returns: dict):
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
    coil_count  = sum(1 for r in results if r["phase"] == "COIL")
    break_count = sum(1 for r in results if r["phase"] == "BREAK")

    key_benches = {"^GSPC":"US","^FTSE":"UK","^GDAXI":"DE","^FCHI":"FR","^GSPTSE":"CA"}
    bench_str   = "  ".join(f"{l}:{ret_fmt(bench_returns.get(s))}"
                            for s, l in key_benches.items() if s in bench_returns)

    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  🔭  BREAKOUT SCANNER  ·  {now}  ·  {break_count} breaking  ·  {coil_count} coiling".ljust(W-2) + "║")
    if bench_str:
        print("║" + f"  Benchmarks (63d):  {bench_str}".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

    if not results:
        print(YLW("  No setups found today.")); return

    # Sort: BREAK first, then by score
    sorted_r = sorted(results[:TOP_N],
                      key=lambda r: (0 if r["phase"]=="BREAK" else 1, -r["score"]))

    for phase_group in ["BREAK", "COIL"]:
        group = [r for r in sorted_r if r["phase"] == phase_group]
        if not group: continue

        if phase_group == "BREAK":
            label = "🚀  BREAKING OUT NOW  —  volume + price confirming breakout today"
            color = GRN
        else:
            label = "🔧  COILING / SETUP  —  energy building, not yet broken out"
            color = YLW

        print(f"\n  {color(BOLD(label))}")
        print()

        if with_backtest:
            hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>8}  {'→PIV':>6}  "
                   f"{'RSI':>5}  {'ADX':>5}  {'ATR%':>5}  {'V5d':>5}  {'M':>3}  {'SCR':>3}  "
                   f"{'#BT':>3}  {'WIN%':>5}  {'AVG':>6}  {'MED':>6}  SIGNALS")
        else:
            hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>8}  {'→PIV':>6}  "
                   f"{'RSI':>5}  {'ADX':>5}  {'ATR%':>5}  {'V5d':>5}  {'M':>3}  {'SCR':>3}  SIGNALS")
        print(BOLD(hdr))
        print("  " + "─"*(W-2))

        for rank, r in enumerate(group, 1):
            cs  = " ".join(r.get("coil_sigs",  []))
            bs  = (" ▶ " + " ".join(r.get("break_sigs",[]))) if r.get("break_sigs") else ""
            sig = CYN(cs) + MAG(bs)

            mkt_s    = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
            ticker_s = BOLD(f"{r['ticker']:<8}")
            atr_s    = f"{r['atr_ratio']:.2f}" if r["atr_ratio"] else "─"
            v5d_s    = f"{r['vol_5d_ratio']:.2f}" if r["vol_5d_ratio"] else "─"
            piv_s    = f"{r['pct_from_piv']:+.1f}%"

            base = (f"  {rank:>3}  {mkt_s}  {ticker_s}  {r['price']:>8.2f}"
                    f"  {piv_s:>6}  {r['rsi']:>5.1f}  {r['adx']:>5.1f}"
                    f"  {atr_s:>5}  {v5d_s:>5}  {r['minervini']:>3}  {r['score']:>3}  ")

            if with_backtest and r.get("n", 0) > 0:
                row = (base + f"{r['n']:>3}  "
                       + f"{wr_fmt(r['wr']):>5}  "
                       + f"{ret_fmt(r['avg']):>6}  "
                       + f"{ret_fmt(r['med']):>6}  " + sig)
            elif with_backtest:
                row = base + DIM("  ─     ─      ─      ─   ") + sig
            else:
                row = base + sig
            print(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n  " + "─"*(W-2))
    if with_backtest:
        rets = [r["avg"] for r in results if r.get("avg") is not None]
        wrs  = [r["wr"]  for r in results if r.get("wr")  is not None]
        brets= [r["avg"] for r in results if r.get("avg") is not None and r["phase"]=="BREAK"]
        bwrs = [r["wr"]  for r in results if r.get("wr")  is not None and r["phase"]=="BREAK"]
        if rets:
            print(f"  ALL  ·  {len(rets)} tickers  ·  "
                  f"median win rate {wr_fmt(float(np.median(wrs)))}  ·  "
                  f"median avg return {ret_fmt(float(np.median(rets)))}")
        if brets:
            print(f"  BREAK only  ·  {len(brets)} tickers  ·  "
                  f"median win rate {wr_fmt(float(np.median(bwrs)))}  ·  "
                  f"median avg return {ret_fmt(float(np.median(brets)))}")

    print(f"\n  {DIM('⚠  Not financial advice. COIL = pre-breakout setup; BREAK = breakout firing today.')}")
    print(f"  {DIM('--legend for signal definitions.')}")
    print("╚" + "═"*(W-2) + "╝\n")


def print_legend():
    print()
    print(BOLD("━"*W))
    print(BOLD("  APPENDIX — Breakout Scanner: Column & Signal Reference"))
    print(BOLD("━"*W))
    print("""
  PHASE LABELS
  ────────────
  BREAK  Stock is actively breaking out TODAY — volume surge + price above pivot.
         Enter with tight stop just below today's low or the pivot high.
  COIL   Stock is building energy in a tight base — not yet broken out.
         Watch it. Set a price alert at the pivot high. Enter on the breakout day.

  COLUMNS
  ───────
  MKT     Market: US · UK · DE · FR · CA · etc.
  TICKER  yfinance symbol
  PRICE   Current closing price (local currency)
  →PIV    % distance from price to 10-day pivot high (+0% = AT the pivot)
  RSI     14-day RSI (38–62 ideal for coil; higher ok on breakout)
  ADX     Trend strength — low (10-22) during coil, rising on break
  ATR%    Recent 5d ATR ÷ prior 15d ATR  (lower = tighter range = better coil)
  V5d     Avg volume last 5 days ÷ 20d avg  (lower = drier = better coil setup)
  M       Minervini Trend Template /8 (≥5 required)
  SCR     Total score  =  coil signals + break signals × 2
  #BT     Historical backtest entries (same signal rules applied to history)
  WIN%    % profitable after 5 trading days
  AVG/MED Average and median 5-day return from backtest

  COIL SIGNALS  (building energy)
  ────────────────────────────────
  ATR<     Recent ATR < 72% of prior ATR — range contracting
  VOLdry   Avg vol last 5d < 65% of 20d avg — volume drying up
  NearPiv  Price within 4% of 10-day high — coiling near resistance
  ADXcurl  ADX between 10-25 and rising — trend beginning to form
  RSIbase  RSI between 38-62 — momentum neutral, not extended
  RS~ok    Stock 63d return within 5% of its local benchmark
  RngTght  Today's High-Low range < 70% of 10d avg range

  BREAKOUT SIGNALS  (shown after ▶, weighted ×2 in score)
  ─────────────────────────────────────────────────────────
  VolSurge  Today's volume > 1.8× 20d avg — institutional participation
  PriceBrk  Price closed above prior 10-day high — resistance broken
  ADXaccel  ADX > 115% of its value 3 bars ago — trend accelerating
  EMAxover  9 EMA crossed above 21 EMA in last 2 bars

  HARD FILTERS
  ────────────
  Minervini ≥ 5/8   Stock in uptrend base (relaxed vs momentum_scanner's 6)
  Price > 50 SMA    Still above key trend line (embedded in Minervini score)
  Min coil score 4  Need majority of coil signals
  Min break score 7 Need coil foundation + at least 2 break signals

  HOW TO USE
  ──────────
  1. Each morning before market open, run with --no-backtest for speed.
  2. BREAK stocks: check the chart. If the candle looks clean, enter at open.
  3. COIL stocks: set a price alert at 'Pivot High' shown in output.
     When price breaks above with volume, that's your entry.
  4. Stop loss: 5-7% below entry (or below the base low for tighter risk).
  5. Target: hold 5 days then reassess (use momentum_scanner.py to track).
""")
    print(BOLD("━"*W)); print()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args          = set(sys.argv[1:])
    fast_mode     = "--fast"        in args
    with_backtest = "--no-backtest" not in args
    show_legend   = "--legend"      in args

    if show_legend:
        print_legend(); return

    universe = get_universe(fast_mode)

    bench_syms = set(universe.values())
    print(DIM(f"  Fetching {len(bench_syms)} benchmark indices..."), flush=True)
    bench_returns = fetch_benchmark_returns(bench_syms)
    fetched = [f"{k.replace('^','')}:{bench_returns[k]:+.1f}%"
               for k in sorted(bench_returns)]
    print(DIM("  " + "  ".join(fetched)))

    bt_label = "backtest ON" if with_backtest else "backtest OFF"
    print(DIM(f"  Scanning {len(universe)} tickers  ·  {bt_label}  ·  hold={HOLD_DAYS}d"))
    print(DIM(f"  Looking for: ATR contraction + volume dry-up + coil near pivot"))
    print()

    t0, results, done = time.time(), [], 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=300):
            done += 1
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r: results.append(r)
            if done % 150 == 0:
                print(DIM(f"  {done}/{len(universe)}  hits={len(results)}"
                          f"  elapsed={time.time()-t0:.0f}s"), flush=True)

    results.sort(key=lambda x: (0 if x["phase"]=="BREAK" else 1, -(x.get("wr") or 0), -x["score"]))
    print(DIM(f"  Done in {time.time()-t0:.0f}s — "
              f"{sum(1 for r in results if r['phase']=='BREAK')} breaking  "
              f"{sum(1 for r in results if r['phase']=='COIL')} coiling\n"))
    print_results(results, with_backtest, bench_returns)


if __name__ == "__main__":
    main()


# ── PUBLIC API ────────────────────────────────────────────────────────────────
build_universe        = get_universe
compute_bench_returns = fetch_benchmark_returns


def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """Run the scanning loop silently; return list of result dicts with strategy tag."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(bench), with_backtest): t
            for t, bench in universe.items()
        }
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                r["strategy"] = "breakout"
                results.append(r)
    results.sort(key=lambda x: (0 if x["phase"] == "BREAK" else 1,
                                 -(x.get("wr") or 0), -x["score"]))
    return results
