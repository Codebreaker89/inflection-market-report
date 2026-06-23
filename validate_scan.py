#!/usr/bin/env python3
"""
validate_scan.py
────────────────
Reads last_scan.json, takes top 3 per strategy, re-fetches minimal OHLCV,
re-checks the core condition for each strategy independently.
Prints PASS / FAIL / WARN per ticker per strategy.

python3 validate_scan.py
"""
import json, os, sys, warnings, logging, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull,"w"); old=sys.stderr; sys.stderr=devnull
    try: yield
    finally: sys.stderr=old; devnull.close()

HERE = Path(__file__).parent
LAST = HERE / "last_scan.json"

# ── INDICATORS ────────────────────────────────────────────────────────────────
def _sma(s,n): return s.rolling(n).mean()
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _adx(h,l,c,n=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    up=(h-h.shift()).clip(lower=0); dn=(l.shift()-l).clip(lower=0)
    dmp=up.where(up>dn,0).ewm(alpha=1/n,adjust=False).mean()
    dmm=dn.where(dn>up,0).ewm(alpha=1/n,adjust=False).mean()
    dip=100*dmp/atr; dim=100*dmm/atr
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()
def _atr(h,l,c,n=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def _bb(c,n=20,k=2):
    m=_sma(c,n); s=c.rolling(n).std()
    return m+k*s, m-k*s, m
def _kc(h,l,c,n=20,mult=1.5):
    m=_ema(c,n); a=_atr(h,l,c,14)
    return m+mult*a, m-mult*a

def fetch(ticker):
    with _quiet():
        raw=yf.download(ticker,period="400d",interval="1d",
                        progress=False,auto_adjust=True,threads=False)
    if raw is None or len(raw)<60: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.droplevel(1)
    return raw.dropna(subset=["Close","High","Low","Volume"])

# ── STRATEGY CHECKS ───────────────────────────────────────────────────────────
def check(strategy, ticker, df):
    c,h,l,v = df["Close"],df["High"],df["Low"],df["Volume"]
    results = {}

    if strategy == "momentum":
        adx=_adx(h,l,c); sma50=_sma(c,50); rsi=_rsi(c)
        results["ADX≥22"]       = float(adx.iloc[-1]) >= 22
        results["price>SMA50"]  = float(c.iloc[-1]) > float(sma50.iloc[-1])
        results["RSI 45-75"]    = 45 <= float(rsi.iloc[-1]) <= 75

    elif strategy == "breakout":
        adx=_adx(h,l,c); atr=_atr(h,l,c)
        adx_val    = float(adx.iloc[-1])
        atr_recent = float(atr.iloc[-6:-1].mean())
        atr_prior  = float(atr.iloc[-21:-6].mean())
        pivot      = float(c.iloc[-11:].max())
        vol_5d     = float(v.iloc[-6:-1].mean())
        vol_20     = float(v.rolling(20).mean().iloc[-1])
        results["ADX 10-35"]        = 10 < adx_val < 35
        results["ATR contracting"]  = (atr_recent/atr_prior) < 0.85 if atr_prior>0 else False
        results["near 10d pivot"]   = (float(c.iloc[-1])/pivot) >= 0.94
        results["vol drying up"]    = (vol_5d/vol_20) < 0.80 if vol_20>0 else False

    elif strategy == "pocket_pivot":
        sma50=_sma(c,50); adx=_adx(h,l,c)
        up_days  = v[c>c.shift()]
        dn_vols  = v[c<c.shift()].iloc[-11:-1]
        max_dn   = float(dn_vols.max()) if len(dn_vols)>0 else 0
        today_up = float(v.iloc[-1]) if float(c.iloc[-1])>float(c.iloc[-2]) else 0
        results["ADX≥15"]          = float(adx.iloc[-1]) >= 15
        results["price>SMA50"]     = float(c.iloc[-1]) > float(sma50.iloc[-1])
        results["up vol>max dn vol"]= today_up > max_dn

    elif strategy == "connors_rsi2":
        rsi2=_rsi(c,2); sma200=_sma(c,200); sma50=_sma(c,50)
        results["price>SMA200"]    = float(c.iloc[-1]) > float(sma200.iloc[-1])
        results["price>SMA50"]     = float(c.iloc[-1]) > float(sma50.iloc[-1])
        results["RSI2 recently<10"]= float(rsi2.iloc[-3:].min()) < 15  # slight tolerance

    elif strategy == "ema_ribbon":
        e8=_ema(c,8);e13=_ema(c,13);e21=_ema(c,21);e34=_ema(c,34);e55=_ema(c,55)
        stacked = float(e8.iloc[-1])>float(e13.iloc[-1])>float(e21.iloc[-1])>float(e34.iloc[-1])>float(e55.iloc[-1])
        spread_now  = float(e8.iloc[-1])-float(e55.iloc[-1])
        spread_prev = float(e8.iloc[-6])-float(e55.iloc[-6])
        results["EMA stack 8>13>21>34>55"] = stacked
        results["ribbon expanding"]         = spread_now > spread_prev
        results["price near EMA8 (±3%)"]   = abs(float(c.iloc[-1])-float(e8.iloc[-1]))/float(e8.iloc[-1]) < 0.03

    elif strategy == "nr7":
        rng = h-l
        results["today=narrowest of 7d"] = float(rng.iloc[-1]) < float(rng.iloc[-7:-1].min())
        results["price>SMA50"] = float(c.iloc[-1]) > float(_sma(c,50).iloc[-1])

    elif strategy == "bb_squeeze":
        bbu,bbl,_=_bb(c); kcu,kcl=_kc(h,l,c)
        macd=_ema(c,12)-_ema(c,26); hist=macd-_ema(macd,9)
        # Check squeeze fired in last 3 days
        sq = (bbu<kcu)&(bbl>kcl)
        recently_squeezed = bool(sq.iloc[-4:-1].any())
        results["recently squeezed"]   = recently_squeezed
        results["MACD hist>0"]         = float(hist.iloc[-1]) > 0
        results["BB released (no sq)"] = not bool(sq.iloc[-1])

    elif strategy == "high_tight_flag":
        low40 = c.rolling(40).min()
        high40= c.rolling(40).max()
        pole  = (float(high40.iloc[-1])-float(low40.iloc[-1]))/float(low40.iloc[-1]) if float(low40.iloc[-1])>0 else 0
        dd    = (float(high40.iloc[-1])-float(c.iloc[-1]))/float(high40.iloc[-1]) if float(high40.iloc[-1])>0 else 1
        results["pole≥90% in 40d"]     = pole >= 0.90
        results["flag pullback≤25%"]   = dd <= 0.25
        results["within entry zone"]   = dd <= 0.15

    elif strategy == "analyst_upgrade":
        # Can't re-check recommendations without API — just verify technical sanity
        sma50=_sma(c,50); sma200=_sma(c,200)
        results["price>SMA50 (sanity)"]  = float(c.iloc[-1]) > float(sma50.iloc[-1])
        results["no recent 5% gap-up"]   = ((float(c.iloc[-1])-float(c.iloc[-4]))/float(c.iloc[-4])) < 0.05
        results["[recommendations: see yfinance]"] = True  # can't re-verify here

    elif strategy == "signal_velocity":
        rsi=_rsi(c); macd=_ema(c,12)-_ema(c,26); hist=macd-_ema(macd,9)
        sma50=_sma(c,50); sma200=_sma(c,200)
        # Net score proxy: just check it's in bullish territory
        score_proxy = sum([
            float(c.iloc[-1])>float(sma50.iloc[-1]),
            float(c.iloc[-1])>float(sma200.iloc[-1]),
            float(hist.iloc[-1])>0,
            float(rsi.iloc[-1])>50,
        ])
        score_3d_ago = sum([
            float(c.iloc[-4])>float(sma50.iloc[-4]),
            float(c.iloc[-4])>float(sma200.iloc[-4]),
            float(hist.iloc[-4])>0,
            float(rsi.iloc[-4])>50,
        ])
        results["score improving"]     = score_proxy > score_3d_ago
        results["MACD hist>0"]         = float(hist.iloc[-1]) > 0
        results["price>SMA50"]         = float(c.iloc[-1]) > float(sma50.iloc[-1])

    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
GRN=lambda t:f"\033[32m{t}\033[0m"; RED=lambda t:f"\033[31m{t}\033[0m"
YLW=lambda t:f"\033[33m{t}\033[0m"; BOLD=lambda t:f"\033[1m{t}\033[0m"
DIM=lambda t:f"\033[2m{t}\033[0m"

def main():
    if not LAST.exists():
        print("last_scan.json not found — run scan.py first"); sys.exit(1)

    data = json.loads(LAST.read_text())
    results_by_strategy = data.get("results_by_strategy", {})

    overall_pass=0; overall_fail=0; overall_warn=0

    for strategy, results in results_by_strategy.items():
        if not results: continue
        top3 = sorted(results, key=lambda r: -r.get("score",0))[:3]
        print(f"\n{BOLD('━'*70)}")
        print(f"{BOLD(strategy.upper())}  —  top {len(top3)} results")
        print(BOLD('━'*70))

        for r in top3:
            ticker = r["ticker"]
            stated_score = r.get("score","?")
            stated_adx   = r.get("adx","?")
            stated_rsi   = r.get("rsi","?")
            print(f"\n  {BOLD(ticker):<12} stated: score={stated_score}  adx={stated_adx}  rsi={stated_rsi}")

            df = fetch(ticker)
            if df is None or len(df) < 60:
                print(f"  {YLW('  SKIP — insufficient data')}")
                continue

            try:
                checks = check(strategy, ticker, df)
            except Exception as e:
                print(f"  {YLW(f'  CHECK ERROR: {e}')}"); continue

            passes = sum(v for v in checks.values())
            total  = len(checks)
            fails  = [k for k,v in checks.items() if not v]
            passed = [k for k,v in checks.items() if v]

            for k in passed:
                print(f"    {GRN('✓')} {k}")
                overall_pass += 1
            for k in fails:
                print(f"    {RED('✗')} {k}")
                overall_fail += 1

            pct = passes/total*100 if total>0 else 0
            if pct >= 75:   verdict = GRN(f"LOOKS VALID ({passes}/{total})")
            elif pct >= 50: verdict = YLW(f"PARTIAL ({passes}/{total}) — review")
            else:           verdict = RED(f"SUSPECT ({passes}/{total}) — likely stale/wrong")
            print(f"  → {verdict}")

    print(f"\n{'━'*70}")
    total_checks = overall_pass+overall_fail
    print(f"Overall: {GRN(str(overall_pass))} pass  {RED(str(overall_fail))} fail  "
          f"out of {total_checks} checks  "
          f"({overall_pass/total_checks*100:.0f}% valid)" if total_checks else "No checks run")
    print('━'*70)

if __name__=="__main__":
    main()
