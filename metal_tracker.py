#!/usr/bin/env python3
"""
Metal Supply Tracker — static HTML generator
Generates metal_tracker.html for GitHub Pages (no server required).
Run locally:   python3 metal_tracker.py  →  open metal_tracker.html
GitHub Actions: same command, HTML committed to repo and served via Pages.
"""

import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import html as _html
import sys

# ── Metal futures (spot prices) ───────────────────────────────────────────────
METALS = [
    {"name": "Gold",        "ticker": "GC=F",  "unit": "$/troy oz",   "symbol": "Au",  "proxy": False},
    {"name": "Silver",      "ticker": "SI=F",  "unit": "$/troy oz",   "symbol": "Ag",  "proxy": False},
    {"name": "Platinum",    "ticker": "PL=F",  "unit": "$/troy oz",   "symbol": "Pt",  "proxy": False},
    {"name": "Palladium",   "ticker": "PA=F",  "unit": "$/troy oz",   "symbol": "Pd",  "proxy": False},
    {"name": "Copper",      "ticker": "HG=F",  "unit": "$/lb",        "symbol": "Cu",  "proxy": False},
    {"name": "Aluminum",    "ticker": "ALI=F", "unit": "$/lb",        "symbol": "Al",  "proxy": False},
    {"name": "Steel (HRC)", "ticker": "HRC=F", "unit": "$/ton",       "symbol": "Fe",  "proxy": False},
    {"name": "Crude Oil",   "ticker": "CL=F",  "unit": "$/barrel",    "symbol": "Oil", "proxy": False},
    # No exchange-traded futures for rare earths/lithium — ETF proxies
    {"name": "Rare Earths", "ticker": "REMX",  "unit": "$ (REMX ETF)","symbol": "RE",  "proxy": True},
    {"name": "Lithium",     "ticker": "LIT",   "unit": "$ (LIT ETF)", "symbol": "Li",  "proxy": True},
    {"name": "Cobalt",      "ticker": "VALE",  "unit": "$ (Vale ADR)","symbol": "Co",  "proxy": True},
]

# ── Global metal company stocks ───────────────────────────────────────────────
METAL_STOCKS = {
    "Aluminum": [
        {"name": "Hindalco Industries",  "ticker": "HINDALCO.NS", "country": "India",        "exchange": "NSE"},
        {"name": "Vedanta Ltd",          "ticker": "VEDL.NS",     "country": "India",        "exchange": "NSE"},
        {"name": "NALCO",                "ticker": "NATIONALUM.NS","country": "India",       "exchange": "NSE"},
        {"name": "Alcoa Corp",           "ticker": "AA",          "country": "USA",          "exchange": "NYSE"},
        {"name": "Century Aluminum",     "ticker": "CENX",        "country": "USA",          "exchange": "NASDAQ"},
        {"name": "Norsk Hydro",          "ticker": "NHYDY",       "country": "Norway",       "exchange": "OTC"},
        {"name": "Rio Tinto",            "ticker": "RIO",         "country": "Australia/UK", "exchange": "NYSE"},
    ],
    "Copper": [
        {"name": "Freeport-McMoRan",     "ticker": "FCX",         "country": "USA",          "exchange": "NYSE"},
        {"name": "Southern Copper",      "ticker": "SCCO",        "country": "Peru/USA",     "exchange": "NYSE"},
        {"name": "Antofagasta",          "ticker": "ANTO.L",      "country": "Chile",        "exchange": "LSE"},
        {"name": "First Quantum Min.",   "ticker": "FM.TO",       "country": "Canada/Zambia","exchange": "TSX"},
        {"name": "Ivanhoe Mines",        "ticker": "IVN.TO",      "country": "Canada/DRC",   "exchange": "TSX"},
        {"name": "Hindalco (Cu)",        "ticker": "HINDALCO.NS", "country": "India",        "exchange": "NSE"},
    ],
    "Gold": [
        {"name": "Newmont Corp",         "ticker": "NEM",         "country": "USA",          "exchange": "NYSE"},
        {"name": "Barrick Gold",         "ticker": "GOLD",        "country": "Canada",       "exchange": "NYSE"},
        {"name": "Agnico Eagle Mines",   "ticker": "AEM",         "country": "Canada",       "exchange": "NYSE"},
        {"name": "Kinross Gold",         "ticker": "KGC",         "country": "Canada",       "exchange": "NYSE"},
        {"name": "Gold Fields",          "ticker": "GFI",         "country": "South Africa", "exchange": "NYSE"},
        {"name": "AngloGold Ashanti",    "ticker": "AU",          "country": "South Africa", "exchange": "NYSE"},
        {"name": "Zijin Mining",         "ticker": "ZIJMY",       "country": "China",        "exchange": "OTC"},
    ],
    "Silver": [
        {"name": "Wheaton Prec. Metals", "ticker": "WPM",         "country": "Canada",       "exchange": "NYSE"},
        {"name": "Pan American Silver",  "ticker": "PAAS",        "country": "Canada",       "exchange": "NASDAQ"},
        {"name": "First Majestic Silver","ticker": "AG",          "country": "Canada/Mexico","exchange": "NYSE"},
        {"name": "Coeur Mining",         "ticker": "CDE",         "country": "USA",          "exchange": "NYSE"},
    ],
    "Steel": [
        {"name": "Tata Steel",           "ticker": "TATASTEEL.NS","country": "India",        "exchange": "NSE"},
        {"name": "JSW Steel",            "ticker": "JSWSTEEL.NS", "country": "India",        "exchange": "NSE"},
        {"name": "SAIL",                 "ticker": "SAIL.NS",     "country": "India",        "exchange": "NSE"},
        {"name": "ArcelorMittal",        "ticker": "MT",          "country": "Luxembourg",   "exchange": "NYSE"},
        {"name": "Nucor Corporation",    "ticker": "NUE",         "country": "USA",          "exchange": "NYSE"},
        {"name": "Posco Holdings",       "ticker": "PKX",         "country": "South Korea",  "exchange": "NYSE"},
        {"name": "Steel Dynamics",       "ticker": "STLD",        "country": "USA",          "exchange": "NASDAQ"},
    ],
    "Nickel": [
        {"name": "Vale S.A.",            "ticker": "VALE",        "country": "Brazil",       "exchange": "NYSE"},
        {"name": "IGO Limited",          "ticker": "IGO.AX",      "country": "Australia",    "exchange": "ASX"},
    ],
    "Zinc": [
        {"name": "Hindustan Zinc",       "ticker": "HINDZINC.NS", "country": "India",        "exchange": "NSE"},
        {"name": "Teck Resources",       "ticker": "TECK",        "country": "Canada",       "exchange": "NYSE"},
    ],
    "Platinum/Palladium": [
        {"name": "Sibanye Stillwater",   "ticker": "SBSW",        "country": "South Africa", "exchange": "NYSE"},
        {"name": "Impala Platinum",      "ticker": "IMPUY",       "country": "South Africa", "exchange": "OTC"},
        {"name": "Anglo Amer. Platinum", "ticker": "ANGPY",       "country": "South Africa", "exchange": "OTC"},
    ],
    "Rare Earths": [
        {"name": "MP Materials",         "ticker": "MP",          "country": "USA",          "exchange": "NYSE"},
        {"name": "Energy Fuels",         "ticker": "UUUU",        "country": "USA",          "exchange": "NYSE"},
        {"name": "Lynas Rare Earths",    "ticker": "LYSCF",       "country": "Australia",    "exchange": "OTC"},
        {"name": "Iluka Resources",      "ticker": "ILU.AX",      "country": "Australia",    "exchange": "ASX"},
    ],
    "Lithium": [
        {"name": "Albemarle Corp",       "ticker": "ALB",         "country": "USA",          "exchange": "NYSE"},
        {"name": "SQM",                  "ticker": "SQM",         "country": "Chile",        "exchange": "NYSE"},
        {"name": "Arcadium Lithium",     "ticker": "ALTM",        "country": "USA/Australia","exchange": "NYSE"},
        {"name": "Pilbara Minerals",     "ticker": "PILBF",       "country": "Australia",    "exchange": "OTC"},
        {"name": "Sigma Lithium",        "ticker": "SGML",        "country": "Brazil",       "exchange": "NASDAQ"},
        {"name": "Lithium Americas",     "ticker": "LAC",         "country": "Canada",       "exchange": "NYSE"},
    ],
    "Cobalt": [
        {"name": "Glencore",             "ticker": "GLNCY",       "country": "Switzerland",  "exchange": "OTC"},
        {"name": "Vale S.A.",            "ticker": "VALE",        "country": "Brazil",       "exchange": "NYSE"},
    ],
    "Diversified Miners": [
        {"name": "BHP Group",            "ticker": "BHP",         "country": "Australia",    "exchange": "NYSE"},
        {"name": "Anglo American",       "ticker": "NGLOY",       "country": "UK",           "exchange": "OTC"},
        {"name": "Vedanta Ltd",          "ticker": "VEDL.NS",     "country": "India",        "exchange": "NSE"},
    ],
}

EVENT_TYPES = {
    "Plant/Smelter Closure": {
        "keywords": ["closure","closed","shut down","shutdown","idle","idled","halt","halted",
                     "curtailment","curtailed","suspended","mothballed","offline","production stop"],
        "pressure": "UP", "color": "red",
    },
    "Labor Strike": {
        "keywords": ["strike","labor dispute","labour dispute","worker dispute","union",
                     "walkout","protest","industrial action","work stoppage"],
        "pressure": "UP", "color": "orange",
    },
    "Geopolitical": {
        "keywords": ["sanctions","sanctioned","tariff","trade war","trade ban","embargo",
                     "conflict","war","export ban","export restriction","import ban",
                     "nationalization","nationalized","seized","coup"],
        "pressure": "?", "color": "purple",
    },
    "Natural Disaster": {
        "keywords": ["earthquake","flood","flooded","flooding","hurricane","typhoon",
                     "landslide","storm","wildfire","fire destroyed","tsunami","volcanic","drought"],
        "pressure": "UP", "color": "teal",
    },
    "Energy Crisis": {
        "keywords": ["power shortage","electricity shortage","energy crisis","blackout",
                     "power cut","power outage","power disruption","gas shortage",
                     "energy shortage","load shedding"],
        "pressure": "UP", "color": "yellow",
    },
    "Supply Deficit": {
        "keywords": ["shortage","deficit","undersupply","supply gap","tight supply",
                     "low inventory","inventory drop","stockpile decline","supply crunch"],
        "pressure": "UP", "color": "pink",
    },
    "Demand Surge": {
        "keywords": ["demand surge","demand increase","record demand","electric vehicle demand",
                     "EV demand","battery demand","data center demand","infrastructure boom"],
        "pressure": "UP", "color": "green",
    },
    "New Supply": {
        "keywords": ["new mine","mine opening","capacity expansion","ramp up",
                     "production increase","restart","reopened","resumes production"],
        "pressure": "DOWN", "color": "lime",
    },
    "Regulatory": {
        "keywords": ["permit revoked","permit denied","environmental ruling","EPA",
                     "environmental ban","compliance failure","license suspended",
                     "ordered to stop","court order"],
        "pressure": "UP", "color": "gray",
    },
}

METAL_KEYWORDS = {
    "Gold":        ["gold","bullion"],
    "Silver":      ["silver"],
    "Copper":      ["copper"],
    "Aluminum":    ["aluminum","aluminium","bauxite","qatalum","alcoa"],
    "Nickel":      ["nickel"],
    "Cobalt":      ["cobalt"],
    "Lithium":     ["lithium"],
    "Zinc":        ["zinc"],
    "Lead":        ["lead"],
    "Tin":         ["tin"],
    "Platinum":    ["platinum","pgm","pge"],
    "Palladium":   ["palladium"],
    "Steel":       ["steel","iron ore","hrc","blast furnace"],
    "Crude Oil":   ["crude","crude oil","petroleum"],
    "Rare Earths": ["rare earth","rare-earth","ree","neodymium","dysprosium","lanthanum",
                    "cerium","praseodymium","terbium","erbium","yttrium","scandium",
                    "europium","gadolinium","holmium","samarium"],
    "Manganese":   ["manganese"],
}

COUNTRY_LIST = [
    "China","India","USA","United States","Australia","Russia","Chile","Peru","Brazil",
    "Indonesia","Philippines","Congo","DR Congo","Zambia","Zimbabwe","South Africa",
    "Canada","Norway","Qatar","Bahrain","Guinea","Ghana","Kazakhstan","Myanmar","Mexico",
    "Bolivia","Japan","South Korea","Germany","France","UK","Britain","Iran",
    "Saudi Arabia","UAE","Argentina","Colombia","Papua New Guinea","Mozambique",
    "Tanzania","Mali","Sweden","Finland","Mongolia",
]

NEWS_SOURCES = [
    {"url": "https://www.mining.com/feed/",          "source": "Mining.com"},
    {"url": "https://www.kitco.com/rss/",            "source": "Kitco"},
    {"url": "https://www.miningweekly.com/rss/",     "source": "Mining Weekly"},
    {"url": "https://northernminer.com/feed/",       "source": "Northern Miner"},
    {"url": "https://news.google.com/rss/search?q=smelter+closure+production+halt+shutdown&hl=en-US&gl=US&ceid=US:en",              "source": None},
    {"url": "https://news.google.com/rss/search?q=mine+strike+labor+dispute+copper+zinc+nickel&hl=en-US&gl=US&ceid=US:en",        "source": None},
    {"url": "https://news.google.com/rss/search?q=metal+production+cut+output+reduction+curtailment&hl=en-US&gl=US&ceid=US:en",  "source": None},
    {"url": "https://news.google.com/rss/search?q=mining+flood+earthquake+disaster+production&hl=en-US&gl=US&ceid=US:en",        "source": None},
    {"url": "https://news.google.com/rss/search?q=energy+crisis+smelter+power+shortage+aluminum&hl=en-US&gl=US&ceid=US:en",     "source": None},
    {"url": "https://news.google.com/rss/search?q=metal+commodity+sanctions+tariff+export+ban&hl=en-US&gl=US&ceid=US:en",       "source": None},
    {"url": "https://news.google.com/rss/search?q=copper+aluminum+nickel+supply+disruption+shortage&hl=en-US&gl=US&ceid=US:en", "source": None},
    {"url": "https://news.google.com/rss/search?q=gold+silver+mine+closure+suspended+output&hl=en-US&gl=US&ceid=US:en",         "source": None},
    {"url": "https://news.google.com/rss/search?q=platinum+palladium+supply+deficit+mine&hl=en-US&gl=US&ceid=US:en",            "source": None},
    {"url": "https://news.google.com/rss/search?q=steel+iron+ore+blast+furnace+supply+shortage&hl=en-US&gl=US&ceid=US:en",      "source": None},
    {"url": "https://news.google.com/rss/search?q=copper+mine+chile+peru+zambia+congo+disruption&hl=en-US&gl=US&ceid=US:en",    "source": None},
    {"url": "https://news.google.com/rss/search?q=rare+earth+neodymium+china+supply+ban+export&hl=en-US&gl=US&ceid=US:en",      "source": None},
    {"url": "https://news.google.com/rss/search?q=rare+earth+mine+closure+production+lynas+materials&hl=en-US&gl=US&ceid=US:en","source": None},
    {"url": "https://news.google.com/rss/search?q=lithium+cobalt+supply+shortage+mine+disruption&hl=en-US&gl=US&ceid=US:en",    "source": None},
    {"url": "https://news.google.com/rss/search?q=lithium+price+supply+deficit+australia+chile&hl=en-US&gl=US&ceid=US:en",      "source": None},
    {"url": "https://news.google.com/rss/search?q=cobalt+congo+DRC+mine+supply+disruption&hl=en-US&gl=US&ceid=US:en",           "source": None},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_event(title):
    t = title.lower()
    best, best_score = "General", 0
    for etype, meta in EVENT_TYPES.items():
        score = sum(1 for kw in meta["keywords"] if kw in t)
        if score > best_score:
            best_score, best = score, etype
    return best

def tag_metals(title):
    t = title.lower()
    tags = [m for m, kws in METAL_KEYWORDS.items() if any(kw in t for kw in kws)]
    return tags if tags else ["Metals"]

def extract_cause(title):
    for pat in [r"due to ([^,.\-|]{6,65})", r"following ([^,.\-|]{6,65})",
                r"amid ([^,.\-|]{6,65})", r"after ([^,.\-|]{6,65})",
                r"as ([a-z][^,.\-|]{5,60})", r"because of ([^,.\-|]{6,65})"]:
        m = re.search(pat, title, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip().rstrip(".")
            if len(phrase) > 5:
                return phrase[:70]
    return None

def extract_country(title):
    t = title.lower()
    for c in COUNTRY_LIST:
        if c.lower() in t:
            return c
    return None

def clean_title(raw):
    parts = raw.rsplit(" - ", 1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (raw.strip(), "")


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_prices():
    results = []
    for m in METALS:
        try:
            hist = yf.Ticker(m["ticker"]).history(period="3mo")
            if hist.empty or len(hist) < 5:
                continue
            c    = hist["Close"]
            spot = float(c.iloc[-1])
            p1m  = float(c.iloc[max(0, len(c) - 22)])
            p2m  = float(c.iloc[max(0, len(c) - 43)])
            results.append({**m,
                "spot":  round(spot, 2),
                "ch1m":  round((spot - p1m) / p1m * 100, 1),
                "ch2m":  round((spot - p2m) / p2m * 100, 1),
            })
        except Exception as e:
            print(f"  [price] {m['name']}: {e}", file=sys.stderr)
    return results

def _fetch_one_stock(co):
    try:
        hist = yf.Ticker(co["ticker"]).history(period="3mo")
        if hist.empty or len(hist) < 5:
            return None
        c    = hist["Close"]
        spot = float(c.iloc[-1])
        n    = len(c)
        def pct(days):
            p = float(c.iloc[max(0, n - days - 1)])
            return round((spot - p) / p * 100, 1) if p else None
        return {**co, "price": round(spot, 2),
                "ch10d": pct(10), "ch20d": pct(20), "ch1m": pct(21), "ch2m": pct(42)}
    except Exception as e:
        print(f"  [stock] {co['name']}: {e}", file=sys.stderr)
        return None

def fetch_stock_data():
    all_tasks = [(metal, co) for metal, cos in METAL_STOCKS.items() for co in cos]
    flat = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_one_stock, co): (metal, co) for metal, co in all_tasks}
        for fut in as_completed(futures):
            metal, co = futures[fut]
            r = fut.result()
            if r:
                flat.setdefault(metal, []).append(r)
    ordered = {}
    for metal in METAL_STOCKS:
        if metal in flat:
            orig = {co["ticker"]: i for i, co in enumerate(METAL_STOCKS[metal])}
            ordered[metal] = sorted(flat[metal], key=lambda x: orig.get(x["ticker"], 99))
    return ordered

def fetch_events():
    raw, seen = [], set()
    for src in NEWS_SOURCES:
        try:
            feed = feedparser.parse(src["url"])
            for entry in feed.entries[:6]:
                title_raw = entry.get("title", "").strip()
                if not title_raw or title_raw in seen:
                    continue
                seen.add(title_raw)
                title, source = clean_title(title_raw)
                if src["source"]:
                    source = src["source"]
                pub_sort, pub_str = 0, ""
                try:
                    dt = datetime(*entry.published_parsed[:6])
                    pub_str  = dt.strftime("%b %d")
                    pub_sort = dt.timestamp()
                except Exception:
                    pub_str = entry.get("published", "")[:10]
                raw.append({"title": title, "source": source,
                            "link": entry.get("link", "#"),
                            "pub_str": pub_str, "pub_sort": pub_sort})
        except Exception as e:
            print(f"  [rss] {str(e)[:60]}", file=sys.stderr)

    # Mining.com homepage scrape
    try:
        r = requests.get("https://www.mining.com/",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.select("h2 a, h3 a, h4 a")[:20]:
            text = tag.get_text(strip=True)
            href = tag.get("href", "")
            if len(text) > 30 and href and text not in seen:
                seen.add(text)
                if not href.startswith("http"):
                    href = "https://www.mining.com" + href
                raw.append({"title": text, "source": "Mining.com",
                            "link": href, "pub_str": "", "pub_sort": 0})
    except Exception as e:
        print(f"  [scrape] {e}", file=sys.stderr)

    events = []
    for item in raw:
        etype = classify_event(item["title"])
        meta  = EVENT_TYPES.get(etype, {})
        events.append({**item,
            "etype":    etype,
            "color":    meta.get("color",    "gray"),
            "pressure": meta.get("pressure", "?"),
            "metals":   tag_metals(item["title"]),
            "cause":    extract_cause(item["title"]),
            "country":  extract_country(item["title"]),
        })
    events.sort(key=lambda x: (x["pub_sort"], x["etype"] != "General"), reverse=True)
    return events[:70]

def metal_briefs(prices, events):
    briefs = {}
    for m in prices:
        name  = m["name"]
        match = "Steel" if name == "Steel (HRC)" else name
        rel   = [e for e in events if match in e["metals"] or name in e["metals"]]
        if not rel:
            continue
        up   = sum(1 for e in rel if e["pressure"] == "UP")
        down = sum(1 for e in rel if e["pressure"] == "DOWN")
        sent = "Supply pressure ↑" if up > down else ("Supply easing ↓" if down > up else "Mixed")
        causes = [e["cause"] for e in rel[:4] if e["cause"]]
        briefs[name] = {"sentiment": sent, "up": up, "down": down,
                        "n": len(rel), "causes": causes[:2]}
    return briefs


# ── HTML template ─────────────────────────────────────────────────────────────
# Markers replaced by generate_html():
#   __PRICES_DATA__   __EVENTS_DATA__   __STOCKS_DATA__   __BRIEFS_DATA__
#   __GENERATED__     __UP_COUNT__      __TOTAL__

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Metal Supply Tracker</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;line-height:1.5}
header{background:#161b22;border-bottom:1px solid #30363d;padding:9px 18px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:30}
header h1{font-size:13px;font-weight:700;color:#e6edf3;white-space:nowrap}
header .ts{font-size:11px;color:#6e7681;flex:1}
.hbadge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;white-space:nowrap}
.hbadge.red{background:#3d0e0e;color:#f85149}.hbadge.dim{background:#21262d;color:#8b949e}
.filter-strip{background:#0d1117;border-bottom:1px solid #21262d;padding:8px 18px;position:sticky;top:37px;z-index:29;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.filter-strip .label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#484f58;white-space:nowrap;margin-right:4px}
.mf{font-size:11px;font-family:inherit;background:none;border:1px solid #21262d;color:#6e7681;padding:3px 10px;border-radius:20px;cursor:pointer;white-space:nowrap;transition:all .12s}
.mf:hover{border-color:#58a6ff;color:#c9d1d9}.mf.active{background:#1f3d6e;border-color:#388bfd;color:#e6edf3;font-weight:700}
.page{padding:14px 18px 50px;max-width:1500px;margin:0 auto}
.sec-block{margin-bottom:22px}
.sec-head{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:6px 6px 0 0;cursor:pointer;user-select:none;border:1px solid transparent}
.sec-head:hover{filter:brightness(1.1)}
.sec-head .tog{font-size:10px;min-width:10px}.sec-head .stitle{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em}
.sec-head .snote{font-size:11px;font-weight:400;text-transform:none;letter-spacing:0;opacity:.65}.sec-head .scnt{margin-left:auto;font-size:11px;opacity:.6}
.sec-prices{background:#0d1f35;border-color:#1e3d6e;color:#58a6ff}
.sec-events{background:#2d0f0f;border-color:#5c1a1a;color:#f85149}
.sec-stocks{background:#0d2e17;border-color:#1a5c30;color:#3fb950}
.sec-body{border:1px solid #21262d;border-top:none;border-radius:0 0 6px 6px;overflow:hidden;overflow-x:auto}
.sec-body.collapsed{display:none}
table{width:100%;border-collapse:collapse}
thead th{background:#161b22;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#6e7681;padding:7px 11px;border-bottom:1px solid #21262d;white-space:nowrap;text-align:left}
thead th.r{text-align:right}thead th.c{text-align:center}
tbody tr{border-bottom:1px solid #161b22}tbody tr:last-child{border-bottom:none}
tbody tr:hover td{background:#161b22}tbody td{padding:7px 11px;vertical-align:top}
tbody td.r{text-align:right;vertical-align:middle}tbody td.c{text-align:center;vertical-align:middle}
.chg{font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}
.pos{color:#3fb950}.neg{color:#f85149}.neu{color:#6e7681}
.m-name{font-weight:600;color:#e6edf3}.m-sym{color:#6e7681;font-size:11px}
.spot{font-weight:600;color:#e6edf3;font-variant-numeric:tabular-nums}.unit{font-size:11px;color:#6e7681;display:block}
.proxy-badge{font-size:10px;background:#1a2e1a;color:#3fb950;padding:1px 5px;border-radius:3px;margin-left:4px}
.b-sent{font-size:11px;font-weight:700}.b-sent.up{color:#f85149}.b-sent.down{color:#3fb950}.b-sent.mix{color:#6e7681}
.b-causes{font-size:11px;color:#58a6ff;margin-top:2px}.b-causes span{display:block}.b-causes span::before{content:"→ ";color:#30363d}
.ev-tabs{display:flex;gap:5px;flex-wrap:wrap;padding:8px 11px;background:#0d1117;border-bottom:1px solid #21262d}
.etab{font-size:11px;font-family:inherit;background:none;border:1px solid #21262d;color:#6e7681;padding:2px 9px;border-radius:4px;cursor:pointer;white-space:nowrap}
.etab:hover{border-color:#58a6ff;color:#c9d1d9}.etab.active{border-color:#5c1a1a;background:#2d0f0f;color:#f85149}
.ev-type{font-size:12px;font-weight:700;white-space:nowrap}.ev-ctry{font-size:12px;color:#8b949e}
.ev-metal{font-size:12px;color:#8b949e}.ev-date{font-size:12px;color:#6e7681;white-space:nowrap}
.ev-title a{color:#c9d1d9;text-decoration:none}.ev-title a:hover{color:#58a6ff}
.ev-cause{font-size:11px;color:#58a6ff;margin-top:3px}.ev-cause::before{content:"↳ ";color:#30363d}
.ev-src{font-size:11px;color:#484f58;margin-top:2px}
.eff{font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;white-space:nowrap}
.eff.up{background:#3d0e0e;color:#f85149}.eff.down{background:#0e2e1a;color:#3fb950}.eff.q{background:#1c1c2e;color:#a371f7}
.c-red{color:#f85149}.c-orange{color:#d29922}.c-purple{color:#a371f7}.c-teal{color:#39c5cf}
.c-yellow{color:#e3b341}.c-pink{color:#ec4899}.c-green{color:#3fb950}.c-lime{color:#7ee787}.c-gray{color:#6e7681}
.metal-group td{background:#161b22!important;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;padding:6px 11px;border-top:2px solid #21262d}
.co-name{font-weight:600;color:#e6edf3}.co-tick{font-size:10px;color:#484f58;display:block}
.co-ctry{font-size:12px;color:#8b949e}.co-exch{font-size:11px;color:#484f58}
.hidden{display:none!important}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#30363d;border-radius:2px}
</style>
</head>
<body>

<header>
  <h1>&#x2B21; Metal Supply Tracker</h1>
  <span class="ts">Generated: <strong id="genTime"></strong></span>
  <span class="hbadge red" id="upBadge"></span>
  <span class="hbadge dim" id="totalBadge"></span>
  <a href="market_report.html" style="font-size:11px;color:#58a6ff;text-decoration:none;border:1px solid #1e3d6e;padding:3px 10px;border-radius:4px;">&#8592; Market Report</a>
</header>

<div class="filter-strip">
  <span class="label">Filter:</span>
  <button class="mf active" data-metal="ALL"        onclick="setMetal(this)">All</button>
  <button class="mf" data-metal="Gold"              onclick="setMetal(this)">Gold</button>
  <button class="mf" data-metal="Silver"            onclick="setMetal(this)">Silver</button>
  <button class="mf" data-metal="Copper"            onclick="setMetal(this)">Copper</button>
  <button class="mf" data-metal="Aluminum"          onclick="setMetal(this)">Aluminum</button>
  <button class="mf" data-metal="Steel"             onclick="setMetal(this)">Steel</button>
  <button class="mf" data-metal="Nickel"            onclick="setMetal(this)">Nickel</button>
  <button class="mf" data-metal="Zinc"              onclick="setMetal(this)">Zinc</button>
  <button class="mf" data-metal="Platinum"          onclick="setMetal(this)">Platinum</button>
  <button class="mf" data-metal="Palladium"         onclick="setMetal(this)">Palladium</button>
  <button class="mf" data-metal="Rare Earths"       onclick="setMetal(this)">Rare Earths</button>
  <button class="mf" data-metal="Lithium"           onclick="setMetal(this)">Lithium</button>
  <button class="mf" data-metal="Cobalt"            onclick="setMetal(this)">Cobalt</button>
  <button class="mf" data-metal="Crude Oil"         onclick="setMetal(this)">Crude Oil</button>
</div>

<div class="page">

  <!-- PRICES -->
  <div class="sec-block">
    <div class="sec-head sec-prices" onclick="toggleSec('prices')">
      <span class="tog" id="tog-prices">&#9660;</span>
      <span class="stitle">Prices</span>
      <span class="snote">spot &middot; 1-month &middot; 2-month &middot; supply intelligence</span>
      <span class="scnt" id="priceCount"></span>
    </div>
    <div class="sec-body" id="body-prices">
      <table><thead><tr>
        <th>Metal</th><th class="r">Spot</th><th class="r">1-Month &Delta;</th>
        <th class="r">2-Month &Delta;</th><th>Supply Intelligence</th>
      </tr></thead>
      <tbody id="priceBody"></tbody></table>
    </div>
  </div>

  <!-- EVENTS -->
  <div class="sec-block">
    <div class="sec-head sec-events" onclick="toggleSec('events')">
      <span class="tog" id="tog-events">&#9660;</span>
      <span class="stitle">Supply Shock Events</span>
      <span class="snote">Mining.com &middot; Kitco &middot; Mining Weekly &middot; 20 sources</span>
      <span class="scnt" id="evCount"></span>
    </div>
    <div class="sec-body" id="body-events">
      <div class="ev-tabs" id="evTabs"></div>
      <table style="table-layout:fixed;">
        <colgroup>
          <col style="width:66px"><col style="width:170px"><col style="width:105px">
          <col style="width:105px"><col style="width:auto"><col style="width:185px"><col style="width:85px">
        </colgroup>
        <thead><tr>
          <th>Date</th><th>Event Type</th><th>Metal(s)</th><th>Country</th>
          <th>Headline</th><th>Root Cause</th><th class="c">Effect</th>
        </tr></thead>
        <tbody id="evBody"></tbody>
      </table>
    </div>
  </div>

  <!-- STOCKS -->
  <div class="sec-block">
    <div class="sec-head sec-stocks" onclick="toggleSec('stocks')">
      <span class="tog" id="tog-stocks">&#9660;</span>
      <span class="stitle">Global Metal Company Stocks</span>
      <span class="snote">major producers by metal &middot; 10-day &middot; 20-day &middot; 1M &middot; 2M</span>
      <span class="scnt" id="stkCount"></span>
    </div>
    <div class="sec-body" id="body-stocks">
      <table><thead><tr>
        <th>Company</th><th>Country</th><th>Exchange</th>
        <th class="r">10-Day &Delta;</th><th class="r">20-Day &Delta;</th>
        <th class="r">1-Month &Delta;</th><th class="r">2-Month &Delta;</th>
      </tr></thead>
      <tbody id="stkBody"></tbody></table>
    </div>
  </div>

</div>

<script>
// ── Injected data ──────────────────────────────────────────────────────────
const PRICES = __PRICES_DATA__;
const EVENTS = __EVENTS_DATA__;
const STOCKS = __STOCKS_DATA__;
const BRIEFS = __BRIEFS_DATA__;
const GENERATED  = "__GENERATED__";
const UP_COUNT   = __UP_COUNT__;
const TOTAL      = __TOTAL__;

// ── State ──────────────────────────────────────────────────────────────────
let activeMetal = 'ALL';
let activeEtype = 'ALL';

// ── Helpers ────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function chgSpan(v) {
  if (v===null||v===undefined) return '<span style="color:#484f58">&#8212;</span>';
  const cls = v>0?'pos':v<0?'neg':'neu';
  const sign = v>0?'&#9650; +':v<0?'&#9660; ':'&#8211;';
  return '<span class="chg '+cls+'">'+sign+v+'%</span>';
}
function chgCell(v) { return '<td class="r">'+chgSpan(v)+'</td>'; }
function fmtPrice(n) {
  return n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
}

// ── Render prices ──────────────────────────────────────────────────────────
function renderPrices() {
  const rows = PRICES.map(m => {
    const b = BRIEFS[m.name];
    const proxy = m.proxy ? ' <span class="proxy-badge">ETF proxy</span>' : '';
    let brief = '<span style="color:#484f58">&#8212;</span>';
    if (b) {
      const sc = b.sentiment.includes('pressure')?'up':b.sentiment.includes('easing')?'down':'mix';
      const causes = b.causes.map(c=>'<span>'+esc(c)+'</span>').join('');
      brief = '<span class="b-sent '+sc+'">'+esc(b.sentiment)+'</span>'
            + '<span style="color:#484f58;font-size:11px;"> &middot; '+b.n+' event'+(b.n!==1?'s':'')+'</span>'
            + (causes ? '<div class="b-causes">'+causes+'</div>' : '');
    }
    return '<tr data-metal="'+esc(m.name)+'">'
      + '<td><span class="m-name">'+esc(m.name)+'</span> <span class="m-sym">('+esc(m.symbol)+')</span>'+proxy+'</td>'
      + '<td class="r"><span class="spot">$'+fmtPrice(m.spot)+'</span><span class="unit">'+esc(m.unit)+'</span></td>'
      + '<td class="r">'+chgSpan(m.ch1m)+'</td>'
      + '<td class="r">'+chgSpan(m.ch2m)+'</td>'
      + '<td>'+brief+'</td>'
      + '</tr>';
  });
  rows.push('<tr id="priceEmpty" class="hidden"><td colspan="5" style="text-align:center;padding:1.2rem;color:#484f58;">No price data for this metal (no exchange-traded futures).</td></tr>');
  document.getElementById('priceBody').innerHTML = rows.join('');
  document.getElementById('priceCount').textContent = PRICES.length+' metals';
}

// ── Render events ──────────────────────────────────────────────────────────
function renderEvents() {
  // Build event-type filter tabs
  const typeCounts = {};
  EVENTS.forEach(ev => { typeCounts[ev.etype]=(typeCounts[ev.etype]||0)+1; });
  const sorted = Object.entries(typeCounts).sort((a,b)=>b[1]-a[1]);
  let tabs = '<button class="etab active" data-et="ALL" onclick="filterEtype(this)">All</button>';
  sorted.filter(([t])=>t!=='General').forEach(([t,c]) => {
    tabs += '<button class="etab" data-et="'+esc(t)+'" onclick="filterEtype(this)">'+esc(t)+' ('+c+')</button>';
  });
  sorted.filter(([t])=>t==='General').forEach(([t,c]) => {
    tabs += '<button class="etab" data-et="General" onclick="filterEtype(this)">Other ('+c+')</button>';
  });
  document.getElementById('evTabs').innerHTML = tabs;

  const rows = EVENTS.map(ev => {
    const cause = ev.cause
      ? '<div class="ev-cause">'+esc(ev.cause)+'</div>'
      : '<span style="color:#484f58">&#8212;</span>';
    const src = ev.source ? '<div class="ev-src">'+esc(ev.source)+'</div>' : '';
    const effCls = ev.pressure==='UP'?'up':ev.pressure==='DOWN'?'down':'q';
    const effTxt = ev.pressure==='UP'?'Price &#8593;':ev.pressure==='DOWN'?'Price &#8595;':'Unclear';
    const metalsAttr = JSON.stringify(ev.metals).replace(/"/g,'&quot;');
    return '<tr data-et="'+esc(ev.etype)+'" data-metals=\''+JSON.stringify(ev.metals)+'\'>'
      + '<td><span class="ev-date">'+esc(ev.pub_str)+'</span></td>'
      + '<td><span class="ev-type c-'+esc(ev.color)+'">'+esc(ev.etype)+'</span></td>'
      + '<td><span class="ev-metal">'+esc(ev.metals.join(', '))+'</span></td>'
      + '<td><span class="ev-ctry">'+esc(ev.country||'&#8212;')+'</span></td>'
      + '<td><div class="ev-title"><a href="'+esc(ev.link)+'" target="_blank" rel="noopener">'+esc(ev.title)+'</a></div>'+src+'</td>'
      + '<td>'+cause+'</td>'
      + '<td class="c"><span class="eff '+effCls+'">'+effTxt+'</span></td>'
      + '</tr>';
  });
  rows.push('<tr id="evEmpty" class="hidden"><td colspan="7" style="text-align:center;padding:1.2rem;color:#484f58;">No events match this filter.</td></tr>');
  document.getElementById('evBody').innerHTML = rows.join('');
  document.getElementById('evCount').textContent = EVENTS.length+' events';
}

// ── Render stocks ──────────────────────────────────────────────────────────
function renderStocks() {
  const order = ['Aluminum','Copper','Gold','Silver','Steel','Nickel','Zinc',
                 'Platinum/Palladium','Rare Earths','Lithium','Cobalt','Diversified Miners'];
  let html = '';
  let total = 0;
  order.forEach(metal => {
    const rows = STOCKS[metal];
    if (!rows||!rows.length) return;
    html += '<tr class="metal-group" data-group="'+esc(metal)+'"><td colspan="7">'+esc(metal)+'</td></tr>';
    rows.forEach(co => {
      total++;
      html += '<tr>'
        + '<td><span class="co-name">'+esc(co.name)+'</span><span class="co-tick">'+esc(co.ticker)+'</span></td>'
        + '<td class="co-ctry">'+esc(co.country)+'</td>'
        + '<td class="co-exch">'+esc(co.exchange)+'</td>'
        + chgCell(co.ch10d)+chgCell(co.ch20d)+chgCell(co.ch1m)+chgCell(co.ch2m)
        + '</tr>';
    });
  });
  document.getElementById('stkBody').innerHTML = html;
  document.getElementById('stkCount').textContent = total+' companies';
}

// ── Filters ────────────────────────────────────────────────────────────────
function metalMatch(rowMetal, filter) {
  if (filter==='ALL') return true;
  const r=rowMetal.toLowerCase(), f=filter.toLowerCase();
  return r===f||r.includes(f)||f.includes(r);
}
function eventMetalMatch(metalsJson, filter) {
  if (filter==='ALL') return true;
  try {
    return JSON.parse(metalsJson||'[]').some(m=>metalMatch(m,filter));
  } catch { return true; }
}

function setMetal(btn) {
  document.querySelectorAll('.mf').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  activeMetal = btn.dataset.metal;
  applyFilters();
}
function filterEtype(btn) {
  document.querySelectorAll('.etab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  activeEtype = btn.dataset.et;
  applyFilters();
}
function applyFilters() {
  // Prices
  let pv=0;
  document.querySelectorAll('#priceBody tr[data-metal]').forEach(row=>{
    const show=metalMatch(row.dataset.metal,activeMetal);
    row.classList.toggle('hidden',!show);
    if(show) pv++;
  });
  document.getElementById('priceEmpty').classList.toggle('hidden',pv>0);

  // Events
  let ev=0;
  document.querySelectorAll('#evBody tr[data-et]').forEach(row=>{
    const show=eventMetalMatch(row.dataset.metals,activeMetal)
              &&(activeEtype==='ALL'||row.dataset.et===activeEtype);
    row.classList.toggle('hidden',!show);
    if(show) ev++;
  });
  document.getElementById('evEmpty').classList.toggle('hidden',ev>0);
  document.getElementById('evCount').textContent=ev+' events';

  // Stocks
  let groupMatch=false, sv=0;
  document.querySelectorAll('#stkBody tr').forEach(row=>{
    if(row.classList.contains('metal-group')){
      groupMatch=metalMatch(row.dataset.group||'',activeMetal);
      row.classList.toggle('hidden',!groupMatch);
    } else {
      row.classList.toggle('hidden',!groupMatch);
      if(groupMatch) sv++;
    }
  });
  document.getElementById('stkCount').textContent=sv+' companies';
}

// ── Collapse ───────────────────────────────────────────────────────────────
function toggleSec(id) {
  const body=document.getElementById('body-'+id);
  const tog=document.getElementById('tog-'+id);
  const open=!body.classList.contains('collapsed');
  body.classList.toggle('collapsed',open);
  tog.innerHTML=open?'&#9654;':'&#9660;';
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('genTime').textContent  = GENERATED;
  document.getElementById('upBadge').textContent  = UP_COUNT+' \u2191 pressure';
  document.getElementById('totalBadge').textContent = TOTAL+' events';
  renderPrices();
  renderEvents();
  renderStocks();
});
</script>
</body>
</html>
"""


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_html(prices, events, stocks, briefs, now):
    up_count = sum(1 for e in events if e["pressure"] == "UP")
    html = HTML_TEMPLATE
    html = html.replace("__PRICES_DATA__", json.dumps(prices,  ensure_ascii=False))
    html = html.replace("__EVENTS_DATA__", json.dumps(events,  ensure_ascii=False))
    html = html.replace("__STOCKS_DATA__", json.dumps(stocks,  ensure_ascii=False))
    html = html.replace("__BRIEFS_DATA__", json.dumps(briefs,  ensure_ascii=False))
    html = html.replace("__GENERATED__",   _html.escape(now))
    html = html.replace("__UP_COUNT__",    str(up_count))
    html = html.replace("__TOTAL__",       str(len(events)))
    out = "metal_tracker.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓  {out}  ({len(html)//1024} KB)")


def main():
    print("=== Metal Supply Tracker — generating HTML ===")
    print("Fetching metal prices …")
    prices = fetch_prices()
    print(f"  {len(prices)} metals fetched")

    print("Fetching supply shock events …")
    events = fetch_events()
    print(f"  {len(events)} events fetched")

    print("Fetching company stocks (parallel) …")
    stocks = fetch_stock_data()
    total_cos = sum(len(v) for v in stocks.values())
    print(f"  {total_cos} companies fetched across {len(stocks)} groups")

    briefs = metal_briefs(prices, events)
    now = datetime.utcnow().strftime("%b %d, %Y %H:%M UTC")

    print("Generating HTML …")
    generate_html(prices, events, stocks, briefs, now)
    print("Done.")


if __name__ == "__main__":
    main()
