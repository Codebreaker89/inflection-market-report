#!/usr/bin/env python3
"""
One-off data correction for trades.csv rows whose stored `currency` doesn't
match what the ticker actually trades in.

Background: ticker_ccy() in show_tracker.py now correctly defaults to "USD"
for any ticker with no recognized exchange suffix (GWW, CVS, etc. — plain US
tickers). But rows 13 (GWW) and 14 (CVS) were written before that default was
correct, and got stuck with currency="EUR", fx_at_entry=1.0. Since qty was
computed as `qty = invest_eur * fx / price` with fx wrongly pinned to 1.0,
these trades deployed real USD-equivalent exposure of ~$1000 rather than the
intended EUR-equivalent ~€1000 — a modest (single-digit-percent, not 100x)
sizing error that drifts with however far EURUSD was from parity on those
entry dates.

This script:
  1. Finds every OPEN row where trade["currency"] != ticker_ccy(trade["ticker"]).
  2. Fetches the REAL EUR->USD rate on that row's entry_date (needs network —
     run this on a machine with internet access, not in an offline sandbox).
  3. Recomputes qty/investment_eur so the original ~€1000 intent is preserved
     at the correct exchange rate, and corrects currency + fx_at_entry.
  4. Writes trades.csv atomically (backs up the original to trades.csv.bak
     first; never edits in place without a backup).

Run once:
    python3 fix_legacy_currency.py            # dry run — prints what would change
    python3 fix_legacy_currency.py --apply    # writes the correction
"""
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "trades.csv"

sys.path.insert(0, str(HERE))
from show_tracker import ticker_ccy, fetch_fx_on_date  # noqa: E402
from datetime import datetime  # noqa: E402


def main():
    apply = "--apply" in sys.argv

    if not CSV_PATH.exists():
        print(f"No trades.csv found at {CSV_PATH}")
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changes = []
    for row in rows:
        if row.get("status") != "OPEN":
            continue  # don't rewrite closed history
        ticker = row.get("ticker", "")
        stored_ccy = row.get("currency", "")
        real_ccy = ticker_ccy(ticker)
        if stored_ccy == real_ccy:
            continue  # already correct — e.g. DCC.L, whose currency label
                      # was always right; only its numeric price was wrong,
                      # and that's fixed separately in scan.py + trades.csv.

        try:
            entry_date = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()
        except Exception:
            print(f"  [SKIP] id={row.get('id')} {ticker}: unparseable entry_date")
            continue

        old_price   = float(row["buy_price"])
        old_qty     = float(row["qty"])
        old_inv_eur = float(row.get("investment_eur") or 0)
        old_fx      = float(row.get("fx_at_entry") or 1.0)

        print(f"  Fetching real {real_ccy} FX rate for {ticker} on {entry_date}...", end=" ")
        try:
            new_fx = fetch_fx_on_date(real_ccy, entry_date)
        except Exception as e:
            print(f"FAILED ({e}) — leaving this row untouched")
            continue
        print(f"1 EUR = {new_fx:.4f} {real_ccy}")
        # fetch_fx_on_date() never raises — on a network failure it silently
        # falls back to 1.0 rather than surfacing an error (see show_tracker.py:
        # fetch_fx_now/fetch_fx_on_date both `except: return 1.0`). A EURUSD
        # rate of exactly 1.0000 has essentially never happened in reality, so
        # this is a reliable tell that the fetch didn't actually reach the
        # network — don't silently "correct" these rows with a fake rate.
        if abs(new_fx - 1.0) < 1e-9:
            print(f"    WARNING: got exactly 1.0000 — this is almost certainly an offline")
            print(f"    fallback, not a real quote. Check your network connection and")
            print(f"    re-run. Skipping this row rather than writing a fake rate.")
            continue

        # Preserve the original EUR investment intent at the CORRECT rate.
        # price is unchanged — it was already a genuine USD quote, only the
        # currency label and fx were wrong.
        new_qty     = round(old_inv_eur * new_fx / old_price, 4)
        new_inv_eur = round(new_qty * old_price / new_fx, 2)

        changes.append({
            "row": row,
            "ticker": ticker,
            "old": {"currency": stored_ccy, "fx": old_fx, "qty": old_qty, "inv_eur": old_inv_eur},
            "new": {"currency": real_ccy, "fx": new_fx, "qty": new_qty, "inv_eur": new_inv_eur},
        })

    if not changes:
        print("\nNo currency mismatches found among OPEN rows. Nothing to do.")
        return

    print(f"\n{'='*70}")
    print(f"{len(changes)} row(s) to correct:")
    print(f"{'='*70}")
    for c in changes:
        print(f"\n  {c['ticker']}:")
        print(f"    currency        : {c['old']['currency']:6s} -> {c['new']['currency']}")
        print(f"    fx_at_entry     : {c['old']['fx']:.6f} -> {c['new']['fx']:.6f}")
        print(f"    qty             : {c['old']['qty']:.4f}  -> {c['new']['qty']:.4f}")
        print(f"    investment_eur  : {c['old']['inv_eur']:.2f}  -> {c['new']['inv_eur']:.2f}")

    if not apply:
        print(f"\n{'='*70}")
        print("Dry run only — nothing written. Re-run with --apply to write these changes.")
        return

    backup_path = CSV_PATH.with_suffix(".csv.bak")
    shutil.copy2(CSV_PATH, backup_path)
    print(f"\nBacked up original to {backup_path}")

    for c in changes:
        r = c["row"]
        r["currency"]       = c["new"]["currency"]
        r["fx_at_entry"]    = f"{c['new']['fx']:.6f}"
        r["qty"]            = str(c["new"]["qty"])
        r["investment_eur"] = str(c["new"]["inv_eur"])

    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(CSV_PATH)
    print(f"Wrote corrected trades.csv ({len(changes)} row(s) updated).")


if __name__ == "__main__":
    main()
