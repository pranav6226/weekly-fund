"""Backfill the SEC disk cache for the full universe (one-time, resumable).

Fetches per ticker (each cached on disk, skipped when fresh):
  - metrics(): XBRL companyfacts -> annual 10-K rows
  - filing_events(): 8-K events via EDGAR full-text search

Safe to re-run: the cache layer skips tickers that are already fresh.
Loud about failures; a failed ticker is retried next run, never fatal.

Usage:
    python scripts/backfill_sec.py [--max-tickers 0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.data.clients.free_stack import FreeDataStack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    uni = pd.read_csv(ROOT / "data" / "sp500_constituents.csv")
    tickers = [str(s).replace(".", "-") for s in uni["Symbol"]]
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"backfilling {len(tickers)} tickers", flush=True)

    stack = FreeDataStack(cache_root=str(ROOT / "data"))
    ok_m, ok_f, fail = 0, 0, []
    t0 = time.time()
    for i, t in enumerate(tickers):
        try:
            m = stack.metrics(t)
            ok_m += 1 if m else 0
        except Exception as e:  # noqa: BLE001
            fail.append((t, "metrics", f"{type(e).__name__}: {e}"))
        try:
            f = stack.filing_events(t)
            ok_f += 1 if f else 0
        except Exception as e:  # noqa: BLE001
            fail.append((t, "filings", f"{type(e).__name__}: {e}"))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(tickers)} metrics_ok={ok_m} filings_ok={ok_f} "
                  f"fail={len(fail)} ({el:.0f}s)", flush=True)
        time.sleep(0.4)  # polite to EDGAR
    print(f"done: {len(tickers)} tickers, metrics_ok={ok_m}, filings_ok={ok_f}, "
          f"fail={len(fail)} in {time.time()-t0:.0f}s")
    for t, which, err in fail[:20]:
        print(f"  FAIL {t} {which}: {err}")
        if len(fail) > 20:
            print(f"  ... +{len(fail)-20} more")
            break


if __name__ == "__main__":
    main()
