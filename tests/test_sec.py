"""Unit tests for the SEC client and FreeDataStack math.

All network is stubbed: SECClient._get is overridden to serve canned
fixtures. What these tests prove:
  * XBRL facts parse into annual rows with the right filing_date
    (amended 10-K/A wins over the original 10-K)
  * filing_events() labels Item 2.02 8-Ks as earnings, others as 'other'
  * FreeDataStack.metrics() prices P/E on the filing-date close --
    never a later price (no lookahead through the price leg)

Run: ./.venv/bin/python tests/test_sec.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.data.clients.sec import SECClient  # noqa: E402
from fund.data.clients.free_stack import FreeDataStack  # noqa: E402


def _fact(val, filed, form="10-K", fy=2024):
    return {"val": val, "filed": filed, "form": form, "fy": fy, "fp": "FY"}


CANNED_FACTS = {
    "cik": "0000001234", "entityName": "Test Corp",
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact(100.0, "2024-11-01", fy=2024),
            _fact(90.0, "2023-11-03", fy=2023)]}},
        "GrossProfit": {"units": {"USD": [
            _fact(40.0, "2024-11-01", fy=2024),
            _fact(36.0, "2023-11-03", fy=2023)]}},
        "NetIncomeLoss": {"units": {"USD": [
            _fact(20.0, "2024-11-01", fy=2024),
            _fact(18.0, "2023-11-03", fy=2023)]}},
        "StockholdersEquity": {"units": {"USD": [
            _fact(50.0, "2024-11-01", fy=2024),
            _fact(45.0, "2023-11-03", fy=2023)]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            # amended filing restates 2024 EPS: the /A must win
            _fact(2.0, "2024-11-01", fy=2024),
            _fact(2.5, "2024-12-15", form="10-K/A", fy=2024),
            _fact(1.8, "2023-11-03", fy=2023)]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            _fact(30.0, "2024-11-01", fy=2024),
            _fact(28.0, "2023-11-03", fy=2023)]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            _fact(5.0, "2024-11-01", fy=2024),
            _fact(4.0, "2023-11-03", fy=2023)]}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            _fact(10.0, "2024-11-01", fy=2024),
            _fact(10.0, "2023-11-03", fy=2023)]}},
    }},
}

CANNED_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "8-K", "10-Q", "8-K"],
            "filingDate": ["2024-02-20", "2024-01-15", "2024-02-01", "2023-05-01"],
            "accessionNumber": ["0000000001-24-000001",
                                "0000000001-24-000002",
                                "0000000001-24-000003",
                                "0000000001-23-000001"],
        },
        "files": [],
    }
}

CANNED_FTS = {
    "hits": {"hits": [
        {"_source": {"adsh": "000000000124000001",
                     "items": ["2.02", "9.01"],
                     "file_date": "2024-02-20", "form": "8-K"}},
    ]},
}


class StubSEC(SECClient):
    """SECClient with the network replaced by canned fixtures."""

    def __init__(self):
        from fund.data.clients.cache import TTLCache
        self._cache = TTLCache("/tmp/sec_test_cache")
        self._session = None  # never touched: _get is stubbed
        self._ticker_map = {"TEST": "0000001234"}

    def _get(self, url, params=None, cache_ns=None, ttl=0, throttle=0):
        if "company_tickers" in url:
            return {"0": {"cik_str": 1234, "ticker": "TEST",
                          "title": "Test Corp"}}
        if "companyfacts" in url:
            return CANNED_FACTS
        if "submissions" in url:
            return CANNED_SUBMISSIONS
        if "search-index" in url:
            return CANNED_FTS
        raise AssertionError(f"unexpected URL: {url}")


def test_annual_facts_amendment_wins():
    sec = StubSEC()
    rows = sec.annual_facts("TEST", limit=8)
    assert len(rows) == 2, f"expected 2 FY rows, got {len(rows)}"
    r24 = rows[0]
    assert r24["fy"] == 2024
    # amended 10-K/A filed 2024-12-15 restated EPS to 2.5 and is the
    # latest filed fact -> row filing_date must be the amendment date
    assert r24["eps_diluted"] == 2.5, r24["eps_diluted"]
    assert r24["filing_date"] == "2024-12-15", r24["filing_date"]
    assert r24["revenue"] == 100.0
    r23 = rows[1]
    assert r23["eps_diluted"] == 1.8 and r23["filing_date"] == "2023-11-03"
    print("ok: annual facts parse; 10-K/A amendment wins; filing_date = latest")


def test_filing_events_labels_earnings():
    sec = StubSEC()
    evs = sec.filing_events("TEST")
    assert len(evs) == 3, f"expected 3 8-Ks (10-Q excluded), got {len(evs)}"
    assert evs[0]["date"] == "2024-02-20" and evs[0]["kind"] == "earnings", evs[0]
    assert evs[1]["kind"] == "other" and evs[2]["kind"] == "other"
    print("ok: Item 2.02 8-K labeled earnings, rest other; 10-Q excluded")


def test_metrics_prices_on_filing_date_not_later():
    # price jumps AFTER the 10-K filing: P/E must use the filing-date close
    idx = pd.date_range("2024-11-01", "2025-06-01", freq="B")
    closes = pd.DataFrame({"TEST": [100.0 if d < pd.Timestamp("2025-01-01")
                                    else 200.0 for d in idx]}, index=idx)
    stack = FreeDataStack(cache_root="/tmp/sec_test_cache", closes=closes)
    stack._sec = StubSEC()  # type: ignore[attr-defined]
    rows = stack.metrics("TEST", limit=2)
    r24 = rows[0]
    # amended filing_date 2024-12-15 -> close then was 100, EPS 2.5 -> PE 40
    assert r24["filing_date"] == "2024-12-15"
    assert r24["price_to_earnings_ratio"] == 40.0, \
        f"P/E used a post-filing price? {r24['price_to_earnings_ratio']}"
    # FCF = (30-5)/10 = 2.5/sh; yield = 2.5/100 = 2.5%
    assert abs(r24["free_cash_flow_yield"] - 0.025) < 1e-9
    assert r24["return_on_equity"] == 0.4          # 20/50
    assert r24["gross_margin"] == 0.4              # 40/100
    assert abs(r24["earnings_per_share_growth"] - (2.5 - 1.8) / 1.8) < 1e-9
    assert abs(r24["revenue_growth"] - (100 - 90) / 90) < 1e-9
    print("ok: metrics ratios correct; P/E gated on filing-date close")


def test_metrics_without_closes_still_returns_accounting_ratios():
    stack = FreeDataStack(cache_root="/tmp/sec_test_cache")  # no closes
    stack._sec = StubSEC()  # type: ignore[attr-defined]
    rows = stack.metrics("TEST", limit=1)
    assert rows[0]["price_to_earnings_ratio"] is None
    assert rows[0]["return_on_equity"] == 0.4  # accounting ratios survive
    print("ok: no closes -> valuation ratios None, accounting ratios intact")


if __name__ == "__main__":
    test_annual_facts_amendment_wins()
    test_filing_events_labels_earnings()
    test_metrics_prices_on_filing_date_not_later()
    test_metrics_without_closes_still_returns_accounting_ratios()
    print("ALL SEC TESTS PASSED")
