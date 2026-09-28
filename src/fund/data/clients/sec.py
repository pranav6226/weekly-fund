"""SEC EDGAR client -- free fundamentals and filing events, no key.

Two EDGAR surfaces, both free and keyless (a User-Agent header is required
by SEC policy):
  * XBRL companyfacts (data.sec.gov/api/xbrl/companyfacts/CIK....json):
    as-reported annual facts with `filed` dates -- the point-in-time field.
    This replaces the paid metrics endpoint: value/quality/growth ratios
    are computed here from 10-K facts, gated on filing_date <= as_of.
  * Filing events: the submissions JSON lists every 8-K with its filing
    date; the full-text search API identifies which 8-Ks carry Item 2.02
    (results of operations = earnings announcements).

Nothing here needs estimates data (no free source exists). The filings desk
uses the market's own verdict -- the 2-day price reaction to the 8-K -- as
the surprise proxy, which is a documented variant of post-announcement
drift, not a hack around missing data.
"""
from __future__ import annotations

import time
from datetime import date as _date
from pathlib import Path

import requests

UA = "Alfred weekly-fund research (contact: research@localhost)"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FTS_URL = "https://efts.sec.gov/LATEST/search-index"

TTL_STATIC = 30 * 24 * 3600   # ticker map, filing history: immutable
TTL_FACTS = 7 * 24 * 3600     # companyfacts: new 10-Ks land quarterly

# us-gaap tag fallback chains (companies label the same line differently;
# tags also drift over time -- e.g. Apple switched revenue to the ASC 606
# tag in FY2019, so the modern tag goes first)
TAG_REVENUE = ("RevenueFromContractWithCustomerExcludingAssessedTax",
               "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet")
TAG_GROSS_PROFIT = ("GrossProfit",)
TAG_NET_INCOME = ("NetIncomeLoss",)
TAG_EQUITY = ("StockholdersEquity",)
TAG_EPS = ("EarningsPerShareDiluted",)
TAG_OCF = ("NetCashProvidedByUsedInOperatingActivities",)
TAG_CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment",)
TAG_SHARES = ("WeightedAverageNumberOfDilutedSharesOutstanding",
              "WeightedAverageNumberOfSharesOutstandingBasic")


class SECError(RuntimeError):
    pass


class SECClient:
    def __init__(self, cache_dir: str | Path = "data/sec_cache"):
        from fund.data.clients.cache import TTLCache
        self._cache = TTLCache(cache_dir)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": UA,
            "Accept": "application/json",
        })
        self._ticker_map: dict[str, str] | None = None
        # In-memory memo for a single run: companyfacts + 8-K JSONs are
        # static within a backtest, and re-parsing ~10MB of JSON per
        # ticker per weekly review is what makes the full-universe run
        # take hours. The TTL disk cache still governs freshness across
        # runs; this only dedupes reads inside one process lifetime.
        self._mem: dict[str, object] = {}
        # Static snapshot seeds the map (data/cik_map.json). The live SEC
        # ticker map refreshes it when reachable; the per-ticker FTS
        # lookup is the last resort. Seed file grows as the universe does.
        self._static_map: dict[str, str] = {}
        for cand in (Path("data/cik_map.json"),
                     Path(__file__).resolve().parents[4] / "data" / "cik_map.json"):
            if cand.exists():
                try:
                    import json as _json
                    self._static_map = {k.upper(): v for k, v in
                                        _json.loads(cand.read_text()).items()}
                    break
                except (OSError, ValueError):
                    pass

    # ------------------------------------------------------------ plumbing
    def _get(self, url: str, params: dict | None = None,
             cache_ns: str | None = None, ttl: int = TTL_STATIC,
             throttle: float = 0.12):
        if cache_ns:
            hit = self._cache.get(cache_ns, {"url": url, "p": params}, ttl)
            if hit is not None:
                return hit
        time.sleep(throttle)  # SEC asks for <= 10 req/s
        try:
            r = self._session.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise SECError(f"SEC request failed: {url}: {e}") from e
        if cache_ns:
            self._cache.put(cache_ns, {"url": url, "p": params}, data)
        return data

    def _cik(self, ticker: str) -> str:
        if self._ticker_map is None:
            # seed from the static snapshot so known tickers never need
            # the network at all
            self._ticker_map = dict(self._static_map)
            # www.sec.gov rate-limits shared IPs aggressively; retry with
            # backoff. Once fetched, the 30-day cache makes this moot.
            for attempt in range(4):
                try:
                    raw = self._get(TICKER_MAP_URL, cache_ns="ticker_map",
                                    ttl=TTL_STATIC)
                    live = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                            for v in raw.values()}
                    self._ticker_map.update(live)
                    break
                except SECError:
                    if attempt < 3:
                        time.sleep(5 * (attempt + 1))
            # else: fall through to the per-ticker FTS fallback below
        cik = self._ticker_map.get(ticker.upper())
        if cik:
            return cik
        # Fallback: resolve the CIK through the full-text search API
        # (different host -- survives www.sec.gov throttling).
        ns = f"cik_{ticker.upper()}"
        cached = self._cache.get(ns, {}, TTL_STATIC)
        if cached:
            return cached
        page = self._get(FTS_URL,
                         params={"q": ticker.upper(), "forms": "10-K",
                                 "size": 10},
                         cache_ns=f"fts_cik_{ticker.upper()}",
                         ttl=TTL_STATIC)
        hits = page["hits"]["hits"]
        # pick the filer whose display name carries the exact ticker in
        # parentheses -- a text search for "JPM" alone can match subsidiaries
        want = f"({ticker.upper()})"
        cik = None
        for h in hits:
            names = h["_source"].get("display_names") or []
            if any(want in n.upper() for n in names):
                cik = h["_source"]["ciks"][0].zfill(10)
                break
        if not cik:
            raise SECError(f"{ticker}: no CIK found "
                           f"(non-US filer or delisted?)")
        self._cache.put(ns, {}, cik)
        self._ticker_map[ticker.upper()] = cik
        return cik

    # ------------------------------------------------------------ XBRL facts
    def _facts(self, ticker: str) -> dict:
        cik = self._cik(ticker)
        return self._get(COMPANYFACTS_URL.format(cik=cik),
                         cache_ns=f"facts_{cik}", ttl=TTL_FACTS)

    @staticmethod
    def _annual_value(facts: dict, tags: tuple[str, ...], fy: int):
        """Latest-filed 10-K fact for a tag and fiscal year (amendments win)."""
        usg = facts.get("facts", {}).get("us-gaap", {})
        for tag in tags:
            units = usg.get(tag, {}).get("units", {})
            if not units:
                continue
            unit = next(iter(units))  # USD, USD/shares, shares...
            cands = [f for f in units[unit]
                     if f.get("form") in ("10-K", "10-K/A")
                     and f.get("fp") == "FY" and f.get("fy") == fy
                     and f.get("val") is not None]
            if cands:
                best = max(cands, key=lambda f: f.get("filed", ""))
                return best["val"], best.get("filed")
        return None, None

    def annual_facts(self, ticker: str, limit: int = 8) -> list[dict]:
        """One row per fiscal year (10-K), newest first.

        Each row: {fy, filing_date, revenue, gross_profit, net_income,
        equity, eps_diluted, ocf, capex, diluted_shares}. Missing tags are
        None -- the desk scores on completeness, it never imputes.
        Filing_date is the latest `filed` among the row's facts, so gating
        on it is conservative (nothing visible before the 10-K lands).

        Results are memoized per process: the parsed rows are tiny (a few
        KB per ticker) while the raw companyfacts JSON is ~10MB -- parsing
        it from disk on every weekly review is what makes the
        full-universe backtest take hours. The TTL disk cache still
        governs freshness across runs.
        """
        mem = self.__dict__.setdefault("_mem", {})
        key = f"annual:{ticker.upper()}:{limit}"
        if key not in mem:
            mem[key] = self._annual_facts_uncached(ticker, limit)
        return [dict(r) for r in mem[key]]

    def _annual_facts_uncached(self, ticker: str, limit: int) -> list[dict]:
        """One row per fiscal year (10-K), newest first.

        Each row: {fy, filing_date, revenue, gross_profit, net_income,
        equity, eps_diluted, ocf, capex, diluted_shares}. Missing tags are
        None -- the desk scores on completeness, it never imputes.
        filing_date is the latest `filed` among the row's facts, so gating
        on it is conservative (nothing visible before the 10-K lands).
        """
        facts = self._facts(ticker)
        usg = facts.get("facts", {}).get("us-gaap", {})
        fys = set()
        # revenue tags first, then net income (banks/insurers have no
        # "revenue" line -- NetIncomeLoss is the universal 10-K anchor)
        for tag in TAG_REVENUE + TAG_NET_INCOME:
            for unit in usg.get(tag, {}).get("units", {}).values():
                for f in unit:
                    if f.get("form") in ("10-K", "10-K/A") and f.get("fp") == "FY":
                        fys.add(f.get("fy"))
        rows = []
        for fy in sorted(fys, reverse=True)[:limit]:
            row: dict = {"fy": fy, "filing_date": None}
            latest_filed = ""
            specs = [("revenue", TAG_REVENUE), ("gross_profit", TAG_GROSS_PROFIT),
                     ("net_income", TAG_NET_INCOME), ("equity", TAG_EQUITY),
                     ("eps_diluted", TAG_EPS), ("ocf", TAG_OCF),
                     ("capex", TAG_CAPEX), ("diluted_shares", TAG_SHARES)]
            for key, tags in specs:
                val, filed = self._annual_value(facts, tags, fy)
                row[key] = val
                if filed and filed > latest_filed:
                    latest_filed = filed
            row["filing_date"] = latest_filed or None
            rows.append(row)
        return rows

    # ------------------------------------------------------------ 8-K events
    def _all_8k(self, ticker: str) -> list[dict]:
        """Every 8-K filing: {date, adsh}. From submissions JSON (recent +
        archive files). Immutable -> long TTL."""
        cik = self._cik(ticker)
        ns = f"filings_{cik}"
        cached = self._cache.get(ns, {}, TTL_STATIC)
        if cached is not None:
            return cached
        data = self._get(SUBMISSIONS_URL.format(cik=cik),
                         cache_ns=f"subm_{cik}", ttl=TTL_STATIC)
        out: list[dict] = []

        def harvest(node: dict):
            forms = node.get("form", [])
            dates = node.get("filingDate", [])
            adshs = node.get("accessionNumber", [])
            for i, form in enumerate(forms):
                if form == "8-K":
                    out.append({"date": dates[i],
                                "adsh": adshs[i].replace("-", "")})

        harvest(data["filings"]["recent"])
        for f in data["filings"].get("files", [])[:5]:  # archive pages
            arch = self._get(f"https://data.sec.gov/submissions/{f['name']}",
                             cache_ns=f"subm_arch_{f['name']}", ttl=TTL_STATIC)
            harvest(arch)
        self._cache.put(ns, {}, out)
        return out

    def _earnings_adsh(self, ticker: str) -> set[str]:
        """Accession numbers of 8-Ks carrying Item 2.02 (earnings)."""
        cik = self._cik(ticker)
        ns = f"earn8k_{cik}"
        cached = self._cache.get(ns, {}, TTL_STATIC)
        if cached is not None:
            return set(cached)
        found: set[str] = set()
        params = {"q": '"2.02"', "forms": "8-K", "ciks": cik,
                  "size": 100, "from": 0}
        for _ in range(10):  # hard cap; AAPL has <100 such filings
            page = self._get(FTS_URL, params=dict(params),
                             cache_ns=f"fts_{cik}_{params['from']}",
                             ttl=TTL_STATIC)
            hits = page["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                src = h["_source"]
                if "2.02" in (src.get("items") or []):
                    found.add(src["adsh"].replace("-", ""))
            if len(hits) < params["size"]:
                break
            params["from"] += params["size"]
        self._cache.put(ns, {}, sorted(found))
        return found

    def filing_events(self, ticker: str, limit: int = 500) -> list[dict]:
        """8-K filing events, newest first: {date, kind, items}.

        kind is 'earnings' for Item 2.02 8-Ks, 'other' otherwise. Dates are
        filing dates -- point-in-time by construction. ``date`` is a
        datetime.date, parsed ONCE here (fromisoformat, ~100ns) so the
        per-review desks never pay pd.to_datetime per event -- that was
        87% of a full-universe review.
        """
        key = f"events:{ticker.upper()}:{limit}"
        mem = self.__dict__.setdefault("_mem", {})
        if key not in mem:
            try:
                filings = self._all_8k(ticker)
                earn = self._earnings_adsh(ticker)
            except SECError:
                mem[key] = []
                return []
            events = []
            for f in filings:
                try:
                    d = _date.fromisoformat(str(f["date"])[:10])
                except ValueError:
                    continue
                events.append({"date": d,
                               "kind": "earnings" if f["adsh"] in earn else "other",
                               "adsh": f["adsh"]})
            events.sort(key=lambda e: e["date"], reverse=True)
            mem[key] = events[:limit]
        return list(mem[key])  # type: ignore[arg-type]
