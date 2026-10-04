"""TypeSafe Jev interface (System One decision model).

Jev is the confidence gate for the live fund: it takes research-agent
outputs and produces calibrated position convictions via its Score
judgment primitive.

Contract (same as fund.decision.fusion.fuse_desks): list[AgentScore] in,
{ticker: 0..1} out -- so the simulator and portfolio constructor consume
it with zero code changes.

Design notes:
  * One `score` question per ticker, batched ~100 tickers per request
    (state budget). Each question points at its ticker's slice of the
    shared state; questions are answered in parallel by the API.
  * 5 ordered levels describing situations (per TypeSafe's guidance),
    normalized score/4 -> 0..1 conviction.
  * Model is pinned (jev-1.13.0): aliases can move and thresholds depend
    on model behavior. Bump deliberately, not silently.
  * The key lives in the user's .env (JEV_API_KEY), never in this repo.
    JevClient() with no argument reads it from the environment.
  * 429/529 get two backoff retries. Anything else raises JevError --
    the live loop catches it and falls back to fuse_desks, so a judge
    outage never kills a rebalance.

BACKTEST RULE (stolen from ai-hedge-fund): a judge trained after the
backtest window may remember how a named company did, and that memory
scores as skill. So backtests MUST call convictions(..., blind=True):
tickers become T-0001 labels, dates are withheld, and the returned
convictions are mapped back before portfolio construction. Live runs use
blind=False and name the company. The walk-forward simulator
(fund.backtest.simulator) stays on fuse_desks -- deterministic and
reproducible -- and never touches this client.
"""
from __future__ import annotations

import json
import os
import random
import time
import http.client
import urllib.error
import urllib.request
from collections import defaultdict

from fund.decision.fusion import anonymize_scores, deanonymize_convictions
from fund.research.schema import AgentScore

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"  # pinned; bump deliberately (see module docstring)

_BATCH_TICKERS = 10  # keep responses small: the egress proxy truncates
                   # large bodies mid-read (IncompleteRead), even at 25/batch
_BATCH_PAUSE_SECS = 1.0  # pace batches: rapid sequential CONNECTs through
                         # the proxy draw RemoteDisconnected flakes
_TIMEOUT_SECS = 90
# The sandbox egress proxy drops ~10-15% of requests (RemoteDisconnected /
# IncompleteRead) in bursts. 6 retries with capped exponential backoff +
# jitter make a 50-batch weekly run survive them; a batch that still fails
# raises JevError and the live loop falls back to local fusion.
_MAX_RETRIES = 6
_MAX_BACKOFF_SECS = 60.0

# Ordered low -> high. Situations, not degrees (TypeSafe's guidance:
# the model matches state against each level's description). Kept terse:
# the legend is echoed back in every answer, so long levels bloat
# responses and the egress proxy truncates large bodies mid-read.
_CONVICTION_LEVELS = [
    "Strong avoid: agent evidence clearly negative -- falling momentum, "
    "weak fundamentals, or adverse filings.",
    "Avoid: evidence leans negative overall; no positive catalyst in "
    "the agent scores.",
    "Neutral: mixed or flat evidence; no edge in either direction.",
    "Buy: evidence leans positive -- rising momentum, solid fundamentals, "
    "or supportive filings.",
    "Strong buy: multiple agents clearly positive with high confidence.",
]
_TOP_LEVEL = len(_CONVICTION_LEVELS) - 1


class JevError(Exception):
    """Jev call failed (auth, network, malformed response)."""


def _proxy_opener() -> urllib.request.OpenerDirector:
    proxies = {}
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        if os.environ.get(var):
            proxies["https" if "https" in var.lower() else "http"] = os.environ[var]
            break
    # This VM needs the egress proxy for direct HTTPS; without it the
    # connection times out. Honor env so other environments just work.
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


class JevClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise JevError("no Jev API key: pass api_key or set JEV_API_KEY")
        self._opener = _proxy_opener()
        # Observability for the run log; reset per convictions() call.
        self.last_stats: dict = {}

    # ------------------------------------------------------------- internals
    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Fresh connection per request: the egress proxy has truncated
            # mid-read on reused keep-alive connections (IncompleteRead).
            "Connection": "close",
        }

        def backoff(attempt: int) -> None:
            delay = min(2.0 * (2 ** attempt), _MAX_BACKOFF_SECS)
            time.sleep(delay + random.uniform(0, 1.0))

        for attempt in range(_MAX_RETRIES + 1):
            req = urllib.request.Request(
                JEV_API_URL, data=body, headers=headers, method="POST"
            )
            try:
                with self._opener.open(req, timeout=_TIMEOUT_SECS) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < _MAX_RETRIES:
                    backoff(attempt)
                    continue
                detail = e.read()[:300].decode("utf-8", "replace")
                raise JevError(f"Jev HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as e:
                # Transient transport failure (RemoteDisconnected,
                # IncompleteRead through the proxy): back off and retry
                # before giving up.
                if attempt < _MAX_RETRIES:
                    backoff(attempt)
                    continue
                raise JevError(f"Jev request failed: {e}") from e
        raise JevError("Jev request failed after retries")  # unreachable

    @staticmethod
    def _state_for(labels: list[str], by_ticker: dict[str, list[AgentScore]],
                   blind: bool, as_of, regime_label,
                   regime_factor: float) -> dict:
        from fund.research.desks import AGENT_DESK

        tickers = {}
        for label in labels:
            agents = []
            for s in by_ticker[label]:
                agents.append({
                    "agent": s.agent,
                    "desk": AGENT_DESK.get(s.agent, "unknown"),
                    "score": round(s.score, 3),
                    "confidence": round(s.confidence, 3),
                    "rationale": (s.rationale or "")[:200],
                })
            tickers[label] = agents
        state: dict = {
            "tickers": tickers,
            "regime": {
                "label": regime_label,
                "factor": round(regime_factor, 3),
            },
        }
        if not blind and as_of is not None:
            state["as_of"] = as_of.isoformat()
        return state

    @staticmethod
    def _questions_for(labels: list[str]) -> dict:
        questions = {}
        for label in labels:
            questions[label] = {
                "type": "score",
                "instructions": (
                    f"Position conviction for {label}: rate how attractive "
                    f"a LONG position is, using only the agent evidence at "
                    f"state -> tickers -> {label}. Agent scores are "
                    f"cross-sectionally z-scored (higher = more "
                    f"attractive); weight each agent by its confidence. "
                    f"Ignore evidence belonging to other tickers."
                ),
                "criteria": _CONVICTION_LEVELS,
            }
        return questions

    # ------------------------------------------------------------------ API
    def convictions(self, scores: list[AgentScore], blind: bool = False,
                    regime_label: str | None = None,
                    regime_factor: float = 1.0) -> dict[str, float]:
        """Judge the desk scores -> {ticker: conviction in 0..1}."""
        mapping: dict[str, str] | None = None
        as_of = None
        if blind:
            scores, mapping = anonymize_scores(scores)
        elif scores:
            as_of = scores[0].as_of

        by_ticker: dict[str, list[AgentScore]] = defaultdict(list)
        for s in scores:
            by_ticker[s.ticker].append(s)
        labels = sorted(by_ticker)

        out: dict[str, float] = {}
        confidences: list[float] = []
        model_seen: str | None = None
        n_batches = (len(labels) + _BATCH_TICKERS - 1) // _BATCH_TICKERS
        for bi in range(n_batches):
            batch = labels[bi * _BATCH_TICKERS:(bi + 1) * _BATCH_TICKERS]
            payload = {
                "model": JEV_MODEL,
                "state": self._state_for(batch, by_ticker, blind, as_of,
                                         regime_label, regime_factor),
                "questions": self._questions_for(batch),
            }
            resp = self._post(payload)
            model_seen = resp.get("model", model_seen)
            answers = resp.get("answers", {})
            for label in batch:
                try:
                    ans = answers[label]
                    raw = float(ans["score"])
                except (KeyError, TypeError, ValueError) as e:
                    raise JevError(
                        f"Jev response missing/invalid answer for {label}"
                    ) from e
                out[label] = min(max(raw / _TOP_LEVEL, 0.0), 1.0)
                try:
                    confidences.append(float(ans.get("confidence", 0.0)))
                except (TypeError, ValueError):
                    pass
            if bi < n_batches - 1:
                time.sleep(_BATCH_PAUSE_SECS)
            if n_batches > 1 and (bi + 1) % 10 == 0:
                print(f"  jev: {bi + 1}/{n_batches} batches judged", flush=True)

        if mapping:
            out = deanonymize_convictions(out, mapping)

        self.last_stats = {
            "model": model_seen or JEV_MODEL,
            "n_tickers": len(out),
            "n_requests": (len(labels) + _BATCH_TICKERS - 1) // _BATCH_TICKERS,
            "mean_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "blind": blind,
        }
        return out
