"""Unit tests for the Phase 3 monitor (scripts/monitor.py).

The critical property: the monitor is READ-ONLY. test_monitor_makes_no_write_calls
runs the full check against a real PaperBroker with urlopen monkeypatched to
explode on any POST/DELETE/PUT/PATCH -- completing the run without raising is
the assertion that no order was placed, modified, or cancelled.

Run:  ./.venv/bin/python -m pytest tests/test_monitor.py -v
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import monitor  # noqa: E402
from fund.execution.broker import PaperBroker  # noqa: E402


# ------------------------------------------------------------------ fakes
class _FakeResp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_fake_urlopen(account, positions):
    def fake_urlopen(req, timeout=None):
        method = req.get_method()
        if method in ("POST", "DELETE", "PUT", "PATCH"):
            raise AssertionError(
                f"monitor attempted a WRITE call: {method} {req.full_url} "
                "-- the monitor must never place, modify, or cancel orders"
            )
        if method != "GET":
            raise AssertionError(f"unexpected method {method}")
        if req.full_url.endswith("/v2/account"):
            return _FakeResp(account)
        if req.full_url.endswith("/v2/positions"):
            return _FakeResp(positions)
        raise AssertionError(f"unexpected URL {req.full_url}")

    return fake_urlopen


ACCOUNT = {
    "status": "ACTIVE", "equity": "100000", "cash": "90000",
    "buying_power": "200000", "currency": "USD",
}
POSITIONS = [
    # AAA: 3.5% of equity vs 2.0% target -> weight drift > 1pp
    {"symbol": "AAA", "qty": "10", "market_value": "3500",
     "avg_entry_price": "340", "current_price": "350"},
    # ZZZ: held but not in targets -> extra
    {"symbol": "ZZZ", "qty": "5", "market_value": "1000",
     "avg_entry_price": "200", "current_price": "200"},
    # BBB is targeted but not held -> missing
]


def _run_with_fake(tmp_path, account=ACCOUNT, positions=POSITIONS,
                   targets=None, seed_log=None, today=date(2026, 9, 28)):
    state_dir = tmp_path / "paper_runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "latest.json").write_text(json.dumps({
        "as_of": "2026-09-25",
        "targets": targets if targets is not None else {"AAA": 0.02, "BBB": 0.03},
    }))
    log_path = tmp_path / "monitor.log"
    if seed_log:
        log_path.write_text(seed_log)
    broker = PaperBroker(api_key="k", api_secret="s")  # dummy keys; no env touched
    orig = urllib.request.urlopen
    urllib.request.urlopen = _make_fake_urlopen(account, positions)
    try:
        return monitor.run_check(broker, state_dir=state_dir,
                                 log_path=log_path, today=today), log_path
    finally:
        urllib.request.urlopen = orig


# ------------------------------------------------------------------- tests
def test_monitor_makes_no_write_calls(tmp_path):
    """End-to-end: real PaperBroker, urlopen explodes on POST/DELETE."""
    result, _ = _run_with_fake(tmp_path)
    assert result["status"] == "ACTIVE"
    assert result["equity"] == 100000.0
    assert result["n_positions"] == 2
    assert result["extra"] == ["ZZZ"]
    assert result["missing"] == ["BBB"]
    assert len(result["drifted"]) == 1
    d = result["drifted"][0]
    assert d["symbol"] == "AAA"
    assert abs(d["actual"] - 0.035) < 1e-9
    assert abs(d["target"] - 0.02) < 1e-9
    assert not result["healthy"]
    assert any(a.startswith("EXTRA_POSITIONS") for a in result["anomalies"])
    assert any(a.startswith("MISSING_POSITIONS") for a in result["anomalies"])
    assert any(a.startswith("WEIGHT_DRIFT") for a in result["anomalies"])


def test_monitor_healthy_when_aligned(tmp_path):
    result, _ = _run_with_fake(
        tmp_path,
        positions=[
            {"symbol": "AAA", "qty": "10", "market_value": "2000",
             "avg_entry_price": "200", "current_price": "200"},
        ],
        targets={"AAA": 0.02},
    )
    assert result["extra"] == []
    assert result["missing"] == []
    assert result["drifted"] == []
    assert result["healthy"] is True
    assert result["anomalies"] == []


def test_non_active_account_is_anomaly(tmp_path):
    bad = dict(ACCOUNT, status="SUSPENDED")
    result, _ = _run_with_fake(tmp_path, account=bad, targets={"AAA": 0.035})
    assert not result["healthy"]
    assert "ACCOUNT_STATUS=SUSPENDED" in result["anomalies"]


def test_missing_targets_file_is_anomaly(tmp_path):
    log_path = tmp_path / "monitor.log"
    broker = PaperBroker(api_key="k", api_secret="s")
    orig = urllib.request.urlopen
    urllib.request.urlopen = _make_fake_urlopen(ACCOUNT, [])
    try:
        result = monitor.run_check(broker, state_dir=tmp_path / "nope",
                                   log_path=log_path, today=date(2026, 9, 28))
    finally:
        urllib.request.urlopen = orig
    assert result["n_targets"] == 0
    assert "NO_TARGETS_FILE" in result["anomalies"]
    assert not result["healthy"]


def test_week_over_week_equity_drop_flags_anomaly(tmp_path):
    seed = "date=2026-09-20 equity=110000.00 npos=1 ntargets=1 extra=- missing=- drift=- status=ACTIVE wow=-\n"
    result, log_path = _run_with_fake(tmp_path, seed_log=seed)
    assert result["wow"] is not None
    assert abs(result["wow"] - (100000 / 110000 - 1)) < 1e-9
    assert any(a.startswith("EQUITY_DOWN_WOW") for a in result["anomalies"])
    assert not result["healthy"]
    # the new snapshot was appended after the seeded line
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("date=2026-09-28 equity=100000.00")


def test_snapshot_log_line_format(tmp_path):
    _, log_path = _run_with_fake(tmp_path)
    line = log_path.read_text().splitlines()[0]
    for key in ("date=", "equity=", "npos=", "extra=", "missing=",
                "drift=", "status=", "wow="):
        assert key in line, f"log line missing {key}: {line}"


def test_monitor_source_has_no_order_paths():
    src = (ROOT / "scripts" / "monitor.py").read_text()
    assert "submit_order" not in src, "monitor must not reference submit_order"
    assert "close_position" not in src, "monitor must not reference close_position"


def test_summary_mentions_status_line_contract(tmp_path, capsys):
    result, _ = _run_with_fake(tmp_path)
    print(monitor.format_summary(result))
    print("STATUS: " + ("HEALTHY" if result["healthy"] else "ANOMALY"))
    out = capsys.readouterr().out
    assert out.rstrip().endswith("STATUS: ANOMALY")
    assert "drift" in out
    assert "AAA" in out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_monitor_makes_no_write_calls(tp / "t1")
        test_monitor_healthy_when_aligned(tp / "t2")
        test_non_active_account_is_anomaly(tp / "t3")
        test_missing_targets_file_is_anomaly(tp / "t4")
        test_week_over_week_equity_drop_flags_anomaly(tp / "t5")
        test_snapshot_log_line_format(tp / "t6")
    test_monitor_source_has_no_order_paths()
    print("test_monitor: all passed")
