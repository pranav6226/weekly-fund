"""Tests for the daily stop check (scripts/stop_check.py) and the
worker's weekday schedule helper. Pure logic only -- no network."""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stop_check import evaluate_stops  # noqa: E402
from autonomous_worker import STOP_DAYS, last_weekday  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")


def pos(qty, mv, entry):
    return {"qty": qty, "market_value": mv, "avg_entry_price": entry}


def test_trigger_math():
    positions = {
        "DELL": pos(5.0, 2335.0, 547.80),   # close 467.05 = -14.7%
        "CALM": pos(5.0, 2600.0, 500.0),    # close 520 = +4%
        "EDGE": pos(5.0, 2250.0, 500.0),    # close 450 = exactly -10%
        "NEAR": pos(5.0, 2255.0, 500.0),    # close 451 = -9.8%
    }
    closes = {"DELL": 467.05, "CALM": 520.0, "EDGE": 450.0, "NEAR": 451.0}
    by = {c["symbol"]: c for c in evaluate_stops(positions, closes, 0.10)}
    assert by["DELL"]["triggered"] is True
    assert by["CALM"]["triggered"] is False
    assert by["EDGE"]["triggered"] is True   # boundary: <= entry*(1-stop)
    assert by["NEAR"]["triggered"] is False
    assert by["DELL"]["stop_level"] == round(547.80 * 0.9, 4)


def test_dust_and_shorts_skipped():
    positions = {
        "DUST": pos(0.001, 0.42, 100.0),
        "REAL": pos(5.0, 2000.0, 500.0),
    }
    checks = evaluate_stops(positions, {"DUST": 1.0, "REAL": 400.0}, 0.10)
    assert [c["symbol"] for c in checks] == ["REAL"]


def test_missing_close_never_exits():
    positions = {"GONE": pos(5.0, 2000.0, 500.0)}
    checks = evaluate_stops(positions, {}, 0.10)
    assert checks[0]["status"] == "check_failed"
    assert checks[0]["triggered"] is False


def test_missing_entry_never_exits():
    positions = {"NOE": pos(5.0, 2000.0, 0.0)}
    checks = evaluate_stops(positions, {"NOE": 100.0}, 0.10)
    assert checks[0]["status"] == "check_failed"
    assert checks[0]["triggered"] is False


def test_last_weekday_schedule():
    # Friday 2026-10-16 14:00 PT -> same day 13:35
    fri_pm = datetime(2026, 10, 16, 14, 0, tzinfo=PT)
    assert last_weekday(fri_pm, 13, 35) == datetime(2026, 10, 16, 13, 35,
                                                    tzinfo=PT)
    # Monday 2026-10-19 09:00 PT (before 13:35) -> Friday 2026-10-16
    mon_am = datetime(2026, 10, 19, 9, 0, tzinfo=PT)
    assert last_weekday(mon_am, 13, 35) == datetime(2026, 10, 16, 13, 35,
                                                    tzinfo=PT)
    # Saturday 2026-10-17 10:00 PT -> Friday 2026-10-16
    sat = datetime(2026, 10, 17, 10, 0, tzinfo=PT)
    assert last_weekday(sat, 13, 35) == datetime(2026, 10, 16, 13, 35,
                                                 tzinfo=PT)
    # Wednesday 2026-10-21 13:35 exactly -> same instant
    wed = datetime(2026, 10, 21, 13, 35, tzinfo=PT)
    assert last_weekday(wed, 13, 35) == wed


def test_stop_days_skip_friday():
    # The live stop job uses STOP_DAYS (Mon-Thu): on Friday afternoon
    # the last slot is Thursday's -- Friday breaches belong to the
    # Saturday review, whose orders fill at the same Monday open.
    fri_pm = datetime(2026, 10, 16, 14, 0, tzinfo=PT)
    assert last_weekday(fri_pm, 13, 35, days=STOP_DAYS) == \
        datetime(2026, 10, 15, 13, 35, tzinfo=PT)
    # Saturday resolves to Thursday too (not Friday).
    sat = datetime(2026, 10, 17, 10, 0, tzinfo=PT)
    assert last_weekday(sat, 13, 35, days=STOP_DAYS) == \
        datetime(2026, 10, 15, 13, 35, tzinfo=PT)
    # Monday before 13:35 -> the previous Thursday.
    mon_am = datetime(2026, 10, 19, 9, 0, tzinfo=PT)
    assert last_weekday(mon_am, 13, 35, days=STOP_DAYS) == \
        datetime(2026, 10, 15, 13, 35, tzinfo=PT)
    # Thursday 2026-10-22 14:00 -> same day.
    thu_pm = datetime(2026, 10, 22, 14, 0, tzinfo=PT)
    assert last_weekday(thu_pm, 13, 35, days=STOP_DAYS) == \
        datetime(2026, 10, 22, 13, 35, tzinfo=PT)


if __name__ == "__main__":
    test_trigger_math()
    test_dust_and_shorts_skipped()
    test_missing_close_never_exits()
    test_missing_entry_never_exits()
    test_last_weekday_schedule()
    test_stop_days_skip_friday()
    print("ALL STOP-CHECK TESTS PASSED")
