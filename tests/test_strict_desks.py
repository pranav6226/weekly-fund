"""Strict-mode desk failure handling (audit findings H3 + M8).

score_desks(strict=True) must propagate a desk failure as DeskError so
an autonomous live run aborts; non-strict mode keeps skip-and-
renormalize but must record the degraded desks and warn loudly. A macro
desk with zero FRED reads is a failure in strict mode (it would
otherwise default to maximum risk-on, factor 1.0) and a stamped,
warned-about degradation otherwise.

Run: ./.venv/bin/python tests/test_strict_desks.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fund.research.desks as desks_mod  # noqa: E402
from fund.research.desks import DeskError, score_desks  # noqa: E402

AS_OF = date(2024, 3, 1)


class EmptyMacro:
    """FRED stand-in with no data at all: every history is empty."""

    def yield_curve_history(self, limit=400):
        return []

    def inflation_history(self, limit=48):
        return []

    def unemployment_history(self, limit=48):
        return []


def _mandate(*names):
    return SimpleNamespace(
        desks=[SimpleNamespace(name=n) for n in names],
        desk_params={})


def _frames():
    idx = pd.date_range("2023-12-01", "2024-03-01", freq="B")
    closes = pd.DataFrame({"AAA": np.full(len(idx), 100.0)}, index=idx)
    volumes = pd.DataFrame({"AAA": np.full(len(idx), 1e6)}, index=idx)
    return closes, volumes


def _with_boom_fundamentals(fn):
    orig = desks_mod.fundamentals_scores

    def boom(fd, tickers, as_of, params):
        raise RuntimeError("edgar down")

    desks_mod.fundamentals_scores = boom
    try:
        fn()
    finally:
        desks_mod.fundamentals_scores = orig


def test_strict_desk_failure_aborts():
    closes, volumes = _frames()

    def run():
        try:
            score_desks(EmptyMacro(), closes, volumes, AS_OF,
                        _mandate("fundamentals"), strict=True)
        except DeskError as e:
            assert "fundamentals" in str(e)
            return
        raise AssertionError("strict mode must raise DeskError on desk failure")

    _with_boom_fundamentals(run)
    print("ok: strict mode aborts (DeskError) when a desk fails")


def test_nonstrict_desk_failure_recorded(capsys):
    closes, volumes = _frames()
    report: dict = {}

    def run():
        scores, macro_f = score_desks(
            EmptyMacro(), closes, volumes, AS_OF,
            _mandate("fundamentals"), strict=False, report=report)
        assert scores == []  # desk skipped; fusion would renormalize
        assert macro_f == 1.0

    _with_boom_fundamentals(run)
    assert report["degraded_desks"] == ["fundamentals"], report
    out = capsys.readouterr().out
    assert "DESK FAILURE fundamentals" in out
    print("ok: non-strict records degraded desk + warns, run continues")


def test_strict_missing_macro_aborts():
    closes, volumes = _frames()
    try:
        score_desks(EmptyMacro(), closes, volumes, AS_OF,
                    _mandate("macro"), strict=True)
    except DeskError as e:
        assert "macro" in str(e)
    else:
        raise AssertionError("strict mode must abort when no FRED reads land")
    print("ok: strict mode aborts when the macro desk has zero reads")


def test_nonstrict_missing_macro_stamped(capsys):
    closes, volumes = _frames()
    report: dict = {}
    scores, macro_f = score_desks(
        EmptyMacro(), closes, volumes, AS_OF, _mandate("macro"),
        strict=False, report=report)
    assert macro_f == 1.0  # the no-data default, now visible
    assert report["macro_reads_present"] is False
    assert report["macro_factor"] == 1.0
    assert report["degraded_desks"] == ["macro"], report
    out = capsys.readouterr().out
    assert "no FRED macro reads" in out
    print("ok: non-strict missing macro is stamped in the report + warned")


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout

    class _Cap:
        def readouterr(self):
            return type("R", (), {"out": _buf.getvalue(), "err": ""})()

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        test_strict_desk_failure_aborts()
        test_nonstrict_desk_failure_recorded(_Cap())
        test_strict_missing_macro_aborts()
        test_nonstrict_missing_macro_stamped(_Cap())
    print("ALL STRICT-DESK TESTS PASSED")
