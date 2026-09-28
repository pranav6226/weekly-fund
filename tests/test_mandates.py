"""Unit tests for the mandate system (the ai-hedge-fund steal).

Run: ./.venv/bin/python tests/test_mandates.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.mandates.schema import load_mandate  # noqa: E402

MANDATE = ROOT / "mandates" / "core-4desk.yaml"


def _write_tmp(body: str) -> Path:
    p = Path(tempfile.mkdtemp()) / "m.yaml"
    p.write_text(body)
    return p


BASE = """\
schema_version: 1
name: t
desks:
  - name: technicals
    weight: 1.0
fusion:
  method: fixed
risk: {}
capital: 1000
rebalance: weekly
benchmark: SPY
"""


def test_loads_example_mandate():
    m = load_mandate(MANDATE)
    assert m.name == "core-4desk"
    assert [d.name for d in m.desks] == ["technicals", "fundamentals",
                                        "sentiment", "macro"]
    assert abs(sum(m.desk_weights.values()) - 1.0) < 1e-9
    assert m.desk_weights["technicals"] == 0.40
    assert m.rebalance == "weekly" and m.benchmark == "SPY"
    assert m.risk.max_positions == 20
    print("ok: example mandate loads, weights normalized")


def test_rejects_unknown_top_level_field():
    p = _write_tmp(BASE + "bogus: 1\n")
    try:
        load_mandate(p)
    except ValueError as e:
        assert "unknown field" in str(e)
        print("ok: unknown top-level field rejected")
    else:
        raise AssertionError("should have rejected unknown field")


def test_rejects_unknown_desk():
    body = BASE.replace("- name: technicals", "- name: astrology")
    p = _write_tmp(body)
    try:
        load_mandate(p)
    except ValueError as e:
        assert "registry" in str(e)
        print("ok: unknown desk rejected")
    else:
        raise AssertionError("should have rejected unknown desk")


def test_rejects_duplicate_desk():
    body = BASE.replace(
        "  - name: technicals\n    weight: 1.0",
        "  - name: technicals\n    weight: 1.0\n  - name: technicals\n    weight: 2.0",
    )
    p = _write_tmp(body)
    try:
        load_mandate(p)
    except ValueError as e:
        assert "duplicate" in str(e)
        print("ok: duplicate desk rejected")
    else:
        raise AssertionError("should have rejected duplicate desk")


def test_rejects_nonpositive_weight():
    body = BASE.replace("weight: 1.0", "weight: 0")
    p = _write_tmp(body)
    try:
        load_mandate(p)
    except ValueError as e:
        assert "weight" in str(e)
        print("ok: non-positive weight rejected")
    else:
        raise AssertionError("should have rejected zero weight")


def test_rejects_bad_schema_version():
    p = _write_tmp(BASE.replace("schema_version: 1", "schema_version: 99"))
    try:
        load_mandate(p)
    except ValueError as e:
        assert "schema_version" in str(e)
        print("ok: bad schema_version rejected")
    else:
        raise AssertionError("should have rejected bad schema version")


if __name__ == "__main__":
    test_loads_example_mandate()
    test_rejects_unknown_top_level_field()
    test_rejects_unknown_desk()
    test_rejects_duplicate_desk()
    test_rejects_nonpositive_weight()
    test_rejects_bad_schema_version()
    print("ALL MANDATE TESTS PASSED")
