"""Mandates: the desk, separated from the tickers.

Stolen from virattt/ai-hedge-fund's FundSpec: a mandate is the desk --
desks, fusion policy, risk limits, capital, cadence, benchmark -- and it
never names tickers. The universe is supplied separately per run
(`--universe` / the backtest config), so one mandate can point at any
universe without editing.

Validation is strict: unknown fields are rejected, desk names must come
from the registry, weights must be positive. A mandate that fails to load
fails loudly at startup, never mid-backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Every desk the mandate may reference. New desks register here; the loader
# rejects anything else so a typo can't silently drop a data source.
DESK_REGISTRY = ("technicals", "fundamentals", "sentiment", "macro")

REBALANCE_CHOICES = ("weekly",)


@dataclass
class DeskSpec:
    name: str            # key into DESK_REGISTRY
    weight: float        # > 0; normalized to sum 1 at load
    params: dict = field(default_factory=dict)


@dataclass
class FusionSpec:
    method: str = "regime_aware"   # "regime_aware" | "fixed"


@dataclass
class RiskSpec:
    max_position_weight: float = 0.08
    max_sector_weight: float = 0.30
    max_positions: int = 20
    stop_loss: float = 0.10
    max_hold_days: int = 84
    regime_floor: float = 0.5


@dataclass
class Mandate:
    schema_version: int
    name: str
    desks: list[DeskSpec]
    fusion: FusionSpec
    risk: RiskSpec
    capital: float = 100_000.0
    rebalance: str = "weekly"
    benchmark: str = "SPY"
    desk_weights: dict = field(default_factory=dict)  # name -> normalized weight

    @property
    def desk_params(self) -> dict[str, dict]:
        return {d.name: d.params for d in self.desks}


def _reject_unknown(mapping: dict, allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"mandate {where}: unknown field(s): {sorted(unknown)}")


def load_mandate(path: str | Path) -> Mandate:
    """Load and strictly validate a mandate YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"mandate {path}: top level must be a mapping")
    _reject_unknown(raw, {"schema_version", "name", "desks", "fusion", "risk",
                          "capital", "rebalance", "benchmark"}, "top-level")

    if raw.get("schema_version") != 1:
        raise ValueError(f"mandate {path}: unsupported schema_version "
                         f"{raw.get('schema_version')!r} (want 1)")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"mandate {path}: 'name' must be a non-empty string")

    desks_raw = raw.get("desks")
    if not isinstance(desks_raw, list) or not desks_raw:
        raise ValueError(f"mandate {path}: 'desks' must be a non-empty list")
    desks: list[DeskSpec] = []
    seen: set[str] = set()
    for i, d in enumerate(desks_raw):
        if not isinstance(d, dict):
            raise ValueError(f"mandate {path}: desks[{i}] must be a mapping")
        _reject_unknown(d, {"name", "weight", "params"}, f"desks[{i}]")
        dname = d.get("name")
        if dname not in DESK_REGISTRY:
            raise ValueError(f"mandate {path}: desks[{i}].name {dname!r} not in "
                             f"registry {list(DESK_REGISTRY)}")
        if dname in seen:
            raise ValueError(f"mandate {path}: duplicate desk {dname!r}")
        seen.add(dname)
        w = d.get("weight", 1.0)
        if not isinstance(w, (int, float)) or w <= 0:
            raise ValueError(f"mandate {path}: desks[{i}].weight must be > 0")
        params = d.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"mandate {path}: desks[{i}].params must be a mapping")
        desks.append(DeskSpec(name=dname, weight=float(w), params=params))

    fusion_raw = raw.get("fusion", {})
    if not isinstance(fusion_raw, dict):
        raise ValueError(f"mandate {path}: 'fusion' must be a mapping")
    _reject_unknown(fusion_raw, {"method"}, "fusion")
    method = fusion_raw.get("method", "regime_aware")
    if method not in ("regime_aware", "fixed"):
        raise ValueError(f"mandate {path}: fusion.method {method!r} unknown")
    fusion = FusionSpec(method=method)

    risk_raw = raw.get("risk", {})
    if not isinstance(risk_raw, dict):
        raise ValueError(f"mandate {path}: 'risk' must be a mapping")
    _reject_unknown(risk_raw, {"max_position_weight", "max_sector_weight",
                               "max_positions", "stop_loss", "max_hold_days",
                               "regime_floor"}, "risk")
    risk = RiskSpec(**{k: v for k, v in risk_raw.items()})

    capital = raw.get("capital", 100_000.0)
    if not isinstance(capital, (int, float)) or capital <= 0:
        raise ValueError(f"mandate {path}: 'capital' must be > 0")
    rebalance = raw.get("rebalance", "weekly")
    if rebalance not in REBALANCE_CHOICES:
        raise ValueError(f"mandate {path}: rebalance {rebalance!r} not in "
                         f"{list(REBALANCE_CHOICES)}")
    benchmark = str(raw.get("benchmark", "SPY")).upper()

    total = sum(d.weight for d in desks)
    desk_weights = {d.name: d.weight / total for d in desks}

    return Mandate(
        schema_version=1, name=name, desks=desks, fusion=fusion, risk=risk,
        capital=float(capital), rebalance=rebalance, benchmark=benchmark,
        desk_weights=desk_weights,
    )
