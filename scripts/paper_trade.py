"""Phase 5: weekly paper-trading loop (Alpaca PAPER only).

Runs the v3 4-desk swarm as of the most recent Friday close, builds the
target portfolio (3% per-position cap per the locked risk decision),
diffs against the paper account, and prints the order plan.

DRY-RUN BY DEFAULT: prints the full plan and writes state, submits
nothing. Pass --live to place real orders on the PAPER endpoint --
the broker refuses any non-paper URL, rejects shorts, and rejects any
single order above 3.2% of equity.

Usage:
    python scripts/paper_trade.py [--as-of 2026-09-25] [--live] [--max-tickers 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig  # noqa: E402
from fund.data.clients.free_stack import FreeDataStack  # noqa: E402
from fund.data.db import get_engine  # noqa: E402
from fund.data.store import load_panel  # noqa: E402
from fund.decision.fusion import blend_regime, desk_breakdown, fuse_desks, market_regime  # noqa: E402
from fund.decision.jev import JevClient, JevError, JevPartialError  # noqa: E402
from fund.decision.portfolio import construct_targets  # noqa: E402
from fund.execution.broker import (  # noqa: E402
    PAPER_API_BASE,
    BrokerError,
    DryRunBroker,
    PaperBroker,
)
from fund.live.paper import (  # noqa: E402
    apply_position_cap,
    build_order_plan,
    check_order_sanity,
    most_recent_friday,
    positions_from_snapshot,
    summarize_plan,
)
from fund.mandates.schema import load_mandate  # noqa: E402
from fund.research.desks import score_desks  # noqa: E402

# Locked by the user 2026-09-28: 500-stock universe, $100k paper capital,
# 3% per-position risk cap, Alpaca paper only.
PAPER_CAPITAL = 100_000.0
MAX_POSITION_WEIGHT = 0.03  # overrides mandate yaml (0.08) and FundConfig (0.08)


def load_dotenv_root() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default="",
                    help="review date YYYY-MM-DD (default: most recent Friday close)")
    ap.add_argument("--live", action="store_true",
                    help="actually submit orders to the PAPER endpoint (default: dry-run)")
    ap.add_argument("--max-tickers", type=int, default=500,
                    help="universe size: first N rows of data/universe.csv")
    ap.add_argument("--no-jev", action="store_true",
                    help="skip the Jev decision gate; use local desk fusion instead")
    args = ap.parse_args()

    load_dotenv_root()
    mode = "LIVE (paper)" if args.live else "DRY-RUN"
    print(f"mode: {mode} | endpoint: {PAPER_API_BASE}", flush=True)

    cfg = FundConfig(max_position_weight=MAX_POSITION_WEIGHT)
    print(f"risk cap: max_position_weight={cfg.max_position_weight:.0%} "
          f"(user lock; mandate yaml says 8%)", flush=True)

    mandate = load_mandate(str(ROOT / "mandates" / "core-4desk.yaml"))
    desk_weights = {d.name: d.weight for d in mandate.desks}
    print(f"mandate: core-4desk, weights={desk_weights}", flush=True)

    # ---- universe: first N rows of universe.csv (stable, deterministic) ----
    uni = pd.read_csv(ROOT / cfg.universe_csv).head(args.max_tickers)
    tickers = uni["Symbol"].str.replace(".", "-", regex=False).tolist()
    sectors = dict(zip(tickers, uni["GICS Sector"]))
    print(f"universe: {len(tickers)} tickers (first {args.max_tickers} rows of "
          f"{cfg.universe_csv})", flush=True)

    # ---- price panel ----
    engine = get_engine(str(ROOT / cfg.db_path))
    panel = load_panel(engine, tickers)
    closes_full, volumes_full = panel["close"], panel["volume"]
    panel_max = closes_full.index.max().date()
    if args.as_of:
        as_of = date.fromisoformat(args.as_of)
        if pd.Timestamp(as_of) not in closes_full.index:
            raise SystemExit(f"--as-of {as_of} not in price panel (max {panel_max})")
    else:
        as_of = most_recent_friday(closes_full.index)
    if panel_max < as_of:
        raise SystemExit(
            f"price panel is stale (max {panel_max} < review {as_of}); "
            "run scripts/download_history.py first")
    print(f"review date: {as_of} (panel through {panel_max})", flush=True)

    # ---- 4-desk swarm, strictly point-in-time ----
    closes = closes_full.loc[:pd.Timestamp(as_of)]
    volumes = volumes_full.loc[:pd.Timestamp(as_of)]
    stack = FreeDataStack(cache_root=str(ROOT / "data"))
    stack.set_closes(closes)
    t0 = pd.Timestamp.now()
    scores, macro_f = score_desks(stack, closes, volumes, as_of,
                                  mandate, sectors)
    price_regime = market_regime(closes_full, as_of, cfg)
    regime_factor = blend_regime(price_regime.factor, macro_f, blend=0.5,
                                 floor=cfg.regime_floor)
    convictions = None
    decision_source = "fusion"
    if not args.no_jev and os.environ.get("JEV_API_KEY"):
        jev_cache = (ROOT / cfg.results_dir / "jev_cache"
                     / f"{as_of.isoformat()}.json")
        try:
            jev = JevClient()
            t_jev = pd.Timestamp.now()
            convictions = jev.convictions(
                scores, blind=False,
                regime_label=price_regime.label,
                regime_factor=regime_factor,
                cache_path=jev_cache,
            )
            st = jev.last_stats
            decision_source = f"jev {st['model']}"
            cached_note = (f", {st['n_cached']} resumed from cache"
                           if st.get("n_cached") else "")
            # mean_confidence is None when every ticker resumed from
            # cache (the cache stores convictions, not confidences).
            conf = st["mean_confidence"]
            conf_note = f"{conf:.2f}" if conf is not None else "n/a (cached)"
            print(f"decision: jev {st['model']} | {st['n_tickers']} tickers, "
                  f"{st['n_requests']} requests{cached_note}, "
                  f"mean_conf={conf_note} "
                  f"({(pd.Timestamp.now() - t_jev).total_seconds():.0f}s)",
                  flush=True)
        except JevPartialError as e:
            # Proxy ate some batches: keep what Jev judged, backfill only
            # the missing tickers with local fusion. The cache means a
            # re-run resumes the missing batches instead of restarting.
            fb = fuse_desks(scores, desk_weights, regime_factor)
            need = {s.ticker for s in scores}
            missing = sorted(t for t in need if t not in e.partial)
            convictions = {**{t: fb[t] for t in missing if t in fb},
                           **e.partial}
            decision_source = "jev (partial) + fusion backfill"
            print(f"WARNING: jev judged {len(e.partial)}/{len(need)} tickers; "
                  f"{len(missing)} backfilled with local fusion "
                  f"(cache: {jev_cache.name})", flush=True)
        except JevError as e:
            print(f"WARNING: jev decision gate failed ({e}); "
                  f"falling back to local fusion", flush=True)
    if convictions is None:
        convictions = fuse_desks(scores, desk_weights, regime_factor)
        print("decision: local fusion (fuse_desks)", flush=True)
    print(f"scored {len(convictions)} tickers in "
          f"{(pd.Timestamp.now() - t0).total_seconds():.0f}s | "
          f"regime={price_regime.label} factor={regime_factor:.3f} "
          f"({price_regime.rationale}|macro={macro_f:.2f})", flush=True)

    # Per-desk detail for the audit trail + dashboard. Reporting only:
    # built from the same shared fusion computation, never fed back
    # into any decision. The final conviction (Jev's, when the gate
    # ran) is attached per ticker.
    desk_scores: dict[str, dict] = {}
    for t, b in desk_breakdown(scores, desk_weights, regime_factor).items():
        row = dict(b["desks"])
        row["fused"] = b["fused"]
        if t in convictions:
            row["conviction"] = round(float(convictions[t]), 4)
        desk_scores[t] = row

    # ---- broker / account ----
    state_dir = ROOT / cfg.results_dir / "paper_runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    prev = _load_latest(state_dir)

    if args.live:
        broker: PaperBroker | DryRunBroker = PaperBroker()
        account = broker.get_account()
        if account.get("status") != "ACTIVE":
            raise SystemExit(f"refusing live run: account status={account.get('status')}")
        equity = float(account["equity"])
        raw_positions = broker.get_positions()
        print(f"account: {account['status']} equity=${equity:,.0f} "
              f"cash=${account['cash']:,.0f}", flush=True)
    else:
        broker = DryRunBroker()
        raw_positions, equity = _dry_run_baseline(prev, closes, as_of)
    print(f"current positions: {len(raw_positions)}", flush=True)

    # ---- targets ----
    snap = prev.get("positions", {}) if prev else {}
    # Incumbents for target construction = what the account holds NOW
    # (raw_positions). Building them from the prior snapshot alone can
    # name phantom positions -- names the last run already exited --
    # whose shares and entry dates would corrupt the survivor/slot and
    # stop/time-stop logic in construct_targets. Entry dates carry over
    # from the snapshot for names held across the last run; holdings the
    # last run opened date from that run.
    prev_as_of = str(prev.get("as_of") or as_of.isoformat())
    cur_snap: dict[str, dict] = {}
    for t, p in raw_positions.items():
        if float(p.get("qty") or 0) <= 0:
            continue
        cur_snap[t] = {
            "qty": p.get("qty"),
            "avg_entry_price": p.get("avg_entry_price") or 0,
            "entry_date": (snap.get(t) or {}).get("entry_date") or prev_as_of,
        }
    pos_states = positions_from_snapshot(cur_snap, as_of)
    prices_today = closes.loc[pd.Timestamp(as_of)].dropna().to_dict()
    targets = construct_targets(convictions, pos_states, prices_today,
                                sectors, cfg, as_of=as_of,
                                regime_factor=regime_factor)
    targets = apply_position_cap(targets, MAX_POSITION_WEIGHT)
    over = [t for t, w in targets.items() if w > MAX_POSITION_WEIGHT + 1e-9]
    assert not over, f"cap breach: {over}"

    # ---- order plan ----
    plan = build_order_plan(targets, raw_positions, equity, prices_today,
                            cfg.min_trade_value)
    check_order_sanity(plan, equity, raw_positions)
    print("\n" + summarize_plan(plan, targets, equity), flush=True)

    fills: list[dict] = []
    if args.live:
        assert isinstance(broker, PaperBroker)
        for o in plan:
            kw = {"notional": o.notional} if o.side == "buy" else {"qty": o.qty}
            try:
                r = broker.submit_order(o.symbol, o.side, **kw)
                fills.append({"symbol": o.symbol, "side": o.side, **kw,
                              "order_id": r.get("id"), "status": r.get("status"),
                              "filled_qty": r.get("filled_qty"),
                              "filled_avg_price": r.get("filled_avg_price")})
                print(f"  submitted {o.side.upper()} {o.symbol} -> {r.get('status')}",
                      flush=True)
            except BrokerError as e:
                fills.append({"symbol": o.symbol, "side": o.side, **kw,
                              "error": str(e)})
                print(f"  FAILED {o.side.upper()} {o.symbol}: {e}", flush=True)
        new_positions = broker.get_positions()
    else:
        new_positions = raw_positions
        print("\nDRY-RUN: no orders submitted. Re-run with --live to execute "
              "on the paper endpoint.", flush=True)

    # ---- state ----
    run = {
        "as_of": as_of.isoformat(),
        "mode": "live" if args.live else "dry-run",
        "endpoint": PAPER_API_BASE,
        "universe": {"tickers": len(tickers),
                     "note": f"first {args.max_tickers} rows of {cfg.universe_csv}"},
        "regime": {"label": price_regime.label, "factor": regime_factor,
                   "price_factor": price_regime.factor, "macro_factor": macro_f,
                   "rationale": f"{price_regime.rationale}|macro={macro_f:.2f}"},
        "equity": equity,
        "decision_source": decision_source,
        "desk_scores": desk_scores,
        "targets": {t: round(w, 6) for t, w in targets.items()},
        "orders": [{"symbol": o.symbol, "side": o.side, "notional": o.notional,
                    "qty": o.qty, "est_value": o.est_value, "reason": o.reason}
                   for o in plan],
        "fills": fills,
        "positions": _snapshot_positions(new_positions, snap, as_of),
    }
    # Live runs own the canonical state. A dry-run must never overwrite
    # latest.json / {as_of}.json -- it once clobbered the live Sep-28
    # state with fake $100k equity and 2 stale positions.
    if args.live:
        out = state_dir / f"{as_of.isoformat()}.json"
        out.write_text(json.dumps(run, indent=2))
        (state_dir / "latest.json").write_text(json.dumps(run, indent=2))
        print(f"\nwrote {out.relative_to(ROOT)}", flush=True)
    else:
        out = state_dir / f"dryrun-{as_of.isoformat()}.json"
        out.write_text(json.dumps(run, indent=2))
        print(f"\nwrote {out.relative_to(ROOT)} (dry-run; latest.json untouched)",
              flush=True)


def _load_latest(state_dir: Path) -> dict:
    p = state_dir / "latest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _dry_run_baseline(prev: dict, closes: pd.DataFrame,
                      as_of: date) -> tuple[dict[str, dict], float]:
    """Current-account baseline for a dry-run: read-only broker reads.

    A dry-run previews the live run, so it must diff against the account
    as it stands NOW. The prior code rebuilt positions from latest.json,
    whose snapshot is the account as of the last live run's start --
    silently one rebalance stale (2026-10-10: the plan 'exited' five
    names that are sub-$1 dust in the real account and never mentioned
    six real holdings). GETs only; a dry-run never submits. Falls back
    to the prior snapshot + locked capital, loudly, when the broker is
    unreachable.
    """
    try:
        pb = PaperBroker()
        account = pb.get_account()
        positions = pb.get_positions()
        if account.get("status") == "ACTIVE" and positions:
            print(f"account: DRY-RUN baseline = live paper account "
                  f"(read-only); equity=${float(account['equity']):,.2f}, "
                  f"{len(positions)} positions", flush=True)
            return positions, float(account["equity"])
        print(f"WARNING: broker baseline unusable "
              f"(status={account.get('status')}); falling back to prior "
              f"state -- plan diffs against a STALE snapshot", flush=True)
    except Exception as e:  # broker down / keys missing: preview must go on
        print(f"WARNING: broker read failed ({e}); falling back to prior "
              f"state -- plan diffs against a STALE snapshot", flush=True)
    return _positions_from_state(prev, closes, as_of), PAPER_CAPITAL


def _positions_from_state(prev: dict, closes: pd.DataFrame,
                          as_of: date) -> dict[str, dict]:
    """Rebuild a broker-style positions view from the last saved snapshot."""
    out: dict[str, dict] = {}
    px = closes.loc[pd.Timestamp(as_of)].dropna().to_dict()
    for t, p in (prev.get("positions", {}) if prev else {}).items():
        qty = float(p.get("qty") or 0)
        if qty <= 0:
            continue
        cur = px.get(t) or float(p.get("avg_entry_price") or 0)
        out[t] = {"qty": qty, "market_value": qty * cur,
                  "avg_entry_price": float(p.get("avg_entry_price") or 0),
                  "current_price": cur}
    return out


def _snapshot_positions(current: dict[str, dict], prev_snap: dict,
                        as_of: date) -> dict[str, dict]:
    """Merge broker positions with carried entry dates for the next run."""
    snap: dict[str, dict] = {}
    for t, p in current.items():
        qty = float(p.get("qty") or 0)
        if qty <= 0:
            continue
        prev = prev_snap.get(t, {})
        snap[t] = {
            "qty": qty,
            "avg_entry_price": float(p.get("avg_entry_price") or 0),
            "entry_date": prev.get("entry_date", as_of.isoformat()),
            "current_price": float(p.get("current_price") or 0),
        }
    return snap


if __name__ == "__main__":
    main()
