#!/usr/bin/env python3
"""Read-only web dashboard for the weekly fund (Railway worker service).

Serves a single dark-mode page: account KPIs, holdings vs targets,
the four research desks and regime, the last rebalance, recent orders,
worker schedule/status, and monitor history.

Access control: a single shared token (env DASHBOARD_TOKEN) is the
login credential. It is generated outside this codebase and stored as
a Railway variable, readable only from the owner's Railway account;
it is never printed or logged here. A correct login sets an
HMAC-signed, HttpOnly session cookie (30 days). If DASHBOARD_TOKEN is
unset the dashboard refuses to serve anything but a 503 notice.

The dashboard is strictly read-only: it only issues GETs to the
Alpaca PAPER endpoint and reads the fund's own state files. It has no
order path at all.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PT = ZoneInfo("America/Los_Angeles")
TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
SESSION_COOKIE = "fund_session"
SESSION_TTL_DAYS = 30

_cache: dict = {"at": 0.0, "data": None}
_cache_lock = threading.Lock()
_failures: dict[str, tuple[int, float]] = {}  # ip -> (count, locked_until)


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "-"


def pct(x, digits=2) -> str:
    try:
        return f"{float(x) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------- data

def _broker():
    from fund.execution.broker import PaperBroker
    return PaperBroker()


def fetch_live() -> dict:
    """Live account data, cached 30s. Never raises."""
    now = time.time()
    with _cache_lock:
        if _cache["data"] is not None and now - _cache["at"] < 30:
            return _cache["data"]
    data: dict = {"error": None}
    try:
        b = _broker()
        data["account"] = b.get_account()
        data["positions"] = b.get_positions()
        orders = b._request(
            "GET", "/v2/orders?status=all&limit=25&direction=desc") or []
        data["orders"] = orders
    except Exception as e:  # noqa: BLE001 - dashboard must render anyway
        data["error"] = str(e)[:200]
    with _cache_lock:
        _cache.update(at=now, data=data)
    return data


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def monitor_history(limit: int = 10) -> list[dict]:
    rows = []
    try:
        lines = (ROOT / "results" / "monitor.log").read_text().splitlines()
    except Exception:  # noqa: BLE001
        return rows
    for line in lines[-limit:]:
        kv = {}
        for part in line.split():
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        if "date" in kv:
            rows.append(kv)
    return rows


def desk_weights() -> list[tuple[str, float]]:
    """Parse desk weights from the mandate file (source of truth)."""
    out = []
    try:
        text = (ROOT / "mandates" / "core-4desk.yaml").read_text()
    except Exception:  # noqa: BLE001
        return out
    current = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- name:"):
            current = s.split(":", 1)[1].strip()
        elif s.startswith("weight:") and current:
            try:
                out.append((current, float(s.split(":", 1)[1].strip())))
            except ValueError:
                pass
            current = None
    return out


def next_runs(now: datetime) -> list[tuple[str, datetime]]:
    def nxt(weekday: int | None, hh: int, mm: int) -> datetime:
        cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if weekday is None:
            return cand if cand > now else cand + timedelta(days=1)
        while cand.weekday() != weekday or cand <= now:
            cand += timedelta(days=1)
        return cand
    return [
        ("Daily monitor", nxt(None, 17, 5)),
        ("Monday fill check", nxt(0, 7, 0)),
        ("Weekly rebalance (Sat)", nxt(5, 9, 0)),
    ]


# ---------------------------------------------------------------- auth

def session_value() -> str:
    return hmac.new(TOKEN.encode(), b"fund-session-v1",
                    hashlib.sha256).hexdigest()


def authed(handler) -> bool:
    if not TOKEN:
        return False
    raw = handler.headers.get("Cookie", "")
    jar = cookies.SimpleCookie()
    try:
        jar.load(raw)
    except Exception:  # noqa: BLE001
        return False
    morsel = jar.get(SESSION_COOKIE)
    return bool(morsel) and hmac.compare_digest(morsel.value, session_value())


# ---------------------------------------------------------------- html

CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0e13;color:#dbe2ec;font:14px/1.5 -apple-system,
'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
a{color:#6cb2f5}
header{padding:18px 20px 6px;display:flex;flex-wrap:wrap;gap:10px;
align-items:baseline;justify-content:space-between}
h1{font-size:19px;margin:0;font-weight:650}
.sub{color:#8b96a5;font-size:12.5px}
main{padding:10px 20px 26px;display:grid;gap:14px;max-width:1180px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:#12161e;border:1px solid #1f2733;border-radius:10px;padding:12px 14px}
.card .k{color:#8b96a5;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:19px;font-weight:650;margin-top:2px}
.pos{color:#3fb96c}.neg{color:#e5604f}
section{background:#10141b;border:1px solid #1f2733;border-radius:10px;padding:14px 16px}
section h2{font-size:14px;margin:0 0 10px;font-weight:650;color:#eef2f7}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #1a212c;white-space:nowrap}
th{color:#8b96a5;font-weight:550}
td:first-child,th:first-child{text-align:left}
.wrap{overflow-x:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;
background:#1a2b1e;color:#3fb96c;border:1px solid #265c37}
.pill.warn{background:#2b2113;color:#e5a44f;border-color:#5c471f}
.mut{color:#8b96a5}.small{font-size:12.5px}
.err{background:#2b1613;border:1px solid #5c2a1f;color:#f0a294;
padding:10px 14px;border-radius:10px}
.login{max-width:360px;margin:14vh auto;background:#12161e;border:1px solid #1f2733;
border-radius:12px;padding:22px}
.login input{width:100%;padding:10px;border-radius:8px;border:1px solid #2a3442;
background:#0b0e13;color:#dbe2ec;margin:8px 0 12px}
.login button{width:100%;padding:10px;border:0;border-radius:8px;background:#2563eb;
color:#fff;font-weight:600;cursor:pointer}
footer{color:#5d6875;font-size:12px;text-align:center;padding:0 20px 22px}
"""


def kpi(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="card"><div class="k">{esc(label)}</div>'
            f'<div class="v {cls}">{esc(value)}</div></div>')


def render_dashboard() -> str:
    now = datetime.now(PT)
    live = fetch_live()
    acct = live.get("account") or {}
    positions = live.get("positions") or {}
    orders = live.get("orders") or []
    run = read_json(ROOT / "results" / "paper_runs" / "latest.json")
    targets = run.get("targets") or {}
    regime = run.get("regime") or {}
    worker = read_json(ROOT / "results" / "worker_state.json")
    hist = monitor_history()

    equity = float(acct.get("equity") or 0)
    last_equity = float(acct.get("last_equity") or 0)
    day_pl = equity - last_equity if last_equity else 0.0
    day_cls = "pos" if day_pl >= 0 else "neg"
    invested = sum(float(p.get("market_value") or 0)
                   for p in positions.values())
    real_positions = {s: p for s, p in positions.items()
                      if abs(float(p.get("market_value") or 0)) >= 1.0}
    dust_n = len(positions) - len(real_positions)
    wow = None
    if hist:
        try:
            wow = float(hist[-1].get("wow"))
        except (TypeError, ValueError):
            wow = None

    cards = "".join([
        kpi("Equity", money(equity)),
        kpi("Day P&L", f"{money(day_pl)} ({pct(day_pl / last_equity if last_equity else 0)})", day_cls),
        kpi("Cash", money(acct.get("cash"))),
        kpi("Invested", f"{invested / equity:.1%}" if equity else "-"),
        kpi("Week over week", pct(wow) if wow is not None else "-", "pos" if (wow or 0) >= 0 else "neg"),
        kpi("Regime", f"{regime.get('label', '-')} · {float(regime.get('factor', 0)):.2f}"),
    ])

    err = (f'<div class="err">Live account read failed: {esc(live["error"])}'
           f' — showing saved state only.</div>') if live.get("error") else ""

    # holdings table
    rows = []
    for sym, p in sorted(real_positions.items(),
                         key=lambda kv: -float(kv[1].get("market_value") or 0)):
        mv = float(p.get("market_value") or 0)
        w = mv / equity if equity else 0
        tgt = targets.get(sym)
        drift = (w - float(tgt)) * 100 if tgt is not None else None
        upl = float(p.get("unrealized_pl") or 0)
        tgt_txt = f"{float(tgt):.2%}" if tgt is not None else "-"
        drift_txt = f"{drift:+.2f}pp" if drift is not None else "-"
        rows.append(
            f"<tr><td><b>{esc(sym)}</b></td>"
            f"<td>{float(p.get('qty') or 0):,.4f}</td>"
            f"<td>{money(p.get('current_price'))}</td><td>{money(mv)}</td>"
            f"<td>{w:.2%}</td><td>{tgt_txt}</td><td>{drift_txt}</td>"
            f"<td class='{'pos' if upl >= 0 else 'neg'}'>{money(upl)}</td></tr>")
    holdings = (
        '<section><h2>Holdings vs targets</h2><div class="wrap"><table>'
        '<tr><th>Symbol</th><th>Qty</th><th>Price</th><th>Value</th>'
        '<th>Weight</th><th>Target</th><th>Drift</th><th>Unreal. P&amp;L</th></tr>'
        + "".join(rows) + "</table></div>"
        + (f'<p class="small mut">+ {dust_n} sub-$1 dust leftover(s) not shown.</p>'
           if dust_n else "")
        + "</section>")

    # desks
    desk_rows = "".join(
        f"<tr><td><b>{esc(name)}</b></td><td>{w:.0%}</td></tr>"
        for name, w in desk_weights())
    desks = (
        '<section><h2>Research desks — core-4desk mandate</h2>'
        '<div class="wrap"><table><tr><th>Desk</th><th>Weight</th></tr>'
        + desk_rows + "</table></div>"
        f'<p class="small">Regime: <b>{esc(regime.get("label", "-"))}</b> '
        f'factor {float(regime.get("factor", 0)):.3f} '
        f'(price {float(regime.get("price_factor", 0)):.2f} · '
        f'macro {float(regime.get("macro_factor", 0)):.2f}) — '
        f'{esc(regime.get("rationale", ""))}. Decisions by Jev; '
        'per-ticker desk scores are not persisted yet.</p></section>')

    # last rebalance
    fills = run.get("fills") or []
    filled = sum(1 for f in fills if f.get("status") == "filled")
    accepted = sum(1 for f in fills if f.get("status") == "accepted")
    rebalance = (
        '<section><h2>Last rebalance</h2>'
        f'<p class="small">Review date <b>{esc(run.get("as_of", "-"))}</b> · '
        f'mode {esc(run.get("mode", "-"))} · equity at run {money(run.get("equity"))} · '
        f'{len(run.get("orders") or [])} orders '
        f'({accepted} accepted, {filled} filled)</p></section>')

    # recent orders
    orows = []
    for o in orders[:15]:
        amt = o.get("filled_avg_price")
        orows.append(
            f"<tr><td><b>{esc(o.get('symbol'))}</b></td>"
            f"<td>{esc(o.get('side'))}</td>"
            f"<td>{esc(o.get('qty') or o.get('notional') or '-')}</td>"
            f"<td>{esc(o.get('status'))}</td>"
            f"<td>{money(amt) if amt else '-'}</td>"
            f"<td class='mut'>{esc((o.get('submitted_at') or '')[:16].replace('T', ' '))}</td></tr>")
    orders_html = (
        '<section><h2>Recent orders (Alpaca)</h2><div class="wrap"><table>'
        '<tr><th>Symbol</th><th>Side</th><th>Qty / $</th><th>Status</th>'
        '<th>Fill px</th><th>Submitted (UTC)</th></tr>'
        + "".join(orows) + "</table></div></section>")

    # worker
    done = worker.get("done") or {}
    nxt = "".join(
        f"<tr><td>{esc(name)}</td><td>{when:%a %b %d, %I:%M %p} PT</td></tr>"
        for name, when in next_runs(now))
    last_done = "".join(
        f"<tr><td>{esc(k)}</td><td class='mut'>{esc(v)}</td></tr>"
        for k, v in sorted(done.items()))
    worker_html = (
        '<section><h2>Autonomous worker (Railway)</h2>'
        '<div class="wrap"><table><tr><th>Next runs</th><th>When</th></tr>'
        + nxt + "</table></div>"
        + ('<div class="wrap" style="margin-top:8px"><table>'
           '<tr><th>Last completed</th><th>Scheduled instant</th></tr>'
           + last_done + "</table></div>" if last_done else "")
        + "</section>")

    # monitor history
    hrows = "".join(
        f"<tr><td>{esc(r.get('date'))}</td><td>{money(r.get('equity'))}</td>"
        f"<td>{esc(r.get('dod', '-').split(':')[0])}</td>"
        f"<td>{pct(float(r['wow'])) if r.get('wow') else '-'}</td>"
        f"<td>{esc(r.get('status'))}</td></tr>" for r in reversed(hist))
    hist_html = (
        '<section><h2>Monitor history</h2><div class="wrap"><table>'
        '<tr><th>Date</th><th>Equity</th><th>Day P&amp;L $</th><th>WoW</th>'
        '<th>Account</th></tr>' + hrows + "</table></div></section>")

    status = acct.get("status", "-")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Weekly Fund</title>
<style>{CSS}</style></head><body>
<header><div><h1>Weekly Fund <span class="pill">{esc(status)}</span></h1>
<div class="sub">Alpaca paper account · read-only dashboard · updated {now:%a %b %d, %I:%M %p} PT</div></div>
<div class="sub">benchmark: SPY (mandate)</div></header>
<main>{err}<div class="cards">{cards}</div>
{holdings}{desks}{rebalance}{orders_html}
<div style="display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">
{worker_html}{hist_html}</div></main>
<footer>Paper trading only. This dashboard never places, modifies, or cancels orders.</footer>
</body></html>"""


LOGIN_HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Fund — login</title><style>{CSS}</style></head><body>
<div class="login"><h1 style="font-size:17px">Weekly Fund</h1>
<p class="sub">Private dashboard. Enter your access token.</p>
<form method="post" action="/login">
<input type="password" name="token" placeholder="Access token" autocomplete="current-password" required>
<button type="submit">Log in</button></form></div></body></html>"""


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep logs lean, no auth data
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def _send(self, code: int, body: str, headers: dict | None = None):
        raw = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not TOKEN:
            return self._send(503, "<h1>Dashboard not configured</h1>"
                                   "<p>DASHBOARD_TOKEN is not set.</p>")
        path = self.path.split("?", 1)[0]
        if path == "/login":
            return self._send(200, LOGIN_HTML)
        if path == "/healthz":
            return self._send(200, "ok")
        if not authed(self):
            return self._send(302, "", {"Location": "/login"})
        if path == "/":
            return self._send(200, render_dashboard())
        return self._send(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/login" or not TOKEN:
            return self._send(404, "not found")
        ip = self.client_address[0]
        count, locked_until = _failures.get(ip, (0, 0.0))
        if locked_until > time.time():
            return self._send(429, "Too many attempts. Try again later.")
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        supplied = (form.get("token") or [""])[0]
        if hmac.compare_digest(supplied, TOKEN):
            _failures.pop(ip, None)
            cookie_val = (
                f"{SESSION_COOKIE}={session_value()}; Path=/; "
                f"Max-Age={SESSION_TTL_DAYS * 86400}; HttpOnly; Secure; "
                "SameSite=Lax")
            return self._send(302, "", {"Location": "/",
                                        "Set-Cookie": cookie_val})
        count += 1
        locked = time.time() + 300 if count >= 8 else 0.0
        _failures[ip] = (count, locked)
        time.sleep(1)
        return self._send(401, LOGIN_HTML.replace(
            "</form>", "</form><p class='neg'>Wrong token.</p>"))


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"dashboard listening on :{port} "
          f"(token configured: {bool(TOKEN)})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
