#!/usr/bin/env bash
# Railway entry point for the weekly-fund worker.
#
# Railway's container filesystem is ephemeral except for the volume
# mounted at FUND_VOLUME_DIR (default /data). The fund keeps two
# things that must survive restarts:
#   data/     the SQLite price panel (market.db) + universe CSVs
#   results/  run state (latest.json), Jev cache, worker state
# Both are symlinked onto the volume; the CSV seeds come from the
# image on first boot. Without a volume (local runs) the repo's own
# directories are used untouched.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VOL="${FUND_VOLUME_DIR:-/data}"

if [ -d "$VOL" ]; then
  mkdir -p "$VOL/appdata" "$VOL/results"
  for f in "$APP_DIR"/data/*.csv; do
    base="$(basename "$f")"
    [ -f "$VOL/appdata/$base" ] || cp "$f" "$VOL/appdata/$base"
  done
  rm -rf "$APP_DIR/data" "$APP_DIR/results"
  ln -s "$VOL/appdata" "$APP_DIR/data"
  ln -s "$VOL/results" "$APP_DIR/results"
  echo "[worker_entry] data/ and results/ linked to volume at $VOL"
else
  echo "[worker_entry] no volume at $VOL -- local ephemeral filesystem"
fi

cd "$APP_DIR"
# Scrub proxy env vars: the container image carries http(s)_proxy
# values pointing at a proxy that is unreachable from the running
# container, and urllib honors them -- every API call then hangs
# until timeout while direct connections succeed (verified
# 2026-10-13). Railway egress is direct; nothing here needs a proxy.
# (scripts/ipv4first/sitecustomize.py scrubs again in-process.)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
# Prefer IPv4 for outbound connects (Railway's IPv6 routes can
# blackhole; see scripts/ipv4first/sitecustomize.py). Opt out with
# FUND_IPV4_FIRST=0.
if [ "${FUND_IPV4_FIRST:-1}" != "0" ]; then
  export PYTHONPATH="$APP_DIR/scripts/ipv4first${PYTHONPATH:+:$PYTHONPATH}"
fi
# Optional read-only dashboard (same container, shares the volume).
# Enabled only when DASHBOARD_ENABLED=1; login token comes from the
# DASHBOARD_TOKEN variable. A supervisor loop restarts it if it dies;
# the trading worker below is unaffected either way.
if [ "${DASHBOARD_ENABLED:-0}" = "1" ]; then
  ( while true; do python3 scripts/dashboard.py; sleep 5; done ) &
  echo "[worker_entry] dashboard started (port ${PORT:-8080})"
fi
exec python3 scripts/autonomous_worker.py
