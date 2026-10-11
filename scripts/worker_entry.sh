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
exec python3 scripts/autonomous_worker.py
