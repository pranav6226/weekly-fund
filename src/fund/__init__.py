"""The weekly fund package.

Import-time environment repair (Railway only): the Railway container
image sets http_proxy/https_proxy env vars pointing at a proxy that
is unreachable from the running container. urllib and requests
honor those vars, so every API call hung until timeout while direct
connections succeeded in ~0.2s (root-caused live on 2026-10-13 via
the dashboard's network self-test). Entry-script and sitecustomize
scrubs were shipped first but did not reliably reach every process,
so the scrub lives here too: every fund process -- broker, monitor,
paper_trade, Jev client, data clients -- imports this package before
making HTTP calls, and the scrub runs before any of that code.

Gated on RAILWAY_ENVIRONMENT (set by the platform in every Railway
container) because the dev VM legitimately needs its egress proxy:
the same code must keep honoring proxy env there.
"""
import os

if os.environ.get("RAILWAY_ENVIRONMENT"):
    for _key in ("http_proxy", "https_proxy", "HTTP_PROXY",
                 "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(_key, None)
