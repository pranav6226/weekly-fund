"""Runtime network shims for the Railway worker environment.

Two container quirks, both verified live on 2026-10-13:

1. IPv6 routes can blackhole: urllib burns its whole timeout on the
   AAAA address and Alpaca calls fail with 'urlopen error timed out'
   even though the service is up over IPv4. Sorting AF_INET results
   first restores the long-standing IPv4 behavior.

2. The container image carries proxy env vars (http_proxy /
   https_proxy / HTTP_PROXY / HTTPS_PROXY) pointing at a proxy that
   is unreachable from the running container. urllib honors them and
   every request hangs until timeout, while raw sockets, http.client
   and a no-proxy urllib opener all succeed in ~0.2s. Railway egress
   is direct and this codebase never needs a proxy, so the vars are
   scrubbed from the process environment.

Loaded via PYTHONPATH by scripts/worker_entry.sh (both the worker
and the dashboard inherit it). FUND_IPV4_FIRST=0 disables shim 1;
FUND_KEEP_PROXY=1 disables shim 2.
"""
import os
import socket

if os.environ.get("FUND_KEEP_PROXY") != "1":
    for _key in ("http_proxy", "https_proxy", "HTTP_PROXY",
                 "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(_key, None)

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(results, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


if os.environ.get("FUND_IPV4_FIRST", "1") != "0":
    socket.getaddrinfo = _ipv4_first
