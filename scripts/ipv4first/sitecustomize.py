"""IPv4-first name resolution for the Railway worker runtime.

Railway containers can resolve hosts to IPv6 addresses whose routes
blackhole: urllib then burns its whole timeout on the AAAA address
and the Alpaca calls fail with 'urlopen error timed out' even though
the service is up (verified 2026-10-13: first Railway boot, every
Alpaca GET timed out at 30s while the same calls succeeded over IPv4
elsewhere). Sorting AF_INET results first restores the long-standing
IPv4 behavior. Loaded via PYTHONPATH by scripts/worker_entry.sh;
set FUND_IPV4_FIRST=0 to disable.
"""
import socket

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(results, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


socket.getaddrinfo = _ipv4_first
