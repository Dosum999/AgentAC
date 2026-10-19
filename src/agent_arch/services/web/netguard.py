"""아웃바운드 네트워크 가드 — SSRF 차단, 호스트 정책, DNS 해석 + 연결 고정(pinning).

모델이 고른 URL 을 우리가 대신 여는 순간, 이 프로세스는 사내망 안의 프록시가 된다. 그래서
(1) 사설·루프백·링크로컬·멀티캐스트·예약·CGNAT·벤치마크·클라우드 메타데이터 대역을 막고,
(2) IPv6 안에 박힌 IPv4(v4-mapped, 6to4, Teredo, NAT64, ISATAP, v4-compatible)를 풀어 같은 규칙에 넣고,
(3) "127.1"·"0x7f.1"·"2130706433" 같은 옛 IPv4 축약은 해석하지 않고 **막는다**(getaddrinfo 는 이걸 받아준다),
(4) 호스트명은 운영자 차단목록 → 허용목록 → 내장 차단(localhost·*.local·*.internal·metadata) 순으로 보고,
(5) DNS 로 얻은 주소는 **전부** 통과해야 하며, 그중 하나로 연결을 고정해 해석과 연결 사이의 재바인딩을 막는다.

파싱이 안 되는 것은 열어 두지 않고 막는다(fail-closed).
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from httpcore._backends.auto import AutoBackend

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class BlockedTarget(Exception):
    """정책상 열 수 없는 대상. message 는 모델에게 그대로 보여도 되는 한국어 안내."""


class UnresolvableHost(BlockedTarget):
    """DNS 해석 실패 — 정책 차단이 아니라 죽은 링크·오타일 확률이 높다. 모델이 다른 결과로 넘어가게 코드를 구분한다."""


# ─── IP 분류 ─────────────────────────────────────────────────────────────────

_V4_EXTRA_BLOCKED = (
    ipaddress.ip_network("100.64.0.0/10"),      # CGNAT(공유 주소 공간) — is_private 가 False
    ipaddress.ip_network("198.18.0.0/15"),      # RFC 2544 벤치마크
    ipaddress.ip_network("192.0.0.0/24"),       # IETF 프로토콜 할당
    ipaddress.ip_network("240.0.0.0/4"),        # 예약(클래스 E)
)
_V4_METADATA = {ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("100.100.100.200")}
_V6_EXTRA_BLOCKED = (
    ipaddress.ip_network("fec0::/10"),          # 사이트로컬(폐기, ipaddress 가 private 로 안 봄)
    ipaddress.ip_network("2001:2::/48"),        # 벤치마크
    ipaddress.ip_network("100::/64"),           # discard
    ipaddress.ip_network("2001:20::/28"),       # ORCHIDv2
    ipaddress.ip_network("64:ff9b:1::/48"),     # 로컬용 NAT64 — 배포별 프리픽스 길이를 몰라 통째로 차단
)
_V6_METADATA = {ipaddress.ip_address("fd00:ec2::254")}
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """IPv6 안에 실린 IPv4 를 꺼낸다. 없으면 None."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]                      # (server, client) — client 가 실제 단말
    packed = ip.packed
    if ip in _NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(packed[12:16])
    hextets = [int.from_bytes(packed[i:i + 2], "big") for i in range(0, 16, 2)]
    if (hextets[4] & 0xFCFF) == 0 and hextets[5] == 0x5EFE:   # ISATAP …:0[2]00:5efe:w.x.y.z
        return ipaddress.IPv4Address(packed[12:16])
    if packed[:12] == b"\x00" * 12 and not ip.is_unspecified and not ip.is_loopback:   # v4-compatible ::a.b.c.d
        return ipaddress.IPv4Address(packed[12:16])
    return None


def is_blocked_ip(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if any(ip in n for n in _V6_EXTRA_BLOCKED) or ip in _V6_METADATA:
            return True
        # 전이 주소(v4-mapped·6to4·Teredo·NAT64·ISATAP·v4-compatible)는 **안에 실린 IPv4 로만** 판정한다.
        # ipaddress 는 ::ffff:0:0/96·2002::/16·64:ff9b::/96 대역 전체를 is_private 로 보기 때문에,
        # 먼저 풀지 않으면 ::ffff:8.8.8.8 같은 공개 주소까지 막힌다.
        inner = _embedded_ipv4(ip)
        if inner is not None:
            return is_blocked_ip(inner)
        return (ip.is_unspecified or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_private)
    return (ip.is_unspecified or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_private or ip == ipaddress.IPv4Address("255.255.255.255")
            or ip in _V4_METADATA or any(ip in n for n in _V4_EXTRA_BLOCKED))


_LEGACY_V4_RE = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){0,3}\.?$", re.IGNORECASE)


def parse_ip_literal(host: str) -> Optional[IPAddress]:
    """정식 IP 리터럴이면 주소를, 호스트명이면 None. 옛 IPv4 축약형은 BlockedTarget."""
    h = (host or "").strip().strip("[]")
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        pass
    if _LEGACY_V4_RE.match(h):
        raise BlockedTarget(f"지원하지 않는 IP 표기입니다: {h}. 표준 점-십진 IPv4 나 호스트명을 쓰세요.")
    if ":" in h:                                     # IPv6 처럼 보이는데 파싱 실패 → 열지 않는다
        raise BlockedTarget(f"해석할 수 없는 IPv6 주소입니다: {h}")
    return None


# ─── 호스트명 정책 ────────────────────────────────────────────────────────────

_BUILTIN_BLOCKED_HOSTS = frozenset({"localhost", "localhost.localdomain", "metadata.google.internal"})
_BUILTIN_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")


def _matches(host: str, pattern: str) -> bool:
    """`*.example.com` 은 하위 도메인만(apex 제외), 그 외는 정확 일치."""
    p = pattern.strip().lower().lstrip(".")
    if not p:
        return False
    if p.startswith("*."):
        return host.endswith(p[1:]) and host != p[2:]
    return host == p


def check_hostname_policy(host: str, *, blocked: Iterable[str] = (), allowed: Iterable[str] = ()) -> None:
    """운영자 차단 → 허용목록 → 내장 차단. 위반이면 BlockedTarget."""
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        raise BlockedTarget("URL 에 호스트가 없습니다.")
    for pat in blocked:
        if _matches(h, pat):
            raise BlockedTarget(f"차단된 호스트입니다(운영 정책): {h}. 다른 도메인의 URL 을 쓰세요.")
    allow = [a for a in allowed if a and a.strip()]
    if allow and not any(_matches(h, a) for a in allow):
        raise BlockedTarget(f"허용 목록에 없는 호스트입니다: {h}.")
    if h in _BUILTIN_BLOCKED_HOSTS or any(h.endswith(s) for s in _BUILTIN_BLOCKED_SUFFIXES):
        raise BlockedTarget(f"내부 주소는 열 수 없습니다: {h}.")


# ─── DNS 해석 + 검증 ──────────────────────────────────────────────────────────

async def _default_resolve(host: str) -> list[str]:
    import dns.asyncresolver
    import dns.exception

    out: list[str] = []
    for rtype in ("A", "AAAA"):
        try:
            ans = await dns.asyncresolver.resolve(host, rtype, lifetime=5.0, raise_on_no_answer=False)
        except dns.exception.DNSException:
            continue
        out.extend(str(r) for r in (ans.rrset or []))
    return out


async def resolve_and_validate(host: str, *, resolver=None) -> list[str]:
    """호스트를 주소 목록으로. 리터럴이면 그대로 검증, 아니면 DNS. **하나라도** 막힌 주소면 전체 거부."""
    literal = parse_ip_literal(host)
    if literal is not None:
        if is_blocked_ip(literal):
            raise BlockedTarget(f"사설·내부망 주소는 열 수 없습니다: {literal}")
        return [str(literal)]
    addrs = await (resolver or _default_resolve)(host)
    if not addrs:
        raise UnresolvableHost(f"주소를 찾을 수 없습니다(DNS 해석 실패): {host}. 링크가 죽었거나 주소가 틀렸을 수 있으니 다른 결과를 읽어 보세요.")
    parsed: list[IPAddress] = []
    for a in addrs:
        try:
            parsed.append(ipaddress.ip_address(a))
        except ValueError as e:                      # 해석기가 이상한 값을 주면 열지 않는다
            raise BlockedTarget(f"해석 결과를 검증할 수 없습니다: {host}") from e
    blocked = [str(ip) for ip in parsed if is_blocked_ip(ip)]
    if blocked:
        raise BlockedTarget(f"'{host}' 이(가) 내부망 주소({blocked[0]})로 해석돼 열지 않습니다.")
    # IPv4 를 앞에 — 컨테이너 대다수가 v6 경로가 없다
    return [str(ip) for ip in sorted(parsed, key=lambda ip: ip.version)]


# ─── 연결 고정 백엔드 ─────────────────────────────────────────────────────────

@dataclass
class PinnedBackend(AutoBackend):
    """httpcore 백엔드 — 검증한 주소로만 TCP 를 연다. TLS 의 SNI/인증서 검증은 httpcore 가 원래
    호스트명으로 따로 하므로(start_tls server_hostname) 인증서 불일치가 나지 않는다."""
    pins: dict[str, str] = field(default_factory=dict)

    async def connect_tcp(self, host: str, port: int, timeout: Optional[float] = None,
                          local_address: Optional[str] = None, socket_options=None):
        target = self.pins.get(host.lower(), None)
        if target is None:                            # 고정 안 된 호스트는 열지 않는다(fail-closed)
            raise BlockedTarget(f"검증되지 않은 호스트로의 연결은 막습니다: {host}")
        return await super().connect_tcp(target, port, timeout=timeout, local_address=local_address,
                                         socket_options=socket_options)
