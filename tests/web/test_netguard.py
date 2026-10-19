"""네트워크 가드 — 차단 대역, IPv6 속 IPv4, 옛 IPv4 축약 거부, 호스트 정책, DNS 전원 통과 규칙, 연결 고정."""
import ipaddress

import pytest

from agent_arch.services.web import netguard as G


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.1", "172.16.5.5", "10.0.1.1", "169.254.169.254", "0.0.0.0", "255.255.255.255",
    "100.64.1.1", "198.18.0.1", "100.100.100.200", "224.0.0.1", "240.1.1.1",
    "::1", "::", "fe80::1", "fc00::1", "fd00:ec2::254", "fec0::1", "ff02::1", "2001:2::1", "100::1", "2001:20::1",
    "::ffff:127.0.0.1", "::ffff:10.1.2.3", "2002:7f00:0001::", "64:ff9b::7f00:1", "64:ff9b:1::1", "::0:5efe:7f00:1", "::7f00:1",
])
def test_blocked_addresses(ip):
    assert G.is_blocked_ip(ipaddress.ip_address(ip)) is True


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "211.231.99.80", "2606:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:0808:0808::"])
def test_public_addresses(ip):
    assert G.is_blocked_ip(ipaddress.ip_address(ip)) is False


def test_legacy_ipv4_literals_are_rejected_not_resolved():
    for h in ("127.1", "0x7f.1", "2130706433", "0177.0.0.1", "127.0.0.1."):
        with pytest.raises(G.BlockedTarget):
            G.parse_ip_literal(h)
    assert G.parse_ip_literal("example.com") is None
    assert str(G.parse_ip_literal("[2606:4700::1111]")) == "2606:4700::1111"
    with pytest.raises(G.BlockedTarget):
        G.parse_ip_literal("2606:zz::1")                       # IPv6 모양인데 파싱 실패 → 열지 않는다


def test_hostname_policy_order_and_patterns():
    for h in ("localhost", "LOCALHOST.", "foo.localhost", "printer.local", "svc.internal", "metadata.google.internal"):
        with pytest.raises(G.BlockedTarget):
            G.check_hostname_policy(h)
    G.check_hostname_policy("example.com")
    with pytest.raises(G.BlockedTarget):
        G.check_hostname_policy("chat.example.com", blocked=("*.example.com",))
    G.check_hostname_policy("example.com", blocked=("*.example.com",))         # 와일드카드는 apex 를 포함하지 않는다
    with pytest.raises(G.BlockedTarget):
        G.check_hostname_policy("example.com", blocked=("example.com",))
    G.check_hostname_policy("news.example.com", allowed=("*.example.com",))
    with pytest.raises(G.BlockedTarget):
        G.check_hostname_policy("other.org", allowed=("*.example.com",))
    with pytest.raises(G.BlockedTarget):                                  # 운영자 차단이 허용목록보다 먼저
        G.check_hostname_policy("bad.example.com", blocked=("bad.example.com",), allowed=("*.example.com",))


@pytest.mark.asyncio
async def test_resolve_requires_every_address_to_pass():
    async def all_public(host):
        return ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]
    async def one_private(host):
        return ["93.184.216.34", "10.0.0.7"]
    async def nothing(host):
        return []
    assert (await G.resolve_and_validate("example.com", resolver=all_public))[0] == "93.184.216.34"   # v4 먼저
    with pytest.raises(G.BlockedTarget):
        await G.resolve_and_validate("evil.example.com", resolver=one_private)
    with pytest.raises(G.BlockedTarget):
        await G.resolve_and_validate("nx.example.com", resolver=nothing)
    assert await G.resolve_and_validate("8.8.8.8") == ["8.8.8.8"]           # 리터럴은 DNS 없이
    with pytest.raises(G.BlockedTarget):
        await G.resolve_and_validate("169.254.169.254")


@pytest.mark.asyncio
async def test_pinned_backend_only_connects_to_pinned_hosts(monkeypatch):
    seen = {}
    async def fake_connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        seen["target"] = (host, port); return "stream"
    monkeypatch.setattr(G.AutoBackend, "connect_tcp", fake_connect)
    b = G.PinnedBackend(pins={"example.com": "93.184.216.34"})
    assert await b.connect_tcp("Example.com", 443) == "stream" and seen["target"] == ("93.184.216.34", 443)
    with pytest.raises(G.BlockedTarget):
        await b.connect_tcp("other.com", 443)
