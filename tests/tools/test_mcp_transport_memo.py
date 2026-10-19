"""transport 의 서버별 프로토콜 기억 — 탐침 생략과 자가치유."""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent_arch.agents.tools.mcp import transport as T

URL = "http://example.test/mcp"


@pytest.fixture(autouse=True)
def _clear_memo():
    T._protocol_memo.clear()
    yield
    T._protocol_memo.clear()


def _fake_client(protocol: str, discover):
    return SimpleNamespace(
        protocol_version=protocol,
        session=SimpleNamespace(discover_result=discover),
    )


class TestRemember:
    def test_modern_server_keeps_version_and_discover_result(self):
        discover = object()
        T._remember(URL, _fake_client("2026-07-28", discover))
        assert T._protocol_memo[URL] == ("2026-07-28", discover)

    def test_legacy_server_pins_legacy_without_discover(self):
        """2025-era 서버는 discover_result 가 없다 — 다음 연결에서 탐침을 건너뛴다."""
        T._remember(URL, _fake_client("2025-11-25", None))
        assert T._protocol_memo[URL] == (T._LEGACY, None)

    def test_forget(self):
        T._remember(URL, _fake_client("2026-07-28", object()))
        T.forget_protocol(URL)
        assert URL not in T._protocol_memo


class _FakeClientCM:
    """Client 대역 — 진입 시 mode 를 기록하고, 지시하면 실패한다."""

    calls: list = []

    def __init__(self, transport, *, mode="auto", prior_discover=None, cache=None, fail_modes=()):
        self._mode = mode
        self._fail = mode in fail_modes

    async def __aenter__(self):
        type(self).calls.append(self._mode)
        if self._fail:
            raise RuntimeError("협상 불가 — 서버가 그 프로토콜을 모른다")
        return _fake_client("2026-07-28", object())

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
class TestConnectFallback:
    async def test_uses_memo_and_skips_probe(self, monkeypatch):
        discover = object()
        T._protocol_memo[URL] = ("2026-07-28", discover)
        _FakeClientCM.calls = []
        monkeypatch.setattr(T, "Client", _FakeClientCM)
        monkeypatch.setattr(T, "streamable_http_client", lambda url, http_client=None: object())

        async with T._connect(URL, http_client=None) as client:
            assert client.protocol_version == "2026-07-28"
        assert _FakeClientCM.calls == ["2026-07-28"]  # auto 탐침 없음

    async def test_stale_memo_falls_back_to_renegotiation(self, monkeypatch):
        """서버가 되돌아가 기억이 어긋나면, 기억을 버리고 다시 협상한다."""
        T._protocol_memo[URL] = ("2026-07-28", object())
        _FakeClientCM.calls = []

        def _factory(transport, **kw):
            return _FakeClientCM(transport, fail_modes=("2026-07-28",), **kw)

        monkeypatch.setattr(T, "Client", _factory)
        monkeypatch.setattr(T, "streamable_http_client", lambda url, http_client=None: object())

        async with T._connect(URL, http_client=None) as client:
            assert client.protocol_version == "2026-07-28"
        assert _FakeClientCM.calls == ["2026-07-28", "auto"]  # 실패 후 재협상

    async def test_failure_during_call_drops_memo(self, monkeypatch):
        """호출 도중 실패면 재연결하지 않되, 기억은 버려 다음 호출이 재협상하게 한다."""
        T._protocol_memo[URL] = ("2026-07-28", object())
        _FakeClientCM.calls = []
        monkeypatch.setattr(T, "Client", _FakeClientCM)
        monkeypatch.setattr(T, "streamable_http_client", lambda url, http_client=None: object())

        with pytest.raises(ValueError):
            async with T._connect(URL, http_client=None):
                raise ValueError("도구 호출이 터졌다")

        assert URL not in T._protocol_memo
        assert _FakeClientCM.calls == ["2026-07-28"]  # 재연결 시도 없음


class TestAuthorizationHeader:
    """토큰이 비면 Authorization 을 보내지 않는다 — 'Bearer ' 는 httpx 가 Illegal header value 로 거부한다(브레이브 MCP 실측)."""

    @pytest.mark.asyncio
    async def test_empty_token_omits_authorization(self, monkeypatch):
        seen = {}

        class _FakeHTTP:
            def __init__(self, *, headers, timeout, follow_redirects):
                seen.update(headers)
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        class _CM:
            async def __aenter__(self):
                return object()
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(T.httpx2, "AsyncClient", _FakeHTTP)
        monkeypatch.setattr(T, "_connect", lambda url, http_client: _CM())
        async with T.open_mcp_session("http://brave:8080/mcp", service_token="", user_headers={"X-User-Email": "a@b.kr"}):
            pass
        assert "Authorization" not in seen and seen["X-User-Email"] == "a@b.kr"
        seen.clear()
        async with T.open_mcp_session("http://m:8765/mcp", service_token="tok"):
            pass
        assert seen == {"Authorization": "Bearer tok"}
