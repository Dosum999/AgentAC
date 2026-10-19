"""시스템 오류 이벤트 수집(app.services.monitoring.error_events) 단위 테스트.

핵심 계약:
- 핸들러는 ERROR 이상 + extra 로 지정한 WARNING 만 받고, emit 은 버퍼에만 넣는다(I/O·예외 없음).
- 자기 모듈 로그는 받지 않는다(flush 실패 → 이벤트 → flush 실패 되먹임 방지).
- 버퍼는 키 수 상한이 있고, 넘친 새 키는 버리고 센다(그 수는 다음 flush 에 한 건의 이벤트로).
- 정화: 이메일·토큰·키·긴 숫자·26자 id 가림, 500자.
- fingerprint 는 계약 2장 식 그대로이고, 숫자·id 만 다른 메시지는 같은 값이다.
- 미들웨어는 원경로가 아니라 라우트 템플릿으로 기록한다.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import httpx
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent_arch.services.monitoring import error_events as ee


@pytest.fixture(autouse=True)
def _fresh_buffer(monkeypatch):
    monkeypatch.setattr(ee, "_buffer", ee._EventBuffer())
    monkeypatch.setattr("agent_arch.core.config.settings.enable_error_events", True, raising=False)
    yield


def _items():
    return list(ee._buffer._items.values())


def _logger(name: str = "agent_arch.agents.orchestrator") -> logging.Logger:
    lg = logging.getLogger(name)
    lg.handlers = [ee.ErrorEventHandler()]
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg


# ── 정화 ───────────────────────────────────────────────────────────────────
class TestSanitize:
    def test_masks_email_bearer_key_and_long_numbers(self):
        out = ee.sanitize(
            "user=user@example.com Authorization: Bearer abcdefghijklmnop api_key=sk-XYZ "
            "주문 20260930123 전화 010-1234-5678"
        )
        assert "user@example.com" not in out and "<email>" in out
        assert "abcdefghijklmnop" not in out
        assert "sk-XYZ" not in out
        assert "20260930123" not in out
        assert "010-1234-5678" not in out

    def test_masks_frappe_token_header_and_query_string(self):
        out = ee.sanitize("POST https://cm.test/api/method/x?token=abc&email=a%40b → token key123456:secret789")
        assert "abc&email" not in out and "?<query>" in out
        assert "key123456:secret789" not in out

    def test_masks_26_char_ids_and_uuid(self):
        post_id = "8x3kq9w1ab2cd3ef4gh5ij6kl7"
        assert len(post_id) == 26
        out = ee.sanitize(f"post {post_id} conv 0f9c1e64-0000-4000-8000-000000000001")
        assert post_id not in out
        assert "0f9c1e64" not in out

    def test_truncates_to_500_and_collapses_whitespace(self):
        out = ee.sanitize("가\n\n나 " + "x" * 1000)
        assert len(out) == 500
        assert out.startswith("가 나 ")

    def test_none_is_empty(self):
        assert ee.sanitize(None) == ""


# ── fingerprint ────────────────────────────────────────────────────────────
class TestFingerprint:
    def test_formula_matches_contract(self):
        parts = ["mcp_tool", "agent_arch.agents.tools.mcp.remote_tool", "httpx.ReadTimeout", "", "", "",
                 "mattermost", ee.normalize("호출 실패 12")]
        expected = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
        got = ee.fingerprint("mcp_tool", "agent_arch.agents.tools.mcp.remote_tool", "httpx.ReadTimeout",
                             None, None, None, "mattermost", "호출 실패 12")
        assert got == expected
        assert len(got) == 16 and int(got, 16) >= 0

    def test_numbers_only_difference_gives_same_fingerprint(self):
        a = ee.fingerprint("exception", "agent_arch.x", message="offset=500 실패 (3건)")
        b = ee.fingerprint("exception", "agent_arch.x", message="offset=1000 실패 (27건)")
        assert a == b

    def test_ids_and_quoted_strings_are_normalized(self):
        a = ee.fingerprint("exception", "agent_arch.x", message="tool 'hr_leave' not found 0f9c1e64-0000-4000-8000-000000000001")
        b = ee.fingerprint("exception", "agent_arch.x", message="tool 'drive_list' not found 11111111-2222-4333-8444-555555555555")
        assert a == b

    def test_different_category_or_source_differs(self):
        base = ee.fingerprint("exception", "agent_arch.x", message="m")
        assert ee.fingerprint("timeout", "agent_arch.x", message="m") != base
        assert ee.fingerprint("exception", "agent_arch.y", message="m") != base

    def test_normalize_keeps_case_and_cuts_120(self):
        assert ee.normalize("ReadTimeout AbC") == "ReadTimeout AbC"
        assert len(ee.normalize("a" * 300)) == 120


# ── 분류 ───────────────────────────────────────────────────────────────────
class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class TestClassify:
    def test_timeout_types(self):
        assert ee.classify_exception(asyncio.TimeoutError()) == "timeout"
        import httpx
        assert ee.classify_exception(httpx.ReadTimeout("x")) == "timeout"

    def test_429_is_overload(self):
        assert ee.classify_exception(_StatusError(429)) == "overload"
        assert ee.classify_exception(RuntimeError("vLLM API error: 429 Too Many Requests")) == "overload"

    def test_pool_exhaustion_is_overload(self):
        class TooManyConnectionsError(Exception):
            pass
        assert ee.classify_exception(TooManyConnectionsError("sorry, too many clients")) == "overload"

    def test_plain_error_is_none(self):
        assert ee.classify_exception(ValueError("bad")) is None
        assert ee.classify_exception(_StatusError(503)) is None

    def test_provider_error_resolves_to_llm_provider_unless_429(self):
        lg = _logger("agent_arch.llm.gemini")
        try:
            raise _StatusError(503)
        except Exception as e:
            lg.exception("[gemini_stream] error: %s", e, extra=ee.event_extra(provider="gemini"))
        try:
            raise _StatusError(429)
        except Exception as e:
            lg.exception("[gemini_stream] error: %s", e, extra=ee.event_extra(provider="gemini"))
        cats = sorted(a.category for a in _items())
        assert cats == ["llm_provider", "overload"]
        assert all(a.provider == "gemini" for a in _items())

    def test_mcp_timeout_stays_mcp_tool(self):
        import httpx
        ee.record_event(category="mcp_tool", source="s", exc=httpx.ReadTimeout("t"), mcp_server="hr")
        (a,) = _items()
        assert a.category == "mcp_tool" and a.mcp_server == "hr" and a.exc_type == "httpx.ReadTimeout"

    def test_unknown_category_folds_to_exception(self):
        ee.record_event(category="weird", source="s", message="m")
        assert _items()[0].category == "exception"


# ── 핸들러 ─────────────────────────────────────────────────────────────────
class TestHandler:
    def test_error_is_captured_plain_warning_is_not(self):
        lg = _logger()
        lg.warning("재시도 중")
        lg.info("info")
        lg.error("터졌다 id=%d", 1234567)
        items = _items()
        assert len(items) == 1
        a = items[0]
        assert (a.level, a.category, a.source) == ("ERROR", "exception", "agent_arch.agents.orchestrator")
        assert a.message == "터졌다 id=…"  # 인자는 끼워 넣지 않는다(템플릿)

    def test_tagged_warning_is_captured(self):
        lg = _logger("agent_arch.agents.tools.mcp.remote_tool")
        lg.warning("[mcp] 도구 호출 실패", extra=ee.event_extra("mcp_tool", mcp_server="mattermost"))
        (a,) = _items()
        assert (a.level, a.category, a.mcp_server) == ("WARNING", "mcp_tool", "mattermost")

    def test_extra_false_opts_out(self):
        _logger().error("이미 다른 경로가 기록", extra={ee.EXTRA_KEY: False})
        assert _items() == []

    def test_exc_info_sets_exc_type(self):
        lg = _logger()
        try:
            raise KeyError("k")
        except KeyError:
            lg.exception("[run] EXCEPTION")
        assert _items()[0].exc_type == "KeyError"

    def test_message_override_hides_user_text(self):
        lg = _logger("agent_arch.services.memory_v2.writer")
        lg.error("atom 쓰기 실패: %s", "내 비밀 질문 원문",
                 extra=ee.event_extra("background", message="memory_v2 atom 쓰기 실패"))
        (a,) = _items()
        assert "비밀" not in a.message and a.category == "background"

    def test_same_error_aggregates_in_one_bucket(self):
        lg = _logger()
        for i in range(5):
            lg.error("offset=%d 실패", i * 100)
        items = _items()
        assert len(items) == 1 and items[0].count == 5

    def test_self_module_logs_are_ignored(self):
        """flush 실패 로그가 다시 이벤트가 되면 DB 장애 동안 버퍼가 자기 로그로 찬다."""
        _logger("agent_arch.services.monitoring.error_events").error("flush 실패")
        _logger("agent_arch.services.common_app.error_events_push").error("푸시 주기 실패")
        assert _items() == []

    def test_reentrant_logging_inside_capture_is_dropped(self, monkeypatch):
        """기록 도중 다시 로그가 나도(정화 중 경고 등) 무한 재귀하지 않는다."""
        lg = _logger()
        calls = {"n": 0}
        real = ee.sanitize

        def noisy(text):
            calls["n"] += 1
            lg.error("재귀 로그")  # 이 로그는 재진입 가드에 막혀야 한다
            return real(text)

        monkeypatch.setattr(ee, "sanitize", noisy)
        lg.error("원래 오류")
        assert calls["n"] == 1
        assert [a.message for a in _items()] == ["원래 오류"]

    def test_emit_never_raises(self, monkeypatch):
        """버퍼가 터져도 로깅 호출은 조용히 넘어간다 — 요청을 깨뜨리지 않는다."""
        monkeypatch.setattr(ee._buffer, "add", MagicMock(side_effect=RuntimeError("boom")))
        _logger().error("x")  # 예외가 나오면 테스트 실패

    def test_emit_does_no_io_and_returns_fast(self):
        """emit 은 I/O 를 하지 않는다 — DB 풀이 없어도(=flush 불가) 즉시 돌아온다."""
        import time
        lg = _logger()
        t0 = time.perf_counter()
        for i in range(2000):
            lg.error("반복 오류 %d", i)
        assert time.perf_counter() - t0 < 2.0

    def test_disabled_flag_stops_capture(self, monkeypatch):
        from agent_arch.core.config import settings
        monkeypatch.setattr(settings, "enable_error_events", False, raising=False)
        _logger().error("x")
        ee.record_event(category="exception", source="s")
        assert _items() == []


# ── 버퍼 상한 ──────────────────────────────────────────────────────────────
class TestBufferCap:
    def test_new_keys_beyond_cap_are_dropped_and_counted(self, monkeypatch):
        monkeypatch.setattr(ee, "_buffer", ee._EventBuffer(max_keys=3))
        for i in range(5):
            ee.record_event(category="exception", source=f"src{i}", message="m")
        assert len(ee._buffer) == 3
        assert ee._buffer.dropped == 2
        # 이미 있는 키는 상한이어도 계속 센다.
        ee.record_event(category="exception", source="src0", message="m")
        assert ee._buffer.dropped == 2
        assert max(a.count for a in _items()) == 2

    @pytest.mark.asyncio
    async def test_drop_count_becomes_one_event_on_flush(self, monkeypatch):
        monkeypatch.setattr(ee, "_buffer", ee._EventBuffer(max_keys=1))
        ee.record_event(category="exception", source="a", message="m")
        ee.record_event(category="exception", source="b", message="m")
        pool = MagicMock()
        pool.executemany = AsyncMock()
        from agent_arch.db.connection import db_pool
        monkeypatch.setattr(db_pool, "get_pool", lambda: pool)

        assert await ee.flush_buffer() == 2
        rows = pool.executemany.await_args.args[1]
        dropped = [r for r in rows if r[7] == "overload"]
        assert len(dropped) == 1 and dropped[0][2] == 1  # count=버린 건수


# ── flush ──────────────────────────────────────────────────────────────────
class TestFlush:
    @pytest.mark.asyncio
    async def test_upserts_naive_utc_rows(self, monkeypatch):
        when = datetime(2026, 10, 1, 3, 4, 59, tzinfo=timezone.utc)
        ee._capture(category="timeout", source="s", message="m", when=when)
        pool = MagicMock()
        pool.executemany = AsyncMock()
        from agent_arch.db.connection import db_pool
        monkeypatch.setattr(db_pool, "get_pool", lambda: pool)

        assert await ee.flush_buffer() == 1
        sql, rows = pool.executemany.await_args.args
        assert "ON CONFLICT (fingerprint, bucket_at) DO UPDATE" in sql
        assert "count           = agent.error_events.count + EXCLUDED.count" in sql
        row = rows[0]
        assert row[0] == datetime(2026, 10, 1, 3, 4)  # 분 버킷, naive UTC
        assert row[0].tzinfo is None
        assert len(ee._buffer) == 0

    @pytest.mark.asyncio
    async def test_failed_flush_restores_buffer(self, monkeypatch):
        ee.record_event(category="exception", source="s", message="m")
        pool = MagicMock()
        pool.executemany = AsyncMock(side_effect=RuntimeError("db down"))
        from agent_arch.db.connection import db_pool
        monkeypatch.setattr(db_pool, "get_pool", lambda: pool)

        assert await ee.flush_buffer() == 0
        assert len(ee._buffer) == 1  # 다음 주기 재시도

    @pytest.mark.asyncio
    async def test_no_pool_keeps_buffer(self, monkeypatch):
        ee.record_event(category="exception", source="s", message="m")
        from agent_arch.db.connection import db_pool
        monkeypatch.setattr(db_pool, "get_pool", lambda: None)
        assert await ee.flush_buffer() == 0
        assert len(ee._buffer) == 1


# ── 미들웨어 ───────────────────────────────────────────────────────────────
def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ee.ErrorEventsMiddleware)

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conv(conversation_id: str):
        raise HTTPException(status_code=404, detail="없음")

    @app.get("/api/v1/private")
    async def private():
        raise HTTPException(status_code=401, detail="no")

    @app.get("/api/v1/boom")
    async def boom():
        raise RuntimeError("unhandled")

    @app.get("/api/v1/ok")
    async def ok():
        return {"ok": True}

    @app.get("/api/v1/health")
    async def health():
        raise HTTPException(status_code=503)

    return app


class TestMiddleware:
    def test_records_route_template_not_raw_path(self):
        client = TestClient(_app())
        client.get("/api/v1/conversations/0f9c1e64-0000-4000-8000-000000000001?email=a@b.com")
        client.get("/api/v1/conversations/another-id")
        items = _items()
        assert len(items) == 1  # 두 요청이 한 그룹
        a = items[0]
        assert a.route == "/api/v1/conversations/{conversation_id}"
        assert (a.category, a.status_code, a.count) == ("http_4xx", 404, 2)
        assert "a@b.com" not in a.message and "0f9c1e64" not in a.message

    def test_auth_status_is_auth_category(self):
        TestClient(_app()).get("/api/v1/private")
        assert _items()[0].category == "auth"

    def test_unmatched_route(self):
        TestClient(_app()).get("/wp-admin/setup.php")
        a = _items()[0]
        assert a.route == "(unmatched)" and a.status_code == 404

    def test_unhandled_exception_is_http_5xx(self):
        TestClient(_app(), raise_server_exceptions=False).get("/api/v1/boom")
        a = _items()[0]
        assert (a.category, a.status_code, a.route, a.exc_type) == (
            "http_5xx", 500, "/api/v1/boom", "RuntimeError",
        )

    def test_success_and_health_are_not_recorded(self):
        client = TestClient(_app())
        client.get("/api/v1/ok")
        client.get("/api/v1/health")
        assert _items() == []

    def test_log_inside_request_carries_route(self):
        app = FastAPI()
        app.add_middleware(ee.ErrorEventsMiddleware)
        lg = _logger("agent_arch.api.v1.agent")

        @app.get("/api/v1/items/{item_id}")
        async def item(item_id: str):
            lg.error("핸들러 안 오류")
            return {}

        TestClient(app).get("/api/v1/items/42")
        assert _items()[0].route == "/api/v1/items/{item_id}"


# ── 백그라운드 태스크 ──────────────────────────────────────────────────────
class TestWatchTask:
    @pytest.mark.asyncio
    async def test_crashed_task_records_background(self):
        async def crash():
            raise ValueError("죽음")

        t = ee.watch_task(asyncio.create_task(crash()), "demo")
        with pytest.raises(ValueError):
            await t
        await asyncio.sleep(0)  # done 콜백
        (a,) = _items()
        assert (a.category, a.source, a.exc_type) == ("background", "lifespan:demo", "ValueError")

    @pytest.mark.asyncio
    async def test_cancelled_task_is_silent_and_forever_exit_is_recorded(self):
        async def forever():
            await asyncio.sleep(3600)

        async def quits():
            return None

        t1 = ee.watch_task(asyncio.create_task(forever()), "loop", forever=True)
        t2 = ee.watch_task(asyncio.create_task(quits()), "quitter", forever=True)
        t1.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
        await asyncio.sleep(0)
        items = _items()
        assert [a.source for a in items] == ["lifespan:quitter"]
        assert items[0].level == "WARNING"

    def test_none_passthrough(self):
        assert ee.watch_task(None, "x") is None


# ── 기록 지점: 프로바이더 일시 오류 재시도 ─────────────────────────────────
class TestTransientRetryEvent:
    @pytest.mark.asyncio
    async def test_429_retry_is_overload_event_even_when_it_recovers(self, monkeypatch):
        """재시도로 살아나도 429 가 몰린 시각은 '프로바이더가 흔들렸다'는 근거로 남긴다."""
        from agent_arch.llm import transient

        monkeypatch.setattr(transient.asyncio, "sleep", AsyncMock())
        lg = logging.getLogger("agent_arch.llm.transient")
        handler = ee.ErrorEventHandler()
        lg.addHandler(handler)
        calls = {"n": 0}

        async def open_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _StatusError(429)
            return "stream"

        try:
            assert await transient.open_with_retry(open_fn, provider="gemini_stream") == "stream"
        finally:
            lg.removeHandler(handler)
        (a,) = _items()
        assert (a.category, a.provider, a.level) == ("overload", "gemini", "WARNING")


def test_field_lengths_fit_common_app_contract():
    """common_app ErrorEvent 스키마 상한을 넘으면 배치 500건 전체가 400 으로 거절되고 진행선이 멈춘다."""
    ee.record_event(
        category="http_4xx",
        source="s" * 300,
        route="/r" * 200,
        exc_type="E" * 300,
        provider="p" * 100,
        mcp_server="m" * 100,
        message="x" * 2000,
        status_code=404,
    )
    (item,) = _items()
    assert len(item.source) <= 140
    assert len(item.route) <= 140
    assert len(item.exc_type) <= 140
    assert len(item.provider) <= 64
    assert len(item.mcp_server) <= 64
    assert len(item.message) <= 500


class TestNoUserTextInEvents:
    """자동 수집 경로가 로그 인자·예외 원문으로 사용자 문장을 실어 보내지 않는다(계약 2장)."""

    def test_log_args_are_not_interpolated(self):
        _logger().error("[stream] EXCEPTION: %s", "사용자가 입력한 비밀 질문")
        (item,) = _items()
        assert "비밀 질문" not in item.message
        assert item.message.startswith("[stream] EXCEPTION:")

    def test_plain_exception_text_is_dropped(self):
        try:
            raise ValueError("사용자가 입력한 비밀 질문")
        except ValueError:
            _logger().exception("턴 처리 실패")
        (item,) = _items()
        assert "비밀 질문" not in item.message
        assert item.exc_type == "ValueError"

    def test_record_event_drops_plain_exception_text(self):
        ee.record_event(category="exception", source="s", exc=RuntimeError("invalid credential abc123"))
        (item,) = _items()
        assert "abc123" not in (item.message or "")

    def test_sdk_exception_text_is_kept(self):
        try:
            raise httpx.ConnectError("connection refused")
        except httpx.ConnectError:
            _logger().exception("호출 실패 %s", "x")
        (item,) = _items()
        assert "connection refused" in item.message

    def test_same_template_different_args_one_fingerprint(self):
        lg = _logger()
        lg.error("실패 %s", "질문 A")
        lg.error("실패 %s", "전혀 다른 질문 B")
        assert len(_items()) == 1


def test_wrapped_llm_sdk_error_is_llm_provider():
    class FakeSdkError(Exception):
        pass

    FakeSdkError.__module__ = "openai._exceptions"
    try:
        try:
            raise FakeSdkError("server error")
        except FakeSdkError as inner:
            raise RuntimeError("invoke 실패") from inner
    except RuntimeError as e:
        ee.record_event(category="exception", source="agent_arch.agents.orchestrator", exc=e)
    (item,) = _items()
    assert item.category == "llm_provider"
