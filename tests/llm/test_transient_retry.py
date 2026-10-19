"""일시 오류 분류와 열기 재시도 검증.

dev 로그 실측(2026-09-03)에서 스트림 실패 141건의 성격이 갈렸다. 재시도로 살아나는 것
(연결 단절 39 · Gemini 503 31 · 429 8)과 재시도가 무의미한 것(OpenAI 결제 소진 28 ·
401 8 · 400 8). 그 턴의 사용자는 답변 대신 안내 메시지를 받았으므로, 분류를 틀리면
사용자를 더 기다리게 하거나(무의미한 재시도) 살릴 수 있는 답변을 포기한다.

6기법:
- 동등 분할: 일시 오류 / 영구 오류 / 상태 코드 없는 예외
- 경계값: 첫 시도 성공, 마지막 시도에서 성공, attempts 소진
- 오류 예측: 429 지만 결제 소진(문구로만 구분됨), 상태 코드가 response 에만 있는 예외
- 원인-결과: 영구 오류는 재시도 없이 즉시 올라온다(대기 시간 0)
- 제외: 상태전이 — 순수 함수/짧은 재시도 루프다.
"""
import pytest

from agent_arch.llm.transient import is_transient, open_with_retry


class _StatusError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status_code = status


class _ResponseError(Exception):
    """상태 코드가 response 에만 있는 SDK 예외 모양."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.response = type("R", (), {"status_code": status})()


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_transient_status_codes(status):
    assert is_transient(_StatusError("일시 오류", status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status_codes(status):
    """재시도해도 같은 답이 온다 — 재시도는 지연만 늘린다."""
    assert is_transient(_StatusError("거부", status)) is False


def test_status_from_response_attribute():
    assert is_transient(_ResponseError("일시 오류", 503)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Connection error.",
        "Request timed out",
        "The service is temporarily unavailable",
    ],
)
def test_transient_without_status_code(message):
    """연결 단절·타임아웃은 상태 코드가 없다 — 문구로 판정한다(실측 39건이 이 경로)."""
    assert is_transient(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - You have no credits remaining. Add credits to continue",
        "Error code: 429 - insufficient_quota",
        "You exceeded your current quota, please check your plan and billing details",
    ],
)
def test_quota_exhaustion_is_permanent_even_with_429(message):
    """오류 예측: 429 라도 결제 소진은 기다려서 살아나지 않는다 — 문구로만 구분된다."""
    assert is_transient(_StatusError(message, 429)) is False


def test_unknown_error_is_not_retried():
    """분류 못 하는 실패는 재시도하지 않는다 — 모르면 사용자를 기다리게 하지 않는다."""
    assert is_transient(ValueError("무슨 일인지 모름")) is False


@pytest.mark.asyncio
async def test_retries_until_success():
    calls = {"n": 0}

    async def _open():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _StatusError("503 Service Unavailable", 503)
        return "stream"

    assert await open_with_retry(_open, provider="t", attempts=3, base_delay=0) == "stream"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_raises_after_attempts_exhausted():
    """경계값: attempts 를 다 쓰면 마지막 예외를 그대로 올린다(호출부가 done(error)로 접는다)."""
    calls = {"n": 0}

    async def _open():
        calls["n"] += 1
        raise _StatusError("503", 503)

    with pytest.raises(_StatusError):
        await open_with_retry(_open, provider="t", attempts=3, base_delay=0)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_permanent_error_is_not_retried():
    """원인-결과: 영구 오류는 한 번만 시도한다 — 대기 없이 즉시 실패해야 한다."""
    calls = {"n": 0}

    async def _open():
        calls["n"] += 1
        raise _StatusError("401 Unauthorized", 401)

    with pytest.raises(_StatusError):
        await open_with_retry(_open, provider="t", attempts=3, base_delay=0)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_first_attempt_success_has_no_delay():
    calls = {"n": 0}

    async def _open():
        calls["n"] += 1
        return "ok"

    assert await open_with_retry(_open, provider="t") == "ok"
    assert calls["n"] == 1
