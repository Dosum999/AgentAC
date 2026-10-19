"""LLM 호출 실패 중 **다시 시도하면 살아나는 것**을 구분하고 재시도한다.

2026-09-03 dev 로그 실측에서 스트림 오류 141건이 이렇게 갈렸다.

    vllm  Connection error.            39   ← 자체 서빙 서버 일시 단절
    gemini 503 Service Unavailable     31   ← 일시 장애
    openai 429 no credits remaining    28   ← 결제 소진 (재시도해도 안 살아난다)
    vllm  401 Unauthorized              8   ← 키 문제 (재시도 무의미)
    gemini 429 Too Many Requests        8   ← 쿼터 (짧은 백오프면 살아난다)
    vllm  400 tool_choice / context      8   ← 요청 자체가 거부 (재시도 무의미)

그 턴의 사용자는 "죄송합니다" 안내 메시지를 받았다. 즉 이건 토큰 회계 문제가 아니라
**답변 실패**이고, 여기서 줄이는 것이 곧 사용자 체감 개선이다.

**재시도해도 안 되는 것을 재시도하지 않는다.** 401·403·404·400(파라미터/컨텍스트 초과)과
결제 소진(insufficient_quota)은 같은 요청을 다시 보내도 같은 답이 온다 — 재시도는 지연만
늘리고 사용자를 더 기다리게 한다.

**아직 아무것도 흘리지 않았을 때만 재시도한다.** 토큰을 이미 흘린 뒤 재시도하면 같은 문장이
두 번 나간다. 그 판단은 호출부(각 프로바이더 stream)가 한다 — 이 모듈은 "열기"만 다시 한다.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: 재시도로 살아나는 상태 코드. 408(timeout)·409(conflict)·429(rate limit)·5xx.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: 상태 코드가 재시도 대상이어도 **본문이 이것을 말하면 포기한다.** OpenAI 결제 소진은
#: 429 로 오지만 기다려도 복구되지 않는다(크레딧을 채워야 한다).
_PERMANENT_MARKERS = (
	"insufficient_quota",
	"no credits remaining",
	"billing",
	"exceeded your current quota",
)

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.6


def _status_of(exc: BaseException) -> int | None:
	"""예외에서 HTTP 상태 코드를 뽑는다. 프로바이더 SDK 마다 위치가 다르다."""

	for attr in ("status_code", "status", "code", "http_status"):
		value = getattr(exc, attr, None)
		if isinstance(value, int):
			return value
	response = getattr(exc, "response", None)
	status = getattr(response, "status_code", None)
	return status if isinstance(status, int) else None


def is_transient(exc: BaseException) -> bool:
	"""다시 시도하면 살아날 가능성이 있는 실패인가.

	상태 코드가 없는 예외(연결 단절·타임아웃)는 메시지로 판정한다 — openai SDK 의
	``APIConnectionError`` 는 status 가 없고 문구가 "Connection error." 다.
	"""

	text = str(exc).lower()
	if any(marker in text for marker in _PERMANENT_MARKERS):
		return False

	status = _status_of(exc)
	if status is not None:
		return status in TRANSIENT_STATUS

	return any(
		marker in text
		for marker in ("connection error", "timeout", "timed out", "temporarily unavailable")
	)


async def open_with_retry(
	open_fn: Callable[[], Awaitable[T]],
	*,
	provider: str,
	attempts: int = DEFAULT_ATTEMPTS,
	base_delay: float = DEFAULT_BASE_DELAY,
) -> T:
	"""스트림 열기를 일시 오류에 한해 지수 백오프로 다시 시도한다.

	Args:
		open_fn: 스트림(또는 응답)을 여는 코루틴 팩토리. 매 시도마다 새로 호출된다.
		provider: 로그용 프로바이더 이름.
		attempts: 총 시도 횟수(첫 시도 포함).
		base_delay: 첫 대기 시간(초). 2배씩 늘리고 ±25% 지터를 준다.

	Returns:
		``open_fn`` 의 결과.

	Raises:
		마지막 시도의 예외를 그대로 올린다 — 호출부가 기존처럼 done(error) 로 접는다.
	"""

	last_exc: BaseException | None = None
	for attempt in range(1, attempts + 1):
		try:
			return await open_fn()
		except Exception as exc:  # noqa: BLE001 — 분류는 is_transient 가 한다
			last_exc = exc
			if attempt >= attempts or not is_transient(exc):
				raise
			# 지터를 주지 않으면 동시 요청이 같은 간격으로 재시도해 서버를 다시 밀어붙인다.
			delay = base_delay * (2 ** (attempt - 1))
			delay *= 1 + random.uniform(-0.25, 0.25)
			# 재시도로 살아나도 이벤트로 남긴다 — 429·503 이 몰린 시각이 곧 "프로바이더가 흔들렸다"는
			# 근거다(429 는 overload, 타임아웃은 timeout, 그 밖은 llm_provider 로 분류된다).
			logger.warning(
				"[%s] 일시 오류 재시도 %d/%d (%.1fs 후): %s",
				provider, attempt, attempts - 1, delay, exc,
				extra=event_extra(provider=provider.split("_")[0], exc=exc),
			)
			await asyncio.sleep(delay)

	assert last_exc is not None  # 도달 불가 — 루프는 return 또는 raise 로만 끝난다
	raise last_exc
