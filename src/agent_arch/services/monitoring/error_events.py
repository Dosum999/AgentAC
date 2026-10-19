"""에이전트 시스템 오류 이벤트 수집 — 로그·HTTP 응답·백그라운드 태스크 → agent.error_events.

## 왜

에이전트의 시스템 오류는 지금까지 컨테이너 로그에만 있었다. 답변 실패 턴(agent.messages 의
finish_reason·status_code)은 리뷰 푸시로 common_app 에 가지만, "그 시각에 LLM 프로바이더가
503 을 냈다", "MCP 서버가 죽어 있었다" 같은 **원인 쪽 신호**는 서버에 들어가 grep 해야만
보였다. 관측 「에이전트 모니터링」 화면이 두 축을 같은 시각에 놓고 보도록, 오류를 분 단위로 모아
이 표에 쌓고 common_app 으로 푸시한다(app/services/common_app/error_events_push.py).
계약: 레포 간 계약 문서 2장(event category·fingerprint·payload).

## 수집 경로 세 갈래

1. ``ErrorEventHandler`` — logging 핸들러. ERROR 이상은 전부, WARNING 은 호출부가
   ``extra=event_extra(...)`` 로 **명시한 것만** 받는다(WARNING 은 재시도·가드 등 정상 흐름
   로그가 많아 전부 받으면 소음이 신호를 덮는다).
2. ``record_event()`` — 로그 한 줄로 표현되지 않는 지점(HTTP 응답 코드, 태스크 사망)의 명시 기록.
3. ``ErrorEventsMiddleware`` — 4xx/5xx 응답을 **라우트 템플릿**으로 기록.

## 핫패스에 얹지 않는다

emit·record_event 는 **I/O 를 하지 않는다.** 메모리 버퍼(fingerprint·분 버킷으로 합산하는
dict)에 넣기만 하고, DB 쓰기는 ``run_error_events_flush_loop`` 가 10초마다 몰아서 한다.
버퍼는 키 수 상한이 있어 오류 폭주에도 메모리가 늘지 않는다 — 넘친 새 키는 버리고 그 수를
세어 다음 flush 때 한 건의 이벤트로 남긴다. 어떤 예외도 호출부로 새지 않는다(로깅이 요청을
깨뜨리면 안 된다).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 계약 문서 2장 event category 고정 목록. 여기 없는 값은 ``exception`` 으로 접는다.
CATEGORIES = (
    "exception",
    "http_5xx",
    "http_4xx",
    "auth",
    "llm_provider",
    "mcp_tool",
    "overload",
    "timeout",
    "background",
)

#: 로그 레코드에 이벤트 필드를 실어 보내는 extra 키. dict 면 그 필드로 기록(WARNING 도 받는다),
#: False 면 ERROR 여도 기록하지 않는다(다른 경로가 이미 기록한 경우 이중 계상 방지).
EXTRA_KEY = "error_event"

# 자기 모듈(과 푸시 모듈)의 로그는 받지 않는다 — flush 실패 로그가 다시 이벤트가 되어 버퍼를
# 채우는 되먹임을 막는다. 푸시 실패는 common_app 쪽 신선도(수신 이력)로 드러난다.
_SELF_LOGGERS = ("agent_arch.services.monitoring", "agent_arch.services.common_app.error_events_push")

# 버퍼 키 상한 — (fingerprint, 분 버킷) 조합 수. 정상 운영에선 10초에 몇 개 수준이다.
BUFFER_MAX_KEYS = 1000

FLUSH_INTERVAL_SEC = 10
# 보존 기간과 정리 주기.
RETENTION = timedelta(days=30)
_PURGE_EVERY = timedelta(hours=1)

_MESSAGE_MAX = 500
_SOURCE_MAX = 140
_EXC_TYPE_MAX = 140
_ROUTE_MAX = 140  # common_app ErrorEvent.route max_length 와 같게 — 넘으면 배치 전체가 400
_SHORT_MAX = 32
_MCP_SERVER_MAX = 64
# 정화 정규식을 돌리기 전에 자르는 길이 — 긴 본문(응답 덤프 등)에 정규식을 돌리지 않는다.
_RAW_MAX = 2000


# ── 정화·정규화 ────────────────────────────────────────────────────────────────

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_ID26 = r"\b[A-Za-z0-9]{26}\b"  # Mattermost post/user/channel id

_SANITIZE_RULES: tuple[tuple[re.Pattern, str], ...] = (
    # URL 의 쿼리 문자열 — 토큰·이메일·검색어가 실린다. 경로는 남긴다(어느 API 인지는 필요하다).
    (re.compile(r"(https?://[^\s?#'\"]+)\?[^\s'\"]*"), r"\1?<query>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
    (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+"), "<jwt>"),
    # Authorization 헤더 모양: "Bearer xxx", "token key:secret", "Basic xxx"
    (re.compile(r"(?i)\b(bearer|token|basic)\s+[A-Za-z0-9._~+/=:-]{8,}"), r"\1 <redacted>"),
    # 벤더 키 모양(OpenAI sk-…, Google AIza…).
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "<redacted>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "<redacted>"),
    # key=…, api_key: …, secret=…, password=…, token=… (JSON 따옴표 형태 포함)
    (
        re.compile(
            r"(?i)([\"']?\b[\w-]*(?:api[_-]?key|secret|password|passwd|token|signature|authorization)[\w-]*[\"']?)"
            r"(\s*[=:]\s*)([\"']?)[^\s&\"',;}]+"
        ),
        r"\1\2\3<redacted>",
    ),
    (re.compile(_UUID), "<uuid>"),
    (re.compile(_ID26), "<id>"),
    # 전화번호 모양, 그다음 6자리 이상 숫자(주문번호·사번·전화).
    (re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b"), "<num>"),
    (re.compile(r"\d{6,}"), "<num>"),
)
_WS = re.compile(r"\s+")


def sanitize(text: Any) -> str:
    """이벤트 메시지 정화 — 이메일·토큰·키·긴 숫자·26자 id 를 가리고 500자로 자른다.

    **사용자 질문 원문은 이 함수가 걸러 주지 않는다.** 질문이 섞이는 로그는 호출부가
    ``event_extra(message=...)`` 로 대표 메시지를 따로 준다(예: memory_v2 writer).
    """
    if text is None:
        return ""
    s = str(text)[:_RAW_MAX]
    for pattern, repl in _SANITIZE_RULES:
        s = pattern.sub(repl, s)
    s = _WS.sub(" ", s).strip()
    return s[:_MESSAGE_MAX]


_NORMALIZE_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(_UUID), "*"),
    (re.compile(_ID26), "*"),
    (re.compile(r"\"[^\"]*\"|'[^']*'"), '"…"'),
    (re.compile(r"\d+"), "#"),
)


def normalize(message: Any) -> str:
    """fingerprint 용 메시지 정규화(계약 2장). 소문자화하지 않는다.

    숫자열→``#``, UUID·26자 영숫자 id→``*``, 따옴표 안 문자열→``"…"``, 앞 120자.
    숫자·id 만 다른 같은 오류("offset=500 실패" / "offset=1000 실패")가 한 그룹으로 묶인다.
    """
    s = str(message or "")
    for pattern, repl in _NORMALIZE_RULES:
        s = pattern.sub(repl, s)
    return s[:120]


def fingerprint(
    category: str,
    source: str,
    exc_type: Optional[str] = None,
    route: Optional[str] = None,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    mcp_server: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """계약 2장 식 그대로의 16자 hex fingerprint. common_app 이 이 값으로 그룹을 묶는다."""
    raw = "|".join([
        category,
        source,
        exc_type or "",
        route or "",
        str(status_code or ""),
        provider or "",
        mcp_server or "",
        normalize(message),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ── 예외 분류 ──────────────────────────────────────────────────────────────────

def exc_type_name(exc: BaseException | type | None) -> Optional[str]:
    """예외 타입 이름 — ``httpx.ReadTimeout`` 처럼 최상위 패키지 + 클래스명(내장은 클래스명만)."""
    if exc is None:
        return None
    cls = exc if isinstance(exc, type) else type(exc)
    module = (getattr(cls, "__module__", "") or "").split(".")[0]
    name = getattr(cls, "__qualname__", cls.__name__)
    if not module or module == "builtins":
        return name[:_EXC_TYPE_MAX]
    return f"{module}.{name}"[:_EXC_TYPE_MAX]


def _status_of(exc: BaseException) -> Optional[int]:
    """예외에서 HTTP 상태 코드를 뽑는다(app/llm/transient 와 같은 규칙)."""
    for attr in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


_OVERLOAD_MARKERS = (
    "too many requests",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "too many connections",
    "toomanyconnections",
    "pool is exhausted",
    "pool exhausted",
)


def classify_exception(exc: BaseException | None, status_code: Optional[int] = None) -> Optional[str]:
    """예외(또는 상태 코드)가 ``timeout`` / ``overload`` 에 해당하면 그 category, 아니면 None.

    - timeout: ``TimeoutError``(asyncio 포함) 계열, 또는 타입명에 Timeout 이 든 SDK 예외
      (httpx.ReadTimeout, openai.APITimeoutError …), 상태 408·504.
    - overload: 429(쿼터·레이트리밋), DB 풀 고갈(asyncpg TooManyConnectionsError, 풀 획득 대기
      초과는 TimeoutError 라 timeout 으로 간다).

    결제 소진(OpenAI insufficient_quota)도 429 로 오지만 여기선 overload 로 친다 — 화면에서
    "429 가 몰렸다"는 사실이 먼저 보여야 하고, 구분은 메시지가 한다.
    """
    status = status_code
    if exc is not None and status is None:
        status = _status_of(exc)
    if status == 429:
        return "overload"
    if status in (408, 504):
        return "timeout"
    if exc is None:
        return None
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _OVERLOAD_MARKERS):
        return "overload"
    if "timed out" in text:
        return "timeout"
    return None


# LLM SDK 최상위 패키지 — 비스트리밍 invoke() 는 SDK 예외를 RuntimeError 로 감싸 다시 올리고,
# orchestrator 는 그걸 exception 으로 기록한다. 원인 사슬에 SDK 예외가 있으면 llm_provider 로 본다.
_LLM_SDK_MODULES = ("openai", "anthropic", "google")
# 이 패키지들의 예외 문자열은 서버가 돌려준 오류(상태·코드·사유)라 대표 메시지에 붙여도 된다.
# 그 밖 예외의 str() 은 호출부가 넣은 값(사용자 문장 포함 가능)이라 싣지 않는다.
_SAFE_EXC_TEXT_MODULES = _LLM_SDK_MODULES + ("httpx", "httpcore", "asyncpg", "redis", "qdrant_client", "mcp")


def _exc_chain(exc: BaseException | None, depth: int = 5):
    seen = 0
    while exc is not None and seen < depth:
        yield exc
        exc = exc.__cause__ or exc.__context__
        seen += 1


def _top_module(exc: BaseException) -> str:
    return (getattr(type(exc), "__module__", "") or "").split(".")[0]


def is_llm_sdk_error(exc: BaseException | None) -> bool:
    return any(_top_module(e) in _LLM_SDK_MODULES for e in _exc_chain(exc))


_FORMAT_SPEC = re.compile(r"%(?:\([^)]*\))?[-#0 +]*\d*(?:\.\d+)?[sdrifxXeEgGc]")


def log_template_message(record: logging.LogRecord, exc: BaseException | None) -> str:
    """로그 레코드 → 대표 메시지. **인자를 끼워 넣지 않은 템플릿**을 쓴다.

    ``logger.exception("[stream] EXCEPTION: %s", e)`` 의 인자 자리에는 사용자 질문·대화 본문이
    섞여 들어올 수 있어 정규식 정화로는 다 못 걸러낸다(이메일·토큰만 패턴이 있다). 템플릿만
    남기면 그 위험이 없고, 인자만 다른 같은 오류가 한 fingerprint 로 묶이는 부수 효과도 있다.
    예외 원문은 SDK·드라이버 예외일 때만 덧붙인다(서버가 돌려준 사유라 원인 파악에 필요하다).
    """
    msg = record.msg if isinstance(record.msg, str) else str(record.msg)
    if record.args:
        msg = _FORMAT_SPEC.sub("…", msg).replace("%%", "%")
    if exc is not None:
        if any(_top_module(e) in _SAFE_EXC_TEXT_MODULES for e in _exc_chain(exc)):
            text = str(exc)[:200]
            if text:
                msg = f"{msg} | {text}"
    return msg


def event_extra(
    category: Optional[str] = None,
    *,
    exc: BaseException | None = None,
    provider: Optional[str] = None,
    mcp_server: Optional[str] = None,
    status_code: Optional[int] = None,
    message: Optional[str] = None,
    message_id: Any = None,
    conversation_id: Any = None,
    source: Optional[str] = None,
) -> dict:
    """``logger.xxx(..., extra=event_extra(...))`` 용 extra dict.

    category 를 비우면 예외·provider·mcp_server 로 판정한다(규칙은 ``_resolve_category``).
    ``message`` 를 주면 로그 문구 대신 그 문구를 대표 메시지로 쓴다(질문 원문이
    섞이는 로그용).
    """
    fields: dict[str, Any] = {
        "category": category,
        "exc": exc,
        "provider": provider,
        "mcp_server": mcp_server,
        "status_code": status_code,
        "message": message,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "source": source,
    }
    return {EXTRA_KEY: {k: v for k, v in fields.items() if v is not None}}


def _resolve_category(
    category: Optional[str],
    exc: BaseException | None,
    status_code: Optional[int],
    provider: Optional[str],
    mcp_server: Optional[str],
) -> str:
    """최종 category.

    - 호출부가 정한 HTTP·auth·background·overload·timeout 은 그대로 둔다(상태 코드·기록 지점이
      곧 분류다).
    - MCP 도구 실패는 타임아웃이어도 ``mcp_tool`` 로 둔다 — 느리든 죽었든 대응(도구 서버 점검)이
      같고, 상관 규칙 ``tool_errors_nearby`` 가 이 category 를 본다.
    - 그 밖(exception·llm_provider·미지정)은 예외가 timeout/overload 면 그쪽으로 — "Gemini 가
      429 를 냈다"(쿼터)와 "Gemini 가 오류를 냈다"(장애)는 대응이 다르다.
    """
    if category in ("http_5xx", "http_4xx", "auth", "background", "overload", "timeout"):
        return category
    if category == "mcp_tool" or (category is None and mcp_server and not provider):
        return "mcp_tool"
    classified = classify_exception(exc, status_code)
    if classified:
        return classified
    if category in (None, "exception") and is_llm_sdk_error(exc):
        return "llm_provider"
    if category in CATEGORIES:
        return category
    if provider:
        return "llm_provider"
    return "exception"


# ── 요청 스코프 라우트 ──────────────────────────────────────────────────────────

# 미들웨어가 요청 진입 때 ASGI scope 를 여기 둔다. 라우트는 라우터 매칭 **뒤에** scope["route"]
# 로 채워지므로(같은 dict 가 갱신된다) 이벤트를 기록하는 시점에 꺼내 읽는다.
_request_scope: ContextVar[Optional[dict]] = ContextVar("error_events_request_scope", default=None)


def route_template(scope: Optional[dict]) -> Optional[str]:
    """ASGI scope → 라우트 템플릿(``/api/v1/conversations/{conversation_id}``). 없으면 None.

    원경로(``/api/v1/conversations/3f2a…``)는 id 마다 다른 그룹이 되고 쿼리에는 개인 정보가
    실리므로 절대 쓰지 않는다.
    """
    if not scope:
        return None
    route = scope.get("route")
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    return str(path)[:_ROUTE_MAX] if path else None


def current_route() -> Optional[str]:
    return route_template(_request_scope.get())


# ── 버퍼 ──────────────────────────────────────────────────────────────────────

@dataclass
class _Agg:
    """(fingerprint, 분 버킷) 하나의 누적."""

    bucket_at: datetime
    fingerprint: str
    level: str
    category: str
    source: str
    route: Optional[str]
    status_code: Optional[int]
    exc_type: Optional[str]
    provider: Optional[str]
    mcp_server: Optional[str]
    message: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None


class _EventBuffer:
    """스레드 안전한 합산 버퍼. 잠금 구간에는 dict 연산만 있다(I/O 없음)."""

    def __init__(self, max_keys: int = BUFFER_MAX_KEYS):
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._items: dict[tuple[str, datetime], _Agg] = {}
        self.dropped = 0

    def add(self, agg: _Agg) -> bool:
        key = (agg.fingerprint, agg.bucket_at)
        with self._lock:
            cur = self._items.get(key)
            if cur is None:
                if len(self._items) >= self.max_keys:
                    self.dropped += 1
                    return False
                self._items[key] = agg
                return True
            cur.count += agg.count
            if agg.last_seen > cur.last_seen:
                cur.last_seen = agg.last_seen
                # 대표 메시지는 최신 것으로 — 정규화가 같으니 모양은 같고 값만 최신이다.
                cur.message = agg.message
            if agg.first_seen < cur.first_seen:
                cur.first_seen = agg.first_seen
            cur.message_id = agg.message_id or cur.message_id
            cur.conversation_id = agg.conversation_id or cur.conversation_id
            return True

    def drain(self) -> tuple[list[_Agg], int]:
        with self._lock:
            items = list(self._items.values())
            self._items = {}
            dropped, self.dropped = self.dropped, 0
        return items, dropped

    def restore(self, items: list[_Agg]) -> None:
        """flush 실패분을 되돌린다. 그사이 쌓인 같은 키와 합산하고, 상한을 넘치면 버린다."""
        for agg in items:
            self.add(agg)

    def __len__(self) -> int:
        return len(self._items)


_buffer = _EventBuffer()

# 이벤트 기록 중 다시 로그가 나 재진입하는 것을 막는다(스레드별).
_reentry = threading.local()


def _enabled() -> bool:
    try:
        from agent_arch.core.config import settings

        return bool(getattr(settings, "enable_error_events", True))
    except Exception:  # noqa: BLE001 — 설정 로드 실패여도 로깅은 계속된다
        return False


def _trim(value: Any, limit: int) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)[:limit]


def _capture(
    *,
    category: Optional[str],
    source: str,
    level: str = "ERROR",
    message: Any = None,
    exc: BaseException | None = None,
    exc_type: Optional[str] = None,
    route: Optional[str] = None,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    mcp_server: Optional[str] = None,
    message_id: Any = None,
    conversation_id: Any = None,
    when: Optional[datetime] = None,
) -> None:
    """버퍼에 한 건 넣는다. 호출부 규약: 예외를 올리지 않는다(래퍼가 삼킨다)."""
    if getattr(_reentry, "active", False):
        return
    _reentry.active = True
    try:
        if exc is not None and exc_type is None:
            exc_type = exc_type_name(exc)
        if message in (None, "") and exc is not None:
            # 예외 원문은 SDK·드라이버 예외일 때만 — 그 밖 str(exc) 에는 호출부 값이 섞일 수 있다.
            safe = any(_top_module(e) in _SAFE_EXC_TEXT_MODULES for e in _exc_chain(exc))
            message = (str(exc) if safe else "") or exc_type
        cat = _resolve_category(category, exc, status_code, provider, mcp_server)
        src = (source or "unknown")[:_SOURCE_MAX]
        rt = _trim(route if route is not None else current_route(), _ROUTE_MAX)
        exc_t = _trim(exc_type, _EXC_TYPE_MAX)
        prov = _trim(provider, _SHORT_MAX)
        mcp = _trim(mcp_server, _MCP_SERVER_MAX)
        msg = sanitize(message)
        fp = fingerprint(cat, src, exc_t, rt, status_code, prov, mcp, msg)
        now = when or datetime.now(timezone.utc)
        _buffer.add(_Agg(
            bucket_at=now.replace(second=0, microsecond=0),
            fingerprint=fp,
            level="WARNING" if level == "WARNING" else "ERROR",
            category=cat,
            source=src,
            route=rt,
            status_code=status_code,
            exc_type=exc_t,
            provider=prov,
            mcp_server=mcp,
            message=msg,
            first_seen=now,
            last_seen=now,
            message_id=_trim(message_id, 64),
            conversation_id=_trim(conversation_id, 64),
        ))
    finally:
        _reentry.active = False


def record_event(
    *,
    category: Optional[str],
    source: str,
    message: Any = None,
    level: str = "ERROR",
    exc: BaseException | None = None,
    exc_type: Optional[str] = None,
    route: Optional[str] = None,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    mcp_server: Optional[str] = None,
    message_id: Any = None,
    conversation_id: Any = None,
) -> None:
    """오류 이벤트 한 건을 명시 기록한다. **절대 예외를 올리지 않고 I/O 도 하지 않는다.**

    Args:
        category: 계약 2장 목록 중 하나. None 이면 예외·provider·mcp_server 로 판정한다.
        source: 로거명 또는 기록 지점(≤140).
        message: 대표 메시지 — 정화된다. 사용자 질문 원문을 넘기지 말 것.
        level: "ERROR" | "WARNING".
        route: 라우트 템플릿. None 이면 현재 요청의 라우트(없으면 null).
    """
    if not _enabled():
        return
    try:
        _capture(
            category=category, source=source, level=level, message=message, exc=exc,
            exc_type=exc_type, route=route, status_code=status_code, provider=provider,
            mcp_server=mcp_server, message_id=message_id, conversation_id=conversation_id,
        )
    except Exception:  # noqa: BLE001 — 기록 실패가 호출부를 깨뜨리면 안 된다
        pass


# ── logging 핸들러 ────────────────────────────────────────────────────────────

class ErrorEventHandler(logging.Handler):
    """ERROR 이상 + ``extra=event_extra(...)`` 로 지정한 WARNING 을 이벤트 버퍼에 넣는다.

    config/logging.yaml 이 ``app`` 로거와 root 에 붙인다(app 은 propagate: false 라 둘 다
    필요하다). emit 은 정규식 몇 개와 sha1 한 번 — I/O 도, 대기도 없다.
    """

    def __init__(self, level: int = logging.WARNING):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            spec = record.__dict__.get(EXTRA_KEY)
            if spec is False:
                return
            if record.levelno < logging.ERROR and not isinstance(spec, dict):
                return
            if record.name.startswith(_SELF_LOGGERS):
                return
            if not _enabled():
                return
            spec = spec if isinstance(spec, dict) else {}

            exc = spec.get("exc")
            if exc is None and record.exc_info and record.exc_info[1] is not None:
                exc = record.exc_info[1]
            message = spec.get("message")
            if message is None:
                message = log_template_message(record, exc)
            _capture(
                category=spec.get("category"),
                source=spec.get("source") or record.name,
                level="ERROR" if record.levelno >= logging.ERROR else "WARNING",
                message=message,
                exc=exc,
                status_code=spec.get("status_code"),
                provider=spec.get("provider"),
                mcp_server=spec.get("mcp_server"),
                message_id=spec.get("message_id"),
                conversation_id=spec.get("conversation_id"),
                when=datetime.fromtimestamp(record.created, timezone.utc),
            )
        except Exception:  # noqa: BLE001 — handleError 는 stderr 에 쓰므로 그것도 피한다
            pass


# ── HTTP 미들웨어 ─────────────────────────────────────────────────────────────

# 소음 제외 — 헬스체크·문서·루트. 모니터링 프로브가 분마다 치므로 여기서 나는 4xx 는 신호가 아니다.
_IGNORED_PATH_PREFIXES = ("/api/v1/health", "/docs", "/redoc", "/openapi.json", "/swagger", "/favicon.ico")
_IGNORED_EXACT = ("/",)


def _http_category(status: int) -> Optional[str]:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "overload"
    if status >= 500:
        return "http_5xx"
    if status >= 400:
        return "http_4xx"
    return None


class ErrorEventsMiddleware:
    """4xx/5xx 응답을 라우트 템플릿으로 기록하는 순수 ASGI 미들웨어.

    BaseHTTPMiddleware 를 쓰지 않는다 — 응답 본문을 한 번 더 감싸 StreamingResponse(SSE)
    를 버퍼링·지연시키는 알려진 문제가 있다. 여기서는 ``http.response.start`` 의 상태 코드만
    엿보고 본문은 그대로 흘린다.

    ⚠️ **SSE(/agent/stream) 오류는 여기서 안 잡힌다.** 스트림은 헤더가 200 으로 먼저 나간 뒤
    본문 안에서 error 청크로 실패를 알리기 때문이다. 그 경로는 app/api/v1/agent.py 의 error
    청크 지점과 orchestrator 의 예외 로그가 기록한다.

    라우트가 매칭되지 않은 요청(스캐너의 404 등)은 route="(unmatched)" 로 한 그룹에 모은다 —
    원경로를 쓰면 경로마다 그룹이 생긴다.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path in _IGNORED_EXACT or path.startswith(_IGNORED_PATH_PREFIXES):
            await self.app(scope, receive, send)
            return

        token = _request_scope.set(scope)
        status_holder: dict[str, int] = {}

        async def _send(message):
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status") or 0)
                try:
                    _record_http(scope, status_holder["status"])
                except Exception:  # noqa: BLE001
                    pass
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            # 응답을 시작하기 전에 터졌다 — 바깥 ServerErrorMiddleware 가 500 을 낸다.
            if "status" not in status_holder:
                try:
                    _record_http(scope, 500, exc=exc)
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            _request_scope.reset(token)


def _record_http(scope: dict, status: int, exc: BaseException | None = None) -> None:
    category = _http_category(status)
    if category is None:
        return
    route = route_template(scope) or "(unmatched)"
    method = scope.get("method") or ""
    record_event(
        category=category,
        source="http",
        level="ERROR" if status >= 500 else "WARNING",
        message=f"{method} {route} → {status}",
        exc=exc,
        route=route,
        status_code=status,
    )


# ── 백그라운드 태스크 ─────────────────────────────────────────────────────────

def watch_task(task: Optional[asyncio.Task], name: str, *, forever: bool = False) -> Optional[asyncio.Task]:
    """태스크가 예외로 죽으면 background 이벤트를 남기는 done 콜백을 붙인다.

    ``asyncio.create_task`` 로 띄운 태스크는 예외로 끝나도 아무도 await 하지 않으면 GC 때
    "Task exception was never retrieved" 한 줄이 남을 뿐이다 — 인덱싱이 조용히 죽어 도구가
    검색에 안 걸리는 상태가 로그 grep 없이는 보이지 않았다.

    Args:
        task: 감시할 태스크. None 이면 그대로 None.
        name: 이벤트 source 에 붙일 이름(``lifespan:<name>``).
        forever: 끝나면 안 되는 루프(동기화·모니터)면 True — 예외 없이 끝나도 기록한다.

    Returns:
        받은 태스크 그대로(호출부가 ``return watch_task(asyncio.create_task(...))`` 로 쓴다).
    """
    if task is None:
        return None

    def _done(t: asyncio.Task) -> None:
        try:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                record_event(
                    category="background", source=f"lifespan:{name}", exc=exc,
                    message=f"백그라운드 태스크 {name} 가 예외로 종료: {exc}", route=None,
                )
            elif forever:
                record_event(
                    category="background", source=f"lifespan:{name}", level="WARNING",
                    message=f"백그라운드 루프 {name} 가 스스로 종료됨", route=None,
                )
        except Exception:  # noqa: BLE001
            pass

    try:
        task.add_done_callback(_done)
    except Exception:  # noqa: BLE001
        pass
    return task


# ── flush (버퍼 → PG) ─────────────────────────────────────────────────────────

_UPSERT = """
INSERT INTO agent.error_events
    (bucket_at, fingerprint, count, first_seen, last_seen, updated_at, level, category,
     source, route, status_code, exc_type, provider, mcp_server, message, message_id,
     conversation_id)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
ON CONFLICT (fingerprint, bucket_at) DO UPDATE SET
    count           = agent.error_events.count + EXCLUDED.count,
    first_seen      = LEAST(agent.error_events.first_seen, EXCLUDED.first_seen),
    last_seen       = GREATEST(agent.error_events.last_seen, EXCLUDED.last_seen),
    updated_at      = EXCLUDED.updated_at,
    message         = EXCLUDED.message,
    message_id      = COALESCE(EXCLUDED.message_id, agent.error_events.message_id),
    conversation_id = COALESCE(EXCLUDED.conversation_id, agent.error_events.conversation_id)
"""

_PURGE = "DELETE FROM agent.error_events WHERE bucket_at < $1"


def _naive(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _row(agg: _Agg, updated_at: datetime) -> tuple:
    return (
        _naive(agg.bucket_at), agg.fingerprint, agg.count, _naive(agg.first_seen),
        _naive(agg.last_seen), _naive(updated_at), agg.level, agg.category, agg.source,
        agg.route, agg.status_code, agg.exc_type, agg.provider, agg.mcp_server, agg.message,
        agg.message_id, agg.conversation_id,
    )


def _dropped_event(dropped: int, now: datetime) -> _Agg:
    """버퍼 상한으로 버린 건수를 한 건의 이벤트로 — 폭주 자체가 신호다."""
    src = "agent_arch.services.monitoring.error_events"
    msg = f"오류 이벤트 버퍼 상한({BUFFER_MAX_KEYS}) 초과로 새 이벤트 {dropped}건을 버렸다"
    return _Agg(
        bucket_at=now.replace(second=0, microsecond=0),
        fingerprint=fingerprint("overload", src, None, None, None, None, None, msg),
        level="WARNING", category="overload", source=src, route=None, status_code=None,
        exc_type=None, provider=None, mcp_server=None, message=msg,
        first_seen=now, last_seen=now, count=dropped,
    )


async def flush_buffer() -> int:
    """버퍼를 비워 PG 에 upsert 한다. 실패하면 버퍼로 되돌린다(다음 주기 재시도).

    Returns:
        int: 쓴 행 수. DB 비활성·빈 버퍼면 0.
    """
    from agent_arch.db.connection import db_pool

    items, dropped = _buffer.drain()
    now = datetime.now(timezone.utc)
    if dropped:
        items.append(_dropped_event(dropped, now))
    if not items:
        return 0
    pool = db_pool.get_pool()
    if pool is None:
        # DB 없는 환경(로컬·테스트) — 버리지 않고 쥐고 있다가 상한에서 멈춘다.
        _buffer.restore(items)
        return 0
    try:
        await pool.executemany(_UPSERT, [_row(a, now) for a in items])
    except Exception:
        _buffer.restore(items)
        logger.warning("오류 이벤트 기록 실패 (%d건, 다음 주기 재시도)", len(items), exc_info=True)
        return 0
    return len(items)


async def purge_expired(now: Optional[datetime] = None) -> None:
    from agent_arch.db.connection import db_pool

    pool = db_pool.get_pool()
    if pool is None:
        return
    cutoff = (now or datetime.now(timezone.utc)) - RETENTION
    try:
        await pool.execute(_PURGE, _naive(cutoff))
    except Exception:
        logger.warning("오류 이벤트 보존 정리 실패", exc_info=True)


async def run_error_events_flush_loop(interval_sec: int = FLUSH_INTERVAL_SEC) -> None:
    """버퍼 → PG flush 루프(10초) + 한 시간마다 30일 지난 행 정리. lifespan 이 띄운다."""
    logger.info("오류 이벤트 flush 루프 시작 (주기 %d초)", interval_sec)
    last_purge: Optional[datetime] = None
    while True:
        try:
            await asyncio.sleep(interval_sec)
            await flush_buffer()
            now = datetime.now(timezone.utc)
            if last_purge is None or now - last_purge >= _PURGE_EVERY:
                await purge_expired(now)
                last_purge = now
        except asyncio.CancelledError:
            logger.info("오류 이벤트 flush 루프 종료")
            raise
        except Exception:
            logger.warning("오류 이벤트 flush 주기 실패", exc_info=True)
