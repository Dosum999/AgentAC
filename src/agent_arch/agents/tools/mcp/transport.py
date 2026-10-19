"""MCP streamable HTTP 연결 헬퍼 (mcp SDK 2.x).

서비스 토큰(+선택적 사용자 컨텍스트) 헤더가 실린 연결을 열어준다.
provider(목록 조회)와 RemoteMCPTool(도구 호출)이 공유한다.

사용자 컨텍스트는 묶음으로 전달한다 — 서버마다 쓰는 헤더가 다르다
(mailcow: X-User-Email로 메일박스 식별 / nextcloud: X-Session-Id로
게이트웨이 세션 패스스루). 각 서버는 자기가 안 쓰는 헤더를 무시한다.

⚠ HTTP 클라이언트는 반드시 ``httpx2`` 다. mcp 2.x 는 httpx 를 버리고 httpx2
  (pydantic 이 이어받은 후속)로 갈아탔고, 여기에 ``httpx.AsyncClient`` 를 넘기면
  **예외 없이 조용히 열화**한다(서버발 메시지가 도착하지 않는다). 백엔드의 다른
  코드는 계속 httpx 를 쓰므로 두 패키지가 공존하며, 이 파일만 httpx2 를 쓴다.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote as _url_quote

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

USER_EMAIL_HEADER = "X-User-Email"
USER_SUB_HEADER = "X-User-Sub"
SESSION_ID_HEADER = "X-Session-Id"
COMPANY_HEADER = "X-Company"
# 패널이 보낸 현재 화면 경로(게이트웨이가 검증해 넘김). SCMS 처럼 "이거" 를 화면의 개체로
# 해석해야 하는 MCP 가 쓴다. ASCII 경로만 통과하므로 인코딩 불필요.
SOURCE_PATH_HEADER = "X-Source-Path"
# SCMS 화면에서 고른 활동 역할(발주사/공급사)·활동 회사. 화면이 보내면 게이트웨이가 통과시키고(header_policy
# ALLOWED_FORWARD_HEADERS 의 x-active-role·x-active-company) 여기서 그대로 전달한다 — 없으면 SCMS 가 역할은 회사
# capability, 회사는 첫 소속으로 정해 두 역할·다중 소속 사용자가 엉뚱한 기준으로 조회됐다.
ACTIVE_ROLE_HEADER = "X-Active-Role"
ACTIVE_COMPANY_HEADER = "X-Active-Company"


def _header_safe(value: str) -> str:
    """비ASCII 헤더 값(한글 회사명 등) percent-encoding."""
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return _url_quote(value, safe="")


def user_context_headers(user_ctx: dict) -> dict[str, str]:
    """get_user_context() dict → MCP 서버로 보낼 사용자 헤더 묶음."""
    mapping = {
        USER_EMAIL_HEADER: user_ctx.get("email"),
        USER_SUB_HEADER: user_ctx.get("sub"),
        SESSION_ID_HEADER: user_ctx.get("session_id"),
        COMPANY_HEADER: user_ctx.get("company"),
        SOURCE_PATH_HEADER: user_ctx.get("source_path"),
        ACTIVE_ROLE_HEADER: user_ctx.get("active_role"),
        ACTIVE_COMPANY_HEADER: user_ctx.get("active_company"),
    }
    return {k: _header_safe(str(v)) for k, v in mapping.items() if v}


@asynccontextmanager
async def open_mcp_session(
    url: str,
    *,
    service_token: str,
    user_headers: dict[str, str] | None = None,
    timeout_seconds: float = 60.0,
):
    """연결이 끝난 ``Client``를 yield한다.

    ``Client``는 진입 시 프로토콜을 스스로 협상한다: 먼저 ``server/discover``를
    던져보고, 못 알아듣는 2025-era 서버면 기존 initialize 핸드셰이크로 폴백한다.
    그래서 v2 로 이관한 서버(hr-mcp)는 무핸드셰이크 2026-07-28 경로로, 아직
    1.x 인 서버는 종전과 같은 경로로 각각 동작한다.

    협상은 서버별로 한 번만 한다 — 결과를 기억해 다음 연결부터는 탐침 없이
    바로 붙는다(아래 ``_protocol_memo``).

    캐시는 끈다(``cache=None``). 지금은 도구 호출 1건마다 연결을 새로 열어
    클라이언트가 즉시 버려지므로, 응답 캐시를 켜봐야 재사용될 일이 없다 —
    연결을 풀링하게 되면 그때 서버의 ttl/scope 힌트와 함께 켤 자리다.

    Args:
        url: MCP 엔드포인트 (예: http://mailcow-mcp:8765/mcp)
        service_token: Bearer shared secret. 빈 문자열이면 Authorization 헤더를 생략한다.
        user_headers: ``user_context_headers()`` 결과. 목록 조회 등
            사용자 무관 호출이면 생략.
        timeout_seconds: HTTP 연결/읽기 타임아웃.
    """
    # 토큰이 없는 서버(자체 인증이 없는 내부망 MCP, 예: 브레이브 검색)는 Authorization 을 보내지 않는다 —
    # "Bearer " 처럼 값이 비면 httpx 가 Illegal header value 로 요청 자체를 거부한다(2026-09-15 실측).
    headers = dict(user_headers or {})
    if service_token:
        headers["Authorization"] = f"Bearer {service_token}"

    async with httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(timeout_seconds),
        follow_redirects=True,
    ) as http_client:
        async with _connect(url, http_client) as client:
            yield client


# ── 서버별 프로토콜 기억 ──────────────────────────────────────────────────────
# 호출마다 연결을 새로 여는 구조라, 기억해두지 않으면 도구 호출 1건마다
# server/discover 탐침을 되풀이한다 — 2025-era 서버에서는 그 탐침이 실패가
# 예정된 순수 낭비다(실측: 1.x 서버 호출당 HTTP 3→4건). 한 번 붙어본 서버는
# 협상 결과를 기억해 다음 연결부터 탐침을 건너뛴다.
#
# 프로세스 로컬이고 비어 있어도 정상 동작한다(그때는 auto 로 협상). 서버가
# 도중에 이관돼 기억이 어긋나면 스스로 지우고 다시 협상한다 — 아래 참조.
_AUTO = "auto"
_LEGACY = "legacy"
_protocol_memo: dict[str, tuple[str, Any]] = {}


def _remember(url: str, client: Client) -> None:
    """협상 결과를 기록한다. 모던 서버면 DiscoverResult 까지 함께 둔다."""
    discovered = client.session.discover_result
    if discovered is None:
        _protocol_memo[url] = (_LEGACY, None)
    else:
        _protocol_memo[url] = (client.protocol_version, discovered)


def forget_protocol(url: str) -> None:
    """기억을 버린다 — 다음 연결은 다시 협상한다."""
    _protocol_memo.pop(url, None)


@asynccontextmanager
async def _connect(url: str, http_client: httpx2.AsyncClient):
    """기억한 프로토콜로 붙되, 어긋나면 버리고 다시 협상한다."""
    mode, prior = _protocol_memo.get(url, (_AUTO, None))
    entered = False
    try:
        async with Client(
            streamable_http_client(url, http_client=http_client),
            mode=mode,
            prior_discover=prior,
            cache=None,
        ) as client:
            entered = True
            _remember(url, client)
            yield client
            return
    except Exception:
        # 기억을 썼다가 실패했으면 그 기억을 의심한다. 연결 단계에서 깨졌다면
        # (아직 호출자에게 넘기기 전이라) 여기서 바로 재협상하고, 호출 도중에
        # 깨졌다면 기억만 버려 다음 호출이 다시 협상하게 한다.
        if mode == _AUTO:
            raise
        forget_protocol(url)
        if entered:
            raise
        logger.info("[mcp] 기억한 프로토콜(%s)로 연결 실패 — 재협상한다: %s", mode, url)

    async with Client(
        streamable_http_client(url, http_client=http_client), cache=None
    ) as client:
        _remember(url, client)
        yield client
