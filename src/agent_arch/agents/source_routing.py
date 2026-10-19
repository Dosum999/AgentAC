"""X-Source-Path 헤더 기반 agent_mode/guide_categories fallback 매핑.

게이트웨이가 주입하는 X-Source-Path 헤더(예: "/hr/leave/new")를 받아
agent_mode 와 guide_categories 의 디폴트를 결정한다.
우선순위: Form 명시값 > X-Source-Path 매핑 > settings 디폴트 (호출자에서 적용).
"""
from __future__ import annotations

from typing import Optional, Tuple, TypedDict

from agent_arch.core.config import settings


class _SourceDefaults(TypedDict, total=False):
    agent_mode: str
    guide_categories: str  # 콤마 join 된 화이트리스트


# guide_categories 는 Qdrant user_guide_kb 의 category 와 정확히 일치해야 한다
# (app/services/guide_kb/ingestion/chunker.py 의 _CATEGORY_RULES 참조).
_PREFIX_MAP: dict[str, _SourceDefaults] = {
    "/hr": {"agent_mode": "guide", "guide_categories": "HR System"},
}


def resolve_defaults(source_path: Optional[str]) -> _SourceDefaults:
    """X-Source-Path 값에서 fallback 디폴트 dict 를 도출한다.

    가장 긴 매칭 prefix 가 우선 ("/hr/leave" 가 "/hr" 보다 먼저).

    Args:
        source_path: X-Source-Path 헤더 원본. None/빈 문자열/매칭 미스 → {}.

    Returns:
        {"agent_mode": ..., "guide_categories": ...} (매핑된 키만 존재).
    """
    if not source_path:
        return {}
    for prefix in sorted(_PREFIX_MAP.keys(), key=len, reverse=True):
        if source_path == prefix or source_path.startswith(prefix + "/"):
            return _PREFIX_MAP[prefix]
    return {}


def apply_source_defaults(
    agent_mode: Optional[str],
    guide_categories: Optional[str],
    source_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Form 명시값에 X-Source-Path 매핑 fallback 을 적용한다.

    우선순위: Form 명시값 > X-Source-Path 매핑 > None
    (None 이면 AgentOrchestrator 가 settings.agent_mode 로 추가 fallback).

    Args:
        agent_mode: Form 에서 받은 agent_mode (None 이면 미지정).
        guide_categories: Form 에서 받은 guide_categories 원본 콤마 문자열.
        source_path: X-Source-Path 헤더 값.

    Returns:
        (resolved_agent_mode, resolved_guide_categories) — AgentOrchestrator 인자로 그대로 전달.
    """
    defaults = resolve_defaults(source_path)
    return (
        agent_mode or defaults.get("agent_mode"),
        guide_categories or defaults.get("guide_categories"),
    )


# ── 사용량 관측용 서비스 축 ────────────────────────────────────────────────────
#
# "이 사용량이 어느 면에서 났나"를 요청 경로 하나로 판정한다. 지금까지 이 정보는 존재하지만
# 버려졌다 — `/agent` 와 `/agent/mes`, `/scms/agent`, `/drawing` 은 이미 라우팅 단계에서
# 구분되는데 트레이스에도, DB 에도, 푸시에도 남지 않았다. 그래서 회사별 합계만 있고 "메신저에서
# 얼마, AI 도면에서 얼마"를 물을 수 없었다.
#
# 판정 근거를 `X-Source-Path` 헤더가 아니라 **실제 요청 경로**로 둔다 — 그 헤더는 게이트웨이를
# 경유할 때만 존재해서, 직접 호출이나 웹훅에서는 비어 버린다(그러면 사용량이 통째로 미분류가
# 된다). 헤더는 UX 힌트지 사용량 귀속의 축이 아니다.
#
# 아래 키는 **API 버전 prefix 를 뗀** 라우터 경로다. 라우터가 전부 `settings.api_v1_prefix`
# 아래 걸리므로(app/main.py) 실제 요청은 `/api/v1/agent/stream` 으로 들어온다 — 매핑에
# prefix 를 하드코딩하면 그 값이 환경변수로 바뀌는 순간 조용히 전량 미분류가 된다. 그래서
# 매핑은 prefix 없는 형태로 두고, 판정 직전에 설정값으로 한 번 벗겨낸다.
_SERVICE_BY_PREFIX: dict[str, str] = {
    "/agent/mes": "mes",
    "/scms/agent": "scms",
    "/drawing": "drawing",
    "/guest": "guest",
    "/agent": "datbi_browser",
    "/audio": "datbi_browser",
}

# 메신저(Mattermost) 봇은 HTTP 요청 경로가 웹훅이라 위 매핑으로 판정되지 않는다 —
# 봇 실행 경로가 직접 실어 보낸다.
SERVICE_MESSENGER = "messenger"


# ── 기억의 자리(장소)로서의 앱 전용 창 ────────────────────────────────────────
#
# 기억의 장소는 지금까지 메신저 채널만 있었다(services/memory_v2/place). 앱 전용 창은
# 채널이 아니지만 **일반 채팅과 따로 분리된 자리**다 — 그 창에서 못 박은 규칙이 브라우저
# 패널까지 따라다니면 기억이 섞인 것이다.
#
# 위의 사용량 서비스 축과 **따로** 두는 이유: `/audio` 는 사용량으로는 브라우저 패널
# (datbi_browser)로 묶여 있는데, 기억의 자리로는 MES 앱 창과 같다 — 앱이 같은 창에서
# 음성과 채팅을 둘 다 보내고 서버도 두 경로에 같은 봇(apply_logistics_bot)을 물린다.
# 두 축을 한 매핑으로 합치면 둘 중 하나가 반드시 틀린다.
_MEMORY_SURFACE_BY_PREFIX: dict[str, str] = {
    "/agent/mes": "mes",
    "/audio": "mes",
}



def _strip_api_prefix(path: str) -> str:
    """실제 요청 경로에서 API 버전 prefix 를 뗀다("/api/v1/agent/stream" → "/agent/stream").

    prefix 가 안 붙은 경로(내부 호출·테스트)는 그대로 통과시킨다 — 두 형태를 모두 받아야
    호출부가 경로를 어디서 얻었는지 신경 쓰지 않아도 된다.
    """
    prefix = (settings.api_v1_prefix or "").rstrip("/")
    if prefix and (path == prefix or path.startswith(prefix + "/")):
        return path[len(prefix):] or "/"
    return path


def resolve_service(path: Optional[str]) -> Optional[str]:
    """요청 경로에서 사용량 서비스 축을 판정한다. 매칭이 없으면 None(=미분류).

    입력은 `request.url.path` 원본을 그대로 받는다 — 즉 `settings.api_v1_prefix` 가 붙은
    형태다. 매핑 키는 prefix 없는 라우터 경로라 여기서 한 번 벗겨 내고 비교한다.

    가장 긴 매칭 prefix 가 우선한다 — "/agent/mes" 가 "/agent" 보다 먼저 잡혀야 MES 채널
    사용량이 브라우저 패널로 섞이지 않는다.

    판정하지 못하면 **기본 서비스를 지어내지 않는다.** 미분류로 두면 화면에서 그 버킷을
    들여다보고 새 면을 발견할 수 있지만, 임의로 datbi_browser 를 붙이면 남의 사용량이
    브라우저 패널 청구에 조용히 섞인다.
    """
    if not path:
        return None
    path = _strip_api_prefix(path)
    for prefix in sorted(_SERVICE_BY_PREFIX.keys(), key=len, reverse=True):
        if path == prefix or path.startswith(prefix + "/"):
            return _SERVICE_BY_PREFIX[prefix]
    return None


def resolve_memory_surface(path: Optional[str]) -> Optional[str]:
    """요청 경로에서 기억의 자리(앱 전용 창)를 판정한다. 매칭이 없으면 None(=장소 밖).

    None 은 "장소가 없는 자리"(브라우저 패널 등)를 뜻하며, 그 턴의 기억은 장소 중립이
    된다 — 이 기능 이전과 순위·자격이 같다(place.rank_factor·eligible 참조).

    Args:
        path: ``request.url.path`` 원본(API 버전 prefix 포함 가능).

    Returns:
        자리 이름(예 ``"mes"``) 또는 None.
    """
    if not path:
        return None
    path = _strip_api_prefix(path)
    for prefix in sorted(_MEMORY_SURFACE_BY_PREFIX.keys(), key=len, reverse=True):
        if path == prefix or path.startswith(prefix + "/"):
            return _MEMORY_SURFACE_BY_PREFIX[prefix]
    return None
