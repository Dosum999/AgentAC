"""서비스 축 라우트 커버리지 가드.

관측 화면의 '미분류'는 대부분 **새 라우터가 생겼는데 아무도 축 매핑을 안 붙인 것**이다.
그 실패는 조용하다 — `resolve_service` 가 None 을 돌려주고, 턴은 그냥 축 없이 저장되며,
화면에는 '미분류' 숫자만 늘어난다. 어느 경로인지는 로그 한 줄뿐이고 그마저 컨테이너가
재기동되면 사라진다(`_warn_unresolved_service` 는 프로세스당 경로당 1회).

그래서 여기서 **경로 집합 자체를 고정한다.** 새 라우터를 붙이면 둘 중 하나를 하게 된다:
축을 `_SERVICE_BY_PREFIX` 에 등록하거나, 턴을 만들지 않는 면임을 아래 allowlist 에 적거나.
어느 쪽이든 **의식적인 선택**이 되고, 조용한 미분류가 생기지 않는다.
"""
from __future__ import annotations

from agent_arch.agents.source_routing import _strip_api_prefix, resolve_service

#: 턴(`agent.messages` 행)을 만들지 않는 면. prefix 는 API 버전 prefix 를 뗀 형태다.
#:
#: 여기 적는 기준은 "이 경로로 들어온 요청이 대화 턴을 저장하지 않는다" 하나뿐이다.
#: 조회·프로필·운영 도구가 대부분이고, 예외가 하나 있다 — `/webhooks` (메신저)는 턴을
#: 만들지만 HTTP 경로가 웹훅이라 경로 판정 대상이 아니다. 봇 실행 경로가 `SERVICE_MESSENGER`
#: 를 직접 싣는다(`bot.py` `_BOT_BASE`/`_resolve_requester`, `webhook.py` 버튼 콜백).
_NO_TURN_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/swagger",
    "/me",           # 내 대화·메모리·환경설정·사용량 조회
    "/user",         # 프로필 저장
    "/company",
    "/usage",
    "/jobs",         # 작업 상태 조회·취소
    "/admin",        # KB 갱신 등 운영 도구
    "/odoo",         # 전시회 부스 등록(인증 없음)
    "/workspace",    # 편집·승인된 값으로 MCP 도구만 실행한다(_commit) — 턴 저장 없음
    "/webhooks",     # 메신저 — 위 주석 참조
    "/qa",           # QA 도구 표면 조회(허용 계정 전용, 읽기 전용) — 턴 저장 없음
)


def _is_allowlisted(path: str) -> bool:
    stripped = _strip_api_prefix(path)
    if stripped == "/":
        return True
    return any(
        stripped == p or stripped.startswith(p + "/") for p in _NO_TURN_PREFIXES
    )


def test_every_route_either_resolves_a_service_or_is_declared_turnless():
    """모든 라우트는 축이 판정되거나, 턴을 안 만드는 면으로 선언돼 있어야 한다.

    새 대화 면을 붙이고 매핑을 잊으면 여기서 걸린다 — 배포 뒤 '미분류'가 늘어난 것을
    보고 역추적하는 대신, PR 에서 바로 알게 된다.
    """
    import os

    os.environ.setdefault("SKIP_STARTUP", "1")
    from agent_arch.main import app

    leaked = sorted(
        {
            _strip_api_prefix(route.path)
            for route in app.routes
            if getattr(route, "path", None)
            and resolve_service(route.path) is None
            and not _is_allowlisted(route.path)
        }
    )

    assert not leaked, (
        "서비스 축이 판정되지 않는 라우트가 있다 — 이 경로의 사용량은 '미분류'로 적재된다.\n"
        f"  {leaked}\n"
        "대화 턴을 만드는 면이면 app/agents/source_routing.py 의 _SERVICE_BY_PREFIX 에 "
        "추가하고, 아니면 이 파일의 _NO_TURN_PREFIXES 에 근거와 함께 적을 것."
    )


def test_turn_saving_surfaces_resolve_to_their_own_axis():
    """대화를 저장하는 면들은 각자의 축으로 갈라져야 한다 — 서로 섞이면 청구가 틀린다.

    `/agent/app` 이 `/agent` 보다 먼저 잡혀야 앱 전용 창 사용량이 브라우저 패널에 섞이지 않는다
    (가장 긴 prefix 우선). 그 우선순위를 여기서 고정한다.
    """
    assert resolve_service("/api/v1/agent/stream") == "browser_panel"
    assert resolve_service("/api/v1/agent/app/run") == "app"
    assert resolve_service("/api/v1/guest/agent") == "guest"
    # 메신저는 경로가 아니라 봇 실행 경로가 싣는다 — 여기서 판정되면 안 된다.
    assert resolve_service("/api/v1/webhooks/mattermost") is None
