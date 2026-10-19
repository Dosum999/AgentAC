"""예시 업무 도메인: 사내 헬프데스크 티켓.

코어(라우터·관문·HITL·오케스트레이터)를 한 줄도 고치지 않고 업무 도메인 하나를 끼우는 방법을 보여 준다.
실제 도메인도 같은 순서로 붙인다.

    1. 도구를 만든다 — 조회 도구는 그대로, 비가역 도구는 ``confirm`` 인자를 받아 초안(draft view)을 돌려준다.
    2. ``register_domain`` 으로 관문(find_<domain>_tools)·라우터 카드·도구 접두사를 등록한다.
    3. ``extensions`` 슬롯을 채운다 — 관문이 돌려줄 도구, 승인 필요 도구와 완료 문구, 승인 후처리,
       전용 채널 모드 프롬프트, 용어 사전 출처.
    4. 앱을 ``create_app(setup=install)`` 로 띄운다(``create_example_app``).

권한: 도구는 서비스 키 ``helpdesk`` 의 ACCESS_ONLY 다. 사용자 프로필에 ``{"helpdesk": ["helpdesk.access"]}`` 가 있어야 보인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_arch.agents.permissions import GateMode, PermissionRequirement, permission_index
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.extensions import Extensions, extensions as default_extensions

DOMAIN = "helpdesk"
SERVICE = "helpdesk"
_PERMISSION = PermissionRequirement(service=SERVICE, mode=GateMode.ACCESS_ONLY)


# ── 저장소(예시라 메모리) ────────────────────────────────────────────────────

@dataclass
class Ticket:
    id: int
    title: str
    requester: str
    status: str = "open"
    resolution: str = ""


@dataclass
class TicketStore:
    tickets: Dict[int, Ticket] = field(default_factory=dict)
    notifications: List[str] = field(default_factory=list)

    @classmethod
    def sample(cls) -> "TicketStore":
        s = cls()
        for t in (Ticket(1, "VPN 접속 안 됨", "김민수"), Ticket(2, "모니터 교체 요청", "이영희"),
                  Ticket(3, "메일 용량 초과", "홍길동")):
            s.tickets[t.id] = t
        return s


store = TicketStore.sample()


# ── 도구 ─────────────────────────────────────────────────────────────────────

class _HelpdeskTool(BaseTool):
    permission_required = _PERMISSION
    # 관문(find_helpdesk_tools)이 붙여 준다 — 상시 바인딩하지 않는다.
    exposure = "retrieved"


class ListTicketsTool(_HelpdeskTool):
    def get_name(self) -> str:
        return "helpdesk_list_tickets"

    def get_description(self) -> str:
        return "헬프데스크 티켓 목록을 조회한다. status 로 거를 수 있다(open|closed)."

    def get_parameters(self) -> Dict[str, Any]:
        return {"status": {"type": "string", "enum": ["open", "closed"], "description": "상태 필터(선택)"}}

    def get_required_params(self) -> List[str]:
        return []

    async def _real_execute(self, status: Optional[str] = None, **kwargs) -> ToolResult:
        rows = [vars(t) for t in store.tickets.values() if not status or t.status == status]
        return ToolResult(success=True, output={"tickets": rows, "count": len(rows)})


class GetTicketTool(_HelpdeskTool):
    def get_name(self) -> str:
        return "helpdesk_get_ticket"

    def get_description(self) -> str:
        return "티켓 하나의 상세를 조회한다."

    def get_parameters(self) -> Dict[str, Any]:
        return {"ticket_id": {"type": "integer", "description": "티켓 번호"}}

    def get_required_params(self) -> List[str]:
        return ["ticket_id"]

    async def _real_execute(self, ticket_id: int, **kwargs) -> ToolResult:
        t = store.tickets.get(int(ticket_id))
        if t is None:
            return ToolResult(success=False, output=None, error=f"{ticket_id}번 티켓이 없습니다.")
        return ToolResult(success=True, output=vars(t))


class CloseTicketTool(_HelpdeskTool):
    """비가역 액션 — 승인 필요. confirm=False 면 초안 카드만 돌려주고 아무것도 바꾸지 않는다.

    코어가 비승인 턴에서 confirm 을 항상 False 로 강제하고(hitl), 실제 실행은 승인 재개(/resume)에서
    confirm=True 로 한 번만 일어난다. 도구는 그 계약만 지키면 된다.
    """

    def get_name(self) -> str:
        return "helpdesk_close_ticket"

    def get_description(self) -> str:
        return "티켓을 종료한다. 사람 승인이 필요하다 — 먼저 confirm=false 로 초안을 만든다."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "ticket_id": {"type": "integer", "description": "티켓 번호"},
            "resolution": {"type": "string", "description": "처리 내용"},
            "confirm": {"type": "boolean", "description": "false=초안, true=실행(승인 뒤에만)"},
        }

    def get_required_params(self) -> List[str]:
        return ["ticket_id", "resolution"]

    async def _real_execute(self, ticket_id: int, resolution: str, confirm: bool = False, **kwargs) -> ToolResult:
        t = store.tickets.get(int(ticket_id))
        if t is None:
            return ToolResult(success=False, output=None, error=f"{ticket_id}번 티켓이 없습니다.")
        if not confirm:
            return ToolResult(success=True, output={"view": {
                "app": DOMAIN, "kind": "draft", "title": f"티켓 #{t.id} 종료",
                "preview": f"{t.title} — {resolution}",
                "form": {"values": {"ticket_id": t.id, "resolution": resolution}},
            }})
        t.status, t.resolution = "closed", resolution
        return ToolResult(success=True, output={"view": {
            "app": DOMAIN, "kind": "result", "label": f"티켓 #{t.id} 종료됨", "result_note": "",
        }, "ticket": vars(t)})


TOOLS = (ListTicketsTool, GetTicketTool, CloseTicketTool)

HELPDESK_PROMPT = """당신은 사내 헬프데스크 담당 에이전트입니다.
티켓 조회·상세·종료만 다룹니다. 다른 업무 요청은 이 채널에서 처리할 수 없다고 안내하세요.
티켓 종료는 승인이 필요합니다 — 먼저 초안을 만들고, 사용자가 카드에서 승인해야 실제로 종료됩니다.
"""


async def _notify_requester(out: dict, ctx: dict) -> Optional[str]:
    """승인 후처리 예: 요청자에게 알림. 돌려준 문장은 승인 카드 결과 줄에 붙는다."""
    t = (out or {}).get("ticket") or {}
    if not t:
        return None
    store.notifications.append(f"{t.get('requester')}: #{t.get('id')} 종료 — {t.get('resolution')}")
    return f"{t.get('requester')} 님께 종료 알림을 보냈어요."


async def _glossary_terms() -> list:
    return [{"term_ko": "헬프데스크", "term_en": "Helpdesk", "term_zh": "服务台"}]


# ── 조립 ─────────────────────────────────────────────────────────────────────

def install(ext: Extensions = default_extensions) -> None:
    """헬프데스크 도메인을 코어에 끼운다. ``create_app(setup=install)`` 의 setup 으로 쓴다.

    여러 번 불러도 안전하다(테스트가 앱을 반복 생성한다).
    """
    from agent_arch.agents.tools.tool_search import register_domain

    instances = []
    for cls in TOOLS:
        tool = ToolRegistry.get(cls().name) or cls()
        if ToolRegistry.get(tool.name) is None:
            ToolRegistry.register(tool)
        permission_index.register(tool.name, tool.permission_required)
        instances.append(tool)

    register_domain(
        DOMAIN,
        gateway_tool="find_helpdesk_tools",
        intent="헬프데스크 — 사내 IT 지원 티켓 조회·상세 확인·종료 처리",
        examples=["'열린 티켓 보여줘'", "'3번 티켓 내용 알려줘'", "'2번 티켓 해결됐으니 닫아줘'"],
        tool_prefix="helpdesk_",
    )
    # 관문이 결정적으로 돌려줄 도구(검색 인덱스에 없는 도메인 도구).
    ext.domain_tools[DOMAIN] = lambda: list(instances)
    ext.approval_required.add("helpdesk_close_ticket")
    ext.action_done_messages["helpdesk_close_ticket"] = "티켓을 종료했어요."
    ext.post_approval["helpdesk_close_ticket"] = _notify_requester
    # 전용 채널 모드(예: 헬프데스크 포털에 붙은 채팅). 일반 채널에서는 supervisor 가 이 도메인으로 라우팅한다.
    ext.mode_prompts[DOMAIN] = lambda surface: HELPDESK_PROMPT
    ext.glossary_source = _glossary_terms


def create_example_app():
    """헬프데스크 도메인을 끼운 앱. ``uvicorn agent_arch.examples.helpdesk:create_example_app --factory``."""
    from agent_arch.main import create_app

    return create_app(setup=install)
