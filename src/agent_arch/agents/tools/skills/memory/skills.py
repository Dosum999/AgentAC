"""장기 사용자 메모리 조회 도구 (skill, in-process — 능동 navigation).

Phase 2 Slice 2a. 시스템 프롬프트에 자동 주입되는 메모리(passive)로 부족하거나,
사용자가 자신의 과거·설정을 직접 물을 때, 에이전트가 스스로 호출해 기억을 검색한다.

소유자 격리: get_user_context()의 sub 로 본인 record 만 조회한다(app/services/memory 가
SQL 레벨 격리까지 강제). find_memory_tools 로 발견되며(도메인 태깅은
permission_required.service="memory"), ENABLE_USER_MEMORY 가 꺼져 있으면 빈 결과를 준다.
"""
from typing import Any, Dict, List

from agent_arch.agents.context import get_user_context
from agent_arch.agents.permissions import GateMode, PermissionRequirement
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.registry import register_tool
from agent_arch.core.config import settings

# 로그인만 요구(PUBLIC). service="memory" 는 발견용 도메인 태깅을 겸한다.
_MEMORY = PermissionRequirement(service="memory", mode=GateMode.PUBLIC)


@register_tool
class MemorySearchTool(BaseTool):
    """사용자 장기 메모리에서 관련 기억(지시·별칭·선호·사실·과거 작업)을 검색한다."""

    permission_required = _MEMORY
    # 상시 바인딩(core): 회상 안내가 "부족하면 memory_search" 라고 가리키는데 발견 단계를 거쳐야 하면
    # 모델이 "그런 도구 없다"로 오답한다(v1 실측). 검색 도구 하나는 항상 손에 들려 둔다.
    exposure = "core"

    def get_name(self) -> str:
        return "memory_search"

    def get_description(self) -> str:
        return (
            "**목적:** 이 사용자의 장기 기억에서 관련 정보를 검색한다 — 과거에 밝힌 "
            "표준 지시·선호, 별칭(사용자 표현→실제 대상), 업무 사실, 지난 작업 이력.\n"
            "**사용 시점:** 사용자가 '전에 내가 뭐라 했지', '내 설정/선호 뭐야', '지난번 그거', "
            "'예전에 부탁한 거' 처럼 **자기 과거·설정**을 묻거나, 답하는 데 사용자의 과거 맥락이 "
            "필요한데 **이미 주어진 컨텍스트로 부족할 때** 쓴다.\n"
            "**쓰지 말 것:** 연차·근태·급여·재고·일정처럼 **실시간으로 바뀌는 값**은 여기 없다 — "
            "해당 도메인 도구로 최신 조회할 것. 이미 시스템에 주입된 메모리로 충분하면 호출 불필요.\n"
            "**반환:** [{id, type, content, confidence}] 리스트. 0건이면 빈 리스트. "
            "잘못된 기억을 지워야 하면 여기서 얻은 id 로 memory_forget 을 호출한다."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "기억에서 찾을 내용(자연어). 예: '표 관련 지시', 'scrum 채널', '음료 선호'",
            },
            "record_type": {
                "type": "string",
                "description": (
                    "특정 종류만 조회(생략 시 전체). 지시=instruction(directive) / "
                    "선호·사실=persona(preference, fact) / 사건=episodic / 별칭=alias(entity_alias)"
                ),
            },
        }

    def get_required_params(self) -> List[str]:
        return ["query"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output=None, error="query 가 필요합니다.")

        if not settings.enable_user_memory:
            return ToolResult(
                success=True,
                output={"memories": [], "count": 0, "note": "장기 메모리 기능이 비활성 상태입니다."},
            )

        ctx = get_user_context() or {}
        sub = ctx.get("sub")
        if not sub:
            return ToolResult(
                success=False, output=None,
                error="로그인 사용자만 장기 메모리를 조회할 수 있습니다.",
            )

        rtype_raw = str(kwargs.get("record_type") or "").strip() or None

        # 메모리 v2(MEMORY_V2_MODE=active) — 하이브리드 atom 검색. 종류는 v1/v2 이름 모두 받아 4종으로 정규화.
        from agent_arch.services.memory_v2 import is_active as _memory_v2_active
        if _memory_v2_active():
            from agent_arch.services.memory_v2.models import normalize_atom_type
            from agent_arch.services.memory_v2.recall import search_atoms

            rows = await search_atoms(sub, query, atom_type=normalize_atom_type(rtype_raw), limit=8)
            memories = [
                {
                    "id": r["id"], "type": r["atom_type"], "content": r["content"],
                    "priority": r.get("priority"), "scene": r.get("scene_name"),
                }
                for r in rows
            ]
            return ToolResult(success=True, output={"memories": memories, "count": len(memories)})

        return ToolResult(success=True, output={"note": "장기 메모리(v2)가 꺼져 있습니다."})


@register_tool
class MemoryRememberTool(BaseTool):
    """사용자가 명시적으로 밝힌 지시·선호·사실을 장기 기억에 저장한다(항상 사용 가능)."""

    permission_required = _MEMORY
    exposure = "core"  # "기억해둬"는 언제든 올 수 있으니 발견 없이 상시 바인딩.

    def get_name(self) -> str:
        return "memory_remember"

    def get_description(self) -> str:
        return (
            "**목적:** 사용자가 명시적으로 밝힌 표준 지시·선호·사실을 장기 기억에 저장한다.\n"
            "**사용 시점:** '앞으로 ~해', '표 쓰지 마', '기억해둬', '내 기본값은 ~', "
            "'나는 ~를 선호해' 처럼 **앞으로도 지켜야 할 지시나 안정적 선호**를 사용자가 말할 때. "
            "저장은 이 도구로 즉시 되니 '저장 기능이 없다'고 답하지 말 것.\n"
            "**쓰지 말 것:** 연차·근태·재고·오늘 일정처럼 **실시간으로 바뀌는 값**(시스템이 저장을 "
            "거부한다). 일회성 요청.\n"
            "**참고:** 저장 후 사용자에게 자연스럽게 확인하라. 별도로 시스템이 대화에서 자동으로도 "
            "기억하므로, 애매하면 이 도구를 굳이 부르지 않아도 된다.\n"
            "**반환:** {saved: bool}. saved=false 면 실시간 값이라 저장 대상이 아니라는 뜻."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "content": {
                "type": "string",
                "description": "저장할 내용을 완결된 한 문장으로. 예: '답변에 표를 쓰지 않는다', "
                "'오전 회의를 선호하지 않는다'.",
            },
            "record_type": {
                "type": "string",
                "description": "종류(기본 instruction=지시). instruction = 앞으로 어떻게 해 달라는 규칙"
                "('앞으로 ~해', '~할 때는 ~로 해', 작업별 기본값 — 예: '점심 상신 시 결재자는 김부장으로 한다') / "
                "persona = 사용자에 대한 사실·취향(예: '오전 회의를 선호하지 않는다') / episodic = 사건. "
                "행동 규칙을 persona 로 저장하면 참고 정보로만 쓰인다. v1 이름(directive/preference/fact)도 허용.",
            },
            "scope": {
                "type": "string",
                "enum": ["always", "when_relevant"],
                "description": "지시(instruction)의 적용 범위. always = 어떤 질문·작업에나 적용되는 응답 방식"
                "(서식·표 사용·호칭·언어·말투·답변 길이·서명 — 예: '답변에 표를 쓰지 않는다', '답변 끝에 서명을 붙인다') "
                "→ 매 답변에 항상 적용된다. when_relevant = 특정 작업·대상에만 쓰는 조건·기본값"
                "(예: '점심 상신 시 결재자는 김부장으로 한다') → 그 작업을 할 때만 적용된다. "
                "사용자가 '항상'이라고 말하지 않아도 응답 방식 전반에 관한 규칙이면 always. "
                "**범위를 사용자에게 되묻지 말고 네가 판단해 넣어라** — 애매하면 when_relevant.",
            },
        }

    def get_required_params(self) -> List[str]:
        return ["content"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        # LLM(특히 gemini)이 선언된 'content' 대신 'text'/'value' 로 호출하는 경우가 관측됨
        # → 인자명에 관대하게 받아 명시 저장이 조용히 실패하는 것을 막는다.
        content = str(
            kwargs.get("content") or kwargs.get("text") or kwargs.get("value") or ""
        ).strip()
        if not content:
            return ToolResult(success=False, output=None, error="content 가 필요합니다.")
        if not settings.enable_user_memory:
            return ToolResult(success=True, output={"saved": False, "note": "장기 메모리 기능이 비활성 상태입니다."})
        ctx = get_user_context() or {}
        sub = ctx.get("sub")
        if not sub:
            return ToolResult(success=False, output=None, error="로그인 사용자만 저장할 수 있습니다.")

        rtype_raw = str(kwargs.get("record_type") or "").strip() or None

        # 메모리 v2(active) — 즉시 L1 에 쓰되 자동 추출과 같은 판정기(중복 skip / 모순 update)를 거친다.
        from agent_arch.services.memory_v2 import is_active as _memory_v2_active
        if _memory_v2_active():
            from agent_arch.services.memory_v2.explicit import save_explicit_atom

            scope_raw = str(kwargs.get("scope") or "").strip() or None
            saved = await save_explicit_atom(sub, content, atom_type=rtype_raw, company_scope=ctx.get("company"),
                                             scope=scope_raw)
            if saved is None:
                return ToolResult(
                    success=True,
                    output={"saved": False, "note": "실시간으로 바뀌는 값이라 장기 저장 대상이 아닙니다(도구로 최신 조회하세요)."},
                )
            out: Dict[str, Any] = {"saved": True, "content": saved["content"], "type": saved["atom_type"], "id": saved["id"]}
            if saved["action"] == "skip":
                out["note"] = "이미 같은 내용이 기억돼 있어 그대로 유지했습니다."
            return ToolResult(success=True, output=out)

        return ToolResult(success=True, output={"note": "장기 메모리(v2)가 꺼져 있습니다."})


@register_tool
class MemoryForgetTool(BaseTool):
    """잘못 저장됐거나 더 이상 맞지 않는 기억 1건을 지운다(memory_remember 의 반대)."""

    permission_required = _MEMORY
    exposure = "core"  # "그거 잊어줘/취소"는 언제든 올 수 있으니 발견 없이 상시 바인딩.

    def get_name(self) -> str:
        return "memory_forget"

    def get_description(self) -> str:
        return (
            "**목적:** 이 사용자의 장기 기억 1건을 삭제한다 — 잘못 저장됐거나("
            "예: 일회성 요청이 지시로 굳음) 더 이상 맞지 않는 지시·선호·별칭·사실을 제거.\n"
            "**사용 시점:** '그 기억 지워/잊어줘', '표 쓰지 말라던 거 취소', '이제 그거 아니야', "
            "'그 별칭 틀렸어' 처럼 **저장된 기억을 취소·정정**하려 할 때.\n"
            "**사용법:** 먼저 memory_search 로 대상 record 의 id 를 찾은 뒤, 그 id 로 호출한다. "
            "기억 장면(요약 문서)을 지울 때는 id 대신 장면 이름을 넣는다. "
            "지운 내용을 사용자에게 확인하라. (정정=지우고 memory_remember 로 새로 저장.)\n"
            "**반환:** {forgot: bool, content}. forgot=false 면 해당 기억이 없거나 이미 지워진 것."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "record_id": {
                "type": "string",
                "description": "지울 기억의 id(memory_search 결과의 'id' 필드) 또는 지울 기억 장면의 이름.",
            },
        }

    def get_required_params(self) -> List[str]:
        return ["record_id"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        record_id = str(kwargs.get("record_id") or "").strip()
        if not record_id:
            return ToolResult(success=False, output=None, error="record_id 가 필요합니다.")
        if not settings.enable_user_memory:
            return ToolResult(success=True, output={"forgot": False, "note": "장기 메모리 기능이 비활성 상태입니다."})
        sub = (get_user_context() or {}).get("sub")
        if not sub:
            return ToolResult(success=False, output=None, error="로그인 사용자만 기억을 지울 수 있습니다.")

        from agent_arch.services.memory_v2 import is_active as _memory_v2_active
        if _memory_v2_active():
            from agent_arch.services.memory_v2.explicit import forget_atom, forget_scene, is_atom_id

            if not is_atom_id(record_id):
                # id 자리에 장면 이름이 오면 그 장면을 지운다. 둘 다 아니면 DB 에 넣지 않고 안내만 한다.
                scene = await forget_scene(sub, record_id)
                if scene is not None:
                    return ToolResult(success=True, output={"forgot": True, "type": "scene", "content": scene["title"]})
                return ToolResult(
                    success=True,
                    output={"forgot": False, "note": "id 형식이 아닙니다. memory_search 결과의 'id' 나 기억 장면 이름으로 다시 호출하세요."},
                )
            removed_v2 = await forget_atom(sub, record_id)
            if removed_v2 is None:
                return ToolResult(
                    success=True,
                    output={"forgot": False, "note": "해당 기억을 찾을 수 없거나 이미 지워졌습니다."},
                )
            return ToolResult(
                success=True,
                output={"forgot": True, "type": removed_v2["atom_type"], "content": removed_v2["content"]},
            )

        return ToolResult(success=True, output={"note": "장기 메모리(v2)가 꺼져 있습니다."})


@register_tool
class RecallConversationTool(BaseTool):
    """사용자의 과거 대화 원문(실제로 오갔던 말)을 검색한다 — 정제 메모리로는 못 담는 것."""

    permission_required = _MEMORY
    exposure = "retrieved"

    def get_name(self) -> str:
        return "recall_conversation"

    def get_description(self) -> str:
        return (
            "**목적:** 이 사용자의 **과거 대화 원문**을 검색한다 — 정제된 기억(memory_search)이 "
            "아니라, 실제로 주고받았던 대화 내용 자체. 다른 대화방/이전 세션의 것도 찾는다.\n"
            "**사용 시점:** '지난주에 뭐라고 답했지', '전에 이 얘기했던 대화 찾아줘', "
            "'예전에 물어본 거 다시' 처럼 **과거에 실제로 나눈 대화**를 되짚어야 할 때. "
            "요약된 사실이 아니라 원문 맥락이 필요할 때 memory_search 대신 이걸 쓴다.\n"
            "**쓰지 말 것:** 지금 대화창 안의 내용(이미 컨텍스트에 있음)이나 실시간 값 조회.\n"
            "**반환:** [{when, query, answer}] 리스트(유사도순). 0건이면 빈 리스트."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "과거 대화에서 찾을 주제/키워드(자연어).",
            },
            "limit": {
                "type": "integer",
                "description": "반환할 최대 대화 수 (기본 5, 최대 10).",
                "default": 5,
            },
        }

    def get_required_params(self) -> List[str]:
        return ["query"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output=None, error="query 가 필요합니다.")

        if not settings.enable_user_memory:
            return ToolResult(
                success=True,
                output={"conversations": [], "count": 0, "note": "장기 메모리 기능이 비활성 상태입니다."},
            )

        sub = (get_user_context() or {}).get("sub")
        if not sub:
            return ToolResult(
                success=False, output=None,
                error="로그인 사용자만 과거 대화를 조회할 수 있습니다.",
            )

        try:
            limit = min(int(kwargs.get("limit", 5)), 10)
        except (TypeError, ValueError):
            limit = 5

        from agent_arch.db.repository.memory_records import search_user_conversations

        rows = await search_user_conversations(sub, query, limit=limit)
        convos = [
            {
                "when": r["created_at"].isoformat() if r.get("created_at") else None,
                "query": r.get("query"),
                "answer": (r.get("answer") or "")[:400],
            }
            for r in rows
        ]
        return ToolResult(success=True, output={"conversations": convos, "count": len(convos)})


@register_tool
class MemoryReadSceneTool(BaseTool):
    """장기 메모리 장면 문서(v2 L2) 전문을 읽는다 — 시스템 프롬프트의 장면 내비게이션에서 깊이 읽을 때."""

    permission_required = _MEMORY
    exposure = "retrieved"

    def get_name(self) -> str:
        return "memory_read_scene"

    def get_description(self) -> str:
        return (
            "**목적:** 시스템 프롬프트의 '기억 장면' 목록에 있는 장면 하나의 **문서 전문**을 읽는다 — "
            "프로젝트·루틴·거래처 같은 맥락 덩어리가 서사로 정리돼 있다.\n"
            "**사용 시점:** 장면 목록의 한 줄 요약만으로 부족해 그 맥락 전체(경위·결정·관련 대상)가 "
            "필요할 때. 낱개 사실은 memory_search, 대화 원문은 recall_conversation 을 쓴다.\n"
            "**사용법:** name 에 장면 목록의 콜론 앞 이름을 그대로 넣는다.\n"
            "**반환:** {found, name, title, summary, content}. found=false 면 그 이름의 장면이 없다."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": {
                "type": "string",
                "description": "장면 이름(프롬프트 '기억 장면' 목록의 콜론 앞 값 그대로).",
            },
        }

    def get_required_params(self) -> List[str]:
        return ["name"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        name = str(kwargs.get("name") or kwargs.get("scene") or "").strip()
        if not name:
            return ToolResult(success=False, output=None, error="name 이 필요합니다.")
        if not settings.enable_user_memory:
            return ToolResult(success=True, output={"found": False, "note": "장기 메모리 기능이 비활성 상태입니다."})
        sub = (get_user_context() or {}).get("sub")
        if not sub:
            return ToolResult(success=False, output=None, error="로그인 사용자만 기억을 조회할 수 있습니다.")

        from agent_arch.services.memory_v2 import is_active as _memory_v2_active
        if not _memory_v2_active():
            return ToolResult(success=True, output={"found": False, "note": "장면 문서는 메모리 v2 가 활성일 때만 제공됩니다."})

        from agent_arch.db.repository.memory_v2 import get_scene

        scene = await get_scene(sub, name)
        if not scene:
            return ToolResult(
                success=True,
                output={"found": False, "note": "해당 이름의 장면이 없습니다. 프롬프트의 장면 목록 이름을 그대로 쓰세요."},
            )
        return ToolResult(
            success=True,
            output={
                "found": True, "name": scene["name"], "title": scene["title"], "summary": scene.get("summary"),
                "content": scene["content"],
                "updated_at": scene["updated_at"].isoformat() if scene.get("updated_at") else None,
            },
        )

