"""에이전트 오케스트레이터.

ReAct 루프 실행과 API 레이어 사이의 코디네이터 역할을 담당한다.
실제 루프 로직은 app.agents.react 패키지에 위임하며,
이 파일은 전처리(PII, 커맨드), 메시지 빌드, DB 저장만 책임진다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from enum import Enum
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from langsmith import get_current_run_tree, traceable

from agent_arch.agents.react import (
    LoopResult,
    ProgressEvent,
    guide_loop_events,
    merge_usage,
    react_loop_events,
)
from agent_arch.agents.react.link_policy import TurnLinkPolicy
from agent_arch.agents.graphs.react_graph import run_react_graph, stream_react_graph
from agent_arch.agents.graphs.supervisor import run_supervisor, stream_supervisor
from agent_arch.agents.graphs.router import (
    CROSS_DOMAIN_READERS,
    domain_from_tool_names,
    is_domainless_utility,
)
from agent_arch.agents.tools.action_view import build_action_artifact, extract_action_views
from agent_arch.agents.tools.dynamic_api_tool import DynamicAPITool
from agent_arch.agents.tools.exposure import core_tools, plan_tools
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.agents.react.history import recent_user_turns
from agent_arch.agents.context import (
    get_profile_context,
    set_attachment_context,
    set_conversation_id,
    set_image_context,
    set_profile_context,
    set_collecting_draft_context,
    set_prior_answer,
    set_recent_user_turns,
    reset_company_requests,
    set_rejected_draft_context,
    get_thread_context,
    set_turn_id,
    set_turn_input,
    set_turn_question,
    set_user_context,
    set_withheld_tools,
    withheld_tools,
    set_web_search_enabled,
    web_search_enabled,
)
from agent_arch.agents.permissions.gate import is_allowed
from agent_arch.agents.prompts.system import (
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
    append_screen_context,
    build_system_prompt,
    build_turn_context,
)
from agent_arch.services.user_profile import get_profile
from agent_arch.models.agent import ImageContext
from agent_arch.core.config import settings
from agent_arch.services.web.brave_output import WEB_TOOLS as _WEB_TOOL_NAMES
from agent_arch.db.repository import set_message_action_view
from agent_arch.extensions import domain_available, domain_tools, extensions
from agent_arch.llm.factory import LLMFactory
from agent_arch.memory.conversation import (
    load_history,
    load_prev_tool_names,
    load_collecting_draft,
    load_rejected_draft,
    mark_rejected_pin_used,
    rejected_pin_used,
    save_logged_in_turn,
    tool_outcome,
)
from agent_arch.services.api_registry import api_indexer, tool_loader
from agent_arch.utils.commands import handle_command
from agent_arch.utils.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    detect_language_with_history,
    normalize_language,
)
from agent_arch.utils.pii import mask_pii_text
from agent_arch.llm.pii.policy import global_mode as _pii_guard_mode
from agent_arch.utils.sanitize import StreamSanitizer, sanitize_tool_names
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)

# 하위 호환성: 기존 테스트에서 orchestrator 모듈로 직접 임포트하는 이름 유지
_merge_usage = merge_usage

# 출력 프로바이더 → 병렬 평가 프로바이더 토글 맵
_PARALLEL_PROVIDER_MAP: dict[str, str] = {
    "gemini": "vllm",
    "vllm":   "gemini",
    "openai": "vllm",
}

# 멀티모달(이미지) 입력을 지원하는 프로바이더 집합.
# 이미지 분석 시 주력 LLM 이 이 집합에 속하면 그대로 재사용하고,
# 아니면 settings.vision_llm_provider 인스턴스를 별도 생성한다.
_VISION_PROVIDERS: frozenset[str] = frozenset({"gemini", "openai", "anthropic"})

# 요청별 오버라이드를 허용하는 agent_mode 화이트리스트
# 코어 모드 + 도메인이 등록한 전용 채널 모드(extensions.mode_prompts). 판정 시점에 합친다(_valid_agent_modes).
_CORE_AGENT_MODES: frozenset[str] = frozenset({"default", "guide", "agent"})


def _valid_agent_modes() -> frozenset[str]:
    return _CORE_AGENT_MODES | frozenset(extensions.mode_prompts)


def _is_channel_scoped(mode: str) -> bool:
    """채널 스코핑 모드 — 전용 진입 경로가 도구·프롬프트를 단일 도메인으로 좁힌 모드.

    supervisor 라우팅이 그 스코핑을 우회하므로 제외 대상이다(_use_supervisor 참조). 도메인이
    extensions.mode_prompts 에 등록한 모드가 곧 전용 채널 모드다.
    """
    return mode in extensions.mode_prompts

# 채널 전용 도메인 — {도구 이름 접두사: 그 채널이 얹는 관문}.
# 이 도메인 도구는 전용 진입점에서 채널 봇 구성(exposure.apply_*_bot)이 런타임 주입할 때만
# 존재해야 한다. 관문이 전역 OFF 라 발견 경로는 이미 막혀 있지만, sticky 복원은 "직전 턴에
# 호출된 이름"을 그대로 되살리므로 같은 대화가 일반 채팅으로 이어지면 채널 밖에서 되살아난다
# — 그래서 이 턴에 관문이 바인딩돼 있을 때(= 그 채널일 때)만 복원한다(_sticky_prev_turn_tools).
# 예: {"app_": "find_app_tools"}. 전용 채널 도메인을 붙일 때 그 도메인의 접두사·관문을 여기 등록한다.
_CHANNEL_SCOPED_PREFIXES: dict[str, str] = {}

# 메모리 도구가 바인딩되지 않는 모드 — 이 모드에는 회상 안정 블록의 도구 지시를 빼고
# 준다(없는 도구를 가리키는 죽은 지시 방지).
#   guide: 가이드 검색 도구만 바인딩한다.
# 전용 채널 모드가 도구를 단일 도메인만으로 재구성한다면 그 모드도 여기 넣는다. 반대로 채널이
# 메모리 도구를 유지한다면(일반 채팅과 분리된 창이라 그 안에서 기억을 찾고 남길 경로가 필요할 때)
# 넣지 않는다 — 안내가 가리키는 도구가 실제로 손에 있어야 한다.
_MEMORY_TOOLLESS_MODES: frozenset[str] = frozenset({"guide"})


def _parse_guide_categories(raw: Optional[str]) -> Optional[List[str]]:
    """Form 입력 문자열을 카테고리 화이트리스트로 정규화한다.

    None/""/공백/'all'(대소문자 무시) → None (= 전체 카테고리 검색).
    'Messenger,Drive' → ['Messenger', 'Drive'] (공백 trim, 빈 토큰 제거).
    'all'이 다른 값과 섞여 있어도(예: 'all,Drive') all 우선 → None.

    Args:
        raw: Form 에서 받은 원본 문자열.

    Returns:
        정규화된 카테고리 리스트, 또는 None(전체 검색).
    """
    if not raw:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    if any(t.lower() == "all" for t in tokens):
        return None
    return tokens


class _ExecStrategy(str, Enum):
    REACT       = "react"       # data 모드: ReAct 루프 (실제 API 호출)
    GUIDE_LOOP  = "guide_loop"  # guide 모드: KB 벡터 검색 루프
    DIRECT_LLM  = "direct_llm" # tool calling 미지원 프로바이더: 직접 LLM 호출


_SERVICE_DISPLAY_NAMES: dict[str, str] = {
    "mattermost": "Mattermost",
    "hr": "HR 서비스",
}


def _extract_services(tool_names: list[str]) -> list[str]:
    """도구 이름 목록에서 고유한 서비스명을 추출한다.

    `{service}_service_*` 또는 `{service}_skill_*` 패턴에서 앞부분을 파싱하며,
    매핑 테이블이 있으면 사람이 읽기 좋은 이름으로 변환한다.

    Args:
        tool_names: 도구 이름 목록

    Returns:
        중복 없는 서비스 표시명 리스트
    """
    seen: set[str] = set()
    result: list[str] = []
    for name in tool_names:
        for sep in ("_service_", "_skill_"):
            if sep in name:
                prefix = name.split(sep)[0]
                display = _SERVICE_DISPLAY_NAMES.get(prefix, prefix.capitalize())
                if display not in seen:
                    seen.add(display)
                    result.append(display)
                break
    return result


def _last_assistant_text(messages: list) -> Optional[str]:
    """메시지 이력에서 **마지막 어시스턴트 답변 본문**을 찾는다. 없으면 None.

    도구 호출만 있고 본문이 빈 어시스턴트 턴은 건너뛴다 — 파일로 만들 내용이 아니다.

    Args:
        messages: _build_messages 가 만든 Message 리스트.

    Returns:
        직전 답변 전문. 찾지 못하면 ``None``.
    """
    for m in reversed(messages or []):
        if getattr(m, "role", None) != "assistant":
            continue
        text = (getattr(m, "content", "") or "").strip()
        if text:
            return text
    return None


def _serialize_messages_for_eval(messages: list) -> list[dict]:
    """final_messages를 DB JSONB 저장용 dict 리스트로 직렬화한다.

    replay-based eval은 main이 호출한 도구와 그 결과(role="tool" 메시지)가
    retrieval 컨텍스트의 핵심이므로 모두 포함한다. assistant의 tool_calls와
    tool 메시지의 tool_call_id/name도 함께 직렬화하여 컨텍스트 재현이 가능하다.

    Args:
        messages: Message 객체 리스트.

    Returns:
        agent.llm_evaluations.messages_context(JSONB) 컬럼에 저장할 dict 리스트.
    """
    out: list[dict] = []
    for m in messages:
        d: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        out.append(d)
    return out


def _flatten_tool_messages_for_replay(messages: list) -> list:
    """eval replay용으로 구조적 tool_call/tool 결과를 평문 텍스트로 펼친다.

    Gemini 3.x thinking 모델은 요청 contents 안의 모든 functionCall part 마다
    thought_signature 를 강제 검증한다. 이 검증은 **서버측**이라 thinking_budget=0 으로
    꺼도, AFC(automatic_function_calling)를 비활성해도 우회되지 않는다(실측 확인).
    다른 프로바이더(vllm 등)가 만든 tool_call 은 시그니처가 없어 그대로 replay 하면
    `default_api:<tool>` 위치에서 400 INVALID_ARGUMENT 가 난다.

    eval 은 도구를 바인딩하지 않은 synthesis 전용이므로, 도구 호출/결과를 평문으로
    펼쳐 functionCall/functionResponse part 자체를 없앤다 → 검증 대상이 사라진다.
    프로바이더 불문 동작하며, retrieval 컨텍스트(어떤 도구를 어떤 인자로 호출해 무엇을
    받았는지)는 텍스트로 그대로 보존된다.

    Args:
        messages: main 루프 종료 시점의 Message 리스트.

    Returns:
        functionCall/functionResponse part 가 제거된 Message 리스트.
    """
    from agent_arch.llm.base import Message

    out: list = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            lines: list[str] = []
            if m.content:
                lines.append(m.content)
            for tc in m.tool_calls:
                name = tc.get("name", "?")
                args = tc.get("arguments", {})
                lines.append(f"[도구 호출] {name}({json.dumps(args, ensure_ascii=False)})")
            out.append(Message(role="assistant", content="\n".join(lines)))
        elif m.role == "tool":
            label = m.name or (m.tool_call_id or "").replace("call_", "", 1) or "tool"
            out.append(Message(role="user", content=f"[도구 결과] {label}: {m.content}"))
        else:
            out.append(m)
    return out


@dataclass
class _RequestContext:
    """전처리 완료 후 실행 분기에 전달되는 요청 컨텍스트.

    command_result가 None이 아닌 경우 슬래시 커맨드가 처리된 것이며,
    호출자는 즉시 반환 또는 yield해야 한다.

    conversation_id는 진입 시 None일 수 있다. save_logged_in_turn의 lazy-create로
    첫 메시지 저장 시 확정되며, 호출자가 ctx.conversation_id를 갱신해야 응답·SSE
    done 이벤트에 동봉할 수 있다.
    """

    user_sub: Optional[str]
    user_email: Optional[str]
    user_info: Optional[dict]
    langsmith_run_id: Optional[str]
    messages: list
    input: str                           # 원본 입력 (Redis 저장용)
    augmented_input: str                 # 이미지 컨텍스트가 주입된 입력 (LLM 호출용)
    t_start: float
    command_result: Optional[dict]
    conversation_id: Optional[str] = None         # 진입 시 None 가능, save 후 확정
    image_context: Optional[ImageContext] = None  # 이미지 분석 결과
    profile: Optional[dict] = None                # Frappe profile (권한 게이트용, user_info와 분리)
    target_language: str = DEFAULT_LANGUAGE       # 이 턴의 답변 목표 언어(요청 필드 또는 입력 감지)


class AgentOrchestrator:
    """에이전트 오케스트레이터.

    ReAct 루프 실행 로직(app.agents.react 패키지)을 감싸고
    API 레이어와 분리된 에이전트 실행 인터페이스를 제공한다.

    Attributes:
        llm_provider: 사용할 LLM 프로바이더 이름
        enable_pii_masking: PII 마스킹 활성화 여부
        llm: LLM 인스턴스
        static_tools: 정적으로 등록된 도구 리스트
        guide_tool: 가이드 모드 전용 KB 검색 도구(extensions.guide_tool 이 없으면 None)
    """

    # 요청 단위 옵션의 클래스 기본값 — 엔드포인트(_apply_web_search)가 인스턴스에 덮어쓴다.
    # 여기 두는 이유: 테스트·내부 경로는 __init__ 을 거치지 않고 인스턴스를 만들기도 하는데, 그때도 전략 판정이
    # 안전한 기본값(웹 꺼짐·기존 답변 경로)으로 떨어져야 한다.
    web_search_enabled: bool = False
    # 어느 표면(surface)의 프롬프트를 넣을지 — 메신저 봇이 "messenger" 로 바꾼다(기본은 웹 그대로).
    # 전용 채널 모드 프롬프트 슬롯(extensions.mode_prompts[mode])이 이 값을 받아 표면별 사본을 고른다.
    prompt_surface: str = "web"
    # 이번 턴에 막을 도구 {이름: 사유} — 채널의 공개 범위 정책이 넣는다(요청자에게 보이면 안 되는 조회 등).
    withheld_tools: Optional[dict] = None

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        enable_pii_masking: Optional[bool] = None,
        agent_mode: Optional[str] = None,
        force_parallel_eval: bool = False,
        guide_categories: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """AgentOrchestrator를 초기화한다.

        Args:
            llm_provider: 사용할 LLM 프로바이더 (None이면 기본값 사용)
            enable_pii_masking: PII 마스킹 활성화 여부 (None이면 설정값 사용)
            agent_mode: 사용할 에이전트 모드 (None이면 settings.agent_mode 사용).
                허용값: "default", "guide", "agent" + extensions.mode_prompts 에 등록된 전용 채널 모드.
            force_parallel_eval: settings.enable_parallel_eval 환경 설정과 무관하게
                병렬 평가를 강제 실행할지 여부. 오디오 엔드포인트처럼 입력 채널이
                고정되어 항상 비교 데이터를 쌓고 싶은 경우에 사용한다.
            language: 답변 목표 언어(ko/en/zh/ja). 클라이언트가 명시한 경우에만 넘긴다.
                None(기본)이면 턴마다 사용자 입력에서 감지하고, 판정이 안 되면 직전
                발화 → settings.default_answer_language 순으로 떨어진다.
            guide_categories: Guide 모드에서 검색할 카테고리 화이트리스트 원본 문자열.
                ``None`` / 빈값 / ``"all"`` 은 전체 카테고리 검색. 콤마 구분으로
                ``"Messenger,Drive"`` 처럼 전달한다. ``_parse_guide_categories`` 를 거쳐
                ``self.guide_categories`` 에 ``List[str] | None`` 형태로 저장된다.

        Raises:
            ValueError: agent_mode가 허용 목록에 없는 경우.
        """
        # 빈 문자열·None은 settings.agent_mode로 fallback (llm_provider와 동일 동작)
        if agent_mode and agent_mode not in _valid_agent_modes():
            raise ValueError(
                f"Unknown agent_mode: {agent_mode}. "
                f"Available modes: {', '.join(sorted(_valid_agent_modes()))}"
            )
        self.agent_mode = agent_mode or settings.agent_mode
        # 웹 검색 opt-in(요청 단위). 엔드포인트가 패널 토글 ∧ 회사 설정으로 계산해 넣는다. 기본 닫힘.
        self.web_search_enabled = False
        # 프론트엔드 별칭(예: "gemma" → "vllm")을 canonical 이름으로 정규화한 뒤 저장.
        # _PARALLEL_PROVIDER_MAP 매칭, DB·LangSmith 메타 일관성을 위해 항상 canonical 사용.
        self.llm_provider = LLMFactory.normalize_provider(llm_provider) or settings.default_llm_provider
        self.enable_pii_masking = (
            enable_pii_masking if enable_pii_masking is not None
            else settings.enable_pii_masking
        )
        self.force_parallel_eval = force_parallel_eval
        # 클라이언트가 명시한 답변 언어(지원하지 않는 코드는 None 으로 떨어뜨려 감지에 맡긴다).
        self.language: Optional[str] = normalize_language(language)
        self.guide_categories: Optional[List[str]] = _parse_guide_categories(guide_categories)
        self.llm = LLMFactory.create(self.llm_provider)
        # CORE 티어(상시 바인딩) 도구. 1단계에서는 전 도구 기본값이 core라
        # get_enabled_tools()와 동일하다 — 티어 조정은 tools_config.yaml에서.
        self.static_tools = core_tools()
        # 가이드 모드 KB 검색 도구 — 도메인이 등록한 팩토리가 없으면 가이드 모드는 도구 없이 돈다.
        self.guide_tool = extensions.guide_tool() if extensions.guide_tool is not None else None

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    async def _analyze_image_context(
        self,
        images: list,
        text_input: str,
    ) -> Optional[ImageContext]:
        """이미지를 분석하여 화면 컨텍스트를 추출한다.

        이미지 분석은 멀티모달을 지원하는 프로바이더가 필요하다. 주력 LLM 이
        비전 지원 프로바이더(_VISION_PROVIDERS: gemini·openai)면 그대로 재사용하고,
        아니면 settings.vision_llm_provider 인스턴스를 별도 생성한다.

        Args:
            images: 분석할 이미지 목록
            text_input: 사용자 원본 입력 텍스트

        Returns:
            분석 결과 ImageContext. 실패 시 None
        """
        import json as _json
        from agent_arch.llm.base import ImageContent, Message

        try:
            analysis_llm = (
                self.llm if self.llm_provider in _VISION_PROVIDERS
                else LLMFactory.create(settings.vision_llm_provider)
            )
            image_contents = [
                ImageContent(
                    data=img.data.split(",", 1)[-1] if img.data.startswith("data:") else img.data,
                    mime_type=img.mime_type,
                )
                for img in images
            ]
            user_content = text_input or "이 화면을 분석해 주세요."
            analysis_messages = [
                Message(role="system", content=IMAGE_ANALYSIS_SYSTEM_PROMPT),
                Message(role="user", content=user_content, images=image_contents),
            ]
            response = await analysis_llm.invoke(analysis_messages)
            raw = response.content.strip()
            # ```json ... ``` 코드블록 제거
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            data = _json.loads(raw)
            return ImageContext(**data)
        except Exception as e:
            logger.warning("이미지 컨텍스트 추출 실패 (Stage 2로 fallback): %s", e)
            return None

    async def _resolve_turn_language(self, text: str, conversation_id: Optional[str]) -> str:
        """이 턴의 답변 목표 언어를 정한다: 요청 명시 → 입력 감지 → 직전 발화 → 기본값.

        입력에서 판정이 안 되는 턴("ok", "3번", HITL continuation 의 고정 지시문)은
        히스토리의 최근 사용자 발화로 물려받는다 — 그렇지 않으면 영어로 대화하다
        "ok" 한 마디에 답변이 한국어로 돌아온다. 히스토리 조회는 **판정이 안 될 때만**
        하므로 정상 턴에 Redis 왕복이 추가되지 않는다.

        Args:
            text: 이번 턴의 사용자 발화(없으면 "" — 히스토리 상속만 노린 호출).
            conversation_id: 히스토리 조회용 대화방 UUID. None 이면 상속 없음.

        Returns:
            지원 언어 코드(ko/en/zh/ja).
        """
        explicit = getattr(self, "language", None)
        if explicit:
            return explicit
        fallback = normalize_language(settings.default_answer_language) or DEFAULT_LANGUAGE
        detected = detect_language(text, fallback=None)
        if detected:
            return detected
        # 여기부터가 판정 불가 턴 — 이때만 히스토리를 읽는다(정상 턴에 왕복 추가 금지).
        history_texts: list = []
        if conversation_id:
            try:
                history = await load_history(f"conversation:{conversation_id}")
                history_texts = [
                    (getattr(m, "content", "") or "")
                    for m in reversed(history)
                    if getattr(m, "role", None) == "user"
                ]
            except Exception:
                logger.warning("[language] 히스토리 조회 실패 — 기본 언어로 진행", exc_info=True)
        return detect_language_with_history(text, history_texts, fallback=fallback)

    async def _build_messages(
        self,
        user_info: Optional[dict],
        input: str,
        images: Optional[list] = None,
        conversation_id: Optional[str] = None,
        recall_query: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list:
        """LLM 호출용 메시지 리스트를 구성한다.

        stateless 프로바이더는 user 메시지만, 나머지는
        system prompt + 히스토리 + user 메시지 순으로 조합한다.

        히스토리는 conversation_id 단위 Redis 키(f"conversation:{id}")에서 로드한다.
        conversation_id가 None이면(신규 대화방 lazy-create 경로) 빈 히스토리로 시작.

        Args:
            user_info: 요청자 정보 (sub, email, company 등)
            input: 사용자 입력 텍스트
            images: 첨부 이미지 목록 (ImageContent 또는 ImageInput 호환 객체)
            conversation_id: 대상 대화방 UUID. None이면 빈 히스토리.
            language: 답변 목표 언어(ko/en/zh/ja). 시스템 프롬프트의 답변 언어 지시가
                이 언어로 렌더된다. None 이면 한국어(기존 동작).
            recall_query: 장기 메모리 회상에 쓸 질의. None 이면 input. "" 이면 턴별 회상(별칭·관련
                기억)을 건너뛰고 안정 블록만 넣는다 — HITL 재개처럼 input 이 사용자 발화가 아닌
                고정 지시문일 때(그 문장으로 임베딩 검색하면 무관 기억이 딸려온다).

        Returns:
            Message 리스트
        """
        from agent_arch.llm.base import ImageContent, Message

        if self.llm.stateless:
            return [Message(role="user", content=input)]

        # 장기 사용자 메모리 회상 (ENABLE_USER_MEMORY ∧ MEMORY_V2_MODE=active 게이트).
        # 안정 블록(페르소나·장면 내비)은 system 쪽 턴 컨텍스트, 턴별 블록은 마지막 user 에.
        memory_block = ""
        memory_stable_block = ""
        always_rules: list = []
        if settings.enable_user_memory and user_info and user_info.get("sub"):
            try:
                from agent_arch.services.memory_v2 import is_active as _memory_v2_active
                if _memory_v2_active():
                    from agent_arch.agents.context import set_memory_recall
                    from agent_arch.services.memory_v2.recall import (
                        recall_for_turn as _recall_v2,
                        strip_tool_guidance,
                    )
                    recall_v2 = await _recall_v2(
                        user_info.get("sub"),
                        input if recall_query is None else recall_query,
                        company_scope=user_info.get("company"),
                    )
                    # 구조체를 턴 전체에 흘린다 — 도구 로드아웃·라우터 힌트·워커 스코프·종합기가 읽는다.
                    set_memory_recall(recall_v2)
                    memory_block, memory_stable_block = recall_v2.turn_block, recall_v2.system_block
                    always_rules = [a.get("content") for a in (recall_v2.strict or []) if isinstance(a, dict)]
                    # 메모리 도구를 안 물리는 모드(가이드·도구를 재구성하는 전용 채널)는 안내를 뺀다 —
                    # 부를 수 없는 도구를 가리키는 죽은 지시가 되고, "저장 도구가 없다고 답하지 말
                    # 것"까지 못 박혀 "기억해둬" 요청에서 막다른 길이 된다. 페르소나·전역 지시·장면
                    # 요약은 도구 없이도 유효하므로 남긴다.
                    if self.agent_mode in _MEMORY_TOOLLESS_MODES and memory_stable_block:
                        memory_stable_block = strip_tool_guidance(memory_stable_block)
            except Exception:
                logger.exception("메모리 recall 실패 — 메모리 없이 진행")

        # 가변 블록은 전부 시스템 프롬프트에서 빼 마지막 user 메시지 뒤로 보낸다
        # (build_turn_context 참조). 턴마다 바뀌는 날짜/시각·메모리는 같은 사용자의
        # 다음 턴을 깨고, 사용자마다 다른 요청자 정보·메모리 안정 블록은 그 뒤의 도구
        # 스키마(7,536토큰)까지 사용자 단위로 쪼갠다. 둘 다 빼야 base_prompt + 출력규칙
        # + 도구 스키마가 전 사용자 공유 프리픽스가 된다.
        # 용어 사전은 **질문에 나온 용어만** 턴 컨텍스트에 싣는다. 사전 전체를 시스템
        # 프롬프트에 두면 용어 수만큼 모든 턴이 길어지고, 질문마다 고르면 시스템 프롬프트가
        # 달라져 공유 프리픽스가 깨진다. 질문에 없던 용어가 답변에 나오는 경우는 최종 교정이
        # 사전 전체로 잡는다. 조회는 Redis 캐시, 실패하면 빈 블록이라 답변을 막지 않는다.
        glossary_block = ""
        try:
            from agent_arch.agents.prompts.system import render_glossary_block
            from agent_arch.services.glossary import (
                get_glossary_terms,
                match_terms,
                render_glossary_lines,
            )
            matched = match_terms(input, await get_glossary_terms())
            glossary_block = render_glossary_block(
                render_glossary_lines(matched, language, input), language)
        except Exception:
            logger.warning("용어 사전 주입 실패 — 용어 없이 진행", exc_info=True)

        system_prompt = build_system_prompt(
            user_info, mode=self.agent_mode, include_turn_context=False,
            include_user_context=False,
            language=language,
            surface=getattr(self, "prompt_surface", "web"),
        )
        turn_context = build_turn_context(
            memory_block,
            user_info=user_info,
            memory_stable_block=memory_stable_block or None,
            # 메신저 봇이 실어 보낸 현재 위치(채널·스레드). 브라우저 패널은 None 이다.
            origin=(user_info or {}).get("origin"),
            # 지금 서 있는 스레드의 대화(메신저 전용). 이력에는 에이전트와 주고받은 턴만 있어서
            # 사람들끼리 나눈 말은 여기로만 들어온다.
            thread_posts=get_thread_context(),
            glossary_block=glossary_block,
            always_rules=always_rules,
        )
        # conversation_id/user_sub 를 함께 넘기면 Redis 가 비었을 때 Postgres 에서 되살린다
        # (24시간 지난 대화에서 "어제 그거" 가 되게). 없으면 예전처럼 빈 이력이다.
        history = (
            await load_history(
                f"conversation:{conversation_id}",
                conversation_id=conversation_id,
                user_sub=(user_info or {}).get("sub"),
            )
            if conversation_id
            else []
        )

        image_contents = None
        if images:
            image_contents = [
                ImageContent(
                    data=img.data.split(",", 1)[-1] if img.data.startswith("data:") else img.data,
                    mime_type=img.mime_type,
                )
                for img in images
            ]

        # turn_context 는 질문 **뒤**에 붙인다: 이력에는 raw input 만 저장되므로
        # system + 지난 턴 전체가 바이트 동일해져 그대로 캐시에서 나온다.
        user_content = f"{input}\n\n{turn_context}" if turn_context else input
        # 화면 경로는 전용 채널에서만, 턴 컨텍스트 **뒤에** 싣는다 — 일반 채널에는 '이거' 를
        # 화면 개체로 해석하는 도구가 없어 토큰만 든다. 끝에 붙어야 앞쪽이 바이트 동일하게
        # 유지된다(프리픽스 캐시). 값은 게이트웨이가 검증해 user_info 에 실어 준 경로다.
        screen_path = (user_info or {}).get("source_path") if _is_channel_scoped(self.agent_mode) else None
        user_content = append_screen_context(user_content, screen_path)

        return [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_content, images=image_contents),
        ]

    def _kickoff_eval_replay(self, ctx: _RequestContext) -> None:
        """main 루프 종료 후 호출 — eval 프로바이더에 final messages를 replay한다.

        settings.enable_parallel_eval 또는 self.force_parallel_eval이 True이고
        토글 매핑이 존재할 때만 발사된다.
        asyncio.create_task로 background 실행되어 main 응답 latency에 영향 없음.
        main이 정상 종료된 시점의 messages 스냅샷(tool_call + tool 결과 포함)을
        그대로 복사해 넘기므로 두 프로바이더가 동일한 retrieval 컨텍스트를 본다.

        Args:
            ctx: 전처리 + main 루프 완료 후의 요청 컨텍스트.
        """
        parallel_provider = _PARALLEL_PROVIDER_MAP.get(self.llm_provider)
        if not parallel_provider:
            return
        if not (self.force_parallel_eval or settings.enable_parallel_eval):
            return
        run = get_current_run_tree()
        if run and ctx.langsmith_run_id:
            run.add_metadata({"has_parallel_eval": True})
        # list(...)로 컨테이너 얕은 복사 — main이 이후 mutate해도 격리된다.
        # Pydantic Message 객체는 내부 mutation 패턴이 없으므로 deepcopy 불필요.
        # 평가 단위는 conversation_id. v19에서 agent.llm_evaluations.session_id 컬럼이
        # conversation_id로 RENAME되어 의미가 정합 완료된 상태.
        asyncio.create_task(
            self._run_parallel_eval(
                input=ctx.input,
                final_messages=list(ctx.messages),
                conversation_id=ctx.conversation_id,
                user_sub=ctx.user_sub,
                user_email=ctx.user_email,
                agent_mode=self.agent_mode,
                eval_provider=parallel_provider,
                main_run_id=ctx.langsmith_run_id,
            )
        )

    @traceable(run_type="chain", name="parallel_eval")
    async def _run_parallel_eval(
        self,
        input: str,
        final_messages: list,
        conversation_id: Optional[str],
        agent_mode: str,
        eval_provider: str,
        user_sub: Optional[str] = None,
        user_email: Optional[str] = None,
        main_run_id: Optional[str] = None,
    ) -> None:
        """main의 final messages를 replay하여 eval 프로바이더의 synthesis를 저장한다.

        main이 어떤 도구를 어떤 query로 호출했고 어떤 KB chunks를 받았는지가 그대로
        final_messages에 누적되어 있다(`role="assistant"` tool_calls + `role="tool"`
        결과). eval 프로바이더는 도구를 바인딩하지 않은 채 `tool_mode="NONE"`으로
        synthesis만 수행하여 동일 컨텍스트 위에서 모델 능력을 비교 가능하게 한다.

        실패 시 예외를 외부로 전파하지 않는다 (주 LLM 흐름에 영향 없음).

        Args:
            input: 사용자 원본 질문.
            final_messages: main 루프 종료 시점의 messages 스냅샷.
            conversation_id: 평가 단위 대화방 UUID.
            agent_mode: 현재 에이전트 모드.
            eval_provider: 병렬 평가에 사용할 LLM 프로바이더.
            user_sub: 호출자 Keycloak sub (운영 가시성용, 선택).
            user_email: 호출자 이메일 (audit 스냅샷, 선택).
            main_run_id: 주 LLM의 LangSmith run ID (두 trace 연결용).
        """
        _logger = logging.getLogger(__name__)
        try:
            # 두 trace를 같은 질문으로 LangSmith에서 묶을 수 있도록 main_run_id를 메타데이터로 기록
            if main_run_id:
                run = get_current_run_tree()
                if run:
                    run.add_metadata({"main_run_id": main_run_id})

            local_llm = LLMFactory.create(eval_provider)
            # 도구 바인딩하지 않고 tool_mode="NONE"으로 강제 synthesis.
            # replay 메시지엔 다른 프로바이더가 만든 tool_call(시그니처 없음)이 섞여 있어,
            # gemini 3.x thinking 모델에 functionCall part 그대로 넘기면 thought_signature
            # 누락으로 400 난다. thinking_budget=0·AFC 비활성으로도 우회 불가(서버측 검증)이므로
            # 도구 호출/결과를 평문으로 펼쳐 functionCall part 자체를 제거한다.
            replay_messages = _flatten_tool_messages_for_replay(final_messages)
            t0 = time.monotonic()
            response = await local_llm.invoke(replay_messages, tool_mode="NONE")
            elapsed = time.monotonic() - t0
            # 저장소는 확장 슬롯이다(extensions.eval_sink). 없으면 로그만 남긴다.
            if extensions.eval_sink is not None:
                await extensions.eval_sink(
                    conversation_id=conversation_id,
                    user_sub=user_sub,
                    user_email=user_email,
                    query=input,
                    answer=response.content,
                    llm_provider=eval_provider,
                    model_name=local_llm.model,
                    response_time=elapsed,
                    agent_mode=agent_mode,
                    messages_context=_serialize_messages_for_eval(final_messages),
                    main_run_id=main_run_id,
                )
            _logger.info(
                "[PARALLEL_EVAL] provider=%s 완료 elapsed=%.2fs len=%d msgs=%d",
                eval_provider, elapsed, len(response.content), len(final_messages),
            )
        except Exception as e:
            _logger.warning(
                "[PARALLEL_EVAL] provider=%s 실행 실패 (주 LLM에 영향 없음): %s",
                eval_provider, e,
            )

    @staticmethod
    def _sticky_prev_turn_tools(messages: list, known_names: set) -> list:
        """직전 턴에 사용된 도구를 바인딩 목록에 되살린다 (sticky 보정).

        검색 기반 노출(retrieved 티어·동적 API)은 현재 턴 발화만 보므로
        "그거 상위 3개로 보여줘" 같은 대명사 후속턴에서 직전 턴 도구가
        top_k에서 빠질 수 있다. load_history가 주입한 role="tool" 메시지의
        도구 이름을 ToolRegistry(정적·MCP) → llm_tools 캐시(동적) 순으로
        resolve해 유지한다. 권한 게이트는 호출자에서 일괄 적용된다.

        Args:
            messages: 현재 턴 LLM 메시지 리스트 (tool_ctx 주입 완료 상태)
            known_names: 이미 바인딩 예정인 도구 이름 집합

        Returns:
            복원된 도구 리스트 (어디서도 resolve 안 되는 이름은 스킵)
        """
        recovered = []
        seen = set(known_names)
        for msg in messages:
            if getattr(msg, "role", None) != "tool":
                continue
            name = getattr(msg, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            # 도메인 관문(find_*)은 tools_config enabled 를 존중한다:
            # 관문이 전역 OFF(예: 전용 채널에서만 여는 도메인 관문)인데 과거 대화
            # sticky 로 일반 채팅에 되살아나면, 격리해 둔 도메인이 그 관문을 통해
            # 다시 발견된다. 일반 도구는 종전대로 무조건 복원(후속턴 연속성).
            if name.startswith("find_") and not settings.get_tool_config(name).get(
                "enabled", True
            ):
                logger.debug("[prepare_tools] 비활성 관문 %s sticky 복원 스킵", name)
                continue
            # 채널 전용 도메인은 그 채널일 때만 되살린다 (관문 바인딩 여부로 판별).
            gateway = next(
                (g for pre, g in _CHANNEL_SCOPED_PREFIXES.items() if name.startswith(pre)),
                None,
            )
            if gateway and gateway not in known_names:
                logger.debug("[prepare_tools] 채널 전용 도구 %s sticky 복원 스킵", name)
                continue
            # 웹은 요청 단위 opt-in — 직전 턴이 웹을 썼어도 이번 요청에서 토글이 꺼졌으면 되살리지 않는다.
            if (name in _WEB_TOOL_NAMES or name == "find_web_tools") and not web_search_enabled():
                logger.debug("[prepare_tools] 웹 검색 꺼짐 — %s sticky 복원 스킵", name)
                continue
            # 기능 플래그·요청 단위 opt-in 으로 열리는 도메인(extensions.domain_available)도 같다 — 이번
            # 요청에서 닫혔으면(꺼진 회사·무인 실행 턴 등) 직전 턴이 썼어도 되살리지 않는다.
            if not domain_available(domain_from_tool_names([name])):
                continue
            instance = ToolRegistry.get(name)
            if instance is not None:
                recovered.append(instance)
                continue
            tool_def = tool_loader.get_by_name(name)
            if tool_def is not None:
                recovered.append(DynamicAPITool(tool_def))
            else:
                logger.debug("[prepare_tools] 직전 턴 도구 %s resolve 불가 — 스킵", name)
        if recovered:
            logger.info(
                "[prepare_tools] 직전 턴 도구 유지(sticky): %s",
                [t.name for t in recovered],
            )
        return recovered

    async def _prepare_tools(
        self,
        messages: list,
        profile: Optional[dict] = None,
    ) -> list:
        """동적 도구를 검색·권한 게이트·bind 하고 허용된 도구 리스트를 반환한다.

        bind 직전 권한 게이트로 도구를 필터링한다. 게이트에서 거부된 도구는
        LLM에 노출되지 않으므로 환각 호출도 차단된다.

        가이드 KB 의 **프롬프트 자동주입**은 하지 않는다(매뉴얼 grounding 으로 액션 요청이
        거절로 새던 문제). 대신 가이드 KB 도구가 CORE 도구로 등록되어 static_tools 를 타고
        노출된다 — LLM 이 how-to 질문에서만 선택 호출한다.

        Args:
            messages: 현재 대화 메시지 리스트 (직전 턴 sticky 도구 복원에 사용).
            profile: Frappe profile (권한 게이트 평가용). None 이면 모든 도구 deny.

        Returns:
            권한 게이트를 통과해 bind 된 allowed_tools 리스트.
        """
        # (A안) passive 도구 dump 제거 — 직접호출 도구는 도메인 탐색 도구
        # (find_<도메인>_tools)로만 발견·바인딩한다. 사전 검색이 잡탕 노이즈를
        # 깔아 모델이 헤매고 도메인 탐색 도구가 무용지물 되던 문제 차단.
        known_names = {t.name for t in self.static_tools}
        retrieved = self._sticky_prev_turn_tools(messages, known_names)
        # 기억 로드아웃 — 이 턴에 매칭된 별칭이 못 박는 도메인의 도구를 탐색 홉 없이 미리 든다.
        have = known_names | {t.name for t in retrieved}
        retrieved += [t for t in self._memory_hint_tools() if t.name not in have]
        # 인덱스 밖 도메인 도구 — 기능 플래그·요청 단위 opt-in 으로 켜졌을 때만 결정적으로 든다
        # (검색 인덱스에 없는 도구다). 꺼진 요청에서는 팩토리가 빈 목록을 돌려준다.
        for domain in extensions.domain_tools:
            have |= {t.name for t in retrieved}
            retrieved += [t for t in domain_tools(domain) if t.name not in have]

        allowed_tools = plan_tools(self.static_tools, retrieved, profile)
        # 공개 범위로 이번 턴에 막은 도구 — 직전 턴 도구 복원(sticky)으로도 되살아나지 않게 바인딩에서 뺀다.
        held = withheld_tools()
        if held:
            allowed_tools = [t for t in allowed_tools if t.name not in held]
        await self.llm.bind_tools(allowed_tools)

        all_tools = self.static_tools + retrieved
        denied = [t.name for t in all_tools if t not in allowed_tools]
        logger.debug(
            "[prepare_tools] sticky=%s static=%s allowed=%d denied=%s",
            [t.name for t in retrieved],
            [t.name for t in self.static_tools],
            len(allowed_tools),
            denied,
        )
        return allowed_tools

    def _save_kwargs(
        self,
        user_sub: str,
        input: str,
        result: LoopResult,
        t_start: float,
        user_email: Optional[str],
        dynamic_tools_count: int = 0,
        **extra,
    ) -> dict:
        """save_logged_in_turn 호출에 필요한 키워드 인자를 구성한다.

        Args:
            user_sub: 사용자 식별자
            input: 사용자 입력
            result: ReAct 루프 결과
            t_start: 턴 시작 시각 (time.monotonic())
            user_email: 사용자 이메일
            dynamic_tools_count: 동적 도구 수 (메타데이터용)
            **extra: 추가 키워드 인자 (save_to_redis 등)

        Returns:
            save_logged_in_turn 키워드 인자 딕셔너리
        """
        tu = result.token_usage
        return dict(
            tool_calls=result.tool_calls or None,
            tool_results=result.tool_results or None,
            response_time=time.monotonic() - t_start,
            llm_invoke_count=result.llm_invoke_count,
            llm_provider=self.llm_provider,
            model_name=self.llm.model,
            user_email=user_email,
            prompt_tokens=tu["prompt_tokens"] or None,
            completion_tokens=tu["completion_tokens"] or None,
            total_tokens=tu["total_tokens"] or None,
            finish_reason=result.finish_reason,
            metadata=({"react_steps": result.step_meta} if getattr(result, "step_meta", None) else None),
            **extra,
        )

    async def _preprocess_request(
        self,
        input: str,
        user_info: Optional[dict],
        images: Optional[list],
        conversation_id: Optional[str] = None,
    ) -> _RequestContext:
        """공통 전처리: LangSmith 태그, 슬래시 커맨드, PII 마스킹, 컨텍스트 설정, 메시지 빌드.

        슬래시 커맨드가 감지된 경우 command_result 필드에 결과가 채워지며,
        호출자는 이 경우 즉시 반환 또는 yield해야 한다.

        Args:
            input: 사용자 입력 텍스트
            user_info: 요청자 정보
            images: 첨부 이미지 목록

        Returns:
            전처리 완료된 요청 컨텍스트
        """
        user_sub = user_info.get("sub") if user_info else None
        user_email = user_info.get("email") if user_info else None

        run = get_current_run_tree()
        langsmith_run_id: Optional[str] = str(run.id) if run else None
        if run:
            tags = ["agent_mode", self.llm_provider]
            if self.agent_mode == "guide":
                tags.append("guide")
            run.add_tags(tags)
            if user_info:
                run.add_metadata({
                    "user_id":    user_sub,
                    "user_email": user_email,
                    "company":    user_info.get("company"),
                    "session_id": user_info.get("session_id"),
                    # 사용량 관측의 서비스 축. DB 적재가 정본이지만 트레이스에도 남겨 두면
                    # 특정 면의 트레이스만 골라 보는 게 가능해진다.
                    "service":    user_info.get("service"),
                })

        command_result = await handle_command(input, self.llm_provider, user_sub=user_sub)
        if command_result is not None:
            return _RequestContext(
                user_sub=user_sub,
                user_email=user_email,
                user_info=user_info,
                langsmith_run_id=langsmith_run_id,
                messages=[],
                input=input,
                augmented_input=input,
                t_start=0.0,
                command_result=command_result,
                conversation_id=conversation_id,
            )

        if self.enable_pii_masking and _pii_guard_mode() == "off":
            # 가명화 래퍼(app/llm/pii)가 켜진 턴은 별표 마스킹을 건너뛴다 — 먼저 돌면 복원할 원문이 없다.
            input = mask_pii_text(input)

        set_user_context(user_info or {})
        # 웹 검색 opt-in — 엔드포인트가 계산해 인스턴스에 둔 값을 턴 컨텍스트로(비동기 잡은 컨텍스트가 이어지지 않는다).
        set_web_search_enabled(getattr(self, "web_search_enabled", False))
        if getattr(self, "web_search_enabled", False):
            from agent_arch.agents.context import remember_web_urls
            remember_web_urls(re.findall(r"https?://[^\s<>\"')\]]+", input or ""))   # 사용자가 준 URL 은 web_fetch 로 열 수 있다
        set_withheld_tools(getattr(self, "withheld_tools", None))
        # 대화 ID 를 요청 스코프 ContextVar 에 심는다 — 대화 스코프 Redis stash(영수증 OCR
        # document_body)를 읽고 쓰는 스킬(영수증 판독·문서 상신 래퍼 등)이 참조한다.
        set_conversation_id(conversation_id)
        # 이번 턴 라우트는 답이 저장될 때 확정 대화 id 로 남긴다(memory/conversation.begin_turn_route).
        from agent_arch.memory.conversation import begin_turn_route
        begin_turn_route()
        if _pii_guard_mode() != "off":
            # 가명화 어휘(임직원 디렉터리) 예열 — LLM 호출 직전까지 2초만 기다리고, 늦으면 다음 턴부터 완전해진다.
            from agent_arch.llm.pii.turn import prewarm as _pii_prewarm
            _pii_prewarm()
        # 첨부 이미지를 요청 스코프 ContextVar 에 심는다 — OCR 계열 스킬(예:
        # 영수증 판독)이 tool args 없이 "이번 요청 이미지"를 읽는다.
        # _analyze_image_context 의 gather 및 이후 execute_tools 자식 Task 가 상속.
        set_image_context(images)
        # file-service 로 영속화된 첨부 메타를 심는다 — 엔드포인트가 인스턴스 속성
        # (pending_attachments)으로 전달. async job 은 엔드포인트와 다른 Task 라
        # ContextVar 가 전파되지 않으므로, run/stream 이 실제 실행되는 이 지점에서
        # set 해야 이후 저장 지점(save_logged_in_turn)이 같은 컨텍스트로 읽는다.
        set_attachment_context(getattr(self, "pending_attachments", None))
        # 첨부 사실은 어느 도구를 부르든 변하지 않는다. 증빙 참조를 영수증 OCR 스킬만
        # 만들다 보니, 모델이 사진을 직접 읽고 OCR 을 건너뛴 턴에서는 상신이 증빙 없이
        # 나갔다(실측 2026-09-29 메신저/gemini). 대화 스코프에 남겨 나중 턴의 create 가
        # 이어받게 한다 — 패널·메신저가 같은 이 지점을 지난다.
        from agent_arch.memory.conversation import save_turn_attachments

        await save_turn_attachments(conversation_id, getattr(self, "pending_attachments", None))
        t_start = time.monotonic()

        # 이미지 컨텍스트 추출 + Frappe 프로필 prefetch (병렬)
        #   - 이미지 분석: 이미지 있을 때만
        #   - 프로필 prefetch: 권한 게이트 + guide_kb 가시성 판정용. Redis 캐시 hit이면 ~1ms
        # 두 작업이 독립적이므로 asyncio.gather로 묶어 wall-time을 최대치 RTT 1회로 제한.

        async def _maybe_image_ctx():
            if not images:
                return None
            return await self._analyze_image_context(images, input)

        async def _maybe_fetch_profile():
            # 권한 게이트용 Frappe profile prefetch. user_info와는 분리해서 보관
            # (라인 102-106의 권한 raw 분리 정책 유지). Redis hit이면 ~1ms.
            if not user_sub:
                return None
            try:
                return await get_profile(user_sub)
            except Exception:
                logger.warning(
                    "auth profile prefetch 실패 — 권한 게이트 fail-safe로 도구 차단",
                    exc_info=True,
                )
                return None

        image_context, profile = await asyncio.gather(
            _maybe_image_ctx(),
            _maybe_fetch_profile(),
        )

        # 프로필을 별도 ContextVar 에 흘려보낸다 — guide_kb 청크 가시성 판정의 입력.
        # user_info 와 분리해 두어 권한 raw(service_access/assigned_roles/permissions)
        # 가 system prompt 빌더로 흘러들 위험을 차단한다.
        set_profile_context(profile)
        # 발화 원문 — 도구가 인자에서 떨어진 정보를 되찾는 데 쓴다(회사 토큰 등). 주입 전 raw.
        set_turn_input(input)
        # 첨부 본문이 입력에 붙은 턴이면 엔드포인트가 사용자 질문만 따로 둔다(agents/context.set_turn_question).
        set_turn_question(getattr(self, "turn_question", None))
        set_turn_id(uuid.uuid4().hex)   # 도구의 "물었다" 기록이 같은 턴 재호출과 실제 사용자 답을 가르는 기준
        reset_company_requests()     # 도구(자식 태스크)가 제자리 수정할 기록 dict 를 부모가 만든다

        augmented_input = input
        if image_context:
            context_text = (
                f"\n\n[이미지 컨텍스트]\n"
                f"화면: {image_context.screen_description}\n"
                f"의도: {image_context.user_intent}\n"
                f"검색어: {image_context.suggested_query}"
            )
            augmented_input = (input + context_text).strip()

        # 직전 턴에 [거절]된 초안이 있으면 그 인자 원본을 이번 발화에 얹는다 — "9시 말고
        # 10시로" 같은 정정 턴이 되쓸 값을 갖게 하고, **라우터**가 이 블록을 보고 조회가
        # 아닌 실행으로 분류하게 한다(라우터는 augmented_input 을 그대로 받는다).
        # 워커까지 전달되는 경로는 supervisor._scope_messages 의 이월이 담당한다.
        from agent_arch.agents.hitl.policy import build_rejected_draft_note

        rejected_draft = await load_rejected_draft(conversation_id)
        if rejected_draft:
            # 도구 **지목**은 거절 직후 정정 턴 하나를 위한 장치다. stash 는 TTL 5분이라
            # 그대로 두면 그 사이 모든 턴이 같은 도구로 묶여, 사용자가 그만두겠다고 해도
            # 평문으로 답할 길이 없다(실측 2026-09-22). 한 번 쓴 뒤로는 끈다 —
            # ANY 강제는 유지되므로 모델이 도구를 고를 수는 있다.
            rejected_draft = {
                **rejected_draft,
                "pin_used": await rejected_pin_used(conversation_id),
            }
            await mark_rejected_pin_used(conversation_id)
        set_rejected_draft_context(rejected_draft)
        if rejected_draft:
            augmented_input = (augmented_input + build_rejected_draft_note(rejected_draft)).strip()

        # 되물을 값이 남은 초안이 진행 중이면 그 포인터를 컨텍스트에 심는다 — supervisor 가
        # 짧은 답변 턴("1번")을 실행으로 태우는 근거다. **라우터 입력에는 넣지 않는다**:
        # 값은 이미 stash 에 있고, 블록을 더 얹으면 라우터 출력만 길어져 punt 위험이 커진다
        # (원문보존 A안에서 "라우터에 규칙 추가는 역행"으로 정리한 것과 같은 판단).
        collecting = await load_collecting_draft(conversation_id)
        set_collecting_draft_context(collecting)

        # 답변 목표 언어를 먼저 정한다 — 시스템 프롬프트의 언어 지시가 이 값으로 렌더되고,
        # 같은 값이 그래프의 언어 가드로도 내려간다(프롬프트와 가드가 어긋나면 가드가 이긴다).
        target_language = await self._resolve_turn_language(input, conversation_id)
        messages = await self._build_messages(
            user_info, augmented_input, images, conversation_id=conversation_id,
            language=target_language,
        )
        # 직전 답변을 요청 스코프에 심는다 — create_file 이 본문 인자 없이도 "방금 그 내용"
        # 을 파일로 만들 수 있게 한다(app/agents/context.py 주석의 실측 근거 참조).
        set_prior_answer(_last_assistant_text(messages))
        set_recent_user_turns(recent_user_turns(messages))   # 도구의 근거 검증용(이번 턴 제외 최근 사용자 발화)

        # parallel_eval 발사는 main 루프 종료 후 _kickoff_eval_replay()에서 수행한다.
        # 그래야 main이 호출한 도구·tool 결과가 누적된 final_messages를 eval이 그대로
        # replay할 수 있다 (두 프로바이더가 동일 retrieval 컨텍스트 공유).

        return _RequestContext(
            user_sub=user_sub,
            user_email=user_email,
            user_info=user_info,
            langsmith_run_id=langsmith_run_id,
            messages=messages,
            input=input,
            augmented_input=augmented_input,
            t_start=t_start,
            command_result=None,
            conversation_id=conversation_id,
            image_context=image_context,
            profile=profile,
            target_language=target_language,
        )

    # ── run() 실행 분기 ───────────────────────────────────────────────────────

    async def _run_guide(self, ctx: _RequestContext) -> dict:
        """가이드 모드를 실행하고 결과 딕셔너리를 반환한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트

        Returns:
            에이전트 실행 결과 딕셔너리
        """
        result = None
        async for event in guide_loop_events(
            self.llm,
            ctx.augmented_input,
            ctx.messages,
            self.guide_tool,
            categories=self.guide_categories,
        ):
            if isinstance(event, LoopResult):
                result = event

        final_content = result.content if result else ""
        message_id = None
        if ctx.user_sub and final_content:
            message_id, ctx.conversation_id = await save_logged_in_turn(
                ctx.conversation_id, ctx.user_sub, ctx.input, final_content,
                **self._save_kwargs(ctx.user_sub, ctx.input, result, ctx.t_start, ctx.user_email,
                                   langsmith_run_id=ctx.langsmith_run_id),
            )
        # main 루프 종료 후 eval 프로바이더에 final messages를 replay
        self._kickoff_eval_replay(ctx)
        return {
            "success": True,
            "output": final_content,
            "llm_provider": self.llm_provider,
            "tool_calls": result.tool_calls if result else [],
            "message_id": message_id,
            "conversation_id": ctx.conversation_id,
            "metadata": {
                "finish_reason": result.finish_reason if result else "error",
                "usage": result.token_usage if result else None,
                "dynamic_tools_count": 0,
                "agent_mode": self.agent_mode,
                # 가이드로 답할 수 없는 질문(abstain)이면 프론트에 에이전트 모드 전환을 제안한다.
                # 프론트는 suggested_mode 를 감지해 확인 버튼을 띄우고, 확인 시 같은 질문을
                # agent_mode="agent" 로 재요청한다.
                **(
                    {"suggested_mode": "agent", "abstain_reason": result.abstain_reason}
                    if result and result.abstain
                    else {}
                ),
            },
        }

    def _resolve_strategy(self) -> _ExecStrategy:
        """현재 모드와 프로바이더 능력으로 실행 전략을 결정한다.

        Returns:
            실행 전략 Enum 값
        """
        # 가이드 모드는 KB 검색 도구가 등록돼 있을 때만 전용 루프를 탄다. 없으면 일반 경로로 둔다 —
        # 검색할 KB 없이 "매뉴얼에서 찾아 답하라"는 루프를 돌리면 매번 못 찾았다는 답만 나온다.
        if self.agent_mode == "guide" and getattr(self, "guide_tool", None) is not None:
            return _ExecStrategy.GUIDE_LOOP
        if not self.llm.supports_function_calling:
            return _ExecStrategy.DIRECT_LLM
        return _ExecStrategy.REACT

    async def _run_direct_llm(self, ctx: _RequestContext) -> dict:
        """도구 없이 LLM을 직접 호출한다.

        프로바이더가 tool calling을 지원하지 않을 때 사용한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트

        Returns:
            에이전트 실행 결과 딕셔너리
        """
        logger.debug("[run] provider=%s (no tool calling) → direct LLM invoke (no tools)", self.llm_provider)
        token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        response = await self.llm.invoke(ctx.messages, user_sub=ctx.user_sub)
        merge_usage(token_usage, response.usage)
        # 내부 도구 이름이 본문에 인용되는 경우를 결정적으로 차단
        sanitized_content = sanitize_tool_names(response.content) if response.content else response.content

        body_content = sanitized_content

        message_id = None
        if ctx.user_sub and body_content:
            message_id, ctx.conversation_id = await save_logged_in_turn(
                ctx.conversation_id, ctx.user_sub, ctx.input, body_content,
                response_time=time.monotonic() - ctx.t_start,
                llm_invoke_count=1,
                save_to_redis=not self.llm.stateless,
                llm_provider=self.llm_provider,
                model_name=self.llm.model,
                user_email=ctx.user_email,
                prompt_tokens=token_usage["prompt_tokens"] or None,
                completion_tokens=token_usage["completion_tokens"] or None,
                total_tokens=token_usage["total_tokens"] or None,
                langsmith_run_id=ctx.langsmith_run_id,
                finish_reason=response.finish_reason,
            )
        elif ctx.user_sub:
            # 빈 답변은 저장하지 않는다 — 모니터링에서 이 턴이 사라지지 않도록 이벤트로 남긴다
            # (source 가 orchestrator 라 common_app 의 unsaved_turn 상관 규칙에 걸린다).
            logger.warning(
                "빈 답변 — 턴을 저장하지 않음",
                extra=event_extra("exception", conversation_id=ctx.conversation_id),
            )
        # main 응답 완료 후 eval 프로바이더에 동일 messages를 replay
        self._kickoff_eval_replay(ctx)

        metadata: dict = {
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "dynamic_tools_count": 0,
            "agent_mode": self.agent_mode,
        }

        return {
            "success": True,
            "output": body_content,
            "llm_provider": self.llm_provider,
            "tool_calls": [],
            "message_id": message_id,
            "conversation_id": ctx.conversation_id,
            "metadata": metadata,
        }

    def _router_llm(self):
        """라우터 전용 fresh(언바운드) llm.

        메인 self.llm 은 _prepare_tools 에서 도구가 바인딩돼 있어 라우터 invoke 시 JSON
        대신 도구 호출을 시도할 수 있다. 새 인스턴스로 그 위험을 차단한다. 과부하 분리를
        위해 ROUTER_LLM_PROVIDER 로 다른 프로바이더(예: vllm)를 쓸 수 있다.
        """
        return LLMFactory.create(settings.router_llm_provider or self.llm_provider)

    def _make_worker_llm(self, domain: Optional[str] = None):
        """워커 전용 fresh llm — 병렬 워커가 각자 다른 스코프 도구를 bind 하므로 인스턴스 격리.

        도메인이 주어지면 llm_providers.yaml ``worker_overrides.<domain>.<provider>`` 를 프로바이더 설정 위에 얹는다
        (예: web 워커만 추론 켜기). 항목이 없으면 대화 레인과 같다.
        """
        provider = LLMFactory.normalize_provider(self.llm_provider) or settings.default_llm_provider
        overrides = ((settings.llm_providers.get("worker_overrides") or {}).get(domain or "") or {}).get(provider) or {}
        return LLMFactory.create(self.llm_provider, **dict(overrides))

    @property
    def _use_supervisor(self) -> bool:
        """supervisor(멀티도메인 라우팅)를 쓸지 여부.

        채널 스코핑 모드(extensions.mode_prompts 에 등록된 전용 채널)는
        supervisor 를 건너뛴다 — supervisor 의 router 도메인 분류가 채널 스코핑
        (도메인 전용 도구·프롬프트)을 우회해, 채널 밖 질의(예: 전용 앱 채널에서의 근태)를
        HR 등 워커로 라우팅해 도구를 실제 실행해 버리기 때문. 전용 채널은 단일
        도메인이므로 supervisor 없이 평범한 ReAct 경로로 채널 전용 프롬프트 + 채널
        도구(_prepare_tools 스코핑)만 태운다. 채널 밖 질의는 그 프롬프트대로 안내/
        일반 답변으로 처리된다.
        """
        return settings.use_langgraph_supervisor and not _is_channel_scoped(self.agent_mode)

    @property
    def _pending_draft_force(self) -> bool:
        """진행 중 초안(수집 중·거절됨) 턴에 첫 도구 호출을 강제할지.

        **채널 스코핑 모드에서만** 켠다. 그 모드는 supervisor 를 건너뛰므로
        (`_use_supervisor`) `_bias_to_action_for_pending_draft` 의 실행 승격이 걸리지
        않는다 — 여기서 그 구멍을 메운다.

        `not self._use_supervisor` 로 두지 않는 이유: `use_langgraph_supervisor` 는
        **기본값이 False** 라(`config.py`), 그 플래그를 세우지 않은 배포에서는 모든 채널이
        True 가 된다. 강제(ANY)까지는 논리가 서지만 **도구 지목**이 함께 켜지는 것이 문제다
        — 캘린더 초안을 거절한 뒤 "내일 일정 뭐야?" 가 생성 도구로 지목된다. 게다가
        `collecting_draft`·`rejected_draft` 는 전자결재도 쓰는 공유 stash 라 남의 도메인
        까지 닿는다. 그래서 환경변수가 아니라 **검증한 채널 집합**에 건다.
        """
        return _is_channel_scoped(self.agent_mode)

    def _worker_global_tools(self, profile: Optional[dict]) -> list:
        """supervisor 워커에 항상 얹을 도메인 무관 글로벌 유틸리티(권한 게이트 통과분).

        모놀리식 경로는 CORE 도구를 전부 바인딩하지만 워커는 도메인 스코프만 받는다 →
        calculator/디렉터리/웹검색 같은 횡단 도구가 워커에서 빠져 "김철수한테 메일" 류에서
        수신자 해석 등이 불가해진다. 여기서 골라 supervisor 로 넘겨 워커 스코프에 합친다.
        첨부 읽기(CROSS_DOMAIN_READERS)는 도메인 귀속 도구지만 같은 이유로 함께 넘긴다.
        """
        return [
            t for t in self.static_tools
            if (is_domainless_utility(t) or t.name in CROSS_DOMAIN_READERS)
            and is_allowed(t.name, profile)
        ]

    def _worker_sticky_tools(self, messages: list, profile: Optional[dict]) -> list:
        """직전 턴 sticky 도구(권한 게이트 통과분) — 참조 후속턴 패리티용.

        모놀리식 _prepare_tools 가 붙이던 직전 턴 도구를 워커에도 제공한다(supervisor 가
        도메인별로 필터링). static 도구는 이미 core/global 로 커버되므로 여기선 동적
        도구(직전 턴에 검색·바인딩됐던 것)만 잡힌다.
        """
        known = {t.name for t in self.static_tools}
        sticky = [t for t in self._sticky_prev_turn_tools(messages, known) if is_allowed(t.name, profile)]
        # 기억 로드아웃 — supervisor 가 _domain_sticky 로 도메인별 워커에 나눠 준다(sticky 와 같은 문).
        have = known | {t.name for t in sticky}
        sticky += [t for t in self._memory_hint_tools() if t.name not in have and is_allowed(t.name, profile)]
        return sticky

    def _memory_hint_tools(self) -> list:
        """이 턴 회상 별칭이 못 박는 도메인의 도구(탐색 도구·승인 필요 도구 제외). 실패·비활성 → [].

        승인 필요(쓰기) 도구는 빼는 이유: 사용자가 요청하지도 않은 쓰기 도구를 미리 들려 주면 환각 액션
        표면이 넓어진다. 필요하면 모델이 find_<domain>_tools 로 찾는다.
        """
        if _is_channel_scoped(self.agent_mode):
            # 전용 채널의 static_tools 는 이미 그 도메인 전체다. 기억 별칭이
            # 못 박는 도메인은 정의상 '다른 도메인'이라, 들려 주면 채널 격리가 깨진다
            # (프롬프트는 그 도메인을 모르는데 도구만 들어오는 불일치이기도 하다).
            return []
        if not getattr(settings, "memory_tool_hint", True):
            return []
        try:
            from agent_arch.agents.context import get_memory_recall
            from agent_arch.agents.hitl.policy import requires_approval
            from agent_arch.agents.tools.registry import ToolRegistry
            from agent_arch.agents.tools.tool_search import _DOMAIN_TOOLS
            from agent_arch.services.memory_v2.loadout import alias_domains, select_tools_for_domains

            recall = get_memory_recall()
            domains = alias_domains(recall, _DOMAIN_TOOLS.keys()) if recall is not None else []
            if not domains:
                return []
            tools = select_tools_for_domains(
                domains, ToolRegistry.get_enabled_tools(),
                lambda name: domain_from_tool_names([name]),
                exclude=requires_approval,
            )
            if tools:
                logger.info("[memory_loadout] domains=%s tools=%s", domains, [t.name for t in tools])
            return tools
        except Exception:
            logger.debug("[memory_loadout] 힌트 도구 계산 실패(무시)", exc_info=True)
            return []

    async def _prior_domain(self, conversation_id: Optional[str]) -> Optional[str]:
        """직전 턴 도구에서 도메인을 도출한다(후속턴 상속용). 없으면 기억 별칭이 못 박는 도메인. 실패 → None."""
        try:
            prior = domain_from_tool_names(await load_prev_tool_names(conversation_id))
        except Exception:
            logger.debug("[supervisor] prior_domain 도출 실패(무시)", exc_info=True)
            prior = None
        if prior or not getattr(settings, "memory_tool_hint", True):
            return prior
        try:
            from agent_arch.agents.context import get_memory_recall
            from agent_arch.agents.tools.tool_search import _DOMAIN_TOOLS
            from agent_arch.services.memory_v2.loadout import hint_domain

            recall = get_memory_recall()
            hinted = hint_domain(recall, _DOMAIN_TOOLS.keys()) if recall is not None else None
            if hinted:
                logger.info("[memory_loadout] router hint domain=%s (별칭)", hinted)
            return hinted
        except Exception:
            return None

    async def _collect_react_result(self, input: str, messages: list, all_tools: list,
                                    target_language: str = DEFAULT_LANGUAGE):
        """비스트리밍 ReAct 최종 LoopResult 를 얻는다 — flag 로 그래프/기존 루프 분기.

        USE_LANGGRAPH_REACT=True 면 LangGraph 그래프(run_react_graph), 아니면
        기존 react_loop_events. 두 경로는 동일 LoopResult 를 낸다(패리티 검증됨).
        """
        if settings.use_langgraph_react:
            return await run_react_graph(
                self.llm, input, messages, all_tools,
                self._search_dynamic_tools, settings.max_iterations,
                target_language=target_language,
                pending_draft_force=self._pending_draft_force,
            )
        result = None
        async for event in react_loop_events(
            self.llm, input, messages, all_tools,
            self._search_dynamic_tools, settings.max_iterations,
        ):
            if isinstance(event, LoopResult):
                result = event
        return result

    async def _run_react(self, ctx: _RequestContext) -> dict:
        """표준 ReAct 루프를 실행하고 결과 딕셔너리를 반환한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트

        Returns:
            에이전트 실행 결과 딕셔너리
        """
        all_tools = await self._prepare_tools(ctx.messages, ctx.profile)

        if self._use_supervisor:
            result = await run_supervisor(
                self.llm, ctx.augmented_input, ctx.messages,
                resolve_tools_fn=self._search_dynamic_tools,
                make_worker_llm_fn=self._make_worker_llm,
                fallback_fn=lambda: self._collect_react_result(
                    ctx.augmented_input, ctx.messages, all_tools, ctx.target_language),
                max_iterations=settings.max_iterations,
                confidence_threshold=settings.router_confidence_threshold,
                router_llm=self._router_llm(),
                search_fn=self._search_dynamic_tools,
                prior_domain=await self._prior_domain(ctx.conversation_id),
                global_tools=self._worker_global_tools(ctx.profile),
                sticky_tools=self._worker_sticky_tools(ctx.messages, ctx.profile),
                target_language=ctx.target_language,
            )
        else:
            result = await self._collect_react_result(
                ctx.augmented_input, ctx.messages, all_tools, ctx.target_language)

        message_id = None
        if ctx.user_sub and result and result.content:
            message_id, ctx.conversation_id = await save_logged_in_turn(
                ctx.conversation_id, ctx.user_sub, ctx.input, result.content,
                **self._save_kwargs(ctx.user_sub, ctx.input, result, ctx.t_start, ctx.user_email,
                                   langsmith_run_id=ctx.langsmith_run_id),
            )
        # main 루프 종료 후 eval 프로바이더에 final messages를 replay
        self._kickoff_eval_replay(ctx)
        return {
            "success": True,
            "output": result.content if result else "",
            "llm_provider": self.llm_provider,
            "tool_calls": result.tool_calls if result else [],
            "message_id": message_id,
            "conversation_id": ctx.conversation_id,
            "metadata": {
                "finish_reason": result.finish_reason if result else "error",
                "usage": result.token_usage if result else None,
                "dynamic_tools_count": len(all_tools),
                "agent_mode": self.agent_mode,
                # 도구 결과 요약(개수만) — 예약 실행기가 "답은 냈지만 조회는 못 한 턴"을 실패로 가른다.
                "tool_outcome": tool_outcome(result.tool_results if result else None),
            },
        }

    @traceable(run_type="chain", name="agent_continuation")
    async def _run_react_continuation(
        self, conversation_id: str, user_info: dict, *, profile: Optional[dict] = None
    ) -> dict:
        """HITL 액션 실행 후 '한 턴 더' 이어 돌려(continuation) 다음 승인필요 액션 또는
        마무리 텍스트를 얻는다. 비스트림·ctx-free·eval-free.

        이전에 실행된 액션은 conversation history(평문 확인 메시지)에 영속돼 있으므로 LLM 이
        이미 한 일을 알고 남은 후속 작업만 계획한다. 승인필요 도구를 만나면 loop 가
        confirm=False 초안 큐로 끊고 LoopResult.pending_queue 를 돌려준다.

        Returns:
            {"content": str, "pending_action": dict|None, "pending_queue": list|None}
        """
        # resume continuation 은 agent_run/stream 래퍼 밖(/resume 경로)에서 독립 실행되므로,
        # 트레이스 루트에 회사/사용자를 직접 부착한다(_preprocess_request 와 동일 패턴).
        # 미부착 시 continuation LLM 호출이 company 없는 root 로 기록되어 사용량 대시보드에서
        # "미매핑"으로 집계된다.
        run = get_current_run_tree()
        if run and user_info:
            run.add_tags(["agent_continuation", self.llm_provider])
            run.add_metadata({
                "user_id": user_info.get("sub"),
                "user_email": user_info.get("email"),
                "company": user_info.get("company"),
                "session_id": user_info.get("session_id"),
            })

        empty = {"content": "", "pending_action": None, "pending_queue": None}
        if self.llm.stateless:
            return empty  # 히스토리 없는 프로바이더는 continuation 불가

        instruction = (
            "앞선 작업의 실행 결과를 반영해, 사용자의 원 요청을 완수하기 위한 남은 후속 "
            "작업이 있으면 이어서 진행하세요. 남은 작업이 없으면 한두 문장으로 간단히 "
            "마무리만 하세요. (이미 완료된 작업을 다시 실행하지 마세요.)"
        )
        # continuation 의 instruction 은 코드가 만든 한국어 고정 문장이라 그걸로 언어를
        # 판정하면 영어·중국어 대화가 여기서 한국어로 뒤집힌다. 빈 문자열을 넘겨
        # 히스토리(직전 사용자 발화)에서만 물려받는다.
        target_language = await self._resolve_turn_language("", conversation_id)
        messages = await self._build_messages(
            user_info, instruction, conversation_id=conversation_id, recall_query="",
            language=target_language,
        )
        set_recent_user_turns(recent_user_turns(messages))   # continuation 턴에도 도구 근거 검증용 최근 발화
        if profile is None and user_info and user_info.get("sub"):
            try:
                profile = await get_profile(user_info["sub"])
            except Exception:
                profile = None
        set_profile_context(profile)
        all_tools = await self._prepare_tools(messages, profile)

        # continuation 은 일반 에이전트 턴(읽기/채널해결+다음 액션 초안)이므로 메인 루프와
        # 동일 반복 budget 을 준다. 짧게 캡하면 읽기에 다 쓰고 액션 초안 전에 끊긴다.
        result = await self._collect_react_result(instruction, messages, all_tools, target_language)
        if result is None:
            return empty
        return {
            "content": result.content or "",
            "pending_action": getattr(result, "pending_action", None),
            "pending_queue": getattr(result, "pending_queue", None),
        }

    # ── stream() 실행 분기 ────────────────────────────────────────────────────

    async def _stream_guide(
        self, ctx: _RequestContext, _out: Optional[dict]
    ) -> AsyncIterator[str]:
        """가이드 모드 스트리밍을 실행한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트
            _out: 호출자가 message_id를 받아갈 dict (optional)

        Yields:
            진행 이벤트 JSON 또는 최종 응답 텍스트
        """
        # 링크 허용 정책 — 루프가 도구 결과를 받는 순간 link_policy 이벤트로 알려 온다.
        # 객체를 **제자리로** 갱신하므로 정화기에 넘긴 permits 콜백이 턴 내내 최신을 읽는다.
        link_policy = TurnLinkPolicy()
        sanitizer = StreamSanitizer(link_permits=link_policy.permits)
        emitted: list[str] = []
        async for event in guide_loop_events(
            self.llm,
            ctx.augmented_input,
            ctx.messages,
            self.guide_tool,
            categories=self.guide_categories,
        ):
            if isinstance(event, ProgressEvent):
                if event.event_type == "link_policy":
                    # 정화기에 넘길 내부 신호 — SSE 로는 내보내지 않는다(프론트 계약 무변경).
                    link_policy.update_from(event.content)
                    continue
                if event.event_type == "token":
                    out = sanitizer.feed(event.content)
                    if out:
                        emitted.append(out)
                        yield out
                else:
                    yield json.dumps(
                        {"type": event.event_type, "content": event.content},
                        ensure_ascii=False,
                    )
                    await asyncio.sleep(0.1)
            elif isinstance(event, LoopResult):
                # 사용자 중단(stopped): 부분답변을 PG에만 저장(재시도 표시 + 피드백용),
                # Redis(LLM 맥락)에는 제외(save_to_redis=False). eval replay 는 건너뛴다.
                if event.finish_reason == "stopped":
                    stop_tail = sanitizer.flush()
                    if stop_tail:
                        emitted.append(stop_tail)
                    partial_text = "".join(emitted)
                    if ctx.user_sub and partial_text:
                        stop_mid, ctx.conversation_id = await save_logged_in_turn(
                            ctx.conversation_id, ctx.user_sub, ctx.input, partial_text,
                            **self._save_kwargs(
                                ctx.user_sub, ctx.input, event, ctx.t_start, ctx.user_email,
                                langsmith_run_id=ctx.langsmith_run_id,
                                status_code=499, save_to_redis=False,
                            ),
                        )
                        if _out is not None:
                            _out["message_id"] = stop_mid
                            _out["conversation_id"] = ctx.conversation_id
                    if _out is not None:
                        _out["stopped"] = True
                    return

                tail = sanitizer.flush()
                if tail:
                    emitted.append(tail)
                    yield tail

                # DB 저장: LoopResult.content (이미 sanitize 적용된 최종본) 우선,
                # fallback 으로 토큰 스트림에서 누적한 emitted 사용.
                final_text = event.content or "".join(emitted)
                if ctx.user_sub and final_text:
                    _mid, ctx.conversation_id = await save_logged_in_turn(
                        ctx.conversation_id, ctx.user_sub, ctx.input, final_text,
                        **self._save_kwargs(ctx.user_sub, ctx.input, event, ctx.t_start,
                                           ctx.user_email, langsmith_run_id=ctx.langsmith_run_id),
                    )
                    if _out is not None:
                        _out["message_id"] = _mid
                        _out["conversation_id"] = ctx.conversation_id
                # 가이드로 답할 수 없는 질문(abstain)이면 done 이벤트에 실어 보낼
                # suggested_mode 를 _out 에 남긴다 (엔드포인트가 done chunk 에 포함).
                if _out is not None and event.abstain:
                    _out["suggested_mode"] = "agent"
                    _out["abstain_reason"] = event.abstain_reason
        # main 스트림 종료 후 eval 프로바이더에 final messages를 replay
        self._kickoff_eval_replay(ctx)

    async def _stream_direct_llm(
        self, ctx: _RequestContext, _out: Optional[dict]
    ) -> AsyncIterator[str]:
        """도구 없이 LLM을 직접 스트리밍 호출한다.

        프로바이더가 tool calling을 지원하지 않을 때 사용한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트
            _out: 호출자가 message_id를 받아갈 dict (optional)

        Yields:
            LLM 스트리밍 토큰
        """
        logger.debug("[stream] provider=%s (no tool calling) → direct LLM stream (no tools)", self.llm_provider)

        # StreamSanitizer로 토큰 경계 분할 케이스까지 결정적으로 마스킹한다.
        sanitizer = StreamSanitizer()
        emitted: list[str] = []

        async for event in self.llm.stream(ctx.messages, user_sub=ctx.user_sub):
            # done 이벤트는 finish_reason 메타정보일 뿐이라 사용자에겐 전달하지 않음.
            if event.type != "text" or not event.content:
                continue
            out = sanitizer.feed(event.content)
            if out:
                emitted.append(out)
                yield out

        tail = sanitizer.flush()
        if tail:
            emitted.append(tail)
            yield tail

        if ctx.user_sub and emitted:
            _mid, ctx.conversation_id = await save_logged_in_turn(
                ctx.conversation_id, ctx.user_sub, ctx.input, "".join(emitted),
                response_time=time.monotonic() - ctx.t_start,
                llm_invoke_count=1,
                save_to_redis=not self.llm.stateless,
                llm_provider=self.llm_provider,
                model_name=self.llm.model,
                user_email=ctx.user_email,
                langsmith_run_id=ctx.langsmith_run_id,
            )
            if _out is not None:
                _out["message_id"] = _mid
                _out["conversation_id"] = ctx.conversation_id
        # main 스트림 종료 후 eval 프로바이더에 동일 messages를 replay
        self._kickoff_eval_replay(ctx)

    async def _stream_react(
        self, ctx: _RequestContext, _out: Optional[dict]
    ) -> AsyncIterator[str]:
        """표준 ReAct 루프 스트리밍을 실행한다.

        Args:
            ctx: 전처리 완료된 요청 컨텍스트
            _out: 호출자가 message_id를 받아갈 dict (optional)

        Yields:
            진행 이벤트 JSON 또는 최종 응답 텍스트
        """
        yield json.dumps({"type": "thinking", "content": "사용자 질의를 분석중 ..."}, ensure_ascii=False)
        await asyncio.sleep(0.1)
        yield json.dumps({"type": "searching", "content": "관련 도구 검색중..."}, ensure_ascii=False)
        await asyncio.sleep(0.1)

        all_tools = await self._prepare_tools(ctx.messages, ctx.profile)

        # 링크 허용 정책 — 그래프/슈퍼바이저가 link_policy 이벤트로 알려 온다(가이드 턴만 켜짐).
        link_policy = TurnLinkPolicy()
        sanitizer = StreamSanitizer(link_permits=link_policy.permits)
        emitted: list[str] = []

        # 기본 이벤트원(supervisor OFF 또는 supervisor 의 fallback) — react 그래프/기존 루프.
        # 셋 다 동일 이벤트 계약(ProgressEvent* 다음 LoopResult 1개)이라 아래 소비 로직 공유.
        def _base_events():
            return (
                stream_react_graph(
                    self.llm, ctx.augmented_input, ctx.messages, all_tools,
                    self._search_dynamic_tools, settings.max_iterations,
                    target_language=ctx.target_language,
                    pending_draft_force=self._pending_draft_force,
                )
                if settings.use_langgraph_react
                else react_loop_events(
                    self.llm, ctx.augmented_input, ctx.messages, all_tools,
                    self._search_dynamic_tools, settings.max_iterations,
                )
            )

        if self._use_supervisor:
            react_events = stream_supervisor(
                self.llm, ctx.augmented_input, ctx.messages,
                resolve_tools_fn=self._search_dynamic_tools,
                make_worker_llm_fn=self._make_worker_llm,
                fallback_stream_fn=_base_events,
                max_iterations=settings.max_iterations,
                confidence_threshold=settings.router_confidence_threshold,
                router_llm=self._router_llm(),
                search_fn=self._search_dynamic_tools,
                prior_domain=await self._prior_domain(ctx.conversation_id),
                global_tools=self._worker_global_tools(ctx.profile),
                sticky_tools=self._worker_sticky_tools(ctx.messages, ctx.profile),
                target_language=ctx.target_language,
            )
        else:
            react_events = _base_events()

        async for event in react_events:
            if isinstance(event, ProgressEvent):
                if event.event_type == "link_policy":
                    # 정화기에 넘길 내부 신호 — SSE 로는 내보내지 않는다(프론트 계약 무변경).
                    link_policy.update_from(event.content)
                    continue
                if event.event_type == "token":
                    out = sanitizer.feed(event.content)
                    if out:
                        emitted.append(out)
                        yield out
                else:
                    yield json.dumps(
                        {"type": event.event_type, "content": event.content},
                        ensure_ascii=False,
                    )
                    await asyncio.sleep(0.05)
            elif isinstance(event, LoopResult):
                # 사용자 중단(stopped): 부분답변을 PG에 저장한다(재시도 페이저 표시 + 피드백용).
                # 단 Redis(LLM 맥락)에는 넣지 않는다(save_to_redis=False) — 같은 대화방에서
                # 재시도/다음 턴이 중단된 부분답변에 오염되지 않게(재시도 답변만 맥락에 남음).
                # HITL 체크포인트 / 액션뷰 emit / eval replay 는 건너뛴다.
                if event.finish_reason == "stopped":
                    stop_tail = sanitizer.flush()
                    if stop_tail:
                        emitted.append(stop_tail)
                    partial_text = "".join(emitted)
                    if ctx.user_sub and partial_text:
                        stop_mid, ctx.conversation_id = await save_logged_in_turn(
                            ctx.conversation_id, ctx.user_sub, ctx.input, partial_text,
                            **self._save_kwargs(
                                ctx.user_sub, ctx.input, event, ctx.t_start, ctx.user_email,
                                langsmith_run_id=ctx.langsmith_run_id,
                                status_code=499, save_to_redis=False,
                            ),
                        )
                        if _out is not None:
                            _out["message_id"] = stop_mid
                            _out["conversation_id"] = ctx.conversation_id
                    if _out is not None:
                        _out["stopped"] = True
                    return

                tail = sanitizer.flush()
                if tail:
                    emitted.append(tail)
                    yield tail

                # DB 저장용 텍스트: react_loop_events 가 sanitize_tool_names 적용한
                # LoopResult.content 를 우선 사용. 빈 값일 때만 토큰 스트림 누적 fallback.
                final_text = event.content or "".join(emitted)
                pending_action = getattr(event, "pending_action", None)
                pending_queue = getattr(event, "pending_queue", None)
                # HITL 액션 초안은 텍스트가 비어도(도구 호출만 한 턴) turn 을 저장해
                # conversation_id(=HITL 그래프 thread_id)를 확정해야 재개가 가능하다.
                save_text = final_text
                if not save_text and pending_action:
                    save_text = self._draft_turn_content(pending_action)
                message_id: Optional[str] = None
                if ctx.user_sub and save_text:
                    message_id, ctx.conversation_id = await save_logged_in_turn(
                        ctx.conversation_id, ctx.user_sub, ctx.input, save_text,
                        **self._save_kwargs(ctx.user_sub, ctx.input, event, ctx.t_start,
                                           ctx.user_email, langsmith_run_id=ctx.langsmith_run_id),
                    )
                    if _out is not None:
                        _out["message_id"] = message_id
                        _out["conversation_id"] = ctx.conversation_id
                elif ctx.user_sub:
                    # 빈 답변은 저장하지 않는다 — 모니터링에서 이 턴이 사라지지 않도록 이벤트로 남긴다
                    # (source 가 orchestrator 라 common_app 의 unsaved_turn 상관 규칙에 걸린다).
                    logger.warning(
                        "빈 답변 — 턴을 저장하지 않음",
                        extra=event_extra("exception", conversation_id=ctx.conversation_id),
                    )

                # HITL: 승인필요 액션(큐)이면 HITL 그래프에 큐 전체를 interrupt 체크포인트로
                # 적재해(thread_id=conversation_id) 순차 승인 대기 상태로 둔다. 그리고 카드는
                # **큐 첫 항목만** emit한다(나머지는 /resume 이 순차 surface). 비-HITL 턴은
                # 기존대로 모든 action view 를 emit한다(읽기 결과 iframe 등).
                if pending_queue and ctx.conversation_id:
                    try:
                        await self._checkpoint_pending_action(
                            ctx.conversation_id, pending_queue
                        )
                    except Exception:
                        logger.exception("[hitl] 체크포인트 적재 실패 — 카드는 정상 emit")
                    views_to_emit = [pending_queue[0].get("draft_view")]
                else:
                    views_to_emit = extract_action_views(event.tool_results)

                # action 도구가 동봉한 view → dmp-frontend 워크스페이스 패널 재활용.
                # 워크스페이스 패널이 소비하는 workspace.open/update/complete 시퀀스로 방출.
                emitted_iframe = False  # 이번 턴에 iframe(app-embed) 뷰를 방출했는지
                saved_card_view = None  # 과거 대화 복원용으로 메시지에 영속할 HITL 카드 아티팩트
                for _view in views_to_emit:
                    if not _view:
                        continue
                    artifact = build_action_artifact(_view, ctx.conversation_id, message_id)
                    if artifact is None:
                        continue
                    # 이 턴의 HITL 승인 카드(draft+form)를 첫 1건만 영속 대상으로 잡는다
                    # (iframe 화면이동 뷰는 제외 — 과거 세션에서 재이동시키지 않기 위함).
                    if (
                        saved_card_view is None
                        and artifact.get("actionKind") == "draft"
                        and artifact.get("form")
                    ):
                        saved_card_view = artifact
                    if artifact.get("resourceUrl"):
                        # url 있는 view(미리보기 팝오버·완료 결과 등) → iframe 패널 표시.
                        # draft(actionKind=draft)+form 이면 @org/ui 가 같은 update 에서
                        # 승인/거절 카드도 렌더하므로, 캘린더 제안은 iframe 미리보기 +
                        # 카드가 함께 뜬다.
                        emitted_iframe = True
                        ws_id = f"ws_{uuid.uuid4().hex[:16]}"
                        yield json.dumps(
                            {
                                "type": "workspace.open",
                                "workspaceId": ws_id,
                                "conversationId": ctx.conversation_id or "",
                                "messageId": artifact["messageId"],
                                "title": artifact["title"],
                                "status": "running",
                            },
                            ensure_ascii=False,
                        )
                        await asyncio.sleep(0.05)
                        yield json.dumps(
                            {"type": "workspace.update", "workspaceId": ws_id, "artifact": artifact},
                            ensure_ascii=False,
                        )
                        await asyncio.sleep(0.05)
                        yield json.dumps(
                            {"type": "workspace.complete", "workspaceId": ws_id, "artifactId": artifact["id"]},
                            ensure_ascii=False,
                        )
                    elif artifact.get("actionKind") == "draft" and artifact.get("form"):
                        # url 없는 제안/초안(메일·메신저 등) → 챗 인라인 카드만.
                        # workspaceId 없이 workspace.update (host 무시, @org/ui 카드 렌더).
                        yield json.dumps(
                            {"type": "workspace.update", "artifact": artifact},
                            ensure_ascii=False,
                        )
                        await asyncio.sleep(0.05)

                # 과거 대화 복원용: 이 턴에 뜬 HITL 승인 카드를 메시지에 영속한다(있을 때만).
                # 조회 시 ConversationMessage.action_view 로 내려가 카드 테이블을 읽기전용 재현.
                if saved_card_view and message_id:
                    try:
                        await set_message_action_view(message_id, saved_card_view)
                    except Exception:
                        logger.warning("[history] action_view 저장 실패", exc_info=True)

                # (직전 워크스페이스 아티팩트 '재방출' 로직 제거)
                # 도구를 호출하지 않은 턴(인사·문맥 답변 등)에서 캐시된 iframe 을 새 ws_id 로
                # 재방출하면, 프론트 WorkspaceRouteSync 가 새 workspaceId 를 감지해 페이지를
                # 워크스페이스 라우트로 강제 이동시킨다 → "안녕" 같은 무관한 턴에도 직전(예:
                # 메신저) 화면으로 튀는 버그. 일반 플로우는 호스트가 워크스페이스 뷰를 턴 사이
                # 에 유지하므로(clearWorkspace 는 특정 경로에서만 호출) 재방출 없이도 패널이
                # 남는다. 따라서 tool-less 턴에서 재방출하지 않는다.
        # main 스트림 종료 후 eval 프로바이더에 final messages를 replay
        self._kickoff_eval_replay(ctx)

    @staticmethod
    def _draft_turn_content(pending_action: dict) -> str:
        """HITL 초안 턴의 저장용 텍스트(빈 응답 대비). conversation_id 확정에 사용."""
        view = pending_action.get("draft_view") or {}
        label = view.get("label") or (view.get("preview") or {}).get("title")
        return label or f"{pending_action.get('tool_name', '작업')} 초안을 준비했습니다. 승인해 주세요."

    async def _checkpoint_pending_action(
        self, conversation_id: str, pending_queue: list
    ) -> None:
        """HITL 그래프에 승인필요 액션 큐를 interrupt 상태로 적재한다.

        thread_id=conversation_id 로 ainvoke 하면 ``execute_action`` 노드가 큐 첫 항목에서
        interrupt 로 정지하고 checkpointer 가 상태(큐+index)를 영속한다(실제 실행 0). 이후
        /resume 이 같은 thread_id 로 Command(resume) 할 때마다 한 항목씩 confirm=True 실행한다.

        새 카드가 뜨는 시점에 **직전 거절 초안 stash 를 폐기**한다. 그 stash 는 거절된 값을
        정정 턴에 되쓰기 위한 것인데, 새 초안이 만들어졌다는 건 그 역할이 끝났다는 뜻이다.
        남겨 두면 이후의 **완결된 새 요청**까지 정정으로 오인돼 옛 값이 섞인다(실측 2026-08-10:
        08-13 초안을 거절한 뒤, 한참 뒤의 "다음주 수요일…" 새 요청이 옛 목요일을 물려받아
        08-20 이 됐다). 새 카드를 다시 거절하면 _persist_action_outcome 이 최신 값으로 재저장
        하므로 누적 정정("10시로" → "장소 추가")은 그대로 동작한다.
        """
        from agent_arch.agents.hitl.graph import get_action_graph
        from agent_arch.memory.conversation import clear_rejected_draft

        await clear_rejected_draft(conversation_id)
        # 항목마다 식별자를 붙인다 — 같은 대화에 새 카드가 뜨면 이 큐가 통째로 바뀌는데, 메신저에는 앞 카드의
        # 버튼이 남는다. 버튼이 자기 항목 id 를 들고 와야 '지금 대기 중인 다른 카드'를 승인하지 않는다
        # (2026-10-16: 멘션 없는 스레드 답글로 카드가 다시 뜬 뒤 앞 카드를 눌러 새 카드 값으로 외부 시스템 초안이 만들어졌다).
        pending_queue = [
            {**item, "pending_id": item.get("pending_id") or uuid.uuid4().hex}
            if isinstance(item, dict) else item
            for item in pending_queue
        ]
        graph = get_action_graph()
        config = {"configurable": {"thread_id": conversation_id}}
        # 같은 conversation(=thread_id)을 턴마다 재사용하므로, 이전 턴이 남긴 state
        # (index/results)를 명시적으로 리셋해 새 큐와 충돌(stale index → IndexError)을 막는다.
        await graph.ainvoke(
            {"pending_queue": list(pending_queue), "index": 0, "results": [], "result": None},
            config,
        )
        # 재시작에도 승인 카드가 살도록 같은 큐를 Redis 에 둔다(hitl/resume.save_queue_snapshot).
        from agent_arch.agents.hitl.resume import save_queue_snapshot

        await save_queue_snapshot(conversation_id, list(pending_queue))

    # ── 공개 메서드 ───────────────────────────────────────────────────────────

    @traceable(run_type="chain", name="agent_run")
    async def run(
        self,
        input: str,
        conversation_id: Optional[str] = None,
        user_info: Optional[dict] = None,
        images: Optional[list] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """에이전트를 실행한다.

        Args:
            input: 사용자 입력
            conversation_id: 대상 대화방 UUID. None이면 save 시점에 lazy-create.
            user_info: 요청자 정보 (sub, email, company 포함)
            images: 첨부 이미지 목록 (ImageContent 또는 ImageInput 호환 객체)
            **kwargs: 추가 파라미터 (하위 호환용, 사용하지 않음)

        Returns:
            Dict[str, Any]: 에이전트 실행 결과. conversation_id가 동봉되어 호출자가
            응답으로 그대로 내려주면 프론트가 같은 방에 후속 요청을 보낼 수 있다.
        """
        ctx = await self._preprocess_request(input, user_info, images, conversation_id=conversation_id)
        if ctx.command_result is not None:
            # 슬래시 커맨드는 메시지 저장 없이 종료 — 진입 시 받은 conversation_id를 그대로 동봉
            ctx.command_result["conversation_id"] = ctx.conversation_id
            return ctx.command_result

        strategy = self._resolve_strategy()
        try:
            match strategy:
                case _ExecStrategy.GUIDE_LOOP:
                    result = await self._run_guide(ctx)
                case _ExecStrategy.DIRECT_LLM:
                    result = await self._run_direct_llm(ctx)
                case _:
                    result = await self._run_react(ctx)
        except Exception as e:
            # 턴이 저장되지 않고 끝난다 — 오류 이벤트(unsaved_turn 상관 규칙)가 이 source 를 본다.
            logger.exception(
                "[run] EXCEPTION: %s", e,
                extra=event_extra("exception", conversation_id=ctx.conversation_id),
            )
            return {"success": False, "error": str(e), "llm_provider": self.llm_provider}

        return result

    @traceable(run_type="chain", name="agent_stream")
    async def stream(
        self,
        input: str,
        conversation_id: Optional[str] = None,
        user_info: Optional[dict] = None,
        images: Optional[list] = None,
        _out: Optional[dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """스트리밍 방식으로 에이전트를 실행한다.

        Tool 실행 중에는 진행 이벤트(JSON)를 yield하여 SSE 연결을 유지하고,
        최종 응답은 텍스트로 yield한다. save 시점에 확정된 conversation_id는
        _out["conversation_id"]로 호출자에게 전달되어 SSE done 이벤트에 동봉된다.

        Args:
            input: 사용자 입력
            conversation_id: 대상 대화방 UUID. None이면 save 시점에 lazy-create.
            user_info: 요청자 정보 (sub, email, company 포함)
            images: 첨부 이미지 목록 (ImageContent 또는 ImageInput 호환 객체)
            _out: 호출자가 message_id·conversation_id를 받아갈 dict (optional)
            **kwargs: 추가 파라미터 (하위 호환용, 사용하지 않음)

        Yields:
            str: 진행 이벤트 JSON 또는 최종 응답 텍스트
        """
        if images:
            yield json.dumps({"type": "analyzing_image"}, ensure_ascii=False)
            await asyncio.sleep(0.1)

        ctx = await self._preprocess_request(input, user_info, images, conversation_id=conversation_id)

        if images and ctx.image_context:
            yield json.dumps(
                {"type": "image_analyzed", "context": ctx.image_context.model_dump()},
                ensure_ascii=False,
            )
            await asyncio.sleep(0.1)

        if ctx.command_result is not None:
            yield ctx.command_result["output"]
            return

        strategy = self._resolve_strategy()
        try:
            match strategy:
                case _ExecStrategy.GUIDE_LOOP:
                    async for token in self._stream_guide(ctx, _out):
                        yield token
                case _ExecStrategy.DIRECT_LLM:
                    async for token in self._stream_direct_llm(ctx, _out):
                        yield token
                case _:
                    async for token in self._stream_react(ctx, _out):
                        yield token
        except Exception as e:
            # ⚠️ 예외 문자열을 그대로 흘리면 사용자 화면에 스택 용어가 나간다 — 2026-09-14
            #    메신저에 `Error: peer closed connection without sending complete message
            #    body (incomplete chunked read)` 가 그대로 게시됐다. 원인 추적은 로그가
            #    맡고(위 exception), 사용자에게는 다시 시도할 수 있다는 사실만 알린다.
            logger.exception(
                "[stream] EXCEPTION: %s", e,
                extra=event_extra("exception", conversation_id=ctx.conversation_id),
            )
            yield "답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."

    async def _search_dynamic_tools(self, query: str, domain: Optional[str] = None) -> List:
        """Qdrant에서 관련 도구를 검색해 바인딩 가능한 인스턴스로 반환한다.

        domain 이 주어지면 그 도메인으로 스코프해 검색한다(supervisor 워커 스코핑).

        통합 인덱싱(UNIFIED_TOOL_INDEXING) 시 origin="registry" 히트(정적
        스킬·MCP — retrieved 티어)가 섞여 오며, 이는 ToolRegistry의 실물
        인스턴스로 resolve한다 (인덱스엔 이름+설명만 있으므로). 동적 API
        히트는 기존대로 DynamicAPITool로 래핑.

        Args:
            query: 사용자 쿼리 텍스트

        Returns:
            검색된 도구 리스트 (BaseTool 혼합). 실패 시 빈 리스트
        """
        try:
            tool_defs = await api_indexer.search(
                query,
                top_k=settings.qdrant_top_k,
                include_registry=settings.unified_tool_indexing,
                domain=domain,
            )
        except Exception as e:
            logger.warning("Dynamic tool search failed: %s", e)
            tool_defs = []

        tools = []
        for td in tool_defs:
            if td.origin == "registry":
                instance = ToolRegistry.get(td.tool_name)
                if instance is None:
                    logger.debug("[search_tools] registry 히트 %s 실물 없음 — 스킵", td.tool_name)
                    continue
                tools.append(instance)
            else:
                tools.append(DynamicAPITool(td))
        if domain and domain in extensions.domain_tools:
            # 인덱스 밖 도메인 도구(기능 플래그·요청 단위 opt-in) — 워커 스코프에 결정적으로 넣는다.
            # 검색이 실패해도(임베딩 장애) 그 도메인 워커가 빈손이 되지 않는다.
            have = {t.name for t in tools}
            tools += [t for t in domain_tools(domain) if t.name not in have]

        # 권한 게이트 — react 루프의 중간 재검색(iteration>0) 경로는 본 함수의
        # 결과를 게이트 없이 바인딩하므로 여기서 일괄 필터한다 (deny default).
        # _prepare_tools 경로는 plan_tools에서 한 번 더 걸러지지만 무해(멱등).
        profile = get_profile_context()
        return [t for t in tools if is_allowed(t.name, profile)]

    def get_tools(self):
        """사용 가능한 도구 리스트를 반환한다.

        Returns:
            등록된 도구 리스트 (guide 모드: guide_tool, 그 외: static_tools)
        """
        if self.agent_mode == "guide":
            return [self.guide_tool] if self.guide_tool is not None else []
        return self.static_tools

    def get_llm_info(self):
        """LLM 공급자 정보를 반환한다.

        Returns:
            Dict[str, Any]: LLM 프로바이더 정보 딕셔너리
        """
        return {
            "provider": self.llm_provider,
            "model": self.llm.model,
            "agent_mode": self.agent_mode,
            "supports_function_calling": self.llm.supports_function_calling,
            "supports_streaming": self.llm.supports_streaming,
        }
