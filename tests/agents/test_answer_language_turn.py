"""턴 답변 언어 결정 테스트 (orchestrator).

우선순위: **요청 명시 → 현재 발화 감지 → 직전 발화 상속 → 서버 기본값**.

상속 단계가 없으면 영어로 대화하다 "ok" 한 마디에 답변이 한국어로 돌아온다(HITL
continuation 은 지시문 자체가 한국어 고정이라 더 확실히 뒤집힌다). 그래서 상속은
장식이 아니라 계약이다.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.core.config import settings
from agent_arch.llm.base import Message

_EN = "Could you show me the unread messages from this morning?"
_KO = "오늘 오전에 온 안 읽은 메시지 보여줘"
_ZH = "请给我看今天上午未读的消息"


def _make_orchestrator(language=None) -> AgentOrchestrator:
    """__init__ 을 우회해 인스턴스를 만든다(다른 orchestrator 테스트와 같은 패턴)."""
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = "default"
    orc.llm_provider = "gemini"
    orc.enable_pii_masking = False
    orc.language = language
    orc.llm = MagicMock(stateless=False, supports_function_calling=True)
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    return orc


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [(_KO, "ko"), (_EN, "en"), (_ZH, "zh")])
async def test_detects_language_from_current_utterance(text, expected):
    orc = _make_orchestrator()
    assert await orc._resolve_turn_language(text, None) == expected


@pytest.mark.asyncio
async def test_explicit_request_language_wins_over_detection():
    """클라이언트가 언어를 명시하면 감지하지 않는다(사용자 선택 > 추정)."""
    orc = _make_orchestrator(language="en")
    assert await orc._resolve_turn_language(_KO, "conv-1") == "en"


@pytest.mark.asyncio
async def test_inherits_language_from_previous_user_turn():
    """짧은 확인 발화는 직전 사용자 발화의 언어를 물려받는다."""
    orc = _make_orchestrator()
    history = [
        Message(role="user", content=_EN),
        Message(role="assistant", content="Here are your unread messages."),
    ]
    with patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=history)) as lh:
        assert await orc._resolve_turn_language("ok", "conv-1") == "en"
    lh.assert_awaited_once_with("conversation:conv-1")


@pytest.mark.asyncio
async def test_continuation_style_empty_text_inherits_from_history():
    """HITL continuation 은 빈 문자열로 물어본다 — 고정 지시문 언어에 오염되지 않게."""
    orc = _make_orchestrator()
    history = [Message(role="user", content=_ZH)]
    with patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=history)):
        assert await orc._resolve_turn_language("", "conv-1") == "zh"


@pytest.mark.asyncio
async def test_history_is_not_read_when_current_turn_decides():
    """정상 턴에 Redis 왕복을 추가하지 않는다."""
    orc = _make_orchestrator()
    with patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])) as lh:
        assert await orc._resolve_turn_language(_EN, "conv-1") == "en"
    lh.assert_not_awaited()


@pytest.mark.asyncio
async def test_falls_back_to_server_default_when_nothing_decides():
    orc = _make_orchestrator()
    with patch.object(settings, "default_answer_language", "en"):
        assert await orc._resolve_turn_language("ok", None) == "en"
    with patch.object(settings, "default_answer_language", "ko"):
        assert await orc._resolve_turn_language("ok", None) == "ko"


@pytest.mark.asyncio
async def test_history_failure_does_not_break_the_turn():
    """히스토리 조회가 깨져도 언어 결정은 계속된다(가드가 언어를 못 정한 채 돌지 않게)."""
    orc = _make_orchestrator()
    with patch("agent_arch.agents.orchestrator.load_history",
               new=AsyncMock(side_effect=RuntimeError("redis down"))):
        assert await orc._resolve_turn_language("ok", "conv-1") == "ko"


@pytest.mark.asyncio
async def test_missing_language_attribute_is_tolerated():
    """__init__ 을 우회해 만든 인스턴스(테스트 더블·구버전 호출부)에서도 죽지 않는다."""
    orc = _make_orchestrator()
    del orc.language
    assert await orc._resolve_turn_language(_EN, None) == "en"


@pytest.mark.asyncio
async def test_system_prompt_carries_the_resolved_language():
    """결정된 언어가 시스템 프롬프트의 답변 언어 절로 실제로 렌더된다."""
    orc = _make_orchestrator()
    with patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
         patch.object(settings, "enable_user_memory", False):
        messages = await orc._build_messages(None, _EN, conversation_id=None, language="en")

    system = messages[0]
    assert system.role == "system"
    assert "**Always answer in English.**" in system.content
    assert "항상 한국어로 답변하세요" not in system.content


# ── 전처리 → 컨텍스트 배선 ────────────────────────────────────────────
#
# 언어를 정하는 것만으로는 부족하다. 그 값이 ctx 를 타고 그래프의 언어 가드까지 가야
# 프롬프트(지시)와 가드(강제)가 같은 언어를 본다. 어긋나면 가드가 이긴다.

_PATCHES = {
    "handle_command": "agent_arch.agents.orchestrator.handle_command",
    "load_history": "agent_arch.agents.orchestrator.load_history",
    "build_system_prompt": "agent_arch.agents.orchestrator.build_system_prompt",
    "set_user_context": "agent_arch.agents.orchestrator.set_user_context",
    "get_current_run_tree": "agent_arch.agents.orchestrator.get_current_run_tree",
    "settings": "agent_arch.agents.orchestrator.settings",
}


async def _preprocess(orc, text):
    with patch(_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
         patch(_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
         patch(_PATCHES["build_system_prompt"], return_value="sys") as bsp, \
         patch(_PATCHES["set_user_context"]), \
         patch(_PATCHES["get_current_run_tree"], return_value=None), \
         patch(_PATCHES["settings"]) as mock_settings:
        mock_settings.demo_mode = False
        mock_settings.enable_pii_masking = False
        mock_settings.enable_parallel_eval = False
        mock_settings.enable_user_memory = False
        mock_settings.default_answer_language = "ko"
        ctx = await orc._preprocess_request(text, {"sub": "u1", "email": "user@example.com"}, None)
    return ctx, bsp


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [(_EN, "en"), (_ZH, "zh"), (_KO, "ko")])
async def test_preprocess_puts_language_on_context(text, expected):
    orc = _make_orchestrator()
    ctx, _ = await _preprocess(orc, text)
    assert ctx.target_language == expected


@pytest.mark.asyncio
async def test_preprocess_passes_language_to_prompt_builder():
    """같은 값이 시스템 프롬프트 빌더로도 전달된다(프롬프트와 가드가 한 값을 공유)."""
    orc = _make_orchestrator()
    ctx, bsp = await _preprocess(orc, _EN)
    assert bsp.call_args.kwargs["language"] == "en" == ctx.target_language
