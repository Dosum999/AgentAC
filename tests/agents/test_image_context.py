"""이미지 컨텍스트 분석 파이프라인 테스트.

_analyze_image_context(), _preprocess_request() 이미지 주입,
stream() SSE 이벤트 순서를 검증한다.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.llm.base import LLMResponse, Message
from agent_arch.models.agent import ImageContext, ImageInput


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def _make_orchestrator(llm_provider: str = "gemini") -> AgentOrchestrator:
    """__init__을 우회하여 AgentOrchestrator 인스턴스를 생성한다."""
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = "default"
    orc.llm_provider = llm_provider
    orc.enable_pii_masking = False
    orc.llm = MagicMock(
        stateless=False,
        supports_function_calling=False,
    )
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    return orc


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, finish_reason="stop", usage=None)


def _make_image() -> ImageInput:
    return ImageInput(data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==", mime_type="image/png")


_VALID_JSON = '{"screen_description": "Messenger 채널 목록", "user_intent": "채널에 멤버 초대", "suggested_query": "메신저 채널 멤버 초대 방법"}'

_USER_INFO = {"sub": "u1", "email": "u@test.com"}

_COMMON_PATCHES = {
    "handle_command": "agent_arch.agents.orchestrator.handle_command",
    "load_history": "agent_arch.agents.orchestrator.load_history",
    "build_system_prompt": "agent_arch.agents.orchestrator.build_system_prompt",
    "set_user_context": "agent_arch.agents.orchestrator.set_user_context",
    "get_current_run_tree": "agent_arch.agents.orchestrator.get_current_run_tree",
    "save_logged_in_turn": "agent_arch.agents.orchestrator.save_logged_in_turn",
    "settings": "agent_arch.agents.orchestrator.settings",
}


# ---------------------------------------------------------------------------
# _analyze_image_context() 단위 테스트
# ---------------------------------------------------------------------------

class TestAnalyzeImageContext:
    """_analyze_image_context() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_returns_image_context_on_valid_json(self):
        """Gemini가 유효한 JSON을 반환하면 ImageContext를 반환한다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        result = await orc._analyze_image_context([_make_image()], "이 화면에서 어떻게 해?")

        assert isinstance(result, ImageContext)
        assert result.screen_description == "Messenger 채널 목록"
        assert result.user_intent == "채널에 멤버 초대"
        assert result.suggested_query == "메신저 채널 멤버 초대 방법"

    @pytest.mark.asyncio
    async def test_strips_json_code_block(self):
        """Gemini가 ```json 코드블록으로 감싸도 파싱에 성공한다."""
        orc = _make_orchestrator()
        wrapped = f"```json\n{_VALID_JSON}\n```"
        orc.llm.invoke = AsyncMock(return_value=_make_response(wrapped))

        result = await orc._analyze_image_context([_make_image()], "질문")

        assert isinstance(result, ImageContext)
        assert result.suggested_query == "메신저 채널 멤버 초대 방법"

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self):
        """LLM 호출이 실패하면 None을 반환하고 예외를 전파하지 않는다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(side_effect=RuntimeError("Gemini API 오류"))

        result = await orc._analyze_image_context([_make_image()], "질문")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        """Gemini가 JSON이 아닌 텍스트를 반환하면 None을 반환한다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response("파싱 불가 텍스트"))

        result = await orc._analyze_image_context([_make_image()], "질문")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_field(self):
        """필수 필드가 누락된 JSON이면 None을 반환한다."""
        orc = _make_orchestrator()
        incomplete = '{"screen_description": "화면"}'  # user_intent, suggested_query 누락
        orc.llm.invoke = AsyncMock(return_value=_make_response(incomplete))

        result = await orc._analyze_image_context([_make_image()], "질문")

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_default_content_when_text_input_empty(self):
        """텍스트 입력이 빈 문자열이면 기본 메시지를 사용한다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        await orc._analyze_image_context([_make_image()], "")

        call_messages = orc.llm.invoke.call_args.args[0]
        user_message = call_messages[1]
        assert user_message.content == "이 화면을 분석해 주세요."

    @pytest.mark.asyncio
    async def test_uses_original_text_when_provided(self):
        """텍스트 입력이 있으면 그대로 user 메시지 content로 사용한다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        await orc._analyze_image_context([_make_image()], "이 화면에서 어떻게 해?")

        call_messages = orc.llm.invoke.call_args.args[0]
        user_message = call_messages[1]
        assert user_message.content == "이 화면에서 어떻게 해?"

    @pytest.mark.asyncio
    async def test_creates_vision_instance_for_non_vision_provider(self):
        """주력 LLM이 비전 미지원(vllm)이면 vision_llm_provider 인스턴스를 별도 생성한다."""
        orc = _make_orchestrator(llm_provider="vllm")

        vision_llm = MagicMock()
        vision_llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.agents.orchestrator.settings.vision_llm_provider", "openai"):
            mock_factory.create.return_value = vision_llm
            await orc._analyze_image_context([_make_image()], "질문")

        mock_factory.create.assert_called_once_with("openai")
        vision_llm.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_self_llm_for_gemini_provider(self):
        """llm_provider가 gemini(비전 지원)면 self.llm을 재사용하고 새 인스턴스를 만들지 않는다."""
        orc = _make_orchestrator(llm_provider="gemini")
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory:
            await orc._analyze_image_context([_make_image()], "질문")

        mock_factory.create.assert_not_called()
        orc.llm.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_self_llm_for_openai_provider(self):
        """llm_provider가 openai(비전 지원)면 self.llm을 재사용하고 새 인스턴스를 만들지 않는다."""
        orc = _make_orchestrator(llm_provider="openai")
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory:
            await orc._analyze_image_context([_make_image()], "질문")

        mock_factory.create.assert_not_called()
        orc.llm.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_is_attached_to_user_message(self):
        """user 메시지에 images 필드가 포함되어 Gemini에 전달된다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        await orc._analyze_image_context([_make_image()], "질문")

        call_messages = orc.llm.invoke.call_args.args[0]
        user_message = call_messages[1]
        assert user_message.images is not None
        assert len(user_message.images) == 1

    @pytest.mark.asyncio
    async def test_history_not_included_in_analysis_messages(self):
        """분석용 메시지는 system + user 2개뿐이며 대화 히스토리가 포함되지 않는다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response(_VALID_JSON))

        await orc._analyze_image_context([_make_image()], "질문")

        call_messages = orc.llm.invoke.call_args.args[0]
        assert len(call_messages) == 2
        assert call_messages[0].role == "system"
        assert call_messages[1].role == "user"


# ---------------------------------------------------------------------------
# _preprocess_request() 이미지 컨텍스트 주입 테스트
# ---------------------------------------------------------------------------

class TestPreprocessRequestImageInjection:
    """_preprocess_request()에서 이미지 컨텍스트 주입 동작 검증."""

    def _patch_common(self, mock_settings):
        mock_settings.enable_pii_masking = False
        mock_settings.enable_parallel_eval = False

    @pytest.mark.asyncio
    async def test_augmented_input_contains_context_on_success(self):
        """이미지 분석 성공 시 augmented_input에 [이미지 컨텍스트] 블록이 포함된다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="채널 목록",
            user_intent="멤버 초대",
            suggested_query="채널 멤버 초대",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            ctx = await orc._preprocess_request("원본 텍스트", _USER_INFO, [_make_image()])

        assert "[이미지 컨텍스트]" in ctx.augmented_input
        assert "채널 목록" in ctx.augmented_input
        assert "멤버 초대" in ctx.augmented_input
        assert "채널 멤버 초대" in ctx.augmented_input

    @pytest.mark.asyncio
    async def test_original_input_preserved_in_ctx_input(self):
        """이미지 컨텍스트가 주입되어도 ctx.input은 원본 텍스트를 유지한다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="화면",
            user_intent="작업",
            suggested_query="쿼리",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            ctx = await orc._preprocess_request("원본 텍스트", _USER_INFO, [_make_image()])

        assert ctx.input == "원본 텍스트"
        assert ctx.augmented_input != ctx.input

    @pytest.mark.asyncio
    async def test_augmented_input_equals_input_when_analysis_fails(self):
        """이미지 분석 실패(None 반환) 시 augmented_input은 input과 동일하다."""
        orc = _make_orchestrator()
        orc._analyze_image_context = AsyncMock(return_value=None)

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            ctx = await orc._preprocess_request("원본 텍스트", _USER_INFO, [_make_image()])

        assert ctx.augmented_input == "원본 텍스트"
        assert ctx.image_context is None

    @pytest.mark.asyncio
    async def test_analyze_not_called_without_images(self):
        """이미지가 없으면 _analyze_image_context가 호출되지 않는다."""
        orc = _make_orchestrator()
        orc._analyze_image_context = AsyncMock()

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            ctx = await orc._preprocess_request("텍스트만", _USER_INFO, None)

        orc._analyze_image_context.assert_not_called()
        assert ctx.augmented_input == "텍스트만"
        assert ctx.image_context is None

    @pytest.mark.asyncio
    async def test_augmented_input_used_in_messages(self):
        """_build_messages()는 augmented_input으로 호출된다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="화면",
            user_intent="작업",
            suggested_query="쿼리",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)

        built_messages = []

        async def capture_build(user_info, input_text, images=None, conversation_id=None, **kwargs):
            built_messages.append(input_text)
            return [Message(role="user", content=input_text)]

        orc._build_messages = capture_build

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            await orc._preprocess_request("원본", _USER_INFO, [_make_image()])

        assert len(built_messages) == 1
        assert "[이미지 컨텍스트]" in built_messages[0]

    @pytest.mark.asyncio
    async def test_empty_input_with_image_context(self):
        """input이 빈 문자열이고 이미지 컨텍스트가 있을 때 augmented_input이 올바르게 조립된다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="화면",
            user_intent="작업",
            suggested_query="쿼리",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)

        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._patch_common(mock_settings)
            ctx = await orc._preprocess_request("", _USER_INFO, [_make_image()])

        # 앞뒤 공백 없이 [이미지 컨텍스트]로 시작해야 한다
        assert ctx.augmented_input.startswith("[이미지 컨텍스트]")
        assert "쿼리" in ctx.augmented_input


# ---------------------------------------------------------------------------
# stream() SSE 이벤트 순서 테스트
# ---------------------------------------------------------------------------

class TestStreamImageEvents:
    """stream()에서 이미지 관련 SSE 이벤트 발생 순서 검증."""

    def _base_patches(self, mock_settings):
        mock_settings.enable_pii_masking = False
        mock_settings.enable_parallel_eval = False
        mock_settings.agent_mode = "default"

    async def _collect_events(self, orc, input_text, images=None):
        """stream() 출력 중 JSON 파싱 가능한 이벤트만 수집한다."""
        events = []
        with patch(_COMMON_PATCHES["handle_command"], new=AsyncMock(return_value=None)), \
             patch(_COMMON_PATCHES["load_history"], new=AsyncMock(return_value=[])), \
             patch(_COMMON_PATCHES["build_system_prompt"], return_value="sys"), \
             patch(_COMMON_PATCHES["set_user_context"]), \
             patch(_COMMON_PATCHES["get_current_run_tree"], return_value=None), \
             patch(_COMMON_PATCHES["save_logged_in_turn"], new=AsyncMock(return_value=("msg-id", "conv-id"))), \
             patch(_COMMON_PATCHES["settings"]) as mock_settings:

            self._base_patches(mock_settings)
            async for token in orc.stream(input_text, user_info=_USER_INFO, images=images):
                try:
                    events.append(json.loads(token))
                except (json.JSONDecodeError, TypeError):
                    pass
        return events

    @pytest.mark.asyncio
    async def test_analyzing_image_event_emitted_first(self):
        """이미지가 있으면 analyzing_image 이벤트가 가장 먼저 yield된다."""
        orc = _make_orchestrator()
        orc._analyze_image_context = AsyncMock(return_value=None)
        orc.llm.stream = _empty_stream()

        events = await self._collect_events(orc, "질문", images=[_make_image()])

        types = [e["type"] for e in events]
        assert "analyzing_image" in types
        assert types.index("analyzing_image") == 0

    @pytest.mark.asyncio
    async def test_image_analyzed_event_emitted_after_analyzing(self):
        """분석 성공 시 image_analyzed 이벤트가 analyzing_image 다음에 온다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="채널 목록",
            user_intent="멤버 초대",
            suggested_query="채널 멤버 초대",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)
        orc.llm.stream = _empty_stream()

        events = await self._collect_events(orc, "질문", images=[_make_image()])

        types = [e["type"] for e in events]
        assert "image_analyzed" in types
        assert types.index("analyzing_image") < types.index("image_analyzed")

    @pytest.mark.asyncio
    async def test_image_analyzed_event_contains_context(self):
        """image_analyzed 이벤트의 context 필드에 분석 결과가 담긴다."""
        orc = _make_orchestrator()
        image_context = ImageContext(
            screen_description="채널 목록",
            user_intent="멤버 초대",
            suggested_query="채널 멤버 초대",
        )
        orc._analyze_image_context = AsyncMock(return_value=image_context)
        orc.llm.stream = _empty_stream()

        events = await self._collect_events(orc, "질문", images=[_make_image()])

        analyzed = next(e for e in events if e["type"] == "image_analyzed")
        assert analyzed["context"]["screen_description"] == "채널 목록"
        assert analyzed["context"]["user_intent"] == "멤버 초대"
        assert analyzed["context"]["suggested_query"] == "채널 멤버 초대"

    @pytest.mark.asyncio
    async def test_image_analyzed_not_emitted_when_analysis_fails(self):
        """분석 실패(None) 시 image_analyzed 이벤트가 yield되지 않는다."""
        orc = _make_orchestrator()
        orc._analyze_image_context = AsyncMock(return_value=None)
        orc.llm.stream = _empty_stream()

        events = await self._collect_events(orc, "질문", images=[_make_image()])

        types = [e["type"] for e in events]
        assert "image_analyzed" not in types

    @pytest.mark.asyncio
    async def test_no_image_events_without_images(self):
        """이미지가 없으면 analyzing_image, image_analyzed 이벤트가 발생하지 않는다."""
        orc = _make_orchestrator()
        orc._analyze_image_context = AsyncMock()
        orc.llm.stream = _empty_stream()

        events = await self._collect_events(orc, "텍스트만", images=None)

        types = [e["type"] for e in events]
        assert "analyzing_image" not in types
        assert "image_analyzed" not in types


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

async def aiter(items):
    """리스트를 async iterator로 변환한다."""
    for item in items:
        yield item


def _empty_stream():
    """빈 async generator를 반환하는 Mock용 side_effect 함수."""
    async def _gen(*args, **kwargs):
        return
        yield  # noqa: unreachable — async generator 선언을 위해 필요
    return _gen
