"""OpenAILLM 테스트.

@traceable 적용, 응답 변환, usage 추출, LangSmith run tree 메타데이터 주입,
멀티모달(이미지) 메시지 변환, tool calling(invoke/stream), tool_choice 매핑,
stream() 토큰/도구호출 처리를 검증한다. OpenAI API 호출은 Mock으로 대체한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.llm.openai_provider import OpenAILLM
from agent_arch.llm.base import Message, ImageContent, LLMResponse
from agent_arch.llm.factory import LLMFactory


def _make_openai_llm(supports_function_calling: bool = False) -> OpenAILLM:
    """openai.AsyncOpenAI를 Mock 처리하여 OpenAILLM 인스턴스를 생성한다."""
    with patch("agent_arch.llm.openai_provider.openai.AsyncOpenAI"):
        return OpenAILLM({
            "api_key": "sk-test-key",
            "model": "gpt-5.4-mini",
            "temperature": 0.7,
            "max_tokens": 4096,
            "supports_function_calling": supports_function_calling,
            "supports_streaming": True,
        })


class _FakeTool:
    """_convert_tool_to_schema가 참조하는 BaseTool 최소 형태."""

    def __init__(self):
        self.name = "get_weather"
        self.description = "도시의 날씨를 조회한다."
        self.parameters = {"city": {"type": "string", "description": "도시명"}}
        self.required_params = ["city"]


def _make_openai_response(
    content: str = "응답",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    with_usage: bool = True,
    tool_calls=None,
) -> MagicMock:
    """OpenAI SDK ChatCompletion Mock 객체를 생성한다."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].finish_reason = finish_reason

    if with_usage:
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens
        mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    else:
        mock_response.usage = None

    return mock_response


def _make_tool_call(call_id: str, name: str, arguments: str) -> MagicMock:
    """OpenAI message.tool_calls 항목 Mock."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_stream_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls=None,
) -> MagicMock:
    """스트리밍 응답 청크 Mock 객체를 생성한다."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_usage_chunk(prompt: int = 1, completion: int = 1) -> MagicMock:
    """stream_options=include_usage 마지막 청크 (choices=[]) Mock."""
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = usage
    return chunk


def _make_tool_delta(index: int, call_id=None, name=None, arguments=None) -> MagicMock:
    """스트림 tool_calls 델타 항목 Mock."""
    delta_tc = MagicMock()
    delta_tc.index = index
    delta_tc.id = call_id
    if name is None and arguments is None:
        delta_tc.function = None
    else:
        delta_tc.function = MagicMock()
        delta_tc.function.name = name
        delta_tc.function.arguments = arguments
    return delta_tc


async def _async_chunks(*contents: str):
    """텍스트 청크 시퀀스 + usage 종료 청크 흉내 비동기 제너레이터."""
    last = len(contents) - 1
    for idx, content in enumerate(contents):
        yield _make_stream_chunk(
            content,
            finish_reason="stop" if idx == last else None,
        )
    yield _make_usage_chunk()


class TestOpenAIInvokeTraceable:
    """@traceable 적용 여부 검증."""

    def test_invoke_is_wrapped(self):
        assert hasattr(OpenAILLM.invoke, "__wrapped__"), (
            "invoke()에 @traceable이 적용되지 않았습니다."
        )

    def test_invoke_name_preserved(self):
        assert OpenAILLM.invoke.__name__ == "invoke"

    def test_stream_is_wrapped(self):
        assert hasattr(OpenAILLM.stream, "__wrapped__"), (
            "stream()에 @traceable이 적용되지 않았습니다."
        )

    def test_stream_name_preserved(self):
        assert OpenAILLM.stream.__name__ == "stream"


class TestOpenAIRegistration:
    """LLMFactory 등록 검증."""

    def test_provider_registered(self):
        """import 시 'openai' 프로바이더가 팩토리에 등록된다."""
        assert "openai" in LLMFactory.get_available_providers()


class TestOpenAIInit:
    """초기화/검증 동작."""

    def test_raises_without_api_key(self):
        with patch("agent_arch.llm.openai_provider.openai.AsyncOpenAI"):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAILLM({"model": "gpt-5.4-mini"})


class TestOpenAIInvokeResponse:
    """invoke() 응답 변환 검증."""

    @pytest.mark.asyncio
    async def test_invoke_returns_llm_response(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(content="테스트 응답")
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert isinstance(result, LLMResponse)
        assert result.content == "테스트 응답"

    @pytest.mark.asyncio
    async def test_invoke_finish_reason_stop(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(finish_reason="stop")
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert result.finish_reason == "stop"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_invoke_usage_populated(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(prompt_tokens=15, completion_tokens=30)
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert result.usage == {
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "total_tokens": 45,
        }

    @pytest.mark.asyncio
    async def test_invoke_parses_tool_calls(self):
        """응답에 tool_calls가 있으면 LLMResponse.tool_calls로 파싱된다."""
        llm = _make_openai_llm(supports_function_calling=True)
        tool_calls = [_make_tool_call("call_1", "get_weather", '{"city": "서울"}')]
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(content="", tool_calls=tool_calls)
        )

        result = await llm.invoke([Message(role="user", content="날씨 알려줘")])

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "서울"}

    @pytest.mark.asyncio
    async def test_invoke_raises_on_api_error(self):
        import openai
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            side_effect=openai.OpenAIError("연결 실패")
        )

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            await llm.invoke([Message(role="user", content="질문")])


class TestOpenAIMessageConversion:
    """메시지 → OpenAI 형식 변환 검증 (특히 멀티모달 이미지)."""

    @pytest.mark.asyncio
    async def test_image_message_converted_to_content_parts(self):
        """images가 있는 user 메시지는 text + image_url(data URI) 파트 배열로 변환된다."""
        llm = _make_openai_llm()
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([
            Message(
                role="user",
                content="이 화면 분석해줘",
                images=[ImageContent(data="QUJDRA==", mime_type="image/png")],
            )
        ])

        sent_messages = create.call_args.kwargs["messages"]
        content = sent_messages[0]["content"]
        assert isinstance(content, list)
        # 텍스트 파트
        assert content[0] == {"type": "text", "text": "이 화면 분석해줘"}
        # 이미지 파트 (data URI)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,QUJDRA=="

    @pytest.mark.asyncio
    async def test_plain_text_message_uses_string_content(self):
        """이미지가 없으면 content는 일반 문자열로 전달된다."""
        llm = _make_openai_llm()
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([Message(role="user", content="안녕")])

        sent_messages = create.call_args.kwargs["messages"]
        assert sent_messages[0]["content"] == "안녕"

    @pytest.mark.asyncio
    async def test_uses_max_completion_tokens_not_max_tokens(self):
        """GPT-5 계열 호환: max_tokens 대신 max_completion_tokens 로 전달한다."""
        llm = _make_openai_llm()
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([Message(role="user", content="질문")], max_tokens=123)

        kwargs = create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 123
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_reasoning_effort_is_sent_only_when_asked(self):
        """추론 강도는 옵트인 — 호출자가 줄 때만 싣는다(SCMS 거래 구조 추출기가 쓴다)."""
        llm = _make_openai_llm()
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([Message(role="user", content="질문")])
        assert "reasoning_effort" not in create.call_args.kwargs

        await llm.invoke([Message(role="user", content="질문")], reasoning_effort="low")
        assert create.call_args.kwargs["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_tool_role_message_includes_tool_call_id(self):
        """role='tool' 메시지는 tool_call_id를 포함한 형식으로 변환된다."""
        llm = _make_openai_llm()
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([
            Message(role="tool", content='{"temp": 20}', tool_call_id="call_1"),
        ])

        sent = create.call_args.kwargs["messages"][0]
        assert sent["role"] == "tool"
        assert sent["tool_call_id"] == "call_1"
        assert sent["content"] == '{"temp": 20}'


class TestOpenAIToolChoice:
    """tool_mode → tool_choice 매핑 검증."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_mode,expected", [
        ("AUTO", "auto"),
        ("ANY", "required"),
        ("NONE", "none"),
    ])
    async def test_tool_mode_maps_to_tool_choice(self, tool_mode, expected):
        llm = _make_openai_llm(supports_function_calling=True)
        await llm.bind_tools([_FakeTool()])
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([Message(role="user", content="질문")], tool_mode=tool_mode)

        assert create.call_args.kwargs["tool_choice"] == expected
        # 도구 스키마도 함께 전달된다
        assert create.call_args.kwargs["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_no_tools_when_function_calling_disabled(self):
        """supports_function_calling=False면 tools/tool_choice를 보내지 않는다."""
        llm = _make_openai_llm(supports_function_calling=False)
        await llm.bind_tools([_FakeTool()])
        create = AsyncMock(return_value=_make_openai_response())
        llm.client.chat.completions.create = create

        await llm.invoke([Message(role="user", content="질문")])

        assert "tools" not in create.call_args.kwargs
        assert "tool_choice" not in create.call_args.kwargs


class TestOpenAIInvokeLangSmith:
    """invoke()의 LangSmith run tree 메타데이터·태그 주입 검증."""

    @pytest.mark.asyncio
    async def test_model_tag_added_to_run_tree(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_tags.assert_called_once_with([llm.model])

    @pytest.mark.asyncio
    async def test_no_error_when_run_tree_is_none(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert isinstance(result, LLMResponse)


class TestOpenAIStream:
    """stream()의 텍스트/도구호출/done 이벤트 검증."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("안녕", "하세요")
        )

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="안녕?")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["안녕", "하세요"]
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_accumulates_tool_calls(self):
        """tool_calls 델타가 인덱스별로 누적되어 done(tool_calls) 전에 한 번에 yield된다."""
        llm = _make_openai_llm(supports_function_calling=True)
        await llm.bind_tools([_FakeTool()])

        async def tool_stream():
            yield _make_stream_chunk(
                tool_calls=[_make_tool_delta(0, call_id="call_1", name="get_weather", arguments='{"ci')]
            )
            yield _make_stream_chunk(
                tool_calls=[_make_tool_delta(0, arguments='ty": "서울"}')],
                finish_reason="tool_calls",
            )
            yield _make_usage_chunk()

        llm.client.chat.completions.create = AsyncMock(return_value=tool_stream())

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="날씨?")])]

        tool_events = [e for e in events if e.type == "tool_calls"]
        assert len(tool_events) == 1
        tcs = tool_events[0].tool_calls
        assert len(tcs) == 1
        assert tcs[0].name == "get_weather"
        assert tcs[0].arguments == {"city": "서울"}
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_yields_done_error_on_api_error(self):
        import openai
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            side_effect=openai.OpenAIError("스트리밍 실패")
        )

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert events[-1].type == "done"
        assert events[-1].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_stream_error_keeps_usage_already_received(self):
        """usage 를 받은 뒤 끊기면 버리지 않는다 — 버리면 크레딧이 과소 집계된다."""
        import openai
        llm = _make_openai_llm()

        async def _usage_then_error():
            yield _make_stream_chunk("부분", finish_reason=None)
            yield _make_usage_chunk(prompt=5, completion=2)
            raise openai.OpenAIError("스트림 중단")

        llm.client.chat.completions.create = AsyncMock(return_value=_usage_then_error())

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        done = events[-1]
        assert done.finish_reason == "error"
        assert done.usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    @pytest.mark.asyncio
    async def test_stream_model_tag_added_to_run_tree(self):
        llm = _make_openai_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("토큰")
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            _ = [e async for e in llm.stream([Message(role="user", content="질문")])]

        mock_run.add_tags.assert_called_once_with([llm.model])


class TestParamFallback:
    """모델이 거부하는 인자를 떼고 재시도하는 안전망 (도면 분석 레인 전용).

    회사가 임의의 모델을 고를 수 있는 경로라서 필요하다 — 실측: gpt-5.6-sol 은
    `temperature=0.0` 을 거부하고(400 unsupported_value) 그대로 두면 도면 판독·판정·
    리포트가 전부 죽는다. 반대로 우리가 모델·인자를 통제하는 에이전트 대화 경로에서는
    조용한 재시도가 잘못된 설정을 숨기므로 **켜지지 않아야** 한다.
    """

    _REJECT = (
        "Error code: 400 - {'error': {'message': \"Unsupported value: 'temperature' does "
        "not support 0.0 with this model. Only the default (1) value is supported.\", "
        "'type': 'invalid_request_error', 'param': 'temperature', "
        "'code': 'unsupported_value'}}"
    )

    def _llm(self, *, param_fallback: bool) -> OpenAILLM:
        with patch("agent_arch.llm.openai_provider.openai.AsyncOpenAI"):
            return OpenAILLM({
                "api_key": "sk-test-key",
                "model": "gpt-5.6-sol",
                "temperature": 0.0,
                "max_tokens": 4096,
                "supports_streaming": True,
                "param_fallback": param_fallback,
            })

    @pytest.mark.asyncio
    async def test_invoke_drops_rejected_param_and_retries(self):
        """거부된 temperature 를 떼고 한 번 더 호출해 성공한다."""
        import openai as openai_mod

        llm = self._llm(param_fallback=True)
        ok = _make_openai_response("판독 결과")
        create = AsyncMock(side_effect=[openai_mod.OpenAIError(self._REJECT), ok])
        llm.client.chat.completions.create = create

        out = await llm.invoke([Message(role="user", content="도면 판독")])

        assert out.content == "판독 결과"
        assert create.await_count == 2
        # 1회차엔 temperature 가 있었고, 2회차엔 빠져 있어야 한다.
        assert "temperature" in create.await_args_list[0].kwargs
        assert "temperature" not in create.await_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_invoke_without_fallback_raises(self):
        """param_fallback 이 꺼진 경로(에이전트 대화)는 실패를 그대로 드러낸다."""
        import openai as openai_mod

        llm = self._llm(param_fallback=False)
        create = AsyncMock(side_effect=openai_mod.OpenAIError(self._REJECT))
        llm.client.chat.completions.create = create

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            await llm.invoke([Message(role="user", content="안녕")])
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_model_rejection_is_not_retried(self):
        """model 자체가 거부되면(chat 모델 아님 등) 뗄 인자가 없으므로 재시도하지 않는다."""
        import openai as openai_mod

        llm = self._llm(param_fallback=True)
        err = (
            "Error code: 404 - {'error': {'message': 'This is not a chat model and thus "
            "not supported in the v1/chat/completions endpoint.', 'param': 'model', "
            "'code': None}}"
        )
        create = AsyncMock(side_effect=openai_mod.OpenAIError(err))
        llm.client.chat.completions.create = create

        with pytest.raises(RuntimeError):
            await llm.invoke([Message(role="user", content="x")])
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_stream_drops_rejected_param_and_retries(self):
        """리포트 생성(stream)도 같은 안전망을 탄다 — invoke 만 고치면 여기서 죽는다."""
        import openai as openai_mod

        llm = self._llm(param_fallback=True)

        async def _chunks():
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta.content = "리포트"
            choice.delta.tool_calls = None
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            chunk.usage = None
            yield chunk

        create = AsyncMock(side_effect=[openai_mod.OpenAIError(self._REJECT), _chunks()])
        llm.client.chat.completions.create = create

        texts = [e.content async for e in llm.stream([Message(role="user", content="리포트")])
                 if e.type == "text"]

        assert "리포트" in "".join(t or "" for t in texts)
        assert create.await_count == 2
        assert "temperature" not in create.await_args_list[1].kwargs


class TestResponsesApi:
    """config api=responses — gpt-6.1-sol 은 Chat Completions 에서 함수 도구를 못 써서 이 경로로 부른다."""

    def _llm(self):
        with patch("agent_arch.llm.openai_provider.openai.AsyncOpenAI"):
            llm = OpenAILLM({"api_key": "sk-test-key", "model": "gpt-6.1-sol", "temperature": 0.0,
                             "max_tokens": 4096, "supports_function_calling": True, "supports_streaming": True,
                             "api": "responses", "reasoning_effort": "low"})
        llm._bound_tools = [_FakeTool()]
        return llm

    @staticmethod
    def _usage():
        return MagicMock(input_tokens=11, output_tokens=7, total_tokens=18, input_tokens_details=None)

    def test_input_conversion_keeps_tool_round_trip(self):
        from agent_arch.llm.openai_responses import to_input

        items = to_input([
            Message(role="system", content="규칙"),
            Message(role="user", content="서울 날씨"),
            Message(role="assistant", content="", tool_calls=[{"id": "call_1", "name": "get_weather",
                                                               "arguments": {"city": "서울"}}]),
            Message(role="tool", content="맑음", tool_call_id="call_1"),
        ])
        assert items[0] == {"role": "system", "content": "규칙"}
        assert items[2] == {"type": "function_call", "call_id": "call_1", "name": "get_weather",
                            "arguments": '{"city": "서울"}'}
        assert items[3] == {"type": "function_call_output", "call_id": "call_1", "output": "맑음"}

    @pytest.mark.asyncio
    async def test_invoke_uses_responses_with_tools_and_effort(self):
        llm = self._llm()
        call = MagicMock(type="function_call", call_id="call_9", arguments='{"city": "부산"}')
        call.name = "get_weather"
        resp = MagicMock(output=[call], usage=self._usage(), status="completed")
        llm.client.responses.create = AsyncMock(return_value=resp)
        llm.client.chat.completions.create = AsyncMock()

        out = await llm.invoke([Message(role="user", content="부산 날씨")], tool_mode="ANY",
                               allowed_tools=["get_weather"])

        kw = llm.client.responses.create.await_args.kwargs
        assert kw["reasoning"] == {"effort": "low"} and kw["store"] is False and "temperature" not in kw
        assert kw["tools"][0]["name"] == "get_weather" and "function" not in kw["tools"][0]
        assert kw["tool_choice"] == {"type": "function", "name": "get_weather"}
        assert out.finish_reason == "tool_calls" and out.tool_calls[0].arguments == {"city": "부산"}
        assert out.usage["total_tokens"] == 18
        llm.client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_yields_text_then_tool_calls(self):
        """본문이 먼저 나와도 함수 호출을 버리지 않는다(gemini 스트림과 같은 방침)."""
        llm = self._llm()
        call = MagicMock(type="function_call", call_id="call_2", arguments='{"city": "대구"}')
        call.name = "get_weather"
        done_resp = MagicMock(usage=self._usage(), status="completed")

        async def _events():
            yield MagicMock(type="response.output_text.delta", delta="확인할게요")
            yield MagicMock(type="response.output_item.done", item=call)
            yield MagicMock(type="response.completed", response=done_resp)

        llm.client.responses.create = AsyncMock(return_value=_events())
        events = [e async for e in llm.stream([Message(role="user", content="대구")])]

        assert [e.type for e in events] == ["text", "tool_calls", "done"]
        assert events[1].tool_calls[0].name == "get_weather"
        assert events[2].finish_reason == "tool_calls" and events[2].usage["total_tokens"] == 18
        assert llm.client.responses.create.await_args.kwargs["stream"] is True
