"""GeminiLLM 테스트.

@traceable 적용, 응답 변환, usage_metadata 추출, LangSmith run tree 메타데이터
주입, stream() first token 측정, thought_signature 캡처·복원을 검증한다.
Gemini API 호출은 Mock으로 대체한다.
"""
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from agent_arch.llm.gemini import GeminiLLM
from agent_arch.llm.base import Message, LLMResponse


def _make_mock_response(text: str = "안녕하세요"):
    """Gemini API 응답 Mock 객체를 생성한다. usage_metadata 없음."""
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = MagicMock()  # MALFORMED와 다르므로 stop으로 처리됨
    mock_candidate.content.parts = []

    mock_response = MagicMock()
    mock_response.text = text
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None
    return mock_response


def _make_mock_response_with_usage(
    text: str = "응답",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
):
    """usage_metadata가 포함된 Gemini API 응답 Mock 객체를 생성한다."""
    mock_response = _make_mock_response(text)

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = completion_tokens
    mock_usage.total_token_count = prompt_tokens + completion_tokens
    mock_response.usage_metadata = mock_usage

    return mock_response


def _text_chunk(text: str):
    """text part 가 들어있는 단일 후보의 스트림 청크 Mock 을 생성한다."""
    part = MagicMock()
    part.function_call = None
    part.text = text
    candidate = MagicMock()
    candidate.content.parts = [part]
    candidate.finish_reason = None
    chunk = MagicMock()
    chunk.text = text  # 직접 chunk.text 접근 경로 호환
    chunk.candidates = [candidate]
    chunk.usage_metadata = None
    return chunk


async def _async_chunks(*texts: str):
    """스트리밍 청크 시퀀스를 흉내 내는 비동기 제너레이터.

    stream() 가 candidates[0].content.parts[*].text 구조를 읽으므로
    각 청크에 단일 text part 를 포함시킨다.
    """
    for text in texts:
        yield _text_chunk(text)


def _make_gemini_llm() -> GeminiLLM:
    """genai.Client를 Mock 처리하여 GeminiLLM 인스턴스를 생성한다."""
    with patch("agent_arch.llm.gemini.genai.Client"):
        return GeminiLLM({
            "api_key": "test-api-key",
            "model": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 1024,
            "supports_function_calling": True,
            "supports_streaming": True,
        })


class TestGeminiInvokeTraceable:
    """@traceable 적용 여부 검증."""

    def test_invoke_is_wrapped(self):
        """`invoke`는 @traceable 래핑으로 __wrapped__ 속성을 갖는다."""
        assert hasattr(GeminiLLM.invoke, "__wrapped__"), (
            "invoke()에 @traceable이 적용되지 않았습니다."
        )

    def test_invoke_name_preserved(self):
        """래핑 후에도 함수 이름이 보존된다."""
        assert GeminiLLM.invoke.__name__ == "invoke"


class TestGeminiInvokeResponse:
    """invoke() 응답 변환 검증."""

    @pytest.mark.asyncio
    async def test_invoke_returns_llm_response(self):
        """invoke()는 항상 LLMResponse 인스턴스를 반환한다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="테스트 응답")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="안녕?")])

        assert isinstance(result, LLMResponse)

    @pytest.mark.asyncio
    async def test_invoke_text_content(self):
        """Gemini 텍스트 응답이 LLMResponse.content에 담긴다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="안녕하세요!")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="안녕?")])

        assert result.content == "안녕하세요!"

    @pytest.mark.asyncio
    async def test_invoke_finish_reason_stop(self):
        """일반 텍스트 응답의 finish_reason은 'stop'이다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="응답")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert result.finish_reason == "stop"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_invoke_no_tool_calls_for_text_response(self):
        """텍스트 전용 응답에서는 tool_calls가 None이다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="도구 없는 응답")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_invoke_with_system_message(self):
        """system 메시지가 있어도 invoke()가 정상 동작한다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="시스템 포함 응답")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            messages = [
                Message(role="system", content="당신은 AI 비서입니다."),
                Message(role="user", content="안녕?"),
            ]
            result = await llm.invoke(messages)

        assert result.content == "시스템 포함 응답"

    def test_response_schema_requests_json_output(self):
        """구조화 출력 — 스키마를 넘기면 JSON 응답 모드로 부른다(도면 판독)."""
        llm = _make_gemini_llm()
        schema = {"type": "OBJECT", "properties": {"box": {"type": "ARRAY", "items": {"type": "NUMBER"}}}}
        config = llm._build_generation_config([], {"response_schema": schema})
        assert config["response_mime_type"] == "application/json"
        assert config["response_schema"] is schema
        assert "response_schema" not in llm._build_generation_config([], {})

    @pytest.mark.asyncio
    async def test_invoke_raises_on_api_error(self):
        """Gemini API 오류 시 RuntimeError를 발생시킨다."""
        llm = _make_gemini_llm()

        with patch.object(
            llm.client.aio.models, "generate_content",
            new=AsyncMock(side_effect=Exception("API 연결 실패"))
        ):
            with pytest.raises(RuntimeError, match="Gemini API error"):
                await llm.invoke([Message(role="user", content="질문")])


class TestUsageMetadataExtraction:
    """_parse_response()의 usage_metadata → LLMResponse.usage 변환 검증."""

    @pytest.mark.asyncio
    async def test_usage_populated_when_metadata_present(self):
        """usage_metadata가 있으면 LLMResponse.usage에 토큰 수가 채워진다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage(
            text="응답", prompt_tokens=15, completion_tokens=30
        )

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 30
        assert result.usage["total_tokens"] == 45

    @pytest.mark.asyncio
    async def test_usage_none_when_no_metadata(self):
        """usage_metadata가 없으면 LLMResponse.usage는 None이다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response(text="응답")

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_keys_match_standard_names(self):
        """usage 딕셔너리 키는 prompt_tokens / completion_tokens / total_tokens이다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage(prompt_tokens=5, completion_tokens=10)

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert set(result.usage.keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    @pytest.mark.asyncio
    async def test_usage_total_equals_sum(self):
        """total_tokens는 prompt_tokens + completion_tokens의 합이다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage(prompt_tokens=7, completion_tokens=13)

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert result.usage["total_tokens"] == result.usage["prompt_tokens"] + result.usage["completion_tokens"]


class TestInvokeLangSmithMetadata:
    """invoke()의 LangSmith run tree 메타데이터·태그 주입 검증."""

    @pytest.mark.asyncio
    async def test_model_tag_added_to_run_tree(self):
        """invoke() 호출 시 run tree에 모델명 태그가 추가된다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage()
        mock_run = MagicMock()

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)), \
             patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_tags.assert_called_once_with([llm.model])

    @pytest.mark.asyncio
    async def test_usage_metadata_added_to_run_tree(self):
        """usage_metadata가 있으면 run tree의 metadata에 usage가 기록된다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage(prompt_tokens=10, completion_tokens=20)
        mock_run = MagicMock()

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)), \
             patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_metadata.assert_called_once_with({
            "ls_model_name": llm.model,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        })
        # 표준 usage_metadata(run.set)로도 토큰이 롤업된다 (input/output 네이밍)
        mock_run.set.assert_called_once_with(usage_metadata={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        })

    @pytest.mark.asyncio
    async def test_ls_model_name_recorded_when_usage_absent(self):
        """usage가 없어도 ls_model_name은 기록하되, usage/usage_metadata는 세팅하지 않는다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response()
        mock_run = MagicMock()

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)), \
             patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_metadata.assert_called_once_with({"ls_model_name": llm.model})
        mock_run.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어 run tree가 None이어도 invoke()는 정상 동작한다."""
        llm = _make_gemini_llm()
        mock_response = _make_mock_response_with_usage()

        with patch.object(llm.client.aio.models, "generate_content", new=AsyncMock(return_value=mock_response)), \
             patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert isinstance(result, LLMResponse)


class TestGeminiStreamTraceable:
    """stream()의 @traceable 적용, first token 측정, 토큰 스트리밍 검증."""

    def test_stream_is_wrapped(self):
        """`stream`은 @traceable 래핑으로 __wrapped__ 속성을 갖는다."""
        assert hasattr(GeminiLLM.stream, "__wrapped__"), (
            "stream()에 @traceable이 적용되지 않았습니다."
        )

    def test_stream_name_preserved(self):
        """래핑 후에도 함수 이름이 보존된다."""
        assert GeminiLLM.stream.__name__ == "stream"

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        """stream()은 Gemini 청크의 text 를 StreamEvent(type=text) 로 순서대로 yield 하고 마지막에 done."""
        llm = _make_gemini_llm()

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=_async_chunks("안녕", "하세요")),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="안녕?")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["안녕", "하세요"]
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_text_chunks(self):
        """text가 없는 청크(빈 parts 등)는 text 이벤트를 yield하지 않는다."""
        llm = _make_gemini_llm()

        # 빈 parts 를 가진 청크 (어떤 part 도 없음)
        empty_candidate = MagicMock()
        empty_candidate.content.parts = []
        empty_candidate.finish_reason = None
        empty_chunk = MagicMock()
        empty_chunk.candidates = [empty_candidate]
        empty_chunk.usage_metadata = None

        async def mixed_chunks():
            yield empty_chunk
            yield _text_chunk("실제토큰")

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=mixed_chunks()),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["실제토큰"]
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_records_first_token_ms_in_run_tree(self):
        """스트림 완료 후 run tree metadata에 first_token_ms가 기록된다."""
        llm = _make_gemini_llm()
        mock_run = MagicMock()

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=_async_chunks("첫", "토큰")),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            _ = [t async for t in llm.stream([Message(role="user", content="질문")])]

        mock_run.add_metadata.assert_called_once()
        metadata_arg = mock_run.add_metadata.call_args[0][0]
        assert "first_token_ms" in metadata_arg
        assert isinstance(metadata_arg["first_token_ms"], float)
        assert metadata_arg["first_token_ms"] >= 0

    @pytest.mark.asyncio
    async def test_stream_model_tag_added_to_run_tree(self):
        """stream() 완료 후 run tree에 모델명 태그가 추가된다."""
        llm = _make_gemini_llm()
        mock_run = MagicMock()

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=_async_chunks("토큰")),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            _ = [t async for t in llm.stream([Message(role="user", content="질문")])]

        mock_run.add_tags.assert_called_once_with([llm.model])

    @pytest.mark.asyncio
    async def test_stream_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어 run tree가 None이어도 stream()은 정상 동작한다."""
        llm = _make_gemini_llm()

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=_async_chunks("토큰")),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["토큰"]
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_yields_done_error_on_api_error(self):
        """Gemini 스트리밍 API 오류 시 RuntimeError 를 raise 하지 않고 done(error) 를 yield 한다."""
        llm = _make_gemini_llm()

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(side_effect=Exception("스트리밍 실패")),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert events
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_stream_error_keeps_usage_from_last_chunk(self):
        """청크를 받은 뒤 끊기면 그 청크의 usage_metadata 를 살려 보낸다.

        버리면 토큰 NULL 로 저장되어 '토큰 미기록'으로 잡히고 크레딧이 과소 집계된다.
        """
        llm = _make_gemini_llm()

        chunk = _text_chunk("부분")
        usage = MagicMock()
        usage.prompt_token_count = 11
        usage.candidates_token_count = 4
        usage.total_token_count = 15
        chunk.usage_metadata = usage

        async def _chunk_then_error():
            yield chunk
            raise Exception("스트림 중단")

        with patch.object(
            llm.client.aio.models, "generate_content_stream",
            new=AsyncMock(return_value=_chunk_then_error()),
        ), patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        done = events[-1]
        assert done.finish_reason == "error"
        assert done.usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


class TestThoughtSignatureRoundtrip:
    """thought_signature 캡처(parse_response) → 복원(_convert_messages) 라운드트립 검증."""

    def test_thought_signature_captured_from_response(self):
        """Part.thought_signature(bytes)가 ToolCall에 base64 str로 저장된다."""
        llm = _make_gemini_llm()

        mock_part = MagicMock()
        mock_part.function_call.name = "test_func"
        mock_part.function_call.id = "call_test_func"
        mock_part.function_call.args = {"key": "val"}
        mock_part.thought_signature = b"my-signature-bytes"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        result = llm._parse_response(mock_response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        expected_b64 = base64.b64encode(b"my-signature-bytes").decode("ascii")
        assert result.tool_calls[0].thought_signature == expected_b64

    def test_thought_signature_none_when_missing(self):
        """thought_signature가 없는 Part에서는 ToolCall.thought_signature가 None이다."""
        llm = _make_gemini_llm()

        mock_part = MagicMock()
        mock_part.function_call.name = "no_sig_func"
        mock_part.function_call.id = "call_no_sig_func"
        mock_part.function_call.args = {}
        mock_part.thought_signature = None  # 없는 경우

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        result = llm._parse_response(mock_response)

        assert result.tool_calls is not None
        assert result.tool_calls[0].thought_signature is None

    def test_thought_signature_restored_in_convert_messages(self):
        """Message.tool_calls의 thought_signature가 _convert_messages에서 bytes로 복원된다."""
        llm = _make_gemini_llm()
        sig_bytes = b"round-trip-test"
        sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

        messages = [Message(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_foo",
                "name": "foo",
                "arguments": {"x": 1},
                "thought_signature": sig_b64,
            }]
        )]

        gemini_msgs = llm._convert_messages(messages)

        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "model"
        part = gemini_msgs[0]["parts"][0]
        assert isinstance(part, types.Part)
        assert part.thought_signature == sig_bytes

    def test_thought_signature_absent_stays_clean(self):
        """thought_signature가 None이면 생성된 Part에 thought_signature가 없다."""
        llm = _make_gemini_llm()

        messages = [Message(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_bar",
                "name": "bar",
                "arguments": {},
                "thought_signature": None,
            }]
        )]

        gemini_msgs = llm._convert_messages(messages)

        part = gemini_msgs[0]["parts"][0]
        assert isinstance(part, types.Part)
        # thought_signature가 None이므로 Part에 설정되지 않아야 함
        assert not part.thought_signature


def test_missing_function_call_ids_are_unique_per_call():
    """운영 서버 실측(2026-09-22): 병렬 brave_llm_context 4건이 모두 id `call_brave_llm_context` — id 로 결과를 찾는 코드가 덮였다."""
    from types import SimpleNamespace
    from agent_arch.llm import gemini as G
    a = G._call_id(SimpleNamespace(id=None, name="brave_llm_context"))
    b = G._call_id(SimpleNamespace(id=None, name="brave_llm_context"))
    assert a != b and a.startswith("call_brave_llm_context__")
    assert G._call_id(SimpleNamespace(id="server-id", name="x")) == "server-id"


def test_signature_required_only_for_gemini3_plus():
    from agent_arch.llm import gemini as G
    assert G._requires_signature("gemini-3.1-flash-lite") and G._requires_signature("models/gemini-3.5-flash")
    assert not G._requires_signature("gemini-2.5-flash") and not G._requires_signature(None)
