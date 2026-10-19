"""VllmLLM 테스트.

@traceable 적용, 응답 변환, usage 추출, LangSmith run tree 메타데이터
주입, stream() first token 측정을 검증한다. vLLM API 호출은 Mock으로 대체한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_arch.llm.vllm import VllmLLM
from agent_arch.llm.base import Message, LLMResponse


def _make_vllm_llm() -> VllmLLM:
    """openai.AsyncOpenAI를 Mock 처리하여 VllmLLM 인스턴스를 생성한다."""
    with patch("agent_arch.llm.vllm.openai.AsyncOpenAI"):
        return VllmLLM({
            "base_url": "http://localhost:8001",
            "model": "Qwen/Qwen2.5-7B",
            "temperature": 0.7,
            "max_tokens": 1024,
        })


def _make_openai_response(
    content: str = "응답",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    with_usage: bool = True,
) -> MagicMock:
    """OpenAI SDK ChatCompletion Mock 객체를 생성한다."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = finish_reason

    if with_usage:
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens
        mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    else:
        mock_response.usage = None

    return mock_response


def _make_stream_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """스트리밍 응답 청크 Mock 객체를 생성한다.

    OpenAI 호환 스트림은 각 청크의 delta 에 content 또는 tool_calls 가 들어오고,
    마지막 content 청크에 finish_reason 이 설정된다. 본 헬퍼는 delta.tool_calls
    를 명시적으로 None 으로 두어 본 코드의 tool_calls 분기를 우회한다.
    """
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
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


async def _async_chunks(*contents: str):
    """스트리밍 청크 시퀀스 + 종료 청크 흉내 비동기 제너레이터.

    마지막 content 청크에 finish_reason="stop" 을 설정하고, 그 뒤에 usage 전용
    청크를 한 번 더 yield 하여 실제 OpenAI 스트림 구조를 흉내낸다.
    """
    last = len(contents) - 1
    for idx, content in enumerate(contents):
        yield _make_stream_chunk(
            content,
            finish_reason="stop" if idx == last else None,
        )
    yield _make_usage_chunk()


class TestVllmInvokeTraceable:
    """@traceable 적용 여부 검증."""

    def test_invoke_is_wrapped(self):
        """`invoke`는 @traceable 래핑으로 __wrapped__ 속성을 갖는다."""
        assert hasattr(VllmLLM.invoke, "__wrapped__"), (
            "invoke()에 @traceable이 적용되지 않았습니다."
        )

    def test_invoke_name_preserved(self):
        """래핑 후에도 함수 이름이 보존된다."""
        assert VllmLLM.invoke.__name__ == "invoke"

    def test_stream_is_wrapped(self):
        """`stream`은 @traceable 래핑으로 __wrapped__ 속성을 갖는다."""
        assert hasattr(VllmLLM.stream, "__wrapped__"), (
            "stream()에 @traceable이 적용되지 않았습니다."
        )

    def test_stream_name_preserved(self):
        """래핑 후에도 함수 이름이 보존된다."""
        assert VllmLLM.stream.__name__ == "stream"


class TestVllmInvokeResponse:
    """invoke() 응답 변환 검증."""

    @pytest.mark.asyncio
    async def test_invoke_returns_llm_response(self):
        """invoke()는 항상 LLMResponse 인스턴스를 반환한다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(content="테스트 응답")
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert isinstance(result, LLMResponse)

    @pytest.mark.asyncio
    async def test_invoke_text_content(self):
        """vLLM 텍스트 응답이 LLMResponse.content에 담긴다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(content="안녕하세요!")
        )

        result = await llm.invoke([Message(role="user", content="안녕?")])

        assert result.content == "안녕하세요!"

    @pytest.mark.asyncio
    async def test_invoke_finish_reason_stop(self):
        """일반 텍스트 응답의 finish_reason은 'stop'이다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(finish_reason="stop")
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert result.finish_reason == "stop"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_invoke_usage_populated(self):
        """usage가 있으면 LLMResponse.usage에 토큰 수가 채워진다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(prompt_tokens=15, completion_tokens=30)
        )

        result = await llm.invoke([Message(role="user", content="질문")])

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 30
        assert result.usage["total_tokens"] == 45

    @pytest.mark.asyncio
    async def test_invoke_raises_on_api_error(self):
        """vLLM API 오류 시 RuntimeError를 발생시킨다."""
        import openai
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            side_effect=openai.OpenAIError("연결 실패")
        )

        with pytest.raises(RuntimeError, match="vLLM API error"):
            await llm.invoke([Message(role="user", content="질문")])


class TestVllmInvokeLangSmith:
    """invoke()의 LangSmith run tree 메타데이터·태그 주입 검증."""

    @pytest.mark.asyncio
    async def test_model_tag_added_to_run_tree(self):
        """invoke() 호출 시 run tree에 모델명 태그가 추가된다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_tags.assert_called_once_with([llm.model])

    @pytest.mark.asyncio
    async def test_usage_added_to_run_tree(self):
        """usage가 있으면 run tree의 metadata에 usage가 기록된다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(prompt_tokens=10, completion_tokens=20)
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
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
    async def test_ls_model_name_recorded_when_usage_is_none(self):
        """usage가 없어도 ls_model_name은 기록하되, usage/usage_metadata는 세팅하지 않는다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(with_usage=False)
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            await llm.invoke([Message(role="user", content="질문")])

        mock_run.add_metadata.assert_called_once_with({"ls_model_name": llm.model})
        mock_run.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어 run tree가 None이어도 invoke()는 정상 동작한다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            result = await llm.invoke([Message(role="user", content="질문")])

        assert isinstance(result, LLMResponse)


class TestVllmStream:
    """stream()의 토큰 yield, first token 측정, LangSmith 기록 검증."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        """stream()은 청크의 content를 StreamEvent(type=text) 로 순서대로 yield 하고 마지막에 done."""
        llm = _make_vllm_llm()
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
    async def test_stream_skips_empty_content_chunks(self):
        """content가 None 인 청크는 text 이벤트를 yield하지 않는다."""
        llm = _make_vllm_llm()

        async def mixed():
            yield _make_stream_chunk(content=None)
            yield _make_stream_chunk("실제토큰", finish_reason="stop")
            yield _make_usage_chunk()

        llm.client.chat.completions.create = AsyncMock(return_value=mixed())

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["실제토큰"]
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_records_first_token_ms(self):
        """스트림 완료 후 run tree metadata에 first_token_ms가 기록된다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("첫", "토큰")
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            _ = [e async for e in llm.stream([Message(role="user", content="질문")])]

        mock_run.add_metadata.assert_called_once()
        metadata_arg = mock_run.add_metadata.call_args[0][0]
        assert "first_token_ms" in metadata_arg
        assert isinstance(metadata_arg["first_token_ms"], float)
        assert metadata_arg["first_token_ms"] >= 0

    @pytest.mark.asyncio
    async def test_stream_model_tag_added_to_run_tree(self):
        """stream() 완료 후 run tree에 모델명 태그가 추가된다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("토큰")
        )
        mock_run = MagicMock()

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=mock_run):
            _ = [e async for e in llm.stream([Message(role="user", content="질문")])]

        mock_run.add_tags.assert_called_once_with([llm.model])

    @pytest.mark.asyncio
    async def test_stream_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어도 stream()은 정상 동작한다."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("토큰")
        )

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        text_events = [e for e in events if e.type == "text"]
        assert [e.content for e in text_events] == ["토큰"]

    @pytest.mark.asyncio
    async def test_stream_yields_done_error_on_api_error(self):
        """vLLM 스트리밍 API 오류 시 RuntimeError 를 raise 하지 않고 done(error) 를 yield 한다."""
        import openai
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            side_effect=openai.OpenAIError("스트리밍 실패")
        )

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert events
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_stream_retries_without_tools_when_server_rejects_tool_choice(self):
        """도구 파서 없이 뜬 vLLM 서버가 tool_choice 를 거부하면 도구를 떼고 답을 받는다.

        실측 문구: '"auto" tool choice requires --enable-auto-tool-choice and
        --tool-call-parser to be set'. 서버 기동 옵션 문제라 클라이언트가 고칠 수 없고,
        그대로 포기하면 사용자는 "죄송합니다" 안내로 끝난다.
        """
        import openai
        llm = _make_vllm_llm()
        llm._bound_tools = [MagicMock()]
        llm.supports_function_calling = True
        llm._convert_tool_to_schema = MagicMock(return_value={"type": "function"})

        seen: list[dict] = []

        async def _create(**params):
            seen.append(params)
            if "tool_choice" in params:
                raise openai.OpenAIError(
                    'Error code: 400 - {"error": {"message": "\'auto\' tool choice '
                    'requires --enable-auto-tool-choice and --tool-call-parser to be set"}}'
                )
            return _async_chunks("도구 없이 답변")

        llm.client.chat.completions.create = _create

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert [p.get("tool_choice") for p in seen] == ["auto", None]
        assert "도구 없이 답변" in "".join(e.content or "" for e in events if e.type == "text")
        assert events[-1].finish_reason != "error"

    @pytest.mark.asyncio
    async def test_stream_drops_max_tokens_on_context_overflow(self):
        """컨텍스트 초과(400)면 max_tokens 를 **떼고** 한 번 더 — 프롬프트는 건드리지 않는다.

        예전에는 ``max_tokens // 4`` 로 줄여 재시도했는데, 프롬프트가 한도에 붙은
        대화에서는 줄여도 또 넘쳤다(실측 2026-09-21: 한도 65,536 / 입력 63,489 →
        8192→2048 로도 1토큰 초과). 거부 문구의 "prompt contains at least N" 으로
        여유를 계산하는 것도 안 된다 — 그 N 은 실제 입력 길이가 아니라 넘침을
        유발하는 하한(= 한도 − 요청한 출력 + 1)이라 계산하면 매번 1씩만 깎인다.
        max_tokens 를 빼면 vLLM 이 ``max_model_len − prompt_tokens`` 를 상한으로
        잡아 준다.
        """
        import openai
        llm = _make_vllm_llm()
        seen: list[object] = []

        async def _create(**params):
            seen.append(params.get("max_tokens", "<없음>"))
            if len(seen) == 1:
                raise openai.OpenAIError(
                    "Error code: 400 - {'error': {'message': \"This model's maximum context "
                    'length is 16384 tokens. However, you requested 1024 output tokens and '
                    'your prompt contains at least 15361 input tokens"}}'
                )
            return _async_chunks("짧게 답변")

        llm.client.chat.completions.create = _create

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert len(seen) == 2
        assert isinstance(seen[0], int)   # 1회차는 설정값 그대로
        assert seen[1] == "<없음>"         # 2회차는 아예 빠진다
        assert events[-1].finish_reason != "error"

    @pytest.mark.asyncio
    async def test_invoke_drops_max_tokens_on_context_overflow(self):
        """invoke 에도 같은 계약을 적용한다 — 이 경로에는 원래 재시도가 없었다.

        stream 만 살려 두면 긴 대화에서 도구 없는 단발 호출(메모리 추출·요약 등)이
        그대로 RuntimeError 로 떨어진다.
        """
        import openai
        llm = _make_vllm_llm()
        seen: list[object] = []

        async def _create(**params):
            seen.append(params.get("max_tokens", "<없음>"))
            if len(seen) == 1:
                raise openai.OpenAIError(
                    "Error code: 400 - {'error': {'message': \"This model's maximum context "
                    'length is 65536 tokens. However, you requested 8192 output tokens and '
                    'your prompt contains at least 57345 input tokens"}}'
                )
            return _make_openai_response(content="짧게 답변")

        llm.client.chat.completions.create = _create

        result = await llm.invoke([Message(role="user", content="질문")])

        assert len(seen) == 2
        assert isinstance(seen[0], int)
        assert seen[1] == "<없음>"
        assert "짧게 답변" in (result.content or "")

    @pytest.mark.asyncio
    async def test_stream_error_keeps_usage_already_received(self):
        """usage 청크를 받은 뒤 끊기면 그 usage 를 버리지 않는다.

        버리면 그 턴이 토큰 NULL 로 저장되어 '토큰 미기록'으로 잡히고, 크레딧이 실제보다
        적게 잡힌다(관측 대시보드가 과소 집계로 경고하는 그 건수에 들어간다). 이미 받은
        만큼은 실어 보내야 한다.
        """
        import openai
        llm = _make_vllm_llm()

        async def _usage_then_error():
            yield _make_stream_chunk("부분", finish_reason=None)
            yield _make_usage_chunk(prompt=7, completion=3)
            raise openai.OpenAIError("스트림 중단")

        llm.client.chat.completions.create = AsyncMock(return_value=_usage_then_error())

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        done = events[-1]
        assert done.type == "done" and done.finish_reason == "error"
        assert done.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    @pytest.mark.asyncio
    async def test_stream_error_before_usage_reports_none(self):
        """경계값: usage 를 받기 전에 끊기면 None 이다 — 없는 값을 지어내지 않는다."""
        import openai
        llm = _make_vllm_llm()

        async def _error_immediately():
            yield _make_stream_chunk("부분", finish_reason=None)
            raise openai.OpenAIError("스트림 중단")

        llm.client.chat.completions.create = AsyncMock(return_value=_error_immediately())

        events = [e async for e in llm.stream([Message(role="user", content="질문")])]

        assert events[-1].finish_reason == "error"
        assert events[-1].usage is None


class TestThinkingFSM:
    """_ThinkingFSM 의 두 형식 처리 검증 (invoke 의 _strip_thinking 과 동등)."""

    def test_full_form_yields_response_only(self):
        """완전 형식: thinking 섹션은 버리고 response 마커 뒤만 emit."""
        from agent_arch.llm.vllm import _ThinkingFSM
        fsm = _ThinkingFSM()
        out = list(fsm.feed("<|channel>thought<channel|>비공개 사고<|channel>response<channel|>실제 답변"))
        out.append(fsm.flush())
        assert "".join(out) == "실제 답변"

    def test_short_form_yields_buffer_at_flush(self):
        """단축 형식: response 마커 없이 thinking 마커 뒤 내용을 응답으로 살린다."""
        from agent_arch.llm.vllm import _ThinkingFSM
        fsm = _ThinkingFSM()
        out = list(fsm.feed("<|channel>thought<channel|>실제로는 답변 본문"))
        out.append(fsm.flush())
        assert "".join(out) == "실제로는 답변 본문"

    def test_short_form_streamed_chunks(self):
        """단축 형식을 청크 단위로 흘려도 flush 에서 본문이 살아남는다."""
        from agent_arch.llm.vllm import _ThinkingFSM
        fsm = _ThinkingFSM()
        out = list(fsm.feed("<|channel>thought<channel|>답변"))
        out.extend(fsm.feed(" 이어짐"))
        out.append(fsm.flush())
        assert "".join(out) == "답변 이어짐"

    def test_plain_text_passes_through(self):
        """thinking 마커가 없는 일반 텍스트는 그대로 통과."""
        from agent_arch.llm.vllm import _ThinkingFSM
        fsm = _ThinkingFSM()
        out = list(fsm.feed("일반 응답"))
        out.append(fsm.flush())
        assert "".join(out) == "일반 응답"


class TestParseGemmaCallArgs:
    """평문 'call:NAME{...}' 누출 인자 파서 (비표준 JSON, best-effort)."""

    def test_flat_korean_args(self):
        """콜론 포함 값(08:00)·한국어 자유텍스트·정수가 정확히 파싱된다."""
        from agent_arch.llm.vllm import _parse_gemma_call_args
        body = "attendance_id:7644,clock_in:08:00,reason:출근 시간 정정 요청,shift_id:1"
        assert _parse_gemma_call_args(body) == {
            "attendance_id": 7644, "clock_in": "08:00",
            "reason": "출근 시간 정정 요청", "shift_id": 1,
        }

    def test_strips_gemma_string_delim(self):
        """Gemma4 문자열 구분자 <|\"|> 는 제거된다."""
        from agent_arch.llm.vllm import _parse_gemma_call_args
        body = 'location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>'
        assert _parse_gemma_call_args(body) == {
            "location": "San Francisco", "unit": "celsius",
        }

    def test_bool_and_null_coercion(self):
        """true/false/null 은 타입 변환된다."""
        from agent_arch.llm.vllm import _parse_gemma_call_args
        assert _parse_gemma_call_args("confirm:false,note:none") == {
            "confirm": False, "note": None,
        }

    def test_leading_zero_kept_as_string(self):
        """선두 0 숫자열(전화·사번)은 int 변환하지 않고 문자열로 보존한다."""
        from agent_arch.llm.vllm import _parse_gemma_call_args
        assert _parse_gemma_call_args("phone:01012345678,count:0,n:42") == {
            "phone": "01012345678", "count": 0, "n": 42,
        }


class TestToolCallLeakGuard:
    """평문 call: 누출 가드 — 화면 억제 + ToolCall 복구, 일반 텍스트는 통과."""

    @staticmethod
    def _drain(guard, *chunks) -> str:
        out = []
        for c in chunks:
            out.extend(guard.feed(c))
        out.extend(guard.flush())
        return "".join(out)

    def test_prose_passthrough(self):
        """일반 텍스트는 그대로 통과하고 복구 도구호출 없음."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        assert self._drain(g, "안녕", "하세요") == "안녕하세요"
        assert g.recovered_tool_call() is None

    def test_bare_call_leak_suppressed_and_recovered(self):
        """평문 call:NAME{...} 는 화면에 안 나오고 ToolCall 로 복구된다."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(
            g, "call:hr_request_attendance_correction{attendance_id:7644,reason:출근 정정,shift_id:1}"
        )
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc is not None
        assert tc.name == "hr_request_attendance_correction"
        assert tc.arguments == {"attendance_id": 7644, "reason": "출근 정정", "shift_id": 1}

    def test_thought_lead_then_call_recovered(self):
        """깨진 'thought' 마커 + call: 누출 → 둘 다 억제, 호출 복구."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "thought\n", "call:foo{a:1}")
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc.name == "foo" and tc.arguments == {"a": 1}

    def test_thought_lead_then_prose_strips_only_marker(self):
        """한 줄짜리 'thought' 마커만 떼고 본문은 보존한다."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "thought\n", "실제 답변입니다")
        assert shown == "실제 답변입니다"
        assert g.recovered_tool_call() is None

    def test_word_thought_prose_not_eaten(self):
        """개행 없는 'thought ...' 정상 문장은 마커로 오인해 먹지 않는다."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "thought leadership ", "matters here")
        assert shown == "thought leadership matters here"

    def test_call_open_split_across_chunks(self):
        """call: 시작이 여러 청크로 쪼개져 와도 누출로 인식·복구한다."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "ca", "ll:", "hr_x", "{id:5}")
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc.name == "hr_x" and tc.arguments == {"id": 5}

    def test_call_word_without_colon_is_prose(self):
        """콜론 없는 'call ...' 은 일반 텍스트로 통과."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        assert self._drain(g, "call your manager for details") == "call your manager for details"

    def test_call_colon_space_is_prose(self):
        """'call: 010-...' 처럼 식별자 외 문자가 오면 도구호출 아님 → 텍스트로 통과 (M2)."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        assert self._drain(g, "call: 02-1234-5678 로 연락하세요") == "call: 02-1234-5678 로 연락하세요"
        assert g.recovered_tool_call() is None

    def test_trailing_prose_brace_not_captured(self):
        """call 뒤에 평문이 더 오고 '}'가 또 있어도 인자에 빨려들지 않는다 (H1)."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "call:foo{a:1} 그리고 추가로 } 설명")
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc.name == "foo" and tc.arguments == {"a": 1}

    def test_empty_args_call_recovered(self):
        """무인자 호출 call:NAME{} 도 드롭하지 않고 복구한다 (H3)."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "call:list_all_items{}")
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc is not None and tc.name == "list_all_items" and tc.arguments == {}

    def test_whitespace_lead_chunk_then_call_suppressed(self):
        """선두 공백이 따로 청크로 와도 뒤이은 call: 가 누출되지 않는다 (M1)."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, " " * 30, "call:foo{a:1}")
        assert shown == ""
        tc = g.recovered_tool_call()
        assert tc.name == "foo" and tc.arguments == {"a": 1}

    def test_truncated_call_suppressed_no_recovery(self):
        """닫는 중괄호 없이 잘린 누출은 억제하되 복구는 None (H2: 루프 재시도 유도)."""
        from agent_arch.llm.vllm import _ToolCallLeakGuard
        g = _ToolCallLeakGuard()
        shown = self._drain(g, "call:foo{a:1,b:2")
        assert shown == ""
        assert g.leak_detected() is True
        assert g.recovered_tool_call() is None


class TestVllmStreamLeakFallback:
    """stream() 통합 — 서버가 평문 call: 로 흘려도 사용자 노출 없이 tool_calls 로 복구."""

    @pytest.mark.asyncio
    async def test_streamed_bare_call_recovered_as_tool_calls(self):
        """thought\\ncall:NAME{...} 누출 → text 이벤트 0개 + 구조화 tool_calls + finish=tool_calls."""
        llm = _make_vllm_llm()
        pieces = [
            "thought\n",
            "call:hr_request_attendance_correction{",
            "attendance_id:7644,clock_in:08:00,",
            "reason:출근 시간 정정 요청,shift_id:1}",
        ]

        async def leak():
            last = len(pieces) - 1
            for i, p in enumerate(pieces):
                yield _make_stream_chunk(p, finish_reason="stop" if i == last else None)
            yield _make_usage_chunk()

        llm.client.chat.completions.create = AsyncMock(return_value=leak())

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="근태 정정")])]

        text_events = [e for e in events if e.type == "text"]
        tool_events = [e for e in events if e.type == "tool_calls"]
        assert text_events == []  # 누출 텍스트가 사용자에게 노출되지 않음
        assert len(tool_events) == 1
        tcs = tool_events[0].tool_calls
        assert len(tcs) == 1
        assert tcs[0].name == "hr_request_attendance_correction"
        assert tcs[0].arguments == {
            "attendance_id": 7644, "clock_in": "08:00",
            "reason": "출근 시간 정정 요청", "shift_id": 1,
        }
        assert events[-1].type == "done"
        assert events[-1].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_normal_prose_still_streams(self):
        """일반 답변은 기존대로 토큰 스트리밍된다(회귀 방지)."""
        llm = _make_vllm_llm()
        llm.client.chat.completions.create = AsyncMock(
            return_value=_async_chunks("오늘 ", "날씨는 ", "맑습니다")
        )

        with patch("agent_arch.services.langsmith.langsmith_utils.get_current_run_tree", return_value=None):
            events = [e async for e in llm.stream([Message(role="user", content="날씨")])]

        text = "".join(e.content for e in events if e.type == "text")
        assert text == "오늘 날씨는 맑습니다"
        assert events[-1].finish_reason == "stop"


class TestVllmApiKey:
    """`--api-key` 로 띄운 서버에는 VLLM_API_KEY 를 보내고, 없으면 더미값으로 기존 동작을 지킨다."""

    def _create(self, monkeypatch, key):
        from agent_arch.core.config import settings
        from agent_arch.llm.factory import LLMFactory

        monkeypatch.setattr(settings, "vllm_api_key", key, raising=False)
        monkeypatch.setattr(settings, "vllm_base_url", "http://vllm.test:8000", raising=False)
        return LLMFactory.create("vllm")

    def test_env_key_is_sent(self, monkeypatch):
        llm = self._create(monkeypatch, "secret-key")
        assert llm.client.api_key == "secret-key"

    @pytest.mark.parametrize("key", [None, ""])
    def test_missing_key_falls_back_to_dummy(self, monkeypatch, key):
        llm = self._create(monkeypatch, key)
        assert llm.client.api_key == "token-abc"
