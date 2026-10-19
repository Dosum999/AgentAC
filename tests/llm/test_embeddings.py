"""agent_arch.llm.embeddings 공통 임베딩 헬퍼 테스트.

OpenAI ``embeddings.create`` 호출 파라미터(model/input/dimensions)와 반환 파싱,
lazy 클라이언트 캐시, API 키 미설정 시 에러를 검증한다. OpenAI SDK 호출은 Mock으로
대체한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai
from google.genai import errors as genai_errors

from agent_arch.llm import embeddings


def _make_embeddings_response(vector):
    """OpenAI SDK embeddings.create 응답 Mock 객체를 생성한다."""
    resp = MagicMock()
    item = MagicMock()
    item.embedding = vector
    resp.data = [item]
    return resp


@pytest.fixture(autouse=True)
def _reset_client():
    """각 테스트마다 모듈 전역 lazy 클라이언트 캐시(OpenAI/Gemini)를 초기화한다."""
    embeddings._openai_client = None
    embeddings._gemini_client = None
    yield
    embeddings._openai_client = None
    embeddings._gemini_client = None


class TestEmbedText:
    """embed_text 동작 검증."""

    @pytest.mark.asyncio
    async def test_calls_create_with_model_and_dimensions(self):
        """embeddings.create 가 settings.embedding_model, input, dimensions 로 호출된다."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(
            return_value=_make_embeddings_response([0.1] * 3072)
        )
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            result = await embeddings.embed_text("안녕하세요", dimensions=3072)

        fake_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-large",
            input="안녕하세요",
            dimensions=3072,
        )
        assert result == [0.1] * 3072
        assert len(result) == 3072

    @pytest.mark.asyncio
    async def test_default_dimensions_is_3072(self):
        """dimensions 미지정 시 기본값 3072 로 호출된다."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(
            return_value=_make_embeddings_response([0.0] * 3072)
        )
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            await embeddings.embed_text("질의")

        _, kwargs = fake_client.embeddings.create.call_args
        assert kwargs["dimensions"] == 3072

    @pytest.mark.asyncio
    async def test_client_is_cached_across_calls(self):
        """AsyncOpenAI 클라이언트는 lazy-init 후 재사용된다 (1회만 생성)."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(
            return_value=_make_embeddings_response([0.0] * 3072)
        )
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client) as ctor, \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            await embeddings.embed_text("첫 번째")
            await embeddings.embed_text("두 번째")

        assert ctor.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_without_api_key(self):
        """OPENAI_API_KEY 가 없으면 ValueError 를 던진다."""
        with patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.embedding_model = "text-embedding-3-large"

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await embeddings.embed_text("키 없음")

    @pytest.mark.asyncio
    async def test_wraps_openai_error_as_runtime_error(self):
        """OpenAIError 는 RuntimeError 로 래핑된다."""
        import openai

        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(
            side_effect=openai.OpenAIError("rate limited")
        )
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            with pytest.raises(RuntimeError, match="OpenAI embedding error"):
                await embeddings.embed_text("실패")


class TestProviderSelection:
    """모델명 기반 프로바이더 판별."""

    def test_is_gemini_model(self):
        assert embeddings._is_gemini_model("models/gemini-embedding-001") is True
        assert embeddings._is_gemini_model("gemini-embedding-001") is True
        assert embeddings._is_gemini_model("text-embedding-3-large") is False
        assert embeddings._is_gemini_model("") is False


def _make_gemini_response(vector):
    """Gemini embed_content 응답 Mock 객체를 생성한다."""
    result = MagicMock()
    emb = MagicMock()
    emb.values = vector
    result.embeddings = [emb]
    return result


class TestGeminiPath:
    """embedding_model 이 Gemini 모델일 때 genai 경로로 동작하는지 검증."""

    @pytest.mark.asyncio
    async def test_gemini_model_uses_embed_content_with_task_type(self):
        """Gemini 모델명이면 genai embed_content 가 model/contents/task_type 으로 호출된다."""
        fake_client = MagicMock()
        fake_client.aio.models.embed_content = AsyncMock(
            return_value=_make_gemini_response([0.2] * 3072)
        )
        with patch("agent_arch.llm.embeddings.genai.Client", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.gemini_api_key = "g-test"
            mock_settings.embedding_model = "models/gemini-embedding-001"

            result = await embeddings.embed_text("문서", task_type="RETRIEVAL_QUERY")

        assert result == [0.2] * 3072
        _, kwargs = fake_client.aio.models.embed_content.call_args
        assert kwargs["model"] == "models/gemini-embedding-001"
        assert kwargs["contents"] == "문서"
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_gemini_raises_without_key(self):
        """Gemini 모델인데 GEMINI_API_KEY 가 없으면 ValueError."""
        with patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.gemini_api_key = None
            mock_settings.embedding_model = "models/gemini-embedding-001"

            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                await embeddings.embed_text("키 없음")

    @pytest.mark.asyncio
    async def test_gemini_error_wrapped_with_prefix(self):
        """Gemini 호출 실패는 'Gemini embedding error' 접두어로 래핑된다(빈 메시지 방지)."""
        fake_client = MagicMock()
        fake_client.aio.models.embed_content = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("agent_arch.llm.embeddings.genai.Client", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.gemini_api_key = "g-test"
            mock_settings.embedding_model = "models/gemini-embedding-001"

            with pytest.raises(RuntimeError, match="Gemini embedding error"):
                await embeddings.embed_text("실패")


class _FakeAPIStatusError(openai.APIStatusError):
    """status_code만 지정하는 경량 OpenAI APIStatusError 더미."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeGeminiAPIError(genai_errors.APIError):
    """code(HTTP 상태)만 지정하는 경량 Gemini APIError 더미."""

    def __init__(self, code: int):
        self.code = code


class TestIsRetryable:
    """_is_retryable: rate limit(429)·일시 오류만 재시도 대상으로 분류."""

    def test_openai_rate_limit_is_retryable(self):
        class _RL(openai.RateLimitError):
            def __init__(self):
                pass

        assert embeddings._is_retryable(_RL()) is True

    def test_openai_5xx_status_is_retryable(self):
        assert embeddings._is_retryable(_FakeAPIStatusError(503)) is True
        assert embeddings._is_retryable(_FakeAPIStatusError(429)) is True

    def test_openai_4xx_status_is_not_retryable(self):
        """400(잘못된 입력) 등은 재시도 대상이 아니다."""
        assert embeddings._is_retryable(_FakeAPIStatusError(400)) is False

    def test_gemini_429_is_retryable(self):
        assert embeddings._is_retryable(_FakeGeminiAPIError(429)) is True
        assert embeddings._is_retryable(_FakeGeminiAPIError(503)) is True

    def test_gemini_4xx_is_not_retryable(self):
        assert embeddings._is_retryable(_FakeGeminiAPIError(400)) is False

    def test_generic_exception_is_not_retryable(self):
        assert embeddings._is_retryable(ValueError("nope")) is False
        assert embeddings._is_retryable(RuntimeError("boom")) is False


class TestEmbedRetry:
    """embed_text의 429/일시 오류 지수 백오프 재시도."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        """일시 오류 1회 후 성공하면 재시도하여 정상 결과를 반환한다."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(
            side_effect=[
                _FakeAPIStatusError(429),
                _make_embeddings_response([0.5] * 3072),
            ]
        )
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.asyncio.sleep", AsyncMock()) as mock_sleep, \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            result = await embeddings.embed_text("재시도")

        assert result == [0.5] * 3072
        assert fake_client.embeddings.create.await_count == 2
        assert mock_sleep.await_count == 1  # 1회 백오프

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(self):
        """일시 오류가 계속되면 _MAX_RETRIES만큼 시도 후 RuntimeError를 던진다."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=_FakeAPIStatusError(429))
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.asyncio.sleep", AsyncMock()) as mock_sleep, \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            with pytest.raises(RuntimeError, match="OpenAI embedding error"):
                await embeddings.embed_text("계속 실패")

        assert fake_client.embeddings.create.await_count == embeddings._MAX_RETRIES
        assert mock_sleep.await_count == embeddings._MAX_RETRIES - 1

    @pytest.mark.asyncio
    async def test_non_retryable_fails_immediately(self):
        """비재시도 오류(400 등)는 즉시 실패하고 백오프하지 않는다."""
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=_FakeAPIStatusError(400))
        with patch("agent_arch.llm.embeddings.openai.AsyncOpenAI", return_value=fake_client), \
             patch("agent_arch.llm.embeddings.asyncio.sleep", AsyncMock()) as mock_sleep, \
             patch("agent_arch.llm.embeddings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.embedding_model = "text-embedding-3-large"

            with pytest.raises(RuntimeError, match="OpenAI embedding error"):
                await embeddings.embed_text("잘못된 입력")

        assert fake_client.embeddings.create.await_count == 1
        mock_sleep.assert_not_awaited()
