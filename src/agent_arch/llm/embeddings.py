"""임베딩 공통 헬퍼 (프로바이더는 ``settings.embedding_model`` 값으로 결정).

모델명만 바꾸면 임베딩 프로바이더가 따라가도록 설계한다:
  - ``"models/..."`` 또는 이름에 ``gemini`` 포함 → Google Gemini (genai)
  - 그 외 (예: ``text-embedding-3-large``)               → OpenAI

이렇게 하면 ``EMBEDDING_MODEL`` 환경변수만으로 Gemini↔OpenAI 전환이 가능하다(코드 변경 불필요).

차이점:
  - Gemini 는 ``task_type``(RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)으로 문서·쿼리를 구분하고
    출력 차원이 모델 고정(gemini-embedding-001 = 3072)이라 ``dimensions`` 를 무시한다.
  - OpenAI 는 ``task_type`` 개념이 없고 ``dimensions`` 로 출력 차원을 지정한다.
"""
import asyncio
import logging
from typing import List, Optional

import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from agent_arch.core.config import settings
from agent_arch.services.usage.meter import meter_llm_call

logger = logging.getLogger(__name__)

# 프로바이더별 클라이언트 (lazy init, 프로세스 단위 공유)
_openai_client: Optional[openai.AsyncOpenAI] = None
_gemini_client: Optional[genai.Client] = None

# rate limit(429)·일시적 서버/네트워크 오류에 대한 재시도 설정.
# 인덱싱을 병렬화하면 동시 요청이 늘어 RPM 한도에 닿을 수 있으므로,
# 임베딩 호출 단위에서 지수 백오프로 재시도해 일시적 한도 초과를 흡수한다.
_MAX_RETRIES = 4  # 최초 1회 + 재시도 3회
_BACKOFF_CAP = 30  # 백오프 상한(초)
# 재시도 대상 HTTP 상태 코드 (429=rate limit, 5xx=일시적 서버 오류)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(e: Exception) -> bool:
    """rate limit(429)·일시적 서버/네트워크 오류인지 판별한다(재시도 대상).

    인증 실패·잘못된 입력 등 영구 오류는 재시도해도 동일하게 실패하므로 제외한다.

    Args:
        e: 임베딩 호출 중 발생한 예외.

    Returns:
        재시도해야 하는 일시적 오류면 True.
    """
    # OpenAI — 타입으로 분류되는 일시 오류
    if isinstance(
        e,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ),
    ):
        return True
    # OpenAI — 그 외 상태 코드 기반 분류
    if isinstance(e, openai.APIStatusError):
        return e.status_code in _RETRYABLE_STATUS
    # Gemini — APIError.code 에 HTTP 상태 코드가 담긴다
    if isinstance(e, genai_errors.APIError):
        return getattr(e, "code", None) in _RETRYABLE_STATUS
    return False


def _is_gemini_model(model: str) -> bool:
    """모델명이 Gemini 임베딩 모델인지 판별한다."""
    m = (model or "").lower()
    return m.startswith("models/") or "gemini" in m


def _get_openai_client() -> openai.AsyncOpenAI:
    """OpenAI 임베딩 클라이언트를 lazy-init 으로 반환한다."""
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        _openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_gemini_client() -> genai.Client:
    """Gemini 임베딩 클라이언트를 lazy-init 으로 반환한다."""
    global _gemini_client
    if _gemini_client is None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


async def _embed_once(
    client,
    model: str,
    text: str,
    *,
    task_type: str,
    dimensions: int,
    is_gemini: bool,
) -> List[float]:
    """확보된 클라이언트로 임베딩 API를 한 번 호출한다(재시도 없음).

    클라이언트 확보(= 키 검증)는 호출자가 루프 밖에서 끝낸 상태로 들어온다.
    여기서는 순수하게 네트워크 호출만 수행하므로, 발생하는 예외는 곧 일시적
    API 오류 후보다. 재시도 여부 판별(``_is_retryable``)은 호출자가 수행한다.
    """
    if is_gemini:
        result = await client.aio.models.embed_content(
            model=model,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        # Gemini Developer API 의 임베딩 응답에는 토큰 수가 없다. 글자 수를 토큰으로 본다 —
        # 한국어는 대략 맞고 영문은 과대 집계지만, 단가가 $0.15/1M 라 오차가 청구를 흔들지 않는다.
        meter_llm_call(
            "gemini", model, {"prompt_tokens": len(text), "completion_tokens": 0},
            source="embedding.gemini_estimate",
        )
        return list(result.embeddings[0].values)

    resp = await client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions,
    )
    used = getattr(resp, "usage", None)
    meter_llm_call(
        "openai", model,
        {"prompt_tokens": getattr(used, "prompt_tokens", 0) or 0, "completion_tokens": 0} if used else None,
        source="embedding.openai",
    )
    return list(resp.data[0].embedding)


async def embed_text(
    text: str,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    dimensions: int = 3072,
) -> List[float]:
    """텍스트를 임베딩 벡터로 변환한다 (프로바이더는 모델명으로 결정).

    rate limit(429)·일시적 서버/네트워크 오류는 지수 백오프로 재시도한다.
    인증 실패·잘못된 입력 등 영구 오류는 재시도하지 않고 즉시 실패한다.

    Args:
        text: 임베딩할 텍스트.
        task_type: Gemini 전용 — RETRIEVAL_DOCUMENT(인덱싱) / RETRIEVAL_QUERY(검색).
            OpenAI 에서는 무시된다.
        dimensions: OpenAI 전용 — 출력 차원(기존 컬렉션 스키마 3072 에 맞춤).
            Gemini 에서는 모델 고정 차원이라 무시된다.

    Returns:
        임베딩 벡터 리스트.

    Raises:
        RuntimeError: 임베딩 API 호출 실패 시(메시지에 프로바이더/원인 접두어 포함).
        ValueError: 해당 프로바이더의 API 키가 없는 경우.
    """
    model = settings.embedding_model
    is_gemini = _is_gemini_model(model)
    provider = "Gemini" if is_gemini else "OpenAI"

    # 클라이언트 확보(= 키 검증)는 재시도 루프 밖에서 1회만 수행한다.
    # 키 누락 ValueError는 일시 오류가 아니므로 재시도/래핑 대상이 아니며,
    # 여기서 그대로 호출자에게 전파된다(docstring의 ValueError 계약 유지).
    client = _get_gemini_client() if is_gemini else _get_openai_client()

    last_error: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            return await _embed_once(
                client,
                model,
                text,
                task_type=task_type,
                dimensions=dimensions,
                is_gemini=is_gemini,
            )
        except Exception as e:
            last_error = e
            # 일시 오류이고 재시도 여유가 있으면 지수 백오프 후 재시도
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                wait_time = min(2 ** attempt, _BACKOFF_CAP)
                logger.warning(
                    "%s 임베딩 일시 오류 재시도 %d/%d (%ds 대기): %s",
                    provider,
                    attempt + 1,
                    _MAX_RETRIES - 1,
                    wait_time,
                    type(e).__name__,
                )
                await asyncio.sleep(wait_time)
                continue
            break

    raise RuntimeError(
        f"{provider} embedding error: {type(last_error).__name__}: {last_error!r}"
    )
