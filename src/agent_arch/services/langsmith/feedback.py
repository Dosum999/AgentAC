"""LangSmith 피드백 및 데이터셋 연동 서비스.

사용자 rating을 LangSmith 피드백으로 전송하고,
(query, answer) 쌍을 dataset example로 추가한다.
"""
import logging
from typing import Optional

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

# rating 텍스트 → LangSmith score 매핑
# "good" 포함 → 1.0, 나머지(bad 계열) → 0.0
_GOOD_SCORE = 1.0
_BAD_SCORE = 0.0


def _to_score(rating: str) -> float:
    """rating 텍스트를 LangSmith score로 변환한다.

    판정은 ``app.utils.rating.classify_rating`` 단일 출처에 위임한다. 예전엔
    ``"good" in rating.lower()`` 로만 봤는데, 프론트 '좋아요' 버튼이 보내는 코드값이
    ``"thumbs_up"`` 이라 **긍정 평가가 전부 0.0(부정)으로 올라가고 있었다** —
    DB/API 분류(positive)와 정반대다.

    Args:
        rating: 사용자 평가 텍스트. preset 코드값 또는 자유 입력 사유.

    Returns:
        1.0 (긍정) 또는 0.0 (부정·미평가).
    """
    from agent_arch.utils.rating import classify_rating

    polarity, _ = classify_rating(rating)
    return _GOOD_SCORE if polarity == "positive" else _BAD_SCORE


def _get_client():
    """LangSmith Client를 반환한다. 미설정 시 None 반환.

    Returns:
        langsmith.Client 인스턴스 또는 None
    """
    if not settings.enable_langsmith or not settings.langsmith_api_key:
        return None
    try:
        from langsmith import Client
        return Client(api_key=settings.langsmith_api_key)
    except Exception:
        logger.warning("LangSmith Client 초기화 실패", exc_info=True)
        return None


def _ensure_dataset(client, dataset_name: str):
    """dataset이 없으면 새로 생성하고 반환한다.

    Args:
        client: langsmith.Client 인스턴스
        dataset_name: dataset 이름

    Returns:
        Dataset 객체
    """
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        return client.create_dataset(dataset_name=dataset_name)


async def send_feedback(
    langsmith_run_id: str,
    rating: str,
    query: str,
    answer: str,
    action_view: Optional[dict] = None,
) -> None:
    """LangSmith에 피드백을 전송하고 dataset에 example을 추가한다.

    enable_langsmith가 False이거나 run_id가 없으면 아무 동작도 하지 않는다.
    실패 시 예외를 전파하지 않고 로그만 남긴다.

    Args:
        langsmith_run_id: @traceable이 생성한 run UUID 문자열
        rating: 사용자 평가 텍스트 ("good", "bad" 등)
        query: 사용자 입력 텍스트
        answer: 어시스턴트 응답 텍스트
        action_view: 이 턴에 뜬 HITL 승인 카드(제목/폼/표시 필드). 있으면 example
            metadata에 실어, common_app '회사 리뷰' 화면이 답변 카드를 복원하게 한다.
            카드는 run/answer 텍스트엔 없고 DB(agent.messages.action_view)에만 있으므로,
            여기서 metadata로 넘겨야 리뷰 화면에서 볼 수 있다.
    """
    client = _get_client()
    if client is None:
        return

    score = _to_score(rating)

    try:
        # 피드백 전송
        client.create_feedback(
            run_id=langsmith_run_id,
            key="user_rating",
            score=score,
            value=rating,
        )
        logger.debug(
            "LangSmith 피드백 전송 완료 (run_id=%s, rating=%s, score=%s)",
            langsmith_run_id, rating, score,
        )
    except Exception:
        logger.warning(
            "LangSmith 피드백 전송 실패 (run_id=%s)", langsmith_run_id, exc_info=True
        )

    # retrieval 메타데이터 수집 (실패/부재 시 None으로 진행)
    retrieval_meta: Optional[dict] = None
    try:
        run = client.read_run(langsmith_run_id)
        run_metadata = getattr(run, "metadata", None)
        if isinstance(run_metadata, dict):
            candidate = run_metadata.get("retrieval")
            # MagicMock/잘못된 타입 방어: dict여야만 수용
            if isinstance(candidate, dict):
                retrieval_meta = candidate
    except Exception:
        logger.debug(
            "LangSmith run 조회 실패 — retrieval metadata 없이 진행 (run_id=%s)",
            langsmith_run_id, exc_info=True,
        )

    try:
        # dataset example 추가
        dataset = _ensure_dataset(client, settings.langsmith_dataset_name)
        example_metadata: dict = {
            "rating": rating,
            "score": score,
            "run_id": langsmith_run_id,
        }
        if retrieval_meta:
            example_metadata["retrieval"] = retrieval_meta
        if isinstance(action_view, dict) and action_view:
            example_metadata["action_view"] = action_view
        client.create_example(
            inputs={"input": query},
            outputs={"output": answer},
            dataset_id=dataset.id,
            metadata=example_metadata,
        )
        logger.debug(
            "LangSmith dataset example 추가 완료 (dataset=%s, rating=%s, retrieval=%s, card=%s)",
            settings.langsmith_dataset_name,
            rating,
            bool(retrieval_meta),
            bool(isinstance(action_view, dict) and action_view),
        )
    except Exception:
        logger.warning(
            "LangSmith dataset example 추가 실패 (run_id=%s)", langsmith_run_id, exc_info=True
        )
