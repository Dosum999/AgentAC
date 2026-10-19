from typing import Optional, Dict, Any
from langsmith import get_current_run_tree


def record_llm_run(
    model: str,
    usage: Optional[Dict[str, int]] = None,
    first_token_secs: Optional[float] = None,
) -> None:
    """LangSmith 현재 run에 LLM 실행 정보를 기록합니다.

    토큰은 **두 곳**에 기록한다:

    1. 표준 ``usage_metadata`` (``run.set``) — LangSmith 네이티브 토큰 롤업 + 비용
       자동계산용. 스트리밍(@traceable async generator)은 반환값이 없어 데코레이터가
       usage를 자동 추출하지 못하므로, 여기서 명시적으로 세팅해야 root 트레이스의
       ``total_tokens`` 와 "Tokens" 컬럼에 반영된다. 스키마 키가 input/output 네이밍이라
       prompt→input, completion→output 으로 매핑한다.
    2. 커스텀 ``metadata.usage`` (``run.add_metadata``) — 기존 호환 유지.

    비용 자동계산을 위해 ``ls_model_name`` 메타데이터로 모델명을 노출한다(LangSmith
    Model Pricing 매칭 키). 모델 태그도 계속 단다(기존 동작 유지).

    Args:
        model: 모델 이름 (LangSmith 태그 + ls_model_name 으로 기록)
        usage: 토큰 사용량 딕셔너리. 키: prompt_tokens, completion_tokens, total_tokens
        first_token_secs: 첫 토큰까지 걸린 시간 (초 단위, 내부에서 ms로 변환)
    """
    run = get_current_run_tree()
    if not run:
        return
    run.add_tags([model])

    # 커스텀 메타데이터 + 비용 매칭용 모델명
    metadata: Dict[str, Any] = {"ls_model_name": model}
    if usage is not None:
        metadata["usage"] = usage
    if first_token_secs is not None:
        metadata["first_token_ms"] = round(first_token_secs * 1000, 1)
    run.add_metadata(metadata)

    # 표준 usage_metadata 세팅 → 네이티브 롤업/비용 (input/output 네이밍으로 매핑)
    if usage is not None:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        std: Dict[str, Any] = {}
        if prompt is not None:
            std["input_tokens"] = prompt
        if completion is not None:
            std["output_tokens"] = completion
        if total is not None:
            std["total_tokens"] = total
        if std:
            run.set(usage_metadata=std)
