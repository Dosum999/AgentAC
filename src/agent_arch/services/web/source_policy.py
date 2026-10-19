"""웹 검색 출처 정책 — Brave Goggles 정의를 설정 파일에서 읽어 모든 brave_* 호출에 강제값으로 싣는다.

왜 재순위인가(2026-09-23 벤치): 한국어 '시장 동향' 질의 상위 결과가 거의 전부 시장규모 리포트 판매 사이트였고, 모델은 받은 것
중에서 고를 수밖에 없다. 프롬프트로 "리포트팜을 쓰지 마라"고 해도 고를 대안이 결과에 없으면 소용없다 — 결과 자체를 바꾼다.
판단(무엇이 좋은 출처인가)은 사람이 파일로 정하고, 코드는 그대로 전달만 한다.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[4] / "config" / "web_source_goggle.txt"


def parse_goggle(text: str) -> Optional[str]:
    """설명 줄('!')·빈 줄을 뺀 규칙만 줄바꿈으로 잇는다. 규칙이 없으면 None."""
    rules = [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("!")]
    return "\n".join(rules) or None


@lru_cache(maxsize=1)
def load_goggle() -> Optional[str]:
    """설정 경로(WEB_SOURCE_GOGGLE_PATH, 빈 값이면 기본 파일)의 Goggle. 파일이 없거나 비면 None(재순위 없음)."""
    raw = str(getattr(settings, "web_source_goggle_path", "") or "").strip()
    path = Path(raw) if raw else _DEFAULT_PATH
    try:
        goggle = parse_goggle(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("[web] 출처 정책 파일 없음 — 재순위 없이 검색 (%s)", path)
        return None
    if goggle:
        logger.info("[web] 출처 정책 %d개 규칙 적용 (%s)", goggle.count("\n") + 1, path)
    return goggle
