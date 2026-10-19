"""web_fetch 공용 조각 — 입력 오류 형과 타임아웃 정규화.

2026-09-15 브레이브 MCP 전환 때 자체 프로바이더와 함께 이 모듈을 걷어냈다가, 2026-09-23 web_fetch 를 되살리며 fetch 가 쓰는 두
가지만 남겼다(검색 파라미터 읽기·사이트명 등 프로바이더 전용 조각은 MCP 서버가 대신한다).
"""
from __future__ import annotations

from typing import Any

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


class ToolInputError(ValueError):
    """모델이 보낸 인자가 스키마 밖일 때. 외부 실패와 구분해 모델이 고쳐 다시 부를 수 있게 한다."""


def resolve_timeout_seconds(value: Any, fallback: int = DEFAULT_TIMEOUT_SECONDS) -> int:
    parsed = value if isinstance(value, (int, float)) and value == value else fallback
    return max(1, min(MAX_TIMEOUT_SECONDS, int(parsed)))
