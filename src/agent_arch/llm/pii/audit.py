"""shadow·enforce 기록 — 값은 남기지 않는다.

shadow 는 "치환했다면 무엇이 바뀌었을지" 를 로그로 본다. 호출당 1건, 종류·출처·형태별
건수와 샘플(최대 20)을 남긴다. 샘플의 문맥은 치환된 텍스트 기준(앞뒤 12자 + 토큰)이라
원문 값이 없다 — "…담당자는 [[PERSON_3f9a1c2d]]님께…" 로 오탐을 판별한다.
개발 환경에서 ``PII_GUARD_SHADOW_REVEAL=1`` 일 때만 값을 함께 남긴다(운영은 강제 off).

``redact_for_log`` 는 복원된 도구 인자를 로그에 찍을 때 값 대신 모양(타입·길이)만 남긴다.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from typing import Any, Optional

from agent_arch.core.config import settings
from agent_arch.llm.pii.engine import Substitution

logger = logging.getLogger("agent_arch.llm.pii.audit")

_SAMPLE_MAX = 20
#: 한글 이름 1개(≈3토큰)가 토큰(≈10토큰)으로 바뀌는 대략치.
_EXTRA_TOKENS_PER_PLACEHOLDER = 7


def reveal_allowed() -> bool:
    if getattr(settings, "is_production", False) or str(getattr(settings, "environment", "")).lower() == "production":
        return False
    return bool(getattr(settings, "pii_guard_shadow_reveal", False))


def summarize(sub: Substitution) -> dict[str, Any]:
    by_kind, by_source, by_form = Counter(), Counter(), Counter()
    for f in sub.findings:
        if not f.applied and f.kind not in sub.blocked:
            by_kind[f"{f.kind}(observe)"] += 1
        else:
            by_kind[f.kind] += 1
        by_source[f.source] += 1
        by_form[f.form] += 1
    return {
        "tokens": sub.token_count,
        "blocked": dict(sub.blocked),
        "by_kind": dict(by_kind),
        "by_source": dict(by_source),
        "by_form": dict(by_form),
        "est_extra_tokens": sub.token_count * _EXTRA_TOKENS_PER_PLACEHOLDER,
        "map_size": len(sub.restore_map),
    }


def _samples(sub: Substitution) -> list[dict]:
    out = []
    for f in sub.findings[:_SAMPLE_MAX]:
        d = asdict(f)
        if reveal_allowed():
            d["value"] = sub.restore_map.get(f.token.strip("[]").split("@")[0], None)
        out.append(d)
    return out


def log_shadow(*, provider: str, model: Optional[str], scope: str, sub: Substitution, scan_ms: float,
               lexicon_size: int, echo: Optional[dict] = None, images: int = 0) -> None:
    if not sub.findings:
        logger.debug("[pii_guard] shadow provider=%s model=%s scope=%s 치환 대상 없음 scan_ms=%.1f", provider, model, scope, scan_ms)
        return
    logger.info(
        "[pii_guard] shadow provider=%s model=%s scope=%s summary=%s echo=%s images=%d scan_ms=%.1f lexicon=%d samples=%s",
        provider, model, scope, summarize(sub), echo or {}, images, scan_ms, lexicon_size, _samples(sub),
    )


def log_enforce(*, provider: str, model: Optional[str], scope: str, sub: Substitution, scan_ms: float,
                lexicon_size: int, unresolved: Optional[list[str]] = None, images: int = 0) -> None:
    if not sub.changed and not unresolved:
        return
    level = logging.WARNING if unresolved else logging.INFO
    logger.log(
        level,
        "[pii_guard] enforce provider=%s model=%s scope=%s summary=%s unresolved=%s images=%d scan_ms=%.1f lexicon=%d",
        provider, model, scope, summarize(sub), unresolved or [], images, scan_ms, lexicon_size,
    )


def echo_stats(sub_response: Substitution) -> dict[str, int]:
    """응답(원문)을 스캔한 결과에서 모델이 되돌려 말한 식별자 수 — 복원 경로의 부하 예측."""
    c = Counter(f.kind for f in sub_response.findings)
    return dict(c)


def redact_for_log(obj: Any, *, max_items: int = 20) -> Any:
    """값 대신 모양만. {"to": "<str:14>", "n": 1, "list": "<list:3>"}"""
    if isinstance(obj, str):
        return f"<str:{len(obj)}>"
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        items = list(obj.items())[:max_items]
        out = {str(k): redact_for_log(v, max_items=max_items) for k, v in items}
        if len(obj) > max_items:
            out["…"] = f"+{len(obj) - max_items}"
        return out
    if isinstance(obj, (list, tuple)):
        return f"<{type(obj).__name__}:{len(obj)}>"
    return f"<{type(obj).__name__}>"
