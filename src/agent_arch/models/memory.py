"""장기 사용자 메모리 record 도메인 모델 (NapMem Phase 1).

agent.memory_records(docker/postgres/migration_v24.sql) 층의 모델.
저장소=app/db/repository/memory_records.py, 추출/reconcile=app/services/memory/.

record_type(7종)·source_kind·confidence 정책을 한 곳에 모은다. 읽기는 repository가
dict를 반환하므로(하우스 컨벤션) 별도 read-model은 두지 않고, 쓰기/추출 입력만
MemoryRecordDraft로 검증한다.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# 7종 record_type — DB CHECK(memory_records_type_chk)와 정확히 일치해야 한다.
RecordType = Literal[
    "entity_alias",         # T1: 별칭→canonical 대상(채널/인물/회사스코프)
    "directive",            # 표준 지시(출력형식 등): "표 쓰지마"
    "tool_arg_convention",  # T3: 도구 인자 관습(메일 제목 자동 등)
    "workflow_macro",       # T2: 반복 다단계 태스크
    "task_event",           # T4: 세션 넘는 작업 이력/연속성
    "preference",           # T5: 약한 선호(과일반화 위험 → 저 confidence)
    "fact",                 # 일반 context-fact(SoR 아닌 대화발 사실)
]
SourceKind = Literal["explicit", "inferred"]

RECORD_TYPES: tuple[str, ...] = (
    "entity_alias", "directive", "tool_arg_convention",
    "workflow_macro", "task_event", "preference", "fact",
)

# confidence 정책 상수 (튜닝 대상)
_EXPLICIT_CONFIDENCE = 0.9
_INFERRED_BASE = 0.3
_INFERRED_STEP = 0.1
_INFERRED_CAP = 0.85


def confidence_for(source_kind: str, support_count: int) -> float:
    """source_kind·관측 빈도로 confidence[0,1]를 산정한다.

    Phase 0 실증(명시 지시 ≈0, 신호는 반복)에 맞춰 inferred는 support_count에
    비례해 상승하되 상한을 둔다. explicit(사용자 명시)은 높은 고정값에서 시작.

    Args:
        source_kind: "explicit" | "inferred".
        support_count: 이 기억을 강화한 관측 수(>=1).

    Returns:
        [0,1] confidence. 소수 3자리(DB NUMERIC(4,3) 정합).
    """
    if source_kind == "explicit":
        return _EXPLICIT_CONFIDENCE
    return round(min(_INFERRED_CAP, _INFERRED_BASE + _INFERRED_STEP * support_count), 3)


class MemoryRecordDraft(BaseModel):
    """추출 파이프라인이 만드는 메모리 후보(영속 이전).

    LLM 추출 결과 또는 빈도 집계 결과를 표준화한 입력. reconcile 단계를 거쳐
    agent.memory_records로 영속되거나(store), 기존 record를 강화(reinforce)한다.

    sor_derivable=True인 후보는 파이프라인이 저장을 거부한다(HR/directory/profile에서
    조회 가능한 정보는 라이브 SoR이 SoT이므로 캐시 금지 — Phase 0로 확증한 저장 금지선).

    Attributes:
        record_type: 7종 중 하나.
        content: 자연어 진술(Qdrant 임베딩 대상 + 에이전트가 읽는 본문).
        payload: 타입별 구조화 필드. entity_alias는
            {canonical_kind, canonical_id, canonical_name, aliases:[...]}.
        source_kind: explicit(명시) | inferred(반복행동 추론, 기본).
        confidence: [0,1]. 미지정 시 confidence_for로 산정.
        support_count: 강화 관측 수(>=1).
        company_scope: None=유저 전역, 값=해당 회사에서만.
        domain_scope: None=전역, 값=툴 도메인(mattermost/mail/calendar/...).
        source_message_ids: 근거 agent.messages id(provenance).
        source_conversation_ids: 근거 대화방 id.
        sor_derivable: True면 저장 거부(SoR 파생).
    """

    record_type: RecordType
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    source_kind: SourceKind = "inferred"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    support_count: int = Field(default=1, ge=1)
    company_scope: Optional[str] = None
    domain_scope: Optional[str] = None
    source_message_ids: list[str] = Field(default_factory=list)
    source_conversation_ids: list[str] = Field(default_factory=list)
    sor_derivable: bool = False

    def resolved_confidence(self) -> float:
        """명시 confidence가 없으면 정책값으로 채워 반환한다."""
        if self.confidence is not None:
            return self.confidence
        return confidence_for(self.source_kind, self.support_count)

    def canonical_key(self) -> Optional[tuple[str, str]]:
        """entity_alias의 (canonical_kind, canonical_id) 키. 없으면 None.

        reconcile 단계에서 동일 대상 중복 판정에 쓴다.
        """
        if self.record_type != "entity_alias":
            return None
        kind = self.payload.get("canonical_kind")
        cid = self.payload.get("canonical_id")
        if kind and cid:
            return (str(kind), str(cid))
        return None
