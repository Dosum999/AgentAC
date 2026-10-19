"""장기 사용자 메모리 v2 도메인 모델.

4층(L0 원문 → L1 atom → L2 장면 → L3 페르소나) 중 L1 atom 의 쓰기 입력(AtomDraft),
중복 판정 결과(DedupDecision), 잡 결과 타입을 한 곳에 모은다. 읽기는 repository 가
dict 를 돌려준다(하우스 컨벤션).

v1(app/models/memory.py)과 달리 confidence/support 가 없다. 빈도 신호는 판정(merge)이
priority 를 올리는 것으로 대체한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from agent_arch.services.memory_v2.place import ORIGIN_USER

# atom_type 4종 — DB CHECK(memory_atoms_type_chk)와 정확히 일치해야 한다.
ATOM_TYPES: tuple[str, ...] = ("persona", "episodic", "instruction", "alias")
AtomType = Literal["persona", "episodic", "instruction", "alias"]

# v1 7종·모델이 흔히 내는 변형 → v2 4종 매핑(추출 결과 정규화).
_LEGACY_TYPE_MAP = {
    "preference": "persona", "fact": "persona", "profile": "persona", "trait": "persona",
    "directive": "instruction", "instruct": "instruction", "rule": "instruction",
    "tool_arg_convention": "instruction", "workflow_macro": "instruction",
    "task_event": "episodic", "event": "episodic", "episode": "episodic", "task": "episodic",
    "entity_alias": "alias", "aliases": "alias",
}

PRIORITY_STRICT = -1        # 전역 강제 지시: 질의와 무관하게 항상 주입
PRIORITY_MIN, PRIORITY_MAX = -1, 100
DEFAULT_PRIORITY = 50
MAX_CONTENT_CHARS = 1000
MAX_ATOMS_PER_BATCH = 20


def normalize_atom_type(value: Any) -> Optional[str]:
    """LLM 이 낸 타입 문자열을 v2 4종으로 정규화한다. 모르면 None."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ATOM_TYPES:
        return v
    return _LEGACY_TYPE_MAP.get(v)


def clamp_priority(value: Any, default: int = DEFAULT_PRIORITY) -> int:
    """priority 를 [-1, 100] 정수로 강제한다. 숫자가 아니면 default.

    -1 은 특별값(전역 강제)이라 그대로 두고, 그 외 음수는 0 으로 올린다.
    """
    try:
        p = int(float(value))
    except (TypeError, ValueError):
        return default
    if p == PRIORITY_STRICT:
        return p
    return max(0, min(PRIORITY_MAX, p))


class AtomDraft(BaseModel):
    """추출기가 만드는 L1 atom 후보(영속 이전)."""

    atom_type: AtomType
    content: str = Field(..., min_length=1)
    priority: int = DEFAULT_PRIORITY
    source_message_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    scene_name: Optional[str] = None
    company_scope: Optional[str] = None
    # 아래 둘은 **코드가 채운다**(services/memory_v2/place). 추출 모델은 내지 않는다 —
    # 근거 턴의 metadata 만 보면 결정되고, 모델이 자기 출처를 주장하지 못하게 하려는 것이다.
    places: list[str] = Field(default_factory=list)
    origin: str = ORIGIN_USER

    @field_validator("content", mode="before")
    @classmethod
    def _strip_content(cls, v: Any) -> str:
        s = str(v or "").strip()
        return s[:MAX_CONTENT_CHARS]

    @field_validator("priority", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> int:
        return clamp_priority(v)

    @field_validator("payload", mode="before")
    @classmethod
    def _payload_dict(cls, v: Any) -> dict:
        return v if isinstance(v, dict) else {}


DedupAction = Literal["store", "update", "merge", "skip"]
DEDUP_ACTIONS: tuple[str, ...] = ("store", "update", "merge", "skip")


class DedupDecision(BaseModel):
    """신규 draft 하나에 대한 중복·모순 판정.

    index 는 drafts 리스트 인덱스. update/merge 는 target_ids(기존 atom id)를 대체·병합하고
    content/atom_type/priority 가 있으면 최종본을 덮는다. 실패·누락은 전부 store(fail-open).
    """

    index: int
    action: DedupAction = "store"
    target_ids: list[str] = Field(default_factory=list)
    content: Optional[str] = None
    atom_type: Optional[AtomType] = None
    priority: Optional[int] = None

    @field_validator("priority", mode="before")
    @classmethod
    def _clamp_opt(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        return clamp_priority(v)


@dataclass
class ExtractionResult:
    """L1 추출 1회의 결과(LLM 원문·모델·지연 포함 — 생성 로그용)."""

    scene_name: Optional[str]
    drafts: list[AtomDraft]
    raw: str
    model: str
    prompt_sha: str
    latency_ms: int
    input_message_ids: list[str]
    error: Optional[str] = None  # LLM 호출 실패 / parse_failed. raw 가 비어 있으면 호출 실패.


@dataclass
class JudgeResult:
    """배치 중복 판정 1회의 결과."""

    decisions: list[DedupDecision]
    pool_size: int = 0
    raw: str = ""
    model: str = ""
    latency_ms: int = 0
    error: Optional[str] = None
    llm_called: bool = False


@dataclass
class WriteStats:
    stored: int = 0
    updated: int = 0
    merged: int = 0
    skipped: int = 0
    id_grounded: int = 0          # skip 판정이었지만 alias 에 정식 id 를 새로 채워 update 로 올린 수
    output_atom_ids: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "store": self.stored, "update": self.updated,
            "merge": self.merged, "skip": self.skipped, "id_grounded": self.id_grounded,
        }

    @property
    def written(self) -> int:
        return self.stored + self.updated + self.merged


@dataclass
class L1JobResult:
    """run_l1_job 의 결과. status: done | empty | no_new | no_checkpoint | skipped_locked | failed."""

    user_sub: str
    status: str
    processed: int = 0
    stats: Optional[WriteStats] = None
    has_more: bool = False
    scene_name: Optional[str] = None
    error: Optional[str] = None
