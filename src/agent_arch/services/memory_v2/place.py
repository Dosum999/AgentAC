"""기억의 장소와 출처 등급 — 어디서 비롯됐고 누구 말이 섞였는가.

**둘 다 코드가 정한다.** 추출 모델은 이 값을 보지도 쓰지도 않는다. 근거가 된 L0 턴의
metadata 만 보면 결정되므로 판단이 필요 없고, 모델이 산문으로 자기 출처를 주장할 수도 없다.

- **장소**는 채널 단위다. 스레드는 너무 잘아 기억의 경계로 쓰지 않는다. **DM 은 장소를 만들지
  않는다** — 본인과 에이전트 둘만 있는 자리는 특정 장소에 매인 맥락이 아니라 그 사람 자체의
  맥락이고, 그것을 채널로 묶으면 다른 자리에서 꺼내 쓸 수 없게 된다.
- **출처 등급**은 둘뿐이다. 근거 턴에 도구로 읽어 온 대화 발췌가 하나도 없으면 ``user``,
  하나라도 있으면 ``mixed``. ``mixed`` 는 저장·검색·회상 전부 되지만 **항상 주입되는 등급
  (priority -1)에는 오를 수 없다** — 남이 채널에서 한 말이 에이전트의 상시 규칙이 되면 안 된다.
  점수 감점이 아니라 선행 조건이다.

장소가 없는 기억(패널 턴, 이 기능 이전의 기억)은 전부 중립이다 — 순위도 자격도 예전 그대로다.
"""
from __future__ import annotations

import json
from typing import Any, Optional

PAYLOAD_PLACES = "places"
PAYLOAD_ORIGIN = "origin"

ORIGIN_USER = "user"
ORIGIN_MIXED = "mixed"

# 순위 배수. 같은 장소는 올리고 다른 장소는 살짝 내린다. 장소 없는 기억은 건드리지 않는다.
FACTOR_SAME = 1.15
FACTOR_OTHER = 0.9
FACTOR_NEUTRAL = 1.0

MAX_PLACES_PER_ATOM = 4

# 앱 전용 창(메신저 채널이 아닌 자리)의 provider. 장소 키는 provider:channel 한 줄이라,
# 창 이름을 채널 자리에 넣으면 나머지 로직(순위·자격·저장)을 그대로 쓴다.
SURFACE_PROVIDER = "agent"


def key_of(place: Any) -> Optional[str]:
    """장소 dict → 장소 키. DM 과 채널 없는 값은 None(장소 없음).

    저장된 모양(``metadata.place``: is_dm)과 런타임 모양(메신저 ``user_info["origin"]``:
    is_direct) 둘 다 받는다 — 같은 사실을 두 곳에서 다르게 부르는 것뿐이라 여기서 흡수한다.
    """
    if not isinstance(place, dict) or place.get("is_dm") or place.get("is_direct"):
        return None
    channel = str(place.get("channel_id") or "").strip()
    if not channel:
        return None
    provider = str(place.get("provider") or "mattermost").strip() or "mattermost"
    return f"{provider}:{channel}"


def surface_place(surface: Any) -> Optional[dict]:
    """앱 전용 창 이름 → 저장용 장소 dict. 이름이 없으면 None(장소 없음).

    전용 앱처럼 일반 채팅과 따로 분리된 창은 채널이 아니지만 자리다 — 그 창에서 못 박은
    규칙이 브라우저 패널까지 따라다니면 기억이 섞인다. 창 이름 판정은 요청 경로가
    소유한다(``source_routing.resolve_memory_surface``).

    Args:
        surface: ``user_info["memory_surface"]`` 값(예 ``"app"``).

    Returns:
        ``{"provider", "channel_id", "is_dm"}`` 또는 None.
    """
    name = str(surface or "").strip()
    if not name:
        return None
    return {"provider": SURFACE_PROVIDER, "channel_id": name, "is_dm": False}


def current_key(user_info: Optional[dict]) -> Optional[str]:
    """이번 턴의 장소 키. 앱 전용 창과 메신저 채널 턴만 값이 있고 패널·DM 은 None.

    앱 창을 먼저 본다 — 두 값이 같이 실릴 일은 없지만(경로가 갈린다), 순서를 고정해 두면
    나중에 앱 안에 메신저를 끼워 넣어도 판정이 흔들리지 않는다.
    """
    info = user_info or {}
    return key_of(surface_place(info.get("memory_surface")) or info.get("origin"))


def _metadata(row: Any) -> dict:
    meta = (row or {}).get("metadata") if isinstance(row, dict) else None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            meta = None
    return meta if isinstance(meta, dict) else {}


def _has_excerpt(meta: dict) -> bool:
    """이 턴이 도구로 남의 대화를 읽어 왔는가(digest.CONVERSATION_TOOLS 결과의 messages)."""
    digest = meta.get("tool_digest")
    if not isinstance(digest, list):
        return False
    return any(isinstance(e, dict) and e.get("messages") for e in digest)


def places_of(turns: list[dict]) -> list[str]:
    """근거 L0 턴들 → 장소 키 목록. 없으면 빈 목록(장소 없음).

    출처 등급은 여기서 정하지 않는다 — 턴이 아니라 **인용된 근거** 단위로 갈라야 하고, 그건
    태그를 보는 extractor.build_drafts 가 한다(place 모듈은 장소만 안다).
    """
    keys: list[str] = []
    for row in turns or []:
        key = key_of(_metadata(row).get("place"))
        if key and key not in keys:
            keys.append(key)
    return keys[:MAX_PLACES_PER_ATOM]


def atom_places(atom: Any) -> list[str]:
    payload = (atom or {}).get("payload") if isinstance(atom, dict) else None
    values = payload.get(PAYLOAD_PLACES) if isinstance(payload, dict) else None
    return [str(v) for v in values if v] if isinstance(values, list) else []


def atom_origin(atom: Any) -> str:
    payload = (atom or {}).get("payload") if isinstance(atom, dict) else None
    value = payload.get(PAYLOAD_ORIGIN) if isinstance(payload, dict) else None
    return ORIGIN_MIXED if value == ORIGIN_MIXED else ORIGIN_USER


def rank_factor(atom: Any, current: Optional[str]) -> float:
    """회상 순위 배수.

    장소가 없는 기억은 언제나 중립이다. **장소가 없는 자리(패널·DM)에서도 모두 중립이다** —
    거기는 "다른 채널"이 아니라 "장소 밖"이라, 채널에서 배운 것을 깎을 이유가 없다. 깎으면
    잘 돌던 패널 회상이 이 기능 때문에 나빠진다. 내리는 것은 **다른 채널에 있을 때뿐**이다.
    """
    places = atom_places(atom)
    if not places or not current:
        return FACTOR_NEUTRAL
    return FACTOR_SAME if current in places else FACTOR_OTHER


def eligible(atom: Any, current: Optional[str]) -> bool:
    """상시 블록(매 턴 강제 주입)에 들어갈 자격 — 그 장소에 있을 때만.

    순위와 자격은 성격이 다르다. 순위는 재배열이라 장소 밖에서 깎으면 손해만 나지만(rank_factor
    참조), 상시 주입은 **매 턴 강제로 읽히는 규칙**이다. 특정 채널에서 못 박은 규칙이 패널과
    다른 모든 자리에 따라다니면 그건 기억이 섞인 것이다. 그래서 자격은 장소 밖에서도 닫는다.

    잃는 것은 아니다 — 이 기억은 여전히 검색으로 회상되어 턴 블록에 올라온다. 막는 것은
    "묻지 않아도 매번 따르라"는 자리 하나뿐이다.
    """
    places = atom_places(atom)
    return not places or bool(current and current in places)
