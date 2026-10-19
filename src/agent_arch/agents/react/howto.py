"""방법·설정·증상 질문 판별 — 가이드 검색 강제(guide-first) 트리거.

배경(2026-09-02 실측, 이슈): 같은 대화의 후속 질문에서 LLM 이 도구를 한 번도
부르지 않고(iterations=0) 앞 턴 맥락만으로 "근무조정으로 하루씩 수정하라"고 답해,
가이드에 있던 근태 설정(조기 출근 연장 제한)을 안내하지 못했다. 프롬프트의 3갈래
지침(GuideKBTool.prompt_guidance)만으로는 후속 턴에서 검색 생략을 막지 못했으므로,
질문 유형이 "방법·절차·정책·기능 유무·증상" 이면 첫 iteration 의 도구 호출을
가이드 검색으로 고정한다.

판별은 규칙 기반이다(LLM 호출 없음 → 지연 0). 과탐(실행 요청을 가이드 검색으로 보냄)은
검색 한 번 더 하는 비용에 그치지만, 미탐은 오답으로 이어지므로 애매하면 강제한다.
"""
import json
import re
from typing import Iterable, Optional

GUIDE_TOOL_NAME = "search_user_guide_kb"

# 방법·절차·정책·기능 유무·증상 신호. 종결어미 변형(띄어쓰기·오타) 을 넉넉히 허용한다.
_HOWTO = re.compile(
    r"(어떻게|어떡|방법|절차|규정|정책|설정|어디서|어디에|어느 메뉴"
    r"|없나요|없을까|없어요|없는가|없습니까|없나\b|없지"
    r"|있나요|있을까|있어요\?|있는가|있습니까|있나\b"
    r"|가능한가|가능해|가능합니까|가능할까|되나요|되는가|되나\?|됩니까|될까요"
    r"|해야\s*하나|해야\s*하는|하는가요|하는건가|인가요|인가\?|건가요|입니까|일까요|까요\?|나요\??|는가\?"
    r"|왜\s|왜요|이유|안\s?돼|안\s?되|않아요|않나요|안\s?나와|안\s?보여|안\s?떠|안\s?잡|잡히|잡혀|잡힌"
    r"|계산되|처리되|반영되|넘어가|찍히|뭐예요|뭐에요|뭔가요|무엇인가요|뭐야\?|뭐죠)"
)
# 실행 요청 신호 — 단독이면 강제하지 않는다(수행 도구가 우선).
_ACTION = re.compile(
    r"(해\s?줘|해\s?주세요|해\s?주실|해\s?줄래|보내\s?줘|보내\s?주세요|신청해|등록해|삭제해|취소해|승인해"
    r"|올려\s?줘|만들어\s?줘|찾아\s?줘|조회해\s?줘|보여\s?줘|정리해\s?줘|요약해\s?줘|상신해|발송해)"
)
# 실행 요청 문장 안에서도 "방법" 을 물은 것이면 가이드 우선(prompt_guidance 3번 규칙).
_STRONG = re.compile(r"(어떻게|어떡|방법|절차|어디서|어디에|규정|정책)")
_MIN_LEN = 4


def _looks_like_json(t: str) -> bool:
    if t[0] not in "{[":
        return False
    try:
        json.loads(t)
        return True
    except ValueError:
        return False


def is_howto_question(text: str) -> bool:
    """사용자 발화가 가이드 근거가 필요한 방법·절차·정책·기능 유무·증상 질문인지 판정한다.

    Args:
        text: 사용자 입력(augmented 입력이어도 무방 — 부분 일치).

    Returns:
        가이드 검색을 강제해야 하면 True.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < _MIN_LEN:
        return False
    if _looks_like_json(t):
        # HITL 재개 페이로드({"goto":..,"resume":..}) 등 구조화 입력 — 질문이 아니다.
        return False
    if not _HOWTO.search(t):
        return False
    if _ACTION.search(t) and not _STRONG.search(t):
        return False
    return True


def pick_forced_tool(text: str, known_tool_names: Iterable[str]) -> Optional[str]:
    """첫 iteration 에 강제할 도구 이름을 고른다. 없으면 None.

    가이드 도구가 바인딩돼 있지 않은 경로(채널 스코프 모드 등)에서는 강제하지 않는다.
    """
    if GUIDE_TOOL_NAME not in set(known_tool_names or ()):
        return None
    return GUIDE_TOOL_NAME if is_howto_question(text) else None
