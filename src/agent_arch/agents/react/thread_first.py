"""메신저 "이 스레드" 질문을 **그 스레드 조회로 고정**한다 (thread-first).

2026-09-14 운영: 테넌트 A>개발팀 스레드에서 "이 스레드에 여러 PDF 가 있는데 각각 정리해
주세요" 라고 물었더니 에이전트가 **회사 scrum 채널**의 PDF 를 읽어 와 "이 스레드(회사 scrum
채널)에 올라온 PDF 는 2개입니다" 라고 답했다. 권한 문제가 아니다 — ACL 필터는
fail-closed 이고 요청자가 실제로 볼 수 있는 채널이었다. **맥락을 잘못 고른 것**이다.

`[현재 위치]` 블록으로 채널·스레드를 알려주고 `mattermost_read_thread` 도구도 만들었지만,
그것만으로는 모델이 그 도구를 **고른다는 보장이 없다**: "PDF 정리해줘" 는 사내 지식검색
(`mattermost_search_knowledge`, core 상시 바인딩)과도 의미가 잘 맞고, supervisor 가 워커
서브쿼리를 다시 쓰면서 "이 스레드" 라는 지시어가 빠지기도 한다(실측 로그의 서브쿼리는
`현재 맥락의 파일을 워드 파일로 생성하여 전송` 이었다).

그래서 guide-first 와 **같은 방식**으로 첫 호출을 고정한다. 판정은 결정적(정규식)이고,
도구가 안 붙은 경로·스레드가 아닌 턴에서는 아무것도 하지 않는다.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# 스레드를 가리키는 지시어. **"이 채널"은 넣지 않는다** — 채널 질문의 정답은 스레드 조회가
# 아니라 채널 조회이고, 그쪽은 `[현재 위치]` 블록이 준 channel_id 로 이미 정확해진다.
_THREAD_DEIXIS_RE = re.compile(
    r"(이|본|해당|현재)\s*(스레드|쓰레드|글타래|대화\s*스레드)"
    r"|여기\s*(에|에서)?\s*(올라온|공유된|있는|첨부[된한]?)"
    r"|위\s*(에|에서)\s*(올라온|공유한|공유된|말한|언급한)"
    r"|this\s+thread",
    re.IGNORECASE,
)

THREAD_TOOL_NAME = "mattermost_read_thread"

# 채널을 가리키는 지시어. 스레드와 **따로** 둔다 — 채널 질문의 정답은 스레드 조회가 아니라
# 채널 조회이고, 검색을 묶는 범위도 채널이다.
_CHANNEL_DEIXIS_RE = re.compile(
    r"(이|본|해당|현재)\s*(채널|방|대화방|톡방)"
    r"|여기\s*채널"
    r"|this\s+channel",
    re.IGNORECASE,
)

# "넓혀 달라"는 신호. 지시어보다 **우선한다** — "이 스레드에서 말한 그 이슈, 다른 데서도
# 논의됐나?" 는 지시어가 있어도 전사 검색이 정답이다.
_BROADER_SCOPE_RE = re.compile(
    r"다른\s*(채널|방|팀|데|곳|쪽)"
    r"|전사|회사\s*전체|전체\s*(검색|조회)|사내\s*전체"
    r"|어디(서든|에서든|든)"
    r"|예전에도|과거에도|이전에도"
    r"|다른\s*곳에서도",
    re.IGNORECASE,
)


def is_thread_question(text: str) -> bool:
    """사용자가 **지금 이 스레드**를 가리키고 있는지.

    Args:
        text: 사용자 입력 원문.

    Returns:
        스레드 지시어가 있으면 True.
    """
    return bool(text and _THREAD_DEIXIS_RE.search(text))


def pick_thread_tool(
    text: str, known_tool_names: Iterable[str], root_id: Optional[str]
) -> Optional[str]:
    """첫 iteration 에 강제할 스레드 도구 이름. 조건이 안 맞으면 None.

    세 가지가 모두 참이어야 한다 — 도구가 바인딩돼 있고, 지금이 스레드 안이며(root_id),
    사용자가 그 스레드를 가리켰을 때. 하나라도 빠지면 평소대로 모델이 고른다.

    **스레드 안 멘션의 맥락은 이제 여기서 해결하지 않는다.** 봇이 그 스레드를 직접 읽어
    턴 컨텍스트로 주입한다(services/mattermost/bot._thread_messages). 도구 고정은 라우팅이
    도메인을 바꾸면 그 도구가 아예 안 붙어 무력해지기 때문이다(2026-09-22 실측). 여기 남은
    것은 "이 스레드의 파일 정리해줘" 처럼 **주입된 20줄보다 더 읽어야 하는** 경우의 지름길이다.

    Args:
        text: 사용자 입력 원문.
        known_tool_names: 이 턴에 바인딩된 도구 이름들.
        root_id: 현재 스레드 루트 post_id(`[현재 위치]`). 스레드 밖이면 None.

    Returns:
        ``"mattermost_read_thread"`` 또는 None.
    """
    if not root_id or THREAD_TOOL_NAME not in set(known_tool_names or ()):
        return None
    return THREAD_TOOL_NAME if is_thread_question(text) else None


def is_channel_question(text: str) -> bool:
    """사용자가 **지금 이 채널**을 가리키고 있는지.

    Args:
        text: 사용자 입력 원문.

    Returns:
        채널 지시어가 있으면 True.
    """
    return bool(text and _CHANNEL_DEIXIS_RE.search(text))


def wants_broader_scope(text: str) -> bool:
    """사용자가 **범위를 넓히자**고 말했는지(지시어보다 우선).

    Args:
        text: 사용자 입력 원문.

    Returns:
        "다른 채널에서도"·"전사"·"예전에도" 류 표현이 있으면 True.
    """
    return bool(text and _BROADER_SCOPE_RE.search(text))


def scope_to_current(text: str) -> bool:
    """이 턴의 조회를 **지금 채널로 묶을지** 판정한다.

    구분은 추론이 아니라 사용자가 준 말로 한다:

        확장 의도어 있음        → 묶지 않음 (전사 검색)
        스레드·채널 지시어 있음 → 묶음
        아무것도 없음           → 묶지 않음 (사내 지식검색 본래 용도)

    ⚠️ **지시어가 없을 때 묶으면 사내 지식검색이 죽는다.** "지난주 품질 이슈 찾아줘" 는
    채널을 말하지 않지만 전사 검색이 정답이다. 그래서 묶는 건 지시어가 있을 때뿐이고,
    그마저 0건이면 호출부가 범위를 풀어 다시 검색한다(3층 안전망).

    Args:
        text: 사용자 입력 원문.

    Returns:
        현재 채널로 묶어야 하면 True.
    """
    if wants_broader_scope(text):
        return False
    return is_thread_question(text) or is_channel_question(text)
