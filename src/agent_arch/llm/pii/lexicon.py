"""어휘 — 치환할 이름·이메일·핸들·사번 목록과 그것을 찾는 패턴.

어휘 출처(``Alias.source``)에 따라 엄격도가 다르다:
- ``directory``(임직원 디렉터리): 정식 이름(full_name)만. 수백~수천 명의 성 뗀 이름·
  영문 ID 까지 풀면 일반 단어와 충돌이 폭발한다.
- ``requester``(요청자 본인)·``tool``(이번 대화 도구 결과에서 수확): 정식 이름 + 성 뗀
  이름(3자 이상 이름에서만) + 이메일 local-part(username) + ``@handle`` 까지.

한글 이름 경계는 ``patterns.NAME_TAIL`` 이 조사·직급을 lookahead 로만 보므로
"김보라님께서는" → ``[[PERSON_x]]님께서는`` 처럼 조사가 보존된다. 2자 이름과
흔한 단어 이름(``COMMON_WORD_NAMES``)은 직급·존칭이 뒤따를 때만 이름으로 본다.

이 모듈은 I/O 가 없다. 디렉터리 조회·Redis 캐시는 배선 모듈이 맡고, 여기엔
완성된 (이름, 이메일) 목록이 들어온다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from agent_arch.llm.pii.patterns import (
    ASCII_ID_LEFT,
    ASCII_ID_RIGHT,
    NAME_LEFT,
    NAME_TAIL,
    NAME_TAIL_STRICT,
)
from agent_arch.llm.pii.tokens import Kind, normalize

Form = Literal["full_name", "given_name", "latin_name", "username", "handle", "email", "empno"]
Source = Literal["requester", "directory", "tool"]

#: 이름이면서 흔한 단어 — 직급·존칭 없이는 치환하지 않는다(shadow 로 보강).
COMMON_WORD_NAMES: frozenset[str] = frozenset({
    "한글", "미래", "하나", "보라", "지원", "사랑", "세계", "우리", "나라", "소망", "희망", "진리", "다정",
    "정의", "자유", "평화", "강산", "이슬", "바다", "하늘", "구름", "아침", "저녁", "오늘", "내일",
    "성공", "행복", "기쁨", "은혜", "믿음", "은별", "새별", "샛별", "노을", "가람", "마루", "누리", "슬기",
    "보람", "아름", "다운", "이름", "주문", "결제", "승인", "신청", "문의", "요청", "검색", "확인", "안내",
})
#: 영문 ID 로 쓰기엔 너무 일반적인 local-part.
COMMON_USERNAMES: frozenset[str] = frozenset({
    "admin", "test", "info", "support", "sales", "hr", "dev", "api", "user", "root", "mail", "noreply",
    "no-reply", "help", "team", "contact", "office", "service", "notice", "bot", "system", "master",
})
_KO_NAME = re.compile(r"^[가-힣]{2,4}$")
_LATIN_NAME = re.compile(r"^[A-Za-z][A-Za-z'.-]{0,29}(?:\s+[A-Za-z][A-Za-z'.-]{0,29}){1,3}$")
_USERNAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{3,}$")


@dataclass(frozen=True)
class Alias:
    """어휘 한 항목 — 본문에서 찾을 표기(text)와 그것이 가리키는 정식 값(canonical)."""

    text: str
    kind: Kind
    canonical: str
    form: Form
    source: Source
    strict: bool = False  # True 면 직급·존칭이 뒤따를 때만 매치(2자·흔한 단어 이름)


def expand_person_aliases(
    name: Optional[str],
    email: Optional[str] = None,
    *,
    source: Source = "directory",
    handle: Optional[str] = None,
) -> list[Alias]:
    """한 사람의 (이름, 이메일, 핸들)에서 어휘 항목들을 만든다.

    - 정식 이름(한글 2~4자 또는 라틴 First Last): PERSON. 2자·흔한 단어는 strict.
    - 성 뗀 이름(given_name): 한글 3자 이상 이름에서만, requester·tool 출처에서만. 항상 strict.
    - 이메일: EMAIL(전체). local-part 가 4자 이상·영문 포함·흔한 ID 가 아니면 username 도
      (requester·tool 출처에서만).
    - 핸들: HANDLE(``@`` 제외한 id). 없으면 username 과 같다.
    """
    out: list[Alias] = []
    name = (name or "").strip()
    email = (email or "").strip()
    rich = source in ("requester", "tool")
    if name:
        canonical = normalize(Kind.PERSON, name)
        if _KO_NAME.match(canonical):
            strict = len(canonical) == 2 or canonical in COMMON_WORD_NAMES
            out.append(Alias(canonical, Kind.PERSON, canonical, "full_name", source, strict))
            if rich and len(canonical) >= 3:
                # 성 뗀 이름은 항상 strict(직급·존칭 필수)라 흔한 단어 블랙리스트와 무관하다.
                out.append(Alias(canonical[1:], Kind.PERSON, canonical, "given_name", source, True))
        elif _LATIN_NAME.match(name):
            spaced = re.sub(r"\s+", " ", name)
            out.append(Alias(spaced, Kind.PERSON, spaced, "latin_name", source, False))
    if email and "@" in email:
        e = normalize(Kind.EMAIL, email)
        out.append(Alias(e, Kind.EMAIL, e, "email", source, False))
        local = e.split("@", 1)[0]
        if rich and _USERNAME.match(local) and local.lower() not in COMMON_USERNAMES:
            out.append(Alias(local, Kind.HANDLE, local, "username", source, False))
            if not handle:
                out.append(Alias(local, Kind.HANDLE, local, "handle", source, False))
    if handle:
        h = normalize(Kind.HANDLE, handle)
        if _USERNAME.match(h):
            out.append(Alias(h, Kind.HANDLE, h, "handle", source, False))
            if rich and h.lower() not in COMMON_USERNAMES:
                out.append(Alias(h, Kind.HANDLE, h, "username", source, False))
    seen: set[tuple[str, str]] = set()
    return [a for a in out if (a.form, a.text.lower()) not in seen and not seen.add((a.form, a.text.lower()))]


_FORM_GROUP = {
    "full_name": "ko",
    "given_name": "ko",
    "latin_name": "latin",
    "username": "user",
    "handle": "handle",
    "email": "email",
    "empno": "empno",
}


def _escape_all(texts: Iterable[str]) -> str:
    # 긴 것부터 — "김보라" 와 "김보" 가 둘 다 있으면 긴 쪽이 이긴다.
    return "|".join(re.escape(t) for t in sorted(set(texts), key=lambda s: (-len(s), s)))


class Lexicon:
    """컴파일된 어휘. ``pattern`` 으로 본문을 훑고 ``lookup()`` 으로 어떤 항목인지 안다."""

    def __init__(self, aliases: Iterable[Alias] = (), company_names: Iterable[str] = ()):
        self.aliases: tuple[Alias, ...] = tuple(aliases)
        self.company_names: frozenset[str] = frozenset(c.strip() for c in company_names if c and c.strip())
        self._by_text: dict[tuple[str, str], Alias] = {}
        self.pattern: Optional[re.Pattern] = None
        self.company_pattern: Optional[re.Pattern] = None
        self._compile()

    def __bool__(self) -> bool:
        return bool(self.aliases)

    def __len__(self) -> int:
        return len(self.aliases)

    @classmethod
    def empty(cls) -> "Lexicon":
        return cls()

    def _compile(self) -> None:
        groups: dict[str, list[str]] = {}
        for a in self.aliases:
            g = _FORM_GROUP[a.form]
            # 한글 이름은 strict 여부로 꼬리 규칙이 달라 그룹을 나눈다.
            key = ("ko_strict" if a.strict else "ko") if g == "ko" else g
            text = a.text if g in ("ko", "ko_strict") else a.text.lower()
            self._by_text[(key, text)] = a
            groups.setdefault(key, []).append(a.text)
        parts: list[str] = []
        if groups.get("ko"):
            parts.append(rf"(?P<ko>{NAME_LEFT}(?:{_escape_all(groups['ko'])}){NAME_TAIL})")
        if groups.get("ko_strict"):
            parts.append(rf"(?P<ko_strict>{NAME_LEFT}(?:{_escape_all(groups['ko_strict'])}){NAME_TAIL_STRICT})")
        if groups.get("latin"):
            # 라틴 이름은 공백 수가 달라도 맞게 escape 된 공백을 \s+ 로.
            alts = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in sorted(set(groups["latin"]), key=lambda s: (-len(s), s)))
            parts.append(rf"(?P<latin>(?<![A-Za-z])(?:{alts})(?![A-Za-z]))")
        if groups.get("email"):
            parts.append(rf"(?P<email>(?<![A-Za-z0-9._%+-])(?:{_escape_all(groups['email'])})(?![A-Za-z0-9-]|\.[A-Za-z]))")
        if groups.get("handle"):
            parts.append(rf"(?P<handle>(?<![A-Za-z0-9._%+-])@(?:{_escape_all(groups['handle'])}){ASCII_ID_RIGHT})")
        if groups.get("user"):
            parts.append(rf"(?P<user>{ASCII_ID_LEFT}(?:{_escape_all(groups['user'])}){ASCII_ID_RIGHT})")
        if groups.get("empno"):
            parts.append(rf"(?P<empno>(?<![A-Za-z0-9-])(?:{_escape_all(groups['empno'])})(?![A-Za-z0-9-]))")
        self.pattern = re.compile("|".join(parts), re.IGNORECASE) if parts else None
        if self.company_names:
            self.company_pattern = re.compile(_escape_all(self.company_names))

    def lookup(self, group: str, matched: str) -> Optional[Alias]:
        if group in ("ko", "ko_strict"):
            return self._by_text.get((group, matched))
        if group == "handle":
            matched = matched.lstrip("@")
        return self._by_text.get((group, matched.lower()))


def build_lexicon(
    persons: Iterable[tuple[Optional[str], Optional[str]]] = (),
    *,
    source: Source = "directory",
    requester: Optional[dict] = None,
    tool_persons: Iterable[tuple[Optional[str], Optional[str]]] = (),
    company_names: Iterable[str] = (),
    empnos: Iterable[str] = (),
) -> Lexicon:
    """어휘를 조립한다.

    Args:
        persons: 디렉터리 (name, email) 목록.
        requester: ``get_user_context()`` 꼴 dict(name, email, username).
        tool_persons: 이번 대화 도구 결과에서 수확한 (name, email).
        company_names: 치환하지 않을 회사명(보호 스팬).
        empnos: 사번 문자열.
    """
    aliases: list[Alias] = []
    for name, email in persons:
        aliases.extend(expand_person_aliases(name, email, source=source))
    if requester:
        aliases.extend(expand_person_aliases(
            requester.get("name"), requester.get("email"), source="requester", handle=requester.get("username"),
        ))
    for name, email in tool_persons:
        aliases.extend(expand_person_aliases(name, email, source="tool"))
    for e in empnos:
        e = (e or "").strip()
        if e:
            aliases.append(Alias(e, Kind.EMPNO, e, "empno", source, False))
    # 같은 (form, text) 는 더 풍부한 출처(requester > tool > directory)가 이긴다.
    rank = {"requester": 0, "tool": 1, "directory": 2}
    best: dict[tuple[str, str], Alias] = {}
    for a in sorted(aliases, key=lambda a: rank[a.source]):
        best.setdefault((a.form, a.text.lower()), a)
    return Lexicon(best.values(), company_names)
