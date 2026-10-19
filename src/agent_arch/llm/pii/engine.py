"""치환·복원 엔진.

**치환(``Substituter``)** — 단일 패스 스팬 스캐너. 원문을 수정하지 않고 모든 규칙으로
스팬을 모은 뒤 tier 순으로 채택하고(겹치면 버림), 토큰을 **일괄 발급**해 재조립한다.
``re.sub`` 를 여러 번 돌리면 앞 단계가 만든 토큰·이메일 도메인을 뒤 단계가 다시
건드리므로 쓰지 않는다.

tier: 0 보호(이미 있는 토큰·차단 라벨·URL·회사명) → 1 차단류 → 2 이메일 → 3 전화
→ 4 사번 → 5 어휘. 상위 tier 가 먼저 자리를 잡으므로 이메일 안의 username, 차단류
안의 전화 조각은 치환되지 않는다.

메시지 필드별:
- ``content``: 텍스트. system 은 기본 스킵(프롬프트엔 사용자 정보가 없다).
- ``assistant.tool_calls[].arguments``: dict 재귀(이력 재전송 시 같은 값→같은 토큰).
- role="tool" ``content``: JSON 이면 **문자열 값 안에서만** 치환하고 다시 직렬화한다
  (gemini 가 ``json.loads`` 로 FunctionResponse 를 만든다). 숫자 값도 9자리 이상이면
  검사해 전화·주민번호면 문자열 토큰으로 바꾼다. 키 이름은 스키마라 건드리지 않는다.
- ``images``·``name``·``tool_call_id``·``thought_signature`` 는 불변. 새 ``Message`` 객체를
  만들어 원본 이력을 오염시키지 않는다.

**복원(``Restorer``)** — 변형 허용 정규식 하나로 ``[[PERSON_x]]``·``[PERSON_x]``·
``PERSON_X``·``\\[\\[PERSON_x\\]\\]`` 를 모두 잡는다. 복원표에 없는 토큰은 본문에선
종류별 라벨로, 도구 인자에선 그대로 두고 ``unresolved`` 로 돌려준다(실행 차단은
executor 몫).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from agent_arch.llm.base import Message
from agent_arch.llm.pii.lexicon import Lexicon
from agent_arch.llm.pii.patterns import (
    EXISTING_TOKEN,
    PARTICLE,
    RULES,
    TIER_LEXICON,
    TIER_PROTECT,
    URL,
    RegexRule,
)
from agent_arch.llm.pii.tokens import (
    BLOCKABLE,
    BLOCKED_LABEL,
    UNKNOWN_LABEL,
    Kind,
    TokenIssuer,
    canonical_key,
    normalize,
    render,
)

logger = logging.getLogger(__name__)

#: enforce 초기 차단 종류. 계좌·카드는 송장·주문번호 오탐을 shadow 로 본 뒤 추가한다.
DEFAULT_BLOCK_KINDS: frozenset[Kind] = frozenset({Kind.SSN, Kind.APIKEY, Kind.PASSWORD})

_PARTICLE_RE = re.compile(rf"^(?:{PARTICLE}){{0,3}}")
_CONTEXT = 12
_MIN_NUMERIC_LEN = 9
_PHONE_KEY = re.compile(r"(?i)phone|mobile|tel|cell|휴대|전화|연락처")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: Optional[Kind]
    action: str          # tokenize | block | protect
    tier: int
    value: Optional[str] = None   # 정규화값(tokenize) — 발급 키
    source: str = "regex"
    form: str = ""
    canonical: Optional[str] = None  # 복원표에 넣을 정식 표기(어휘)


@dataclass
class Finding:
    """치환 1건의 기록. 원문 값은 담지 않는다(``context_masked`` 는 매치 자리에 토큰)."""

    kind: str
    source: str
    form: str
    token: str
    role: str
    msg_index: int
    json_key: Optional[str]
    particle: Optional[str]
    context_masked: str
    applied: bool = True   # False 면 차단 종류가 꺼져 있어 관찰만 했다


@dataclass
class Substitution:
    messages: list[Message]
    restore_map: dict[str, str]
    findings: list[Finding]
    blocked: Counter = field(default_factory=Counter)
    token_count: int = 0

    @property
    def changed(self) -> bool:
        return self.token_count > 0 or bool(self.blocked)


@dataclass
class RestoreResult:
    value: Any
    unresolved: list[str]
    restored: int = 0

    @property
    def text(self) -> str:
        return self.value


class _Pending:
    """한 텍스트 단위의 스캔 결과(2 단계 렌더 전)."""

    __slots__ = ("text", "spans", "json_key", "was_number")

    def __init__(self, text: str, spans: list[Span], json_key: Optional[str], was_number: bool = False):
        self.text, self.spans, self.json_key, self.was_number = text, spans, json_key, was_number


class Substituter:
    """호출 1회분 치환기. 어휘·발급기는 호출별 불변 객체로 받는다(전역 상태 없음)."""

    def __init__(
        self,
        issuer: TokenIssuer,
        lexicon: Optional[Lexicon] = None,
        *,
        rules: Iterable[RegexRule] = RULES,
        block_kinds: Iterable[Kind] = DEFAULT_BLOCK_KINDS,
        scan_system: bool = False,
    ):
        self.issuer = issuer
        self.lexicon = lexicon or Lexicon.empty()
        self.rules = tuple(rules)
        self.block_kinds = frozenset(block_kinds)
        self.scan_system = scan_system

    # ── 1단계: 스캔 ──────────────────────────────────────────────────────────
    def scan(self, text: str, json_key: Optional[str] = None) -> list[Span]:
        """겹침을 해소한 스팬 목록(시작 위치순)."""
        if not text:
            return []
        found: list[Span] = []
        for m in EXISTING_TOKEN.finditer(text):
            found.append(Span(m.start(), m.end(), None, "protect", TIER_PROTECT))
        for m in URL.finditer(text):
            found.append(Span(m.start(), m.end(), None, "protect", TIER_PROTECT, source="url"))
        if self.lexicon.company_pattern is not None:
            for m in self.lexicon.company_pattern.finditer(text):
                found.append(Span(m.start(), m.end(), None, "protect", TIER_PROTECT, source="company"))
        for rule in self.rules:
            if rule.deny_json_keys is not None and json_key and rule.deny_json_keys.search(json_key):
                continue
            for m in rule.pattern.finditer(text):
                if rule.validator is not None and not rule.validator(m):
                    continue
                if rule.deny_before is not None and rule.deny_before.search(text[max(0, m.start() - 20):m.start()]):
                    continue
                s, e = m.span(rule.group) if rule.group else m.span()
                if rule.kind in BLOCKABLE:
                    found.append(Span(s, e, rule.kind, "block", rule.tier, form=rule.kind.value.lower()))
                else:
                    raw = text[s:e]
                    found.append(Span(s, e, rule.kind, "tokenize", rule.tier, normalize(rule.kind, raw),
                                      form=rule.kind.value.lower(), canonical=raw))
        if self.lexicon.pattern is not None:
            for m in self.lexicon.pattern.finditer(text):
                alias = self.lexicon.lookup(m.lastgroup or "", m.group(0))
                if alias is None:
                    continue
                s, e = m.span()
                if m.lastgroup == "handle":
                    s += 1   # '@' 는 토큰 밖에 남긴다
                found.append(Span(s, e, alias.kind, "tokenize", TIER_LEXICON, normalize(alias.kind, alias.canonical),
                                  source=alias.source, form=alias.form, canonical=alias.canonical))
        return _resolve(found)

    # ── 2단계: 발급 + 렌더 ───────────────────────────────────────────────────
    def substitute_messages(self, messages: list[Message]) -> Substitution:
        pending: list[tuple[int, Message, Any, Any, Optional[Any]]] = []
        for i, msg in enumerate(messages):
            content_p: Any = None
            if msg.content and (msg.role != "system" or self.scan_system):
                content_p = self._pend_json(msg.content) if msg.role == "tool" else self._pend_text(msg.content)
            args_p = [self._pend_value(tc.get("arguments")) for tc in (msg.tool_calls or [])] if msg.tool_calls else None
            pending.append((i, msg, content_p, args_p, None))
        assignments = self.issuer.assign(_collect_pairs(p for _, _, c, a, _ in pending for p in _iter_pending(c, a)))
        sub = Substitution(messages=[], restore_map={}, findings=[])
        for i, msg, content_p, args_p, _ in pending:
            update: dict[str, Any] = {}
            if content_p is not None:
                new, changed = self._render_any(content_p, assignments, sub, msg.role, i, tool_json=(msg.role == "tool"))
                if changed:
                    update["content"] = new
            if args_p is not None:
                new_calls = []
                any_changed = False
                for tc, p in zip(msg.tool_calls or [], args_p):
                    new_args, changed = self._render_any(p, assignments, sub, msg.role, i)
                    any_changed |= changed
                    new_calls.append({**tc, "arguments": new_args} if changed else tc)
                if any_changed:
                    update["tool_calls"] = new_calls
            sub.messages.append(msg.model_copy(update=update) if update else msg)
        return sub

    def substitute_text(self, text: str, *, role: str = "user", msg_index: int = 0) -> tuple[str, Substitution]:
        """단일 텍스트 치환(ChatGPT 위탁 레인처럼 ``Message`` 를 쓰지 않는 호출자용)."""
        p = self._pend_text(text)
        assignments = self.issuer.assign(_collect_pairs([p]))
        sub = Substitution(messages=[], restore_map={}, findings=[])
        new, _ = self._render_any(p, assignments, sub, role, msg_index)
        return new, sub

    # ── 내부 ─────────────────────────────────────────────────────────────────
    def _pend_text(self, text: str, json_key: Optional[str] = None) -> _Pending:
        return _Pending(text, self.scan(text, json_key), json_key)

    def _pend_json(self, content: str) -> Any:
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return self._pend_text(content)
        if not isinstance(parsed, (dict, list)):
            return self._pend_text(content)
        return ("json", self._pend_value(parsed))

    def _pend_value(self, obj: Any, key: Optional[str] = None) -> Any:
        if isinstance(obj, str):
            return self._pend_text(obj, key)
        if isinstance(obj, bool) or obj is None:
            return obj
        if isinstance(obj, (int, float)):
            s = str(obj)
            # 전화번호가 숫자형으로 담기면 앞의 0 이 사라진다("phone": 1012345678) — 키 이름이 전화면 되살려 검사.
            if key and _PHONE_KEY.search(key) and s.isdigit() and not s.startswith("0"):
                s = "0" + s
            if len(s) >= _MIN_NUMERIC_LEN:
                spans = self.scan(s, key)
                if spans:
                    return _Pending(s, spans, key, was_number=True)
            return obj
        if isinstance(obj, dict):
            return {k: self._pend_value(v, k if isinstance(k, str) else key) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._pend_value(v, key) for v in obj]
        return obj

    def _render_any(self, p: Any, assignments, sub: Substitution, role: str, idx: int, *, tool_json: bool = False):
        """(렌더 결과, 바뀌었는가)."""
        if isinstance(p, tuple) and len(p) == 2 and p[0] == "json":
            new, changed = self._render_any(p[1], assignments, sub, role, idx)
            return (json.dumps(new, ensure_ascii=False) if changed else None), changed
        if isinstance(p, _Pending):
            if not p.spans:
                return p.text, False
            new = self._render_text(p, assignments, sub, role, idx)
            return new, new != p.text
        if isinstance(p, dict):
            out, changed = {}, False
            for k, v in p.items():
                nv, c = self._render_any(v, assignments, sub, role, idx)
                out[k] = nv if c else _unwrap(v)
                changed |= c
            return out, changed
        if isinstance(p, list):
            out, changed = [], False
            for v in p:
                nv, c = self._render_any(v, assignments, sub, role, idx)
                out.append(nv if c else _unwrap(v))
                changed |= c
            return out, changed
        return p, False

    def _render_text(self, p: _Pending, assignments, sub: Substitution, role: str, idx: int) -> str:
        text, out, pos = p.text, [], 0
        new_len = 0
        marks: list[tuple[Span, str, int, bool]] = []   # (span, 조각, 새 텍스트에서의 시작, 적용 여부)
        for sp in p.spans:
            gap = text[pos:sp.start]
            out.append(gap)
            new_len += len(gap)
            original = text[sp.start:sp.end]
            if sp.action == "protect":
                out.append(original)
                new_len += len(original)
            elif sp.action == "block":
                applied = sp.kind in self.block_kinds
                piece = BLOCKED_LABEL[sp.kind] if applied else original
                out.append(piece)
                if applied:
                    sub.blocked[sp.kind.value] += 1
                marks.append((sp, piece, new_len, applied))
                new_len += len(piece)
            else:
                hexid = assignments[(sp.kind, sp.value)]
                token = render(sp.kind, hexid)
                if sp.kind is Kind.EMAIL:
                    token += "@" + original.split("@", 1)[1]
                out.append(token)
                self._remember(sub, sp, hexid, original)
                sub.token_count += 1
                marks.append((sp, token, new_len, True))
                new_len += len(token)
            pos = sp.end
        out.append(text[pos:])
        rendered = "".join(out)
        # 기록은 **치환된 텍스트** 기준으로 만든다 — 이웃한 다른 식별자도 이미 토큰이라 값이 새지 않는다.
        for sp, piece, start, applied in marks:
            sub.findings.append(self._finding(sp, piece, rendered, start, text, role, idx, p.json_key, applied))
        return rendered

    @staticmethod
    def _remember(sub: Substitution, sp: Span, hexid: str, original: str) -> None:
        key = canonical_key(sp.kind, hexid)
        if sp.kind is Kind.EMAIL:
            value = original            # 도메인 포함 전체 이메일
        elif sp.form == "given_name":
            value = sub.restore_map.get(key, original)   # 정식 이름이 이미 있으면 그것을 유지
        elif sp.form in ("full_name", "latin_name"):
            value = sp.canonical or original
        else:
            value = original
        sub.restore_map[key] = value

    @staticmethod
    def _finding(sp: Span, piece: str, rendered: str, start: int, original_text: str, role: str, idx: int,
                 json_key, applied: bool) -> Finding:
        before = rendered[max(0, start - _CONTEXT):start]
        after = rendered[start + len(piece):start + len(piece) + _CONTEXT]
        particle = None
        if sp.kind is Kind.PERSON:
            m = _PARTICLE_RE.match(original_text[sp.end:sp.end + 8])
            particle = (m.group(0) or None) if m else None
        # 토큰·라벨 밖에 남은 이메일·숫자열(관찰만 한 차단류 등)은 문맥에서도 가린다.
        ctx = _CTX_MASK.sub("…", before) + piece + _CTX_MASK.sub("…", after)
        return Finding(
            kind=(sp.kind.value if sp.kind else "?"), source=sp.source, form=sp.form, token=piece, role=role,
            msg_index=idx, json_key=json_key, particle=particle, context_masked=ctx, applied=applied,
        )


_CTX_MASK = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+|\d{3,}")


def _unwrap(p: Any) -> Any:
    """렌더하지 않은 pending 트리를 원래 값으로 되돌린다."""
    if isinstance(p, _Pending):
        return int(p.text) if p.was_number and p.text.isdigit() else (float(p.text) if p.was_number else p.text)
    if isinstance(p, dict):
        return {k: _unwrap(v) for k, v in p.items()}
    if isinstance(p, list):
        return [_unwrap(v) for v in p]
    if isinstance(p, tuple) and len(p) == 2 and p[0] == "json":
        return _unwrap(p[1])
    return p


def _iter_pending(*trees: Any):
    stack = list(trees)
    while stack:
        t = stack.pop()
        if isinstance(t, _Pending):
            yield t
        elif isinstance(t, dict):
            stack.extend(t.values())
        elif isinstance(t, (list, tuple)):
            stack.extend(t)


def _collect_pairs(pendings: Iterable[_Pending]) -> set[tuple[Kind, str]]:
    pairs: set[tuple[Kind, str]] = set()
    for p in pendings:
        for sp in p.spans:
            if sp.action == "tokenize" and sp.kind is not None and sp.value is not None:
                pairs.add((sp.kind, sp.value))
    return pairs


def _resolve(found: list[Span]) -> list[Span]:
    """tier 오름차순으로 채택, 이미 채택된 스팬과 겹치면 버린다. 반환은 시작 위치순."""
    found.sort(key=lambda sp: (sp.tier, sp.start, -(sp.end - sp.start)))
    chosen: list[Span] = []
    for sp in found:
        if sp.end <= sp.start:
            continue
        if any(sp.start < c.end and c.start < sp.end for c in chosen):
            continue
        chosen.append(sp)
    chosen.sort(key=lambda sp: sp.start)
    return chosen


# ── 복원 ─────────────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(
    r"(?:(?:\\?\[){1,2}\s*)?(?P<kind>PERSON|EMAIL|PHONE|HANDLE|EMPNO)_(?P<hex>[0-9A-Fa-f]{12}|[0-9A-Fa-f]{8})"
    r"(?:\s*(?:\\?\]){1,2})?(?P<domain>@[A-Za-z0-9.-]+\.[A-Za-z]{2,})?",
    re.IGNORECASE,
)


class Restorer:
    """복원표로 토큰을 원문으로 되돌린다."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)
        self._by_kind_prefix: dict[str, list[str]] = {}
        for key in self.mapping:
            kind, _, hexid = key.partition("_")
            self._by_kind_prefix.setdefault(kind, []).append(hexid)

    def lookup(self, kind: str, hexid: str) -> Optional[str]:
        kind, hexid = kind.upper(), hexid.lower()
        key = f"{kind}_{hexid}"
        if key in self.mapping:
            return self.mapping[key]
        if len(hexid) == 12:                      # 12자 토큰을 모델이 그대로 썼지만 복원표는 8자
            return self.mapping.get(f"{kind}_{hexid[:8]}")
        cands = [h for h in self._by_kind_prefix.get(kind, []) if h.startswith(hexid)]
        if len(cands) == 1:                       # 모델이 12자를 8자로 줄인 경우
            return self.mapping[f"{kind}_{cands[0]}"]
        return None

    def restore_text(self, text: str, *, unknown: str = "label") -> RestoreResult:
        if not text:
            return RestoreResult(text, [], 0)
        unresolved: list[str] = []
        restored = 0

        def _sub(m: re.Match) -> str:
            nonlocal restored
            kind, hexid = m.group("kind").upper(), m.group("hex")
            value = self.lookup(kind, hexid)
            if value is None:
                unresolved.append(canonical_key(kind, hexid))
                if unknown == "keep":
                    return m.group(0)
                return UNKNOWN_LABEL[Kind(kind)] + ("" if kind == "EMAIL" else (m.group("domain") or ""))
            restored += 1
            if kind == "EMAIL":
                return value                      # 도메인 그룹까지 통째로 전체 이메일로
            return value + (m.group("domain") or "")

        return RestoreResult(TOKEN_RE.sub(_sub, text), unresolved, restored)

    def restore_value(self, obj: Any) -> RestoreResult:
        """dict/list 재귀 복원. 미등록 토큰은 그대로 두고 ``unresolved`` 로 돌려준다."""
        unresolved: list[str] = []
        restored = 0

        def _walk(o: Any) -> Any:
            nonlocal restored
            if isinstance(o, str):
                r = self.restore_text(o, unknown="keep")
                unresolved.extend(r.unresolved)
                restored += r.restored
                return r.value
            if isinstance(o, dict):
                return {k: _walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_walk(v) for v in o]
            if isinstance(o, tuple):
                return tuple(_walk(v) for v in o)
            return o

        return RestoreResult(_walk(obj), unresolved, restored)


def find_tokens(obj: Any) -> list[str]:
    """값 안에 남아 있는 토큰(정규 키)을 모두 돌려준다 — 복원 뒤에도 남았다면 미복원이다."""
    out: list[str] = []

    def _walk(o: Any) -> None:
        if isinstance(o, str):
            out.extend(canonical_key(m.group("kind"), m.group("hex")) for m in TOKEN_RE.finditer(o))
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(obj)
    return out
