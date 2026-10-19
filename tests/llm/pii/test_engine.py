"""치환·복원 엔진 — 조사 보존, 겹침 해소, JSON 왕복, 중첩 인자, 변형 복원, 미등록 토큰, 왕복 동일성."""
import json

import pytest

from agent_arch.llm.base import Message
from agent_arch.llm.pii.engine import DEFAULT_BLOCK_KINDS, Restorer, Substituter, find_tokens
from agent_arch.llm.pii.lexicon import build_lexicon
from agent_arch.llm.pii.tokens import Kind, TokenIssuer, canonical_key

SECRET = "unit-test-secret"


@pytest.fixture
def lex():
    return build_lexicon(
        [("김보라", "user@example.com"), ("이수", "user@example.com"), ("박미래", "user@example.com"), ("John Smith", "john@acme.com")],
        requester={"name": "백지선", "email": "user@example.com", "username": "jsbaek"},
        company_names=["(주)회사", "김보라식품"],
    )


@pytest.fixture
def sub(lex):
    return Substituter(TokenIssuer(SECRET, "conv:1"), lex)


def _one(sub, text, role="user"):
    r = sub.substitute_messages([Message(role=role, content=text)])
    return r.messages[0].content, r


def _token_of(r, original):
    for k, v in r.restore_map.items():
        if v == original:
            return f"[[{k}]]"
    raise AssertionError(f"{original} not in restore_map {r.restore_map}")


# ── 한글 이름 경계 ─────────────────────────────────────────────────────────────
def test_particles_preserved_after_token(sub):
    out, r = _one(sub, "김보라님께서는 승인했고 김보라가 왔다.")
    tok = _token_of(r, "김보라")
    assert out == f"{tok}님께서는 승인했고 {tok}가 왔다."
    assert r.findings[0].particle == "님께서는"


def test_name_inside_other_word_not_replaced(sub):
    out, r = _one(sub, "김보라색 옷과 홍김보라 씨")
    assert out == "김보라색 옷과 홍김보라 씨" and r.token_count == 0


def test_two_char_name_needs_honorific(sub):
    out, r = _one(sub, "이수 과장님이 이수가 왔다고 했다")
    tok = _token_of(r, "이수")
    assert out == f"{tok} 과장님이 이수가 왔다고 했다"


def test_given_name_only_for_rich_sources(sub):
    out, r = _one(sub, "지선님 안녕하세요. 보라님도 안녕.")
    assert _token_of(r, "지선") in out            # 요청자 성 뗀 이름 — 정식 이름을 본 적 없으면 쓴 표기대로 복원
    assert "보라님" in out                          # 디렉터리 출처는 성 뗀 이름 별칭이 없다


def test_given_name_restores_to_full_name_when_both_seen(sub):
    out, r = _one(sub, "지선님, 백지선 대리")
    assert r.restore_map[list(r.restore_map)[0]] == "백지선"
    assert len(r.restore_map) == 1


def test_company_name_protected(sub):
    out, r = _one(sub, "김보라식품(주)와 (주)회사 에 보낸다")
    assert out == "김보라식품(주)와 (주)회사 에 보낸다"


def test_latin_name(sub):
    out, r = _one(sub, "Contact John Smith or john smith today")
    tok = _token_of(r, "John Smith")
    assert out == f"Contact {tok} or {tok} today"


# ── 이메일·핸들·username 공존 ─────────────────────────────────────────────────
def test_email_keeps_domain_outside_token(sub):
    out, r = _one(sub, "메일user@example.com로 보내")
    tok = _token_of(r, "user@example.com")
    assert out == f"메일{tok}@example.com로 보내"


def test_handle_and_username_and_email_of_requester(sub):
    out, r = _one(sub, "@jsbaek 님, jsbaek 계정, user@example.com 메일")
    h = _token_of(r, "jsbaek")
    e = _token_of(r, "user@example.com")
    assert out == f"@{h} 님, {h} 계정, {e}@example.com 메일"


def test_phone_and_email_regex(sub):
    out, r = _one(sub, "연락처 010-1234-5678, 02-123-4567")
    assert out == f"연락처 {_token_of(r, '010-1234-5678')}, {_token_of(r, '02-123-4567')}"


# ── 차단류·겹침·보호 ─────────────────────────────────────────────────────────
def test_block_kinds_default_and_observe(sub):
    text = "주민번호 900101-1234567, 비밀번호는 abc123!, 계좌 신한 110-123-456789, 카드 4111 1111 1111 1111"
    out, r = _one(sub, text)
    assert out == "주민번호 [차단됨:주민번호], 비밀번호는 [차단됨:비밀번호], 계좌 신한 110-123-456789, 카드 4111 1111 1111 1111"
    assert dict(r.blocked) == {"SSN": 1, "PASSWORD": 1}
    observed = [(f.kind, f.applied) for f in r.findings]
    assert ("ACCOUNT", False) in observed and ("CARD", False) in observed


def test_block_kinds_configurable(lex):
    s = Substituter(TokenIssuer(SECRET, "conv:1"), lex, block_kinds=DEFAULT_BLOCK_KINDS | {Kind.ACCOUNT})
    out, _ = _one(s, "계좌 신한 110-123-456789 입금")
    assert out == "계좌 신한 [차단됨:계좌번호] 입금"


def test_overlap_higher_tier_wins(sub):
    # 이메일 안의 username(jsbaek) 은 이메일 토큰에 흡수되고, 주민번호 안의 숫자열은 전화로 잡히지 않는다.
    out, r = _one(sub, "user@example.com 900101-1234567")
    assert out.count("[[") == 1 and "[차단됨:주민번호]" in out


def test_existing_tokens_and_urls_untouched(sub):
    text = "이미 [[PERSON_3f9a1c2d]] 와 [차단됨:주민번호] 그리고 https://x.com/bkim/jsbaek?u=user@example.com 링크"
    out, r = _one(sub, text)
    assert out == text and r.token_count == 0


def test_idempotent_double_substitution(sub):
    first, r1 = _one(sub, "김보라님과 user@example.com")
    second, r2 = _one(sub, first)
    assert second == first and r2.token_count == 0


# ── 메시지 필드 ───────────────────────────────────────────────────────────────
def test_system_skipped_by_default_and_optional(lex):
    msgs = [Message(role="system", content="백지선 은 요청자다")]
    assert Substituter(TokenIssuer(SECRET, "c"), lex).substitute_messages(msgs).messages[0].content == "백지선 은 요청자다"
    out = Substituter(TokenIssuer(SECRET, "c"), lex, scan_system=True).substitute_messages(msgs).messages[0].content
    assert "[[PERSON_" in out


def test_tool_json_roundtrip_values_only(sub):
    payload = {"colleagues": [{"name": "김보라", "email": "user@example.com", "phone": 1012345678, "count": 3, "company": "(주)회사"}],
               "order_no": "1234567890123", "ok": True, "note": None}
    msg = Message(role="tool", content=json.dumps(payload, ensure_ascii=False), tool_call_id="c1")
    r = sub.substitute_messages([msg])
    out = json.loads(r.messages[0].content)
    row = out["colleagues"][0]
    assert row["name"] == _token_of(r, "김보라")
    assert row["email"] == _token_of(r, "user@example.com") + "@example.com"
    assert row["phone"] == _token_of(r, "01012345678")          # 숫자형 전화는 앞 0 을 살려 검사
    assert row["count"] == 3 and row["company"] == "(주)회사"
    assert out["order_no"] == "1234567890123" and out["ok"] is True and out["note"] is None
    assert r.findings[0].json_key == "name"
    assert msg.content == json.dumps(payload, ensure_ascii=False)   # 원본 불변


def test_tool_error_text_path(sub):
    msg = Message(role="tool", content="[ERROR] 김보라 를 찾지 못함", tool_call_id="c1")
    out = sub.substitute_messages([msg]).messages[0].content
    assert out.startswith("[ERROR] [[PERSON_")


def test_assistant_tool_calls_arguments_recursive(sub):
    tc = {"id": "c1", "name": "mail_send", "arguments": {"to": ["user@example.com"], "body": "김보라님께", "n": 1, "meta": {"cc": "jsbaek"}}}
    msg = Message(role="assistant", content="", tool_calls=[tc])
    r = sub.substitute_messages([msg])
    args = r.messages[0].tool_calls[0]["arguments"]
    assert args["to"] == [_token_of(r, "user@example.com") + "@example.com"]
    assert args["body"].startswith("[[PERSON_") and args["n"] == 1
    assert args["meta"]["cc"] == _token_of(r, "jsbaek")
    assert msg.tool_calls[0]["arguments"]["to"] == ["user@example.com"]      # 원본 불변
    assert r.messages[0].tool_calls[0]["id"] == "c1"


def test_unchanged_message_is_same_object(sub):
    msg = Message(role="user", content="개인정보 없는 문장", images=None)
    r = sub.substitute_messages([msg])
    assert r.messages[0] is msg and not r.changed


def test_same_value_same_token_across_messages_and_calls(lex):
    s1 = Substituter(TokenIssuer(SECRET, "conv:1"), lex)
    r1 = s1.substitute_messages([Message(role="user", content="김보라"), Message(role="assistant", content="김보라님")])
    r2 = Substituter(TokenIssuer(SECRET, "conv:1"), lex).substitute_messages([Message(role="user", content="아무개와 김보라")])
    assert r1.messages[0].content == r1.messages[1].content[:-1]
    assert list(r1.restore_map) == list(r2.restore_map)


def test_findings_do_not_contain_values(sub):
    _, r = _one(sub, "김보라님 연락처 010-1234-5678 메일 user@example.com")
    for f in r.findings:
        assert "김보라" not in f.context_masked and "1234" not in f.context_masked and "bkim@" not in f.context_masked


def test_substitute_text_helper(sub):
    out, r = sub.substitute_text("김보라에게 user@example.com 로")
    assert out.count("[[") == 2 and len(r.restore_map) == 2


# ── 복원 ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def rest():
    return Restorer({"PERSON_3f9a1c2d": "김보라", "EMAIL_0a0b0c0d": "user@example.com", "PHONE_11112222": "010-1234-5678",
                     "HANDLE_abababab": "jsbaek", "PERSON_abcdef012345": "박미래"})


@pytest.mark.parametrize("variant", [
    "[[PERSON_3f9a1c2d]]", "[PERSON_3f9a1c2d]", "PERSON_3f9a1c2d", "[[ PERSON_3F9A1C2D ]]", "\\[\\[PERSON_3f9a1c2d\\]\\]",
    "[[PERSON_3f9a1c2d]", "[PERSON_3f9a1c2d]]",
])
def test_restore_variants(rest, variant):
    r = rest.restore_text(f"담당자는 {variant}님입니다")
    assert r.value == "담당자는 김보라님입니다" and r.restored == 1 and not r.unresolved


def test_restore_email_domain_once(rest):
    assert rest.restore_text("[[EMAIL_0a0b0c0d]]@example.com 로").value == "user@example.com 로"
    assert rest.restore_text("[[EMAIL_0a0b0c0d]] 로").value == "user@example.com 로"
    assert rest.restore_text("[[EMAIL_0a0b0c0d]]@wrong.com 로").value == "user@example.com 로"


def test_restore_handle_keeps_at(rest):
    assert rest.restore_text("@[[HANDLE_abababab]] 님").value == "@jsbaek 님"


def test_restore_12_and_8_hex_fallbacks(rest):
    assert rest.restore_text("[[PERSON_abcdef012345]]").value == "박미래"
    assert rest.restore_text("[[PERSON_abcdef01]]").value == "박미래"          # 12자를 8자로 줄인 경우
    assert rest.restore_text("[[PERSON_3f9a1c2d0000]]").value == "김보라"      # 8자 뒤에 hex 가 붙은 경우


def test_restore_unknown_token_label_vs_keep(rest):
    r = rest.restore_text("[[PERSON_deadbeef]] 와 [[EMAIL_deadbeef]]@x.com")
    assert r.value == "(확인되지 않은 사람) 와 (확인되지 않은 이메일)"
    assert r.unresolved == ["PERSON_deadbeef", "EMAIL_deadbeef"]
    k = rest.restore_text("[[PERSON_deadbeef]]", unknown="keep")
    assert k.value == "[[PERSON_deadbeef]]" and k.unresolved == ["PERSON_deadbeef"]


def test_restore_value_nested_and_types(rest):
    r = rest.restore_value({"to": ["[[EMAIL_0a0b0c0d]]@example.com", "[[PERSON_deadbeef]]"], "n": 1, "t": ("[[PERSON_3f9a1c2d]]",), "b": True})
    assert r.value == {"to": ["user@example.com", "[[PERSON_deadbeef]]"], "n": 1, "t": ("김보라",), "b": True}
    assert r.unresolved == ["PERSON_deadbeef"] and r.restored == 2


def test_blocked_label_not_restored(rest):
    assert rest.restore_text("[차단됨:주민번호] 그대로").value == "[차단됨:주민번호] 그대로"


def test_find_tokens():
    assert find_tokens({"a": "x [[PERSON_3f9a1c2d]] y", "b": ["EMAIL_0A0B0C0D"], "c": 1}) == ["PERSON_3f9a1c2d", "EMAIL_0a0b0c0d"]
    assert find_tokens("김보라") == []


# ── 왕복 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "김보라님께서는 010-1234-5678 로 연락하라고 하셨고, user@example.com 로 메일도 보내라 했다.",
    "@jsbaek 님과 jsbaek 계정, John Smith 와 이수 과장님, 박미래 대리에게 공유.",
    "회의는 2026-10-14 09:30, 주문번호 0123456789012, 금액 1,010,123원.",
    "김보라식품(주)의 김보라 담당자(user@example.com)와 (주)회사 백지선.",
    "메일user@example.com로, 전화010-1234-5678로.",
])
def test_roundtrip(sub, text):
    out, r = _one(sub, text)
    assert Restorer(r.restore_map).restore_text(out).value == text


def test_restore_map_key_format(sub):
    _, r = _one(sub, "김보라")
    (key,) = r.restore_map
    assert key == canonical_key(Kind.PERSON, key.split("_")[1]) and len(key.split("_")[1]) == 8
