"""정규식 규칙 — 한글 붙여쓰기 경계, 숫자류 오탐(날짜·주문번호·송장) 배제, 차단류 검증기."""
import re

import pytest

from agent_arch.llm.pii import patterns as P
from agent_arch.llm.pii.tokens import Kind


def _rule(kind, nth=0):
    return [r for r in P.RULES if r.kind is kind][nth]


def _hits(rule, text, json_key=None):
    out = []
    for m in rule.pattern.finditer(text):
        if rule.validator and not rule.validator(m):
            continue
        if rule.deny_before and rule.deny_before.search(text[max(0, m.start() - 20):m.start()]):
            continue
        s, e = m.span(rule.group) if rule.group else m.span()
        out.append(text[s:e])
    return out


# ── 이메일 ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("text", "expected"), [
    ("user@example.com", ["user@example.com"]),
    ("메일user@example.com로 보내", ["user@example.com"]),
    ("a.b+c@x.co.kr 입니다.", ["a.b+c@x.co.kr"]),
    ("끝에 마침표 test@example.com.", ["test@example.com"]),
    ("@jsbaek 멘션은 이메일이 아니다", []),
    ("foo@bar 는 TLD 가 없다", []),
])
def test_email(text, expected):
    assert _hits(_rule(Kind.EMAIL), text) == expected


# ── 전화 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("text", "expected"), [
    ("010-1234-5678", ["010-1234-5678"]),
    ("01012345678", ["01012345678"]),
    ("02-123-4567 사무실", ["02-123-4567"]),
    ("031)123-4567", ["031)123-4567"]),
    ("+82-10-1234-5678", ["+82-10-1234-5678"]),
    ("010.1234.5678", ["010.1234.5678"]),
    ("연락처는 010 1234 5678 입니다", ["010 1234 5678"]),
    ("2026-10-14 09:30", []),
    ("주문번호 0123456789012", []),
    ("금액 1,010,123,4567원", []),
    ("버전 1.010-1234-5678", []),
])
def test_phone(text, expected):
    assert _hits(_rule(Kind.PHONE), text) == expected


# ── 주민번호 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("text", "expected"), [
    ("900101-1234567", ["900101-1234567"]),
    ("9001011234567", ["9001011234567"]),
    ("외국인 900101-5234567", ["900101-5234567"]),
    ("월이 13 → 901301-1234567", []),
    ("일이 32 → 900132-1234567", []),
    ("뒷자리 9 → 900101-9234567", []),
    ("길면 안 됨 900101-12345678", []),
])
def test_ssn(text, expected):
    assert _hits(_rule(Kind.SSN), text) == expected


# ── 카드 ─────────────────────────────────────────────────────────────────────
def test_card_requires_luhn():
    assert _hits(_rule(Kind.CARD), "카드 4111 1111 1111 1111") == ["4111 1111 1111 1111"]
    assert _hits(_rule(Kind.CARD), "4111-1111-1111-1111") == ["4111-1111-1111-1111"]
    assert _hits(_rule(Kind.CARD), "송장 1234-5678-9012-3456") == []          # Luhn 실패
    assert _hits(_rule(Kind.CARD), "17자리 41111111111111111") == []


# ── 계좌 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("text", "expected"), [
    ("계좌 신한 110-123-456789 입금", ["110-123-456789"]),
    ("국민은행 123456-78-901234 로", ["123456-78-901234"]),
    ("입금계좌: 3333012345678", ["3333012345678"]),
    ("그냥 숫자 110-123-456789 는 키워드 없음", []),
    ("송장번호 110-123-456789", []),
    ("주문 번호: 110-123-456789 계좌 아님", []),
    ("은행 12-34 자리 부족", []),
    ("하나 더 주세요 110-123-456789", []),       # '하나' 단독은 키워드 아님
])
def test_account(text, expected):
    assert _hits(_rule(Kind.ACCOUNT), text) == expected


def test_account_denied_under_order_json_key():
    rule = _rule(Kind.ACCOUNT)
    assert rule.deny_json_keys.search("order_no") and rule.deny_json_keys.search("송장번호")
    assert not rule.deny_json_keys.search("memo")


# ── API 키·비밀번호 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("secret", [
    "sk-" + "a" * 40, "sk-ant-" + "b" * 40, "AIza" + "c" * 35, "ghp_" + "d" * 36, "xoxb-1234567890-abc",
    "AKIA" + "E" * 16, "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcdefghijklmnop",
])
def test_apikey_families(secret):
    assert _hits(_rule(Kind.APIKEY, 0), f"키는 {secret} 입니다") == [secret]


def test_apikey_ignores_uuid_and_base64_blobs():
    rule = _rule(Kind.APIKEY, 0)
    assert _hits(rule, "id 123e4567-e89b-12d3-a456-426614174000") == []
    assert _hits(rule, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA") == []


def test_bearer_value_only():
    tok = "abcdefghijklmnopqrstuvwxyz0123"
    assert _hits(_rule(Kind.APIKEY, 1), f"Authorization: Bearer {tok}") == [tok]


@pytest.mark.parametrize(("text", "expected"), [
    ("비밀번호는 abc123! 입니다", ["abc123!"]),
    ("password: Hunter2x", ["Hunter2x"]),
    ('api_key = "sk12345678"', ["sk12345678"]),
    ("비밀번호를 바꾸세요", []),
    ("비밀번호 재설정합니다", []),
    ("비밀번호는 다음과같이", []),      # 한글만인 값은 비밀번호가 아니다
    ("password: short", []),
])
def test_password(text, expected):
    assert _hits(_rule(Kind.PASSWORD), text) == expected


# ── 보호 스팬·사번 ────────────────────────────────────────────────────────────
def test_existing_token_and_label_are_protected():
    assert P.EXISTING_TOKEN.findall("[[PERSON_3f9a1c2d]] [[EMAIL_abcdef012345]] [차단됨:주민번호]") == [
        "[[PERSON_3f9a1c2d]]", "[[EMAIL_abcdef012345]]", "[차단됨:주민번호]"]
    assert P.EXISTING_TOKEN.findall("[[PERSON_zz]] [[PERSON_3f9a1c2]]") == []


def test_url_span():
    assert P.URL.findall("링크 https://x.com/bkim/profile) 와 http://a.b/c?d=e 끝") == ["https://x.com/bkim/profile", "http://a.b/c?d=e"]


def test_empno_rule():
    rule = P.empno_rule(r"E-\d{4}")
    assert _hits(rule, "사번 E-1001 과 E-10012 과 XE-1001") == ["E-1001"]


def test_luhn():
    assert P.luhn_ok("4111111111111111") and not P.luhn_ok("1234567890123456")


def test_name_tail_keeps_particles_and_rejects_other_hangul():
    pat = re.compile(rf"{P.NAME_LEFT}김보라{P.NAME_TAIL}")
    assert pat.search("김보라님께서는")
    assert pat.search("김보라 과장")
    assert pat.search("김보라과장님께")
    assert pat.search("김보라(bkim)")
    assert pat.search("김보라\"")
    assert not pat.search("김보라색")
    assert not pat.search("홍김보라")
    strict = re.compile(rf"{P.NAME_LEFT}보라{P.NAME_TAIL_STRICT}")
    assert strict.search("보라님과") and strict.search("보라 대리가")
    assert not strict.search("보라색") and not strict.search("보라가 왔다")
