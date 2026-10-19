"""토큰 발급 — 결정성(같은 범위·같은 값→같은 토큰), 정규화, 8자 충돌 시 12자 일괄 발급(순서 독립)."""
import logging

import pytest

from agent_arch.llm.pii.tokens import (
    BLOCKABLE,
    BLOCKED_LABEL,
    RESTORABLE,
    UNKNOWN_LABEL,
    Kind,
    TokenIssuer,
    canonical_key,
    normalize,
    render,
    scope_id,
)


def test_same_scope_same_value_same_token():
    a = TokenIssuer("secret", scope_id("conv-1"))
    b = TokenIssuer("secret", scope_id("conv-1"))
    assert a.digest(Kind.PERSON, "김보라") == b.digest(Kind.PERSON, "김보라")
    assert a.assign([(Kind.PERSON, "김보라")]) == b.assign([(Kind.PERSON, "김보라")])


def test_different_scope_or_secret_differs():
    base = TokenIssuer("secret", scope_id("conv-1")).digest(Kind.PERSON, "김보라")
    assert TokenIssuer("secret", scope_id("conv-2")).digest(Kind.PERSON, "김보라") != base
    assert TokenIssuer("other", scope_id("conv-1")).digest(Kind.PERSON, "김보라") != base


def test_scope_id_falls_back_to_user_then_anon():
    assert scope_id("c1", "u1") == "conv:c1"
    assert scope_id(None, "u1") == "user:u1"
    assert scope_id(None, None) == "anon"


def test_empty_secret_rejected():
    with pytest.raises(ValueError):
        TokenIssuer("", "conv:1")


@pytest.mark.parametrize(("kind", "raw", "expected"), [
    (Kind.EMAIL, " BKim@example.com ", "bkim@example.com"),
    (Kind.PHONE, "010-1234-5678", "01012345678"),
    (Kind.PHONE, "010 1234 5678", "01012345678"),
    (Kind.HANDLE, "@JsBaek", "jsbaek"),
    (Kind.PERSON, "김 보라", "김보라"),
    (Kind.EMPNO, " E-1001 ", "E-1001"),
])
def test_normalize(kind, raw, expected):
    assert normalize(kind, raw) == expected


def test_normalized_variants_share_token():
    iss = TokenIssuer("s", "conv:1")
    pairs = [(Kind.PHONE, normalize(Kind.PHONE, "010-1234-5678")), (Kind.PHONE, normalize(Kind.PHONE, "01012345678"))]
    assert len(set(pairs)) == 1
    assert len(iss.assign(pairs)) == 1


def test_assign_defaults_to_8_hex():
    out = TokenIssuer("s", "conv:1").assign([(Kind.PERSON, "김보라"), (Kind.EMAIL, "a@b.kr")])
    assert all(len(h) == 8 for h in out.values())


def test_collision_group_all_get_12_hex_and_is_order_independent(monkeypatch, caplog):
    iss = TokenIssuer("s", "conv:1")
    full = {"a": "deadbeef" + "aaaa" + "0" * 52, "b": "deadbeef" + "bbbb" + "0" * 52, "c": "cafebabe" + "cccc" + "0" * 52}
    monkeypatch.setattr(iss, "digest", lambda kind, v: full[v])
    with caplog.at_level(logging.WARNING):
        one = iss.assign([(Kind.PERSON, "a"), (Kind.PERSON, "b"), (Kind.PERSON, "c")])
        two = iss.assign([(Kind.PERSON, "c"), (Kind.PERSON, "b"), (Kind.PERSON, "a")])
    assert one == two
    assert one[(Kind.PERSON, "a")] == "deadbeefaaaa" and one[(Kind.PERSON, "b")] == "deadbeefbbbb"
    assert one[(Kind.PERSON, "c")] == "cafebabe"
    assert sum("토큰 8자 충돌" in r.message for r in caplog.records) == 2


def test_render_and_canonical_key():
    assert render(Kind.PERSON, "3f9a1c2d") == "[[PERSON_3f9a1c2d]]"
    assert canonical_key(Kind.PERSON, "3F9A1C2D") == "PERSON_3f9a1c2d"
    assert canonical_key("person", "ABCDEF012345") == "PERSON_abcdef012345"


def test_kind_partitions_and_labels():
    assert RESTORABLE | BLOCKABLE == frozenset(Kind)
    assert not (RESTORABLE & BLOCKABLE)
    assert set(BLOCKED_LABEL) == set(BLOCKABLE)
    assert set(UNKNOWN_LABEL) == set(RESTORABLE)
    assert Kind.PERSON.restorable and not Kind.SSN.restorable
