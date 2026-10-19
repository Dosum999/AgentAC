"""어휘 — 출처별 별칭 생성 규칙, 2자·흔한 단어 strict, 영문 ID 블랙리스트, longest-first."""
from agent_arch.llm.pii.lexicon import Lexicon, build_lexicon, expand_person_aliases
from agent_arch.llm.pii.tokens import Kind


def _forms(aliases):
    return sorted((a.form, a.text) for a in aliases)


def test_directory_source_yields_full_name_and_email_only():
    assert _forms(expand_person_aliases("김보라", "bkim@example.com", source="directory")) == [
        ("email", "bkim@example.com"), ("full_name", "김보라")]


def test_requester_source_adds_given_name_username_handle():
    forms = _forms(expand_person_aliases("백지선", "jsbaek@example.com", source="requester", handle="jsbaek"))
    assert forms == [("email", "jsbaek@example.com"), ("full_name", "백지선"), ("given_name", "지선"),
                     ("handle", "jsbaek"), ("username", "jsbaek")]


def test_two_char_and_common_word_names_are_strict():
    (full,) = [a for a in expand_person_aliases("이수", None) if a.form == "full_name"]
    assert full.strict
    (common,) = [a for a in expand_person_aliases("김미래", None) if a.form == "full_name"]
    assert not common.strict                       # 3자 정식 이름은 느슨
    (mirae,) = [a for a in expand_person_aliases("미래", None) if a.form == "full_name"]
    assert mirae.strict                            # 흔한 단어
    given = [a for a in expand_person_aliases("김보라", None, source="tool") if a.form == "given_name"]
    assert given and given[0].strict and given[0].canonical == "김보라"


def test_short_or_common_local_parts_do_not_become_usernames():
    assert not [a for a in expand_person_aliases(None, "hr@example.com", source="requester") if a.form == "username"]
    assert not [a for a in expand_person_aliases(None, "admin@example.com", source="requester") if a.form == "username"]
    assert not [a for a in expand_person_aliases(None, "12345@example.com", source="requester") if a.form == "username"]
    assert [a for a in expand_person_aliases(None, "jsbaek@example.com", source="requester") if a.form == "username"]


def test_latin_name_alias():
    (a,) = [a for a in expand_person_aliases("John  Smith", None) if a.kind is Kind.PERSON]
    assert a.form == "latin_name" and a.text == "John Smith"


def test_lexicon_pattern_longest_first_and_lookup():
    lex = build_lexicon([("김보", "a@x.kr"), ("김보라", "b@x.kr")])
    m = lex.pattern.search("김보라님 안녕")
    assert m.group(0) == "김보라" and lex.lookup(m.lastgroup, m.group(0)).canonical == "김보라"


def test_lookup_is_case_insensitive_for_ascii_forms():
    lex = build_lexicon(requester={"name": "백지선", "email": "JsBaek@example.com", "username": "jsbaek"})
    m = lex.pattern.search("문의는 JSBAEK@example.com 로")
    assert m and lex.lookup(m.lastgroup, m.group(0)).kind is Kind.EMAIL
    m = lex.pattern.search("계정 JsBaek 확인")
    assert m and lex.lookup(m.lastgroup, m.group(0)).form == "username"


def test_richer_source_wins_for_same_text():
    lex = build_lexicon([("김보라", "bkim@example.com")], tool_persons=[("김보라", "bkim@example.com")])
    full = [a for a in lex.aliases if a.form == "full_name" and a.text == "김보라"]
    assert len(full) == 1 and full[0].source == "tool"
    assert any(a.form == "given_name" for a in lex.aliases)


def test_empty_lexicon_and_company_names():
    assert not Lexicon.empty() and Lexicon.empty().pattern is None
    lex = build_lexicon(company_names=["(주)회사", " ", "김보라식품"])
    assert lex.company_names == {"(주)회사", "김보라식품"}
    assert lex.company_pattern.search("김보라식품(주)") and not lex.pattern


def test_empno_aliases():
    lex = build_lexicon(empnos=["E-1001", ""])
    m = lex.pattern.search("사번 E-1001 입니다")
    assert m and lex.lookup(m.lastgroup, m.group(0)).kind is Kind.EMPNO
