"""가이드 발 링크 차단 정책 — 근거 있는 주소만 통과한다.

배경(2026-09-21 신고): "에이전트 사용방법 안내해줘" 답변 말미에 ``더 자세한 내용은 아래 가이드를
참고해 주세요.`` + 가이드 문서 링크 3개가 붙었는데 **주소가 전부 지어낸 것**이었다(가이드 페이지의
진짜 주소는 LLM 에게 가지 않는다). 그래서 "모델이 실제로 본 텍스트에 있던 주소"만 남긴다.
"""
import json

import pytest

from agent_arch.agents.react import link_policy as LP
from agent_arch.core.config import settings

_FORM = "https://example.com/main/support/inquiry/write"
_FAKE = "https://example.com/user-guide/messenger-channel-settings"   # 실측 환각 — 진짜와 같은 호스트다


def _guide_result(content: str) -> dict:
    groups = [{"category": "General", "category_top_score": 0.8,
               "chunks": [{"title": "고객 문의", "content": content, "score": 0.8}]}]
    return {"tool_call_id": "t1", "name": LP.GUIDE_TOOL, "content": json.dumps(groups, ensure_ascii=False)}


class TestNormalize:
    def test_normalizes_case_trailing_slash_fragment_and_punctuation(self):
        n = LP.normalize_url
        assert n("HTTPS://example.com/Main/Download/") == "https://example.com/Main/Download"
        assert n("https://example.com/a#sec") == n("https://example.com/a")
        assert n("https://example.com/a.") == "https://example.com/a"
        assert n("https://example.com/a?b=1") == "https://example.com/a?b=1"      # 쿼리는 의미가 있다

    def test_rejects_non_http_and_placeholder_hosts(self):
        assert LP.normalize_url("mailto:a@b.com") == ""
        assert LP.normalize_url("https://단말기IP/upload") == ""            # 호스트에 점이 없다
        assert LP.normalize_url(None) == ""


class TestHarvest:
    def test_picks_markdown_and_bare_urls(self):
        got = LP.harvest_urls(f"문의는 [문의 폼]({_FORM}) 또는 {_FORM}/etc 에서.")
        assert got == {_FORM, _FORM + "/etc"}

    def test_empty_for_textless_input(self):
        assert LP.harvest_urls("") == set() and LP.harvest_urls("링크 없음") == set()


class TestPermits:
    def test_inactive_turn_permits_everything(self):
        """가이드 검색을 안 쓴 턴은 이번 변경 전과 완전히 같이 동작한다."""
        policy = LP.policy_for([{"name": "mattermost_search_knowledge", "content": "{}"}])
        assert policy.active is False
        assert policy.permits(_FAKE) is True

    def test_guide_turn_keeps_sourced_and_drops_fabricated(self):
        policy = LP.policy_for([_guide_result(f"회원 문의 폼( {_FORM} )을 이용하세요.")])
        assert policy.active is True
        assert policy.permits(_FORM) is True          # 청크 본문에 실제로 있던 실무 주소
        assert policy.permits(_FAKE) is False         # 같은 호스트지만 근거가 없다

    def test_static_links_are_allowed_by_exact_match(self, monkeypatch):
        monkeypatch.setattr(settings, "dmp_portal_url", "https://example.com", raising=False)
        policy = LP.policy_for([_guide_result("내용")])
        assert policy.permits("https://example.com/main/download") is True   # 프롬프트와 같은 함수에서 온다
        assert policy.permits(LP.APPSTORE_URL) is True
        assert policy.permits("https://example.com/main/") is False          # 호스트로 열지 않는다

    def test_messenger_host_is_allowed_by_host(self, monkeypatch):
        """`{base}/api/v4/files/{id}` 는 코드가 조립해 도구 결과에 없다 — 호스트로 연다."""
        monkeypatch.setattr(settings, "mattermost_base_url", "https://chat.example.com", raising=False)
        policy = LP.policy_for([_guide_result("내용")])
        assert policy.permits("https://chat.example.com/api/v4/files/abc") is True

    def test_user_input_and_prompt_text_count_as_evidence(self):
        policy = LP.policy_for([_guide_result("내용")], texts=[f"이 링크 뭐야? {_FAKE}"])
        assert policy.permits(_FAKE) is True

    def test_non_http_targets_are_not_this_guards_business(self):
        policy = LP.policy_for([_guide_result("내용")])
        assert policy.permits("mailto:user@example.com") is True


class TestSnapshot:
    def test_round_trip_merges_in_place(self):
        src = LP.policy_for([_guide_result(f"폼 {_FORM}")])
        dst = LP.TurnLinkPolicy()
        permits = dst.permits                 # 정화기에 먼저 바인딩된 콜백
        assert permits(_FORM) is True         # 아직 정책이 없다 → 전부 통과
        dst.update_from(src.dumps())
        assert dst.active is True and permits(_FORM) is True and permits(_FAKE) is False

    def test_broken_payload_is_ignored(self):
        policy = LP.TurnLinkPolicy()
        policy.update_from("{not json")
        policy.update_from(json.dumps([1, 2]))
        assert policy.active is False and policy.allowed == set()
