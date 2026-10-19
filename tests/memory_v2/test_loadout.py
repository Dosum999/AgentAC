"""기억 → 흐름 힌트(결정론): 별칭 종류 → 도메인, 라우터 힌트, 도메인 도구 선택."""
from __future__ import annotations

from types import SimpleNamespace

from agent_arch.services.memory_v2 import loadout as L

KNOWN = ["talk", "mail", "calendar", "drive", "hr", "eapproval", "logistics", "memory"]


def _recall(*kinds):
    return SimpleNamespace(aliases=[{"payload": {"canonical_kind": k, "canonical_id": "x"}} for k in kinds])


def test_alias_domains_maps_known_kinds_and_dedupes():
    assert L.alias_domains(_recall("mattermost_channel", "mattermost_channel", "calendar_event"), KNOWN) == ["talk", "calendar"]
    assert L.alias_domains(_recall("person", "company", "ui_element"), KNOWN) == []      # 애매한 종류는 힌트 안 함
    assert L.alias_domains(_recall("scms_order"), KNOWN) == []                            # 등록 도메인이 아니면 무시
    assert L.alias_domains(SimpleNamespace(aliases=[]), KNOWN) == []
    assert L.alias_domains(None, KNOWN) == []


def test_hint_domain_only_when_unambiguous():
    assert L.hint_domain(_recall("mattermost_channel"), KNOWN) == "talk"
    assert L.hint_domain(_recall("mattermost_channel", "mail_folder"), KNOWN) is None
    assert L.hint_domain(_recall("person"), KNOWN) is None


def test_select_tools_for_domains_filters_and_caps():
    tools = [SimpleNamespace(name=n) for n in (
        "find_messenger_tools", "mattermost_skill_get_channel_messages", "mattermost_skill_find_channels",
        "mattermost_post_message", "mail_list_messages", "calculator",
    )]
    domain_of = lambda n: "talk" if n.startswith("mattermost_") else ("mail" if n.startswith("mail_") else None)
    out = L.select_tools_for_domains(["talk"], tools, domain_of, exclude=lambda n: n == "mattermost_post_message")
    assert [t.name for t in out] == ["mattermost_skill_find_channels", "mattermost_skill_get_channel_messages"]
    assert L.select_tools_for_domains([], tools, domain_of) == []
    capped = L.select_tools_for_domains(["talk"], tools, domain_of, cap=1)
    assert len(capped) == 1
