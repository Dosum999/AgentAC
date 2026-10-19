"""도구 섹션 자동 렌더 단위 테스트.

핵심 계약: 시스템 프롬프트의 도구 인벤토리성 텍스트(도메인 탐색도구 목록·폴백 안내
도메인·도구별 규칙 문단)가 tools_config.yaml 의 ``enabled`` 스위치를 그대로 따라가고,
치환 후 마커(``<<...>>``)가 남지 않아야 한다. 도구 바인딩·검색 인덱스와 동일한 필터를
공유해 "도구는 껐는데 프롬프트엔 남는" 불일치를 구조적으로 차단하는 것이 목적이다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sysmod
from agent_arch.agents.tools.tool_search import _DOMAIN_TOOLS


def _guidance_tool_names() -> list[str]:
    """``prompt_guidance`` 를 가진 도구 이름을 레지스트리에서 뽑는다.

    "guidance 가 하나도 없을 때" 를 검증하려면 그런 도구를 **전부** 꺼야 한다.
    이름을 하드코딩하면 guidance 를 가진 도구가 새로 생길 때마다 그 테스트가
    조용히 무의미해지므로(끄지 못한 도구의 문단이 남아 empty 케이스가 아님),
    레지스트리에서 매번 뽑아 따라가게 한다.
    """
    import agent_arch.agents.tools  # noqa: F401 — @register_tool 등록 트리거

    from agent_arch.agents.tools.registry import ToolRegistry

    return [
        name
        for name, tool in ToolRegistry.get_all().items()
        if (getattr(tool, "prompt_guidance", "") or "").strip()
    ]


def _fake_get_tool_config(enabled_map: dict, exposure_map: dict | None = None):
    """이름→설정 dict 를 흉내내는 get_tool_config 대체 함수를 만든다.

    enabled_map/exposure_map 에 없는 이름은 빈 dict 를 돌려준다 — 호출부가
    ``.get("enabled", True)`` 로 기본 활성을 적용하므로 실제 동작과 일치한다.
    """
    exposure_map = exposure_map or {}

    def _get(name: str) -> dict:
        cfg: dict = {}
        if name in enabled_map:
            cfg["enabled"] = enabled_map[name]
        if name in exposure_map:
            cfg["exposure"] = exposure_map[name]
        return cfg

    return _get


@pytest.fixture
def patch_cfg(monkeypatch):
    """settings.get_tool_config 를 테스트 지정 맵으로 교체하는 헬퍼를 제공한다.

    settings 는 단일 인스턴스라, 이 속성을 바꾸면 registry/exposure/prompts 가
    모두 같은 값을 본다(실제 런타임과 동일한 공유 구조).
    """
    def _apply(enabled_map: dict, exposure_map: dict | None = None):
        # 렌더 검증은 PRODUCTION 프롬프트 대상이므로 .env 의 DEMO_MODE 와 독립시킨다.
        monkeypatch.setattr(
            sysmod.settings,
            "get_tool_config",
            _fake_get_tool_config(enabled_map, exposure_map),
        )

    return _apply


class TestDomainLabelContract:
    """_domain_label 추출 규약 — 신규 도메인이 규약을 깨면 여기서 잡힌다."""

    @pytest.mark.parametrize("domain", list(_DOMAIN_TOOLS.keys()))
    def test_intent_has_label_separator(self, domain):
        _tool, intent, _ex = _DOMAIN_TOOLS[domain]
        assert " — " in intent, f"{domain} intent 에 ' — ' 라벨 구분자가 없다"

    @pytest.mark.parametrize("domain", list(_DOMAIN_TOOLS.keys()))
    def test_label_is_nonempty(self, domain):
        _tool, intent, _ex = _DOMAIN_TOOLS[domain]
        assert sysmod._domain_label(intent).strip(), f"{domain} 라벨이 비어있다"

    def test_label_strips_parenthetical(self):
        assert sysmod._domain_label("사내 메신저(Mattermost) — 채널·DM") == "사내 메신저"


class TestDomainCapabilityBullets:
    """기능 안내 불릿 — 사용자에게 보일 문구의 회귀 방지.

    벤더 제품명("Mattermost")은 _DOMAIN_TOOLS intent 원본에서 빼두었다(사용자는
    제품명으로 부르지 않고, 답변에 노출될 이유도 없다). 되살아나면 여기서 잡힌다.
    """

    def test_vendor_name_absent_from_label(self, patch_cfg):
        patch_cfg({})
        bullets = sysmod._render_domain_capability_bullets()
        assert "**사내 메신저**" in bullets
        assert "Mattermost" not in bullets

    def test_helpful_parenthetical_kept(self, patch_cfg):
        # "(캘린더)"·"(파일)" 처럼 사용자에게 도움이 되는 괄호는 남는다
        patch_cfg({})
        bullets = sysmod._render_domain_capability_bullets()
        assert "**일정(캘린더)**" in bullets
        assert "**드라이브(파일)**" in bullets

    def test_router_hint_and_tool_name_removed(self, patch_cfg):
        patch_cfg({})
        bullets = sysmod._render_domain_capability_bullets()
        assert "※" not in bullets
        assert "find_" not in bullets


class TestDomainDiscoveryBullets:
    def test_enabled_domain_appears_disabled_omitted(self, patch_cfg):
        # 메신저 켜고, 메일 끈다
        patch_cfg({"find_messenger_tools": True, "find_mail_tools": False})
        bullets = sysmod._render_domain_discovery_bullets()
        assert "`find_messenger_tools`" in bullets
        assert "`find_mail_tools`" not in bullets

    def test_default_missing_config_is_enabled(self, patch_cfg):
        # 설정에 없는 도메인은 기본 활성(enabled=True) 이어야 한다
        patch_cfg({})
        bullets = sysmod._render_domain_discovery_bullets()
        assert "`find_hr_tools`" in bullets

    def test_bullet_format(self, patch_cfg):
        patch_cfg({})
        bullets = sysmod._render_domain_discovery_bullets()
        for line in bullets.splitlines():
            assert line.startswith("- ") and " → `find_" in line


class TestDomainLabels:
    def test_labels_follow_enabled_filter(self, patch_cfg):
        patch_cfg({"find_calendar_tools": False})
        labels = sysmod._render_domain_labels()
        assert "일정" not in labels
        # 다른 활성 도메인 라벨은 남아있다
        assert "메일" in labels

    def test_labels_joined_by_middot(self, patch_cfg):
        patch_cfg({})
        labels = sysmod._render_domain_labels()
        assert "·" in labels


class TestToolGuidance:

    def test_optimization_guidance_excluded_when_disabled(self, patch_cfg):
        patch_cfg({"optimization_solve": False})
        guidance = sysmod._render_tool_guidance()
        assert "optimization_solve" not in guidance


class TestApplyToolSections:
    def test_guidance_marker_collapsed_when_empty(self, patch_cfg):
        patch_cfg({name: False for name in _guidance_tool_names()})
        out = sysmod._apply_tool_sections("A\n\n<<TOOL_GUIDANCE>>\n\nB")
        # 마커도 앞 빈 줄도 남지 않아 A/B 가 자연스럽게 이어진다
        assert "<<TOOL_GUIDANCE>>" not in out
        assert out == "A\n\nB"


    def test_no_markers_left_in_full_prompt(self, patch_cfg):
        # optimization off (현 운영 상태) — 전체 프롬프트에 미치환 마커가 없어야 한다
        patch_cfg({"optimization_solve": False})
        prompt = sysmod.build_system_prompt()
        assert "<<" not in prompt
        assert ">>" not in prompt
        assert "optimization_solve" not in prompt

