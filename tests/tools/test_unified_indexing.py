"""통합 도구 인덱싱(2단계) 단위 테스트.

- ExternalToolDef origin 확장 (registry는 http 필드 불필요, dynamic은 필수)
- registry_loader 변환 규칙 (이름+설명만, only_names 필터)
- exposure: 플래그 off 시 retrieved → core 승격 폴백
- sticky: 직전 턴 도구 복원 (registry → 동적 캐시 순 resolve)
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from agent_arch.agents.tools import exposure as exp
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.services.api_registry.models import ExternalToolDef


class FakeTool(BaseTool):
    def __init__(self, name: str, exposure: str = "core"):
        self._name = name
        self.exposure = exposure
        super().__init__()

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return f"{self._name} 설명"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def get_required_params(self) -> List[str]:
        return []


class TestExternalToolDefOrigin:
    def test_registry_origin_allows_missing_http_fields(self):
        td = ExternalToolDef(
            tool_name="mail_list_messages", description="메일 목록",
            origin="registry", is_external=False, enabled=True, parameters_schema={},
        )
        assert td.origin == "registry" and td.http_method is None

    def test_dynamic_api_requires_http_fields(self):
        with pytest.raises(ValueError, match="http_method/endpoint_path 누락"):
            ExternalToolDef(
                tool_name="t", description="d",
                is_external=True, enabled=True, parameters_schema={},
            )

    def test_default_origin_is_dynamic_api(self):
        td = ExternalToolDef(
            tool_name="t", description="d", is_external=True, enabled=True,
            http_method="GET", endpoint_path="/x", parameters_schema={},
        )
        assert td.origin == "dynamic_api"


class TestRegistryLoader:
    @pytest.fixture
    def fake_registry(self, monkeypatch):
        tools = [FakeTool("a"), FakeTool("b")]
        from agent_arch.services.api_registry import registry_loader

        monkeypatch.setattr(
            registry_loader.ToolRegistry, "get_enabled_tools", classmethod(lambda cls: tools)
        )
        return tools

    def test_converts_name_and_description_only(self, fake_registry):
        from agent_arch.services.api_registry.registry_loader import defs_from_registry

        defs = defs_from_registry()
        assert [d.tool_name for d in defs] == ["a", "b"]
        assert all(d.origin == "registry" for d in defs)
        assert all(d.http_method is None and d.parameters_schema == {} for d in defs)

    def test_only_names_filter(self, fake_registry):
        from agent_arch.services.api_registry.registry_loader import defs_from_registry

        defs = defs_from_registry(only_names=["b"])
        assert [d.tool_name for d in defs] == ["b"]


class TestFlagOffPromotion:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        exp._cap_warned = False
        exp._flag_off_warned = False
        exp._fallback_warned.clear()
        monkeypatch.setattr(exp.settings, "get_tool_config", lambda name: {})
        yield

    @pytest.fixture
    def fake_registry(self, monkeypatch):
        tools = [FakeTool("core_t", "core"), FakeTool("ret_t", "retrieved")]
        monkeypatch.setattr(
            exp.ToolRegistry, "get_enabled_tools", classmethod(lambda cls: tools)
        )

    def test_flag_off_promotes_retrieved_to_core(self, fake_registry, monkeypatch, caplog):
        monkeypatch.setattr(exp.settings, "unified_tool_indexing", False)
        with caplog.at_level(logging.WARNING):
            names = [t.name for t in exp.core_tools()]
        assert names == ["core_t", "ret_t"]  # 승격 유지 — 도구 소실 없음
        assert "core로 승격" in caplog.text

    def test_flag_on_excludes_retrieved(self, fake_registry, monkeypatch):
        monkeypatch.setattr(exp.settings, "unified_tool_indexing", True)
        names = [t.name for t in exp.core_tools()]
        assert names == ["core_t"]


class TestStickyPrevTurnTools:
    """orchestrator._sticky_prev_turn_tools — registry 우선, 동적 캐시 폴백."""

    @pytest.fixture
    def sticky(self):
        from agent_arch.agents.orchestrator import AgentOrchestrator

        return AgentOrchestrator._sticky_prev_turn_tools

    def _tool_msg(self, name):
        return SimpleNamespace(role="tool", name=name)

    def test_resolves_registry_tool(self, sticky, monkeypatch):
        import agent_arch.agents.orchestrator as orch

        # 채널 전용 도메인(mespoc)은 관문 바인딩을 요구하므로, resolve 경로 자체는
        # 도메인 제약이 없는 이름으로 확인한다 (채널 가드는 test_channel_scoped_sticky).
        registry_tool = FakeTool("hr_today_attendance", "retrieved")
        monkeypatch.setattr(
            orch.ToolRegistry, "get",
            classmethod(lambda cls, n: registry_tool if n == registry_tool.name else None),
        )
        recovered = sticky([self._tool_msg("hr_today_attendance")], set())
        assert recovered == [registry_tool]

    def test_falls_back_to_dynamic_cache(self, sticky, monkeypatch):
        import agent_arch.agents.orchestrator as orch

        td = ExternalToolDef(
            tool_name="dyn_tool", description="d", is_external=True, enabled=True,
            http_method="GET", endpoint_path="/x", parameters_schema={},
        )
        monkeypatch.setattr(orch.ToolRegistry, "get", classmethod(lambda cls, n: None))
        monkeypatch.setattr(orch.tool_loader, "get_by_name", lambda n: td if n == "dyn_tool" else None)
        recovered = sticky([self._tool_msg("dyn_tool")], set())
        assert len(recovered) == 1 and recovered[0].name == "dyn_tool"

    def test_skips_known_and_unresolvable(self, sticky, monkeypatch):
        import agent_arch.agents.orchestrator as orch

        monkeypatch.setattr(orch.ToolRegistry, "get", classmethod(lambda cls, n: None))
        monkeypatch.setattr(orch.tool_loader, "get_by_name", lambda n: None)
        msgs = [self._tool_msg("already_bound"), self._tool_msg("ghost")]
        assert sticky(msgs, {"already_bound"}) == []

    def test_ignores_non_tool_messages(self, sticky):
        msgs = [SimpleNamespace(role="user", name=None), SimpleNamespace(role="assistant")]
        assert sticky(msgs, set()) == []

    def test_skips_disabled_find_gateway(self, sticky, monkeypatch):
        """비활성(enabled:false) find_* 관문은 sticky 로 되살리지 않는다.

        전역 OFF 로 격리한 도메인이 과거 대화의 sticky 로 되살아나면 그 관문을 통해
        다시 발견된다 — 관문만 tools_config enabled 를 존중하고, 같은 도메인의 일반
        도구는 후속턴 연속성을 위해 종전대로 복원한다.
        """
        import agent_arch.agents.orchestrator as orch

        gateway = FakeTool("find_logistics_tools", "core")
        normal = FakeTool("logistics_get_stock", "retrieved")
        monkeypatch.setattr(
            orch.ToolRegistry, "get",
            classmethod(lambda cls, n: {t.name: t for t in (gateway, normal)}.get(n)),
        )
        monkeypatch.setattr(
            orch.settings, "get_tool_config",
            lambda name: {"enabled": False} if name == "find_logistics_tools" else {},
        )
        recovered = sticky(
            [self._tool_msg("find_logistics_tools"), self._tool_msg("logistics_get_stock")],
            set(),
        )
        # 관문은 스킵, 일반 도구는 종전대로 복원
        assert recovered == [normal]

    def test_channel_scoped_tool_needs_its_gateway(self, sticky, monkeypatch):
        """채널 전용 도메인(mespoc)은 그 채널의 관문이 이 턴에 바인딩됐을 때만 복원된다.

        관문 전역 OFF 로 발견은 막혀 있지만 sticky 는 이름을 그대로 되살리므로, MES
        앱 채널 대화가 일반 채팅으로 이어지면 채널 밖에서 되살아났다.
        """
        import agent_arch.agents.orchestrator as orch

        monkeypatch.setitem(orch._CHANNEL_SCOPED_PREFIXES, "mespoc_", "find_mespoc_tools")
        normal = FakeTool("mespoc_skill_lot_progress", "retrieved")
        monkeypatch.setattr(
            orch.ToolRegistry, "get",
            classmethod(lambda cls, n: normal if n == normal.name else None),
        )
        msgs = [self._tool_msg("mespoc_skill_lot_progress")]

        assert sticky(msgs, set()) == []                          # 채널 밖 — 차단
        assert sticky(msgs, {"find_mespoc_tools"}) == [normal]     # 채널 안 — 유지
