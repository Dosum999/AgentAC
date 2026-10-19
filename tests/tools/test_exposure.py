"""도구 노출(Exposure) 정책 단위 테스트.

1단계 핵심 계약: 전 도구 기본 CORE 상태에서 plan_tools 결과가
리팩토링 전 공식(static + dynamic 권한필터)과 동일해야 한다.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from agent_arch.agents.tools import exposure as exp
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.exposure import Exposure, plan_tools, resolve_exposure


class FakeTool(BaseTool):
    """테스트용 최소 도구. 이름·exposure를 인스턴스 단위로 지정."""

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

    async def _demo_execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=None)


@pytest.fixture(autouse=True)
def _reset_warn_state():
    exp._cap_warned = False
    exp._fallback_warned.clear()
    yield


@pytest.fixture
def no_yaml_override(monkeypatch):
    """tools_config.yaml에 exposure 설정이 없는 상태를 모사."""
    monkeypatch.setattr(exp.settings, "get_tool_config", lambda name: {})


class TestResolveExposure:
    def test_class_default_is_core(self, no_yaml_override):
        assert resolve_exposure(FakeTool("t")) is Exposure.CORE

    def test_instance_attribute_respected(self, no_yaml_override):
        assert resolve_exposure(FakeTool("t", exposure="retrieved")) is Exposure.RETRIEVED

    def test_yaml_overrides_class_attribute(self, monkeypatch):
        monkeypatch.setattr(
            exp.settings, "get_tool_config", lambda name: {"exposure": "retrieved"}
        )
        # 클래스/인스턴스는 core지만 yaml이 이긴다
        assert resolve_exposure(FakeTool("t", exposure="core")) is Exposure.RETRIEVED

    def test_invalid_value_falls_back_to_core(self, no_yaml_override, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_exposure(FakeTool("t", exposure="oops"))
        assert result is Exposure.CORE
        assert "알 수 없는 exposure" in caplog.text

    def test_discoverable_unimplemented_falls_back(self, no_yaml_override, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_exposure(FakeTool("t", exposure="discoverable"))
        assert result is Exposure.CORE
        assert "미구현" in caplog.text

    def test_fallback_warning_logged_once_per_tool(self, no_yaml_override, caplog):
        tool = FakeTool("t", exposure="oops")
        with caplog.at_level(logging.WARNING):
            resolve_exposure(tool)
            resolve_exposure(tool)
        assert caplog.text.count("알 수 없는 exposure") == 1


class TestPlanTools:
    @pytest.fixture
    def allow_all(self, monkeypatch):
        monkeypatch.setattr(exp, "is_allowed", lambda name, profile: True)

    def test_equivalent_to_legacy_formula(self, monkeypatch):
        """1단계 동일성 계약: 리팩토링 전 공식과 결과 일치."""
        core = [FakeTool("a"), FakeTool("b"), FakeTool("c")]
        retrieved = [FakeTool("x"), FakeTool("y")]
        denied = {"b", "y"}
        fake_gate = lambda name, profile: name not in denied
        monkeypatch.setattr(exp, "is_allowed", fake_gate)

        # 리팩토링 전: all_tools = static + dynamic → 권한 필터
        legacy = [t for t in core + retrieved if fake_gate(t.name, {})]
        planned = plan_tools(core, retrieved, profile={})

        assert [t.name for t in planned] == [t.name for t in legacy] == ["a", "c", "x"]

    def test_order_core_first_then_retrieved(self, allow_all):
        planned = plan_tools([FakeTool("a")], [FakeTool("x")], profile={})
        assert [t.name for t in planned] == ["a", "x"]

    def test_dedupe_prefers_core_instance(self, allow_all):
        core_inst = FakeTool("same")
        retrieved_inst = FakeTool("same")
        planned = plan_tools([core_inst], [retrieved_inst], profile={})
        assert len(planned) == 1
        assert planned[0] is core_inst

    def test_profile_none_denies_everything(self):
        # 실제 is_allowed 사용 — profile=None이면 fail-safe deny
        planned = plan_tools([FakeTool("a")], [FakeTool("x")], profile=None)
        assert planned == []


class TestCoreCapWarning:
    def test_warns_over_cap_once(self, caplog):
        with caplog.at_level(logging.WARNING):
            exp._warn_core_cap(exp._CORE_CAP + 1)
            exp._warn_core_cap(exp._CORE_CAP + 5)
        assert caplog.text.count("상한") == 1

    def test_no_warning_under_cap(self, caplog):
        with caplog.at_level(logging.WARNING):
            exp._warn_core_cap(exp._CORE_CAP)
        assert "상한" not in caplog.text
