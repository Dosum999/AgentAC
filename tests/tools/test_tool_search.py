"""tool_search 메타도구 + 발견-바인딩 흐름 단위 테스트."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent_arch.agents.permissions import DenyReason, GateDecision
from agent_arch.agents.tools import tool_search as ts
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.agents.tools.tool_search import drain_discovered_tools
from agent_arch.services.api_registry.models import ExternalToolDef


class FakeTool(BaseTool):
    def __init__(self, name: str):
        self._name = name
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


def _registry_def(name: str, domain: str = "logistics") -> ExternalToolDef:
    return ExternalToolDef(
        tool_name=name, description=f"{name} 설명", origin="registry",
        domain=domain, is_external=False, enabled=True, parameters_schema={},
    )


def _dynamic_def(name: str, domain: str = "logistics") -> ExternalToolDef:
    return ExternalToolDef(
        tool_name=name, description=f"{name} 설명", is_external=True, enabled=True,
        external_service=domain, http_method="GET", endpoint_path="/x",
        parameters_schema={}, domain=domain,
    )


@pytest.fixture
def tool():
    return ToolRegistry.get("find_eapproval_tools")


@pytest.fixture(autouse=True)
def _drain_residue():
    drain_discovered_tools()
    yield
    drain_discovered_tools()


@pytest.fixture
def fake_search(monkeypatch):
    """api_indexer.search를 시나리오 주입 가능하게 모킹."""
    holder = {"hits": [], "kwargs": None}

    async def search(query, top_k=7, service_filter=None, include_registry=False, domain=None):
        holder["kwargs"] = dict(query=query, top_k=top_k, include_registry=include_registry, domain=domain)
        return holder["hits"]

    from agent_arch.services.api_registry import api_indexer
    monkeypatch.setattr(api_indexer, "search", search)
    return holder


@pytest.fixture
def allow_all(monkeypatch):
    monkeypatch.setattr(ts, "evaluate", lambda name, profile: GateDecision(allowed=True))


def _deny(reason: DenyReason, service: str = "eapproval"):
    """게이트를 특정 사유로 고정하는 monkeypatch 값."""
    return lambda name, profile: GateDecision(allowed=False, reason=reason, service=service)


class TestToolSearchExecute:
    def test_eapproval_domain_card_mentions_core_directory_approver_labels(self):
        _, summary, _ = ts._DOMAIN_TOOLS["eapproval"]
        assert "search_colleagues" in summary
        assert "이름 <email>" in summary

    def test_leave_domain_cards_have_non_overlapping_ownership(self):
        _, hr_summary, hr_examples = ts._DOMAIN_TOOLS["hr"]
        _, eapproval_summary, eapproval_examples = ts._DOMAIN_TOOLS["eapproval"]

        assert "배정된 휴가 잔여량" in hr_summary
        assert "회사에 정의된 휴가 제도" in hr_summary
        assert "신청 가능한 휴가 종류" in eapproval_summary
        assert "휴가 신청 문서" in eapproval_summary
        assert all("신청 내역" not in example for example in hr_examples)
        assert any("신청 가능" in example for example in eapproval_examples)

    def test_drive_domain_card_advertises_reading_file_contents(self):
        """드라이브 도구는 전부 retrieved 라 이 카드를 보고 LLM 이 호출해야 활성화된다.

        본문 읽기·내용 검색이 카드에서 빠지면 '내용 뭐야'류 질문이 드라이브로 오지 않고,
        에이전트가 파일명만 보고 본문을 지어낸다(2026-08-12 운영 실측). 문구가 조용히
        되돌아가는 것을 막는 회귀 테스트다.
        """
        _, summary, examples = ts._DOMAIN_TOOLS["drive"]
        assert "본문" in summary and "내용 검색" in summary
        joined = " ".join(examples)
        assert "내용" in joined and "요약" in joined

    @pytest.mark.asyncio
    async def test_registers_as_core_public(self, tool):
        assert tool is not None
        assert tool.exposure == "core"
        assert tool.permission_required.mode.value == "public"
        assert ToolRegistry.get("find_eapproval_tools") is tool

    @pytest.mark.asyncio
    async def test_resolves_registry_hit_and_enqueues(self, tool, fake_search, allow_all, monkeypatch):
        fake = FakeTool("eapproval_list_documents")
        monkeypatch.setattr(
            ts.ToolRegistry, "get",
            classmethod(lambda cls, n: fake if n == fake.name else None),
        )
        fake_search["hits"] = [_registry_def(fake.name, domain="eapproval")]

        result = await tool._real_execute(query="미결 결재 문서")
        assert result.success and result.output["found"] == 1
        assert result.output["tools"][0]["name"] == fake.name
        assert drain_discovered_tools() == [fake]

    @pytest.mark.asyncio
    async def test_resolves_dynamic_hit(self, tool, fake_search, allow_all):
        fake_search["hits"] = [_dynamic_def("eapproval_dynamic_list_documents", domain="eapproval")]
        result = await tool._real_execute(query="전자결재 문서 목록")
        assert result.output["found"] == 1
        drained = drain_discovered_tools()
        assert drained[0].name == "eapproval_dynamic_list_documents"

    @pytest.mark.asyncio
    async def test_permission_denied_tools_filtered(self, tool, fake_search, monkeypatch):
        fake_search["hits"] = [_dynamic_def("secret_tool")]
        monkeypatch.setattr(ts, "evaluate", _deny(DenyReason.NO_SERVICE_ACCESS))
        result = await tool._real_execute(query="비밀")
        assert result.success and result.output["found"] == 0
        assert drain_discovered_tools() == []

    @pytest.mark.asyncio
    async def test_fixed_domain_passed_to_search(self, tool, fake_search, allow_all):
        await tool._real_execute(query="미결 결재")
        assert fake_search["kwargs"]["domain"] == "eapproval"

    @pytest.mark.asyncio
    async def test_excludes_itself(self, tool, fake_search, allow_all):
        fake_search["hits"] = [_registry_def("find_eapproval_tools", domain="eapproval")]
        result = await tool._real_execute(query="도구 검색")
        assert result.output["found"] == 0

    @pytest.mark.asyncio
    async def test_top_k_capped(self, tool, fake_search, allow_all):
        await tool._real_execute(query="x", top_k=999)
        assert fake_search["kwargs"]["top_k"] == ts._MAX_TOP_K


class TestEmptyResultDistinguishesCause:
    """found=0 일 때 "권한이 없음"과 "기능이 없음"을 구별해 안내하는지.

    구별하지 않으면 권한만 받으면 되는 사용자에게 "기능이 없습니다"가 나가고,
    사용자는 관리자에게 권한을 요청하면 된다는 사실을 알 수 없다.
    """

    @pytest.mark.asyncio
    async def test_all_denied_by_permission_says_permission_blocked(
        self, tool, fake_search, monkeypatch
    ):
        fake_search["hits"] = [_dynamic_def("eapproval_secret", domain="eapproval")]
        monkeypatch.setattr(ts, "evaluate", _deny(DenyReason.NO_SERVICE_ACCESS))

        out = (await tool._real_execute(query="결재 문서")).output
        assert out["found"] == 0
        assert out["reason"] == "permission_blocked"
        # "기능이 없음을 안내하라"는 지시가 남아 있으면 LLM 이 원인을 오인한다.
        assert "기능이 없음을 사용자에게 안내" not in out["message"]
        assert "권한" in out["message"]
        # 한국어 기능명은 LLM 이 맥락에서 만든다 — 서버가 서비스 라벨을 내려주지 않는다.
        assert "services" not in out

    @pytest.mark.asyncio
    async def test_blocked_message_does_not_claim_the_feature_exists(
        self, tool, fake_search, monkeypatch
    ):
        """권한 차단 안내가 기능의 존재를 단정하면 안 된다.

        indexer.search 에 관련성 임계값이 없어 hits 는 "맞는 도구"가 아니라 그
        도메인의 top_k 최근접일 뿐이다. 따라서 "기능은 존재하지만 권한이 없다"고
        말하면, 아예 없는 기능에 대해서도 권한 문제라고 거짓 안내하게 된다.
        """
        fake_search["hits"] = [_dynamic_def("eapproval_unrelated", domain="eapproval")]
        monkeypatch.setattr(ts, "evaluate", _deny(DenyReason.NO_SERVICE_ACCESS))

        msg = (await tool._real_execute(query="전화 연결해줘")).output["message"]
        assert "기능은 존재" not in msg
        # 확인 불가임을 명시하고, 양방향 단정을 모두 금지해야 한다.
        assert "확인할 수 없" in msg
        assert "단정하지" in msg

    @pytest.mark.asyncio
    async def test_search_miss_keeps_feature_absent_message(self, tool, fake_search, allow_all):
        """검색 미스(권한 차단 0건)는 기존 '기능이 없음' 안내를 유지한다."""
        fake_search["hits"] = []
        out = (await tool._real_execute(query="존재하지 않는 기능")).output
        assert out["found"] == 0
        assert "reason" not in out
        assert "기능이 없음을 사용자에게 안내" in out["message"]

    @pytest.mark.asyncio
    async def test_meta_missing_is_not_reported_as_permission_issue(
        self, tool, fake_search, monkeypatch
    ):
        """NO_META 는 등록 누락(버그)이므로 '권한 요청' 안내 대상이 아니다."""
        fake_search["hits"] = [_dynamic_def("unmapped_tool", domain="eapproval")]
        monkeypatch.setattr(ts, "evaluate", _deny(DenyReason.NO_META, service=""))

        out = (await tool._real_execute(query="결재")).output
        assert out["found"] == 0
        assert "reason" not in out

    @pytest.mark.asyncio
    async def test_partial_denial_keeps_success_shape(self, tool, fake_search, monkeypatch):
        """일부만 차단되면 찾은 도구로 진행 — LLM 에 차단 사실을 알리지 않는다."""
        allowed_name = "eapproval_list_documents"

        def selective(name, profile):
            if name == allowed_name:
                return GateDecision(allowed=True, service="eapproval")
            return GateDecision(
                allowed=False, reason=DenyReason.NO_SERVICE_ACCESS, service="eapproval"
            )

        monkeypatch.setattr(ts, "evaluate", selective)
        fake_search["hits"] = [
            _dynamic_def(allowed_name, domain="eapproval"),
            _dynamic_def("eapproval_secret", domain="eapproval"),
        ]

        out = (await tool._real_execute(query="결재 문서")).output
        assert out["found"] == 1
        assert "reason" not in out
        assert [t["name"] for t in out["tools"]] == [allowed_name]
        assert [t.name for t in drain_discovered_tools()] == [allowed_name]


class TestDrainSemantics:
    def test_drain_empties_queue(self):
        ts._enqueue([FakeTool("a")])
        assert [t.name for t in drain_discovered_tools()] == ["a"]
        assert drain_discovered_tools() == []

    def test_enqueue_accumulates(self):
        ts._enqueue([FakeTool("a")])
        ts._enqueue([FakeTool("b")])
        assert [t.name for t in drain_discovered_tools()] == ["a", "b"]
