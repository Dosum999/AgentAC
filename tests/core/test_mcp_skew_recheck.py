"""목록 불일치는 **기동을 막지 않는다** — 짧게 다시 보고 끝낸다.

경위 (2026-09-15, 하루에 두 번 물린 곳):

1. 배포 순서가 어긋나 백엔드가 MCP 보다 먼저 뜨면 **옛 목록을 성공적으로** 받는다. 조회가
   실패한 게 아니라 재시도가 돌지 않고, 모델이 새 도구를 부르면 ``tool NOT FOUND`` 가 난다.
   → 불일치를 감지해 다시 보게 만들었다.
2. 그런데 그 감지를 **연결 실패와 같은 실패**로 취급했더니, hr MCP 처럼 서버에서 도구를
   주석 처리해 **영구히** 안 내놓는 경우 매 기동 6번 재시도 → "최대 재시도 초과 - 비활성화"
   ERROR 가 뜨고, 그 5분 동안 레지스트리 최종 인덱싱이 통째로 밀렸다(실측).

두 원인(배포 순서 / 정책 잔재)은 그 순간 구분할 수 없다. 그래서 규칙은 하나다 —
**등록은 성공으로 치고, 목록만 짧게 다시 본다.**

MCP 패키지를 통째로 스텁으로 갈아 끼우는 이유: ``app.agents.tools.mcp`` 는 import 시점에
전송 계층을 끌고 들어오는데, 설치된 ``mcp`` 패키지 버전에 따라 그 import 가 실패한다(이
저장소의 tests/tools 일부가 이미 그 이유로 죽는다). 여기서 검증하려는 것은 lifespan 의
제어 흐름이지 전송 계층이 아니다.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import pathlib
import sys
import types

import pytest

from agent_arch.core import lifespan as mod

_SKEW_PATH = pathlib.Path(__file__).resolve().parents[2] / "src/agent_arch/agents/tools/mcp/skew.py"


def _load_skew() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("agent_arch.agents.tools.mcp.skew", _SKEW_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Spec:
    name = "mattermost"


@pytest.fixture
def mcp_stub(monkeypatch):
    """provider·skew 를 스텁으로 갈아 끼우고, register 호출 결과를 조종한다.

    Returns:
        ``install(side_effects) -> calls`` — 호출마다 side_effects 를 순서대로 적용하고
        (예외면 raise, 아니면 반환), 실제 호출 기록 리스트를 돌려준다.
    """
    skew_mod = _load_skew()
    pkg = types.ModuleType("agent_arch.agents.tools.mcp")
    provider = types.ModuleType("agent_arch.agents.tools.mcp.provider")
    monkeypatch.setitem(sys.modules, "agent_arch.agents.tools.mcp", pkg)
    monkeypatch.setitem(sys.modules, "agent_arch.agents.tools.mcp.skew", skew_mod)
    monkeypatch.setitem(sys.modules, "agent_arch.agents.tools.mcp.provider", provider)
    monkeypatch.setattr(mod, "_SKEW_RECHECK_DELAY", 0.0)
    monkeypatch.setattr(mod, "_SKEW_RECHECK_TRIES", 3)

    calls: list = []

    def install(side_effects):
        async def _fake(spec):
            calls.append(spec.name)
            effect = side_effects[min(len(calls) - 1, len(side_effects) - 1)]
            if isinstance(effect, Exception):
                raise effect
            return effect

        provider.register_server_tools = _fake
        return calls

    install.skew = lambda missing: skew_mod.MCPToolListSkew("mattermost", missing)
    return install


@pytest.mark.asyncio
async def test_skew_does_not_fail_registration(mcp_stub):
    """불일치가 예외로 새어 나가면 호출부가 5분짜리 백오프를 돈다 — 그러면 안 된다."""
    mcp_stub([mcp_stub.skew(["mattermost_skill_find_posts_by_author"])])
    await mod._register_mcp_server(_Spec())        # 예외 없이 끝난다
    await asyncio.sleep(0)                          # 백그라운드 재확인 태스크 출발


@pytest.mark.asyncio
async def test_connection_failure_still_propagates(mcp_stub):
    """연결 실패는 종전대로 올린다 — 서버가 늦게 뜨는 경우의 백오프가 그대로 필요하다."""
    mcp_stub([RuntimeError("connection refused")])
    with pytest.raises(RuntimeError):
        await mod._register_mcp_server(_Spec())


@pytest.mark.asyncio
async def test_recheck_stops_once_resolved(mcp_stub):
    """상대가 새 버전으로 올라오면 그 시점에 멈춘다(남은 횟수를 더 돌지 않는다)."""
    calls = mcp_stub([mcp_stub.skew(["x"]), 3])     # 1회차 불일치 → 2회차 성공
    await mod._recheck_mcp_skew(_Spec())
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_recheck_gives_up_with_named_tools(mcp_stub):
    """영구 불일치(서버에서 비활성화)면 이름을 찍어 한 번 경고하고 끝낸다.

    caplog 를 쓰지 않는다 — 앱 로깅 설정이 먼저 돌면 전파가 끊겨 전체 실행에서만 빈
    문자열이 잡힌다(단독 실행은 통과, 전체 실행은 실패). 로거에 직접 붙어서 본다.
    """
    records: list = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    sink = _Sink()
    logger = logging.getLogger("agent_arch.core.lifespan")
    logger.addHandler(sink)
    try:
        calls = mcp_stub([mcp_stub.skew(["hr_request_leave", "hr_cancel_leave"])])
        await mod._recheck_mcp_skew(_Spec())
    finally:
        logger.removeHandler(sink)

    assert len(calls) == 3                          # _SKEW_RECHECK_TRIES 만큼만
    text = " ".join(r.getMessage() for r in records)
    assert "hr_request_leave" in text
    # 영구 불일치는 경고지 장애가 아니다 — ERROR 로 올리면 배포 때마다 사람을 깨운다.
    assert not [r for r in records if r.levelname == "ERROR"]


@pytest.mark.asyncio
async def test_recheck_survives_connection_errors(mcp_stub):
    """재확인 중 연결이 끊겨도 다음 차례를 본다 — 그 사이가 재기동일 수 있다."""
    calls = mcp_stub([RuntimeError("boom"), RuntimeError("boom"), 1])
    await mod._recheck_mcp_skew(_Spec())
    assert len(calls) == 3
