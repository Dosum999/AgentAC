"""턴 장소 기록 — 메신저 경로만 metadata.place 를 남기고, 패널 턴은 그대로여야 한다.

장소(채널·스레드·DM 여부)는 지금까지 시스템 프롬프트에만 있었고 저장되는 곳이 없었다.
그래서 그 턴에서 뽑힌 기억은 자기가 어디서 비롯됐는지 알 수 없었다(2026-09-18 진단).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.context import set_user_context
from agent_arch.memory import conversation as conv_module


@pytest.fixture
def save_mocks(monkeypatch):
    """저장 깔때기의 외부 의존을 mock 으로 치환한다(tests/memory/test_pos_save_hook 과 같은 패턴)."""
    pg = AsyncMock(return_value=("mid-1", "conv-1"))
    monkeypatch.setattr(conv_module, "pg_save_turn", pg)
    monkeypatch.setattr(conv_module, "append_turn", AsyncMock())
    monkeypatch.setitem(
        sys.modules, "agent_arch.services.memory", types.SimpleNamespace(schedule_remember=MagicMock())
    )
    set_user_context({})
    return pg


_CHANNEL_ORIGIN = {"channel_id": "ch-1", "channel_name": "개발팀", "root_id": "post-9",
                   "is_direct": False}


class TestTurnPlace:
    def test_messenger_origin_becomes_place(self):
        assert conv_module.turn_place({"origin": _CHANNEL_ORIGIN}) == {
            "provider": "mattermost", "channel_id": "ch-1", "is_dm": False,
            "channel_name": "개발팀", "thread_root": "post-9",
        }

    def test_dm_without_thread(self):
        place = conv_module.turn_place(
            {"origin": {"channel_id": "ch-dm", "channel_name": "", "root_id": None,
                        "is_direct": True}}
        )
        assert place == {"provider": "mattermost", "channel_id": "ch-dm", "is_dm": True}

    def test_panel_turn_has_no_place(self):
        """패널은 origin 을 채우지 않는다 — 이 기능은 메신저 전용이다."""
        assert conv_module.turn_place({"sub": "u1", "company": "(주)회사"}) is None
        assert conv_module.turn_place({}) is None
        assert conv_module.turn_place(None) is None

    def test_malformed_origin_is_ignored(self):
        assert conv_module.turn_place({"origin": "ch-1"}) is None          # dict 아님
        assert conv_module.turn_place({"origin": {"channel_name": "x"}}) is None  # 채널 없음
        assert conv_module.turn_place({"origin": {"channel_id": "  "}}) is None   # 공백뿐

    def test_long_channel_name_is_clamped(self):
        place = conv_module.turn_place(
            {"origin": {"channel_id": "c", "channel_name": "가" * 500}}
        )
        assert len(place["channel_name"]) == conv_module._PLACE_NAME_MAX


class TestPlaceOnSave:
    @pytest.mark.asyncio
    async def test_messenger_turn_records_place(self, save_mocks):
        set_user_context({"sub": "u1", "company": "(주)회사", "origin": _CHANNEL_ORIGIN})
        await conv_module.save_logged_in_turn("conv-1", "u1", "이거 정리해줘", "정리했습니다.")

        meta = save_mocks.await_args.kwargs["metadata"]
        assert meta["place"]["channel_id"] == "ch-1"
        assert meta["place"]["thread_root"] == "post-9"
        assert meta["company"] == "(주)회사"          # 기존 키는 그대로

    @pytest.mark.asyncio
    async def test_panel_turn_metadata_is_unchanged(self, save_mocks):
        """패널 턴 metadata 에는 키가 하나도 늘지 않는다 — 회귀 판정의 기준선."""
        set_user_context({"sub": "u1", "company": "(주)회사"})
        await conv_module.save_logged_in_turn("conv-1", "u1", "질문", "답변")

        assert save_mocks.await_args.kwargs["metadata"] == {"company": "(주)회사"}


class TestPublicChannelCaution:
    """채널 답변은 모든 참여자가 읽는다 — 기억해 둔 개인 정보를 거기 옮겨 적지 않게 한다.

    회상 블록은 매 턴 점심시간·카드 끝자리 같은 것을 시스템 프롬프트에 넣는다. 패널에서는
    본인만 보지만 채널에서는 팀 전체가 본다. DM 과 패널에는 이 주의를 붙이지 않는다.
    """

    def _block(self, **origin):
        from agent_arch.agents.prompts.system import _build_origin_context

        return _build_origin_context({"channel_id": "ch-1", "channel_name": "개발팀", **origin})

    def test_group_channel_gets_the_caution(self):
        assert "모든 참여자가 봅니다" in self._block(is_direct=False)

    def test_dm_does_not(self):
        assert "모든 참여자가 봅니다" not in self._block(is_direct=True)

    def test_panel_has_no_block_at_all(self):
        from agent_arch.agents.prompts.system import _build_origin_context

        assert _build_origin_context(None) == ""
        assert _build_origin_context({}) == ""


def test_app_surface_turn_records_its_window():
    """MES 앱 창(채팅·음성) 턴은 창 이름을 자리로 남긴다 — origin 이 없어도."""
    from agent_arch.memory.conversation import turn_place

    assert turn_place({"memory_surface": "app"}) == {
        "provider": "agent", "channel_id": "app", "is_dm": False,
    }


def test_panel_turn_still_has_no_place():
    """브라우저 패널은 자리가 없다 — metadata 가 이 기능 이전과 같아야 한다."""
    from agent_arch.memory.conversation import turn_place

    assert turn_place({}) is None
    assert turn_place({"memory_surface": ""}) is None
