"""대화 히스토리 모듈 단위 테스트 (직전 1턴 도구 컨텍스트 + 토큰 기반 자동 압축).

대상:
    - app.memory.conversation.append_turn: tool_ctx 저장 / 삭제 / trim
    - app.memory.conversation.load_history: tool_ctx 주입 (직전 user 뒤)
    - app.memory.conversation.clear_user_conversations: tool_ctx 키 삭제
    - 토큰 기반 자동 압축 동작

Redis 는 tests/services/test_preferences.py 와 동일한 in-memory fake 로 교체한다.
사용자 preferences 적용은 monkeypatch 로 prefs 객체를 강제 주입해 검증한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agent_arch.llm.base import LLMResponse, Message
from agent_arch.memory import conversation as conv_module
from agent_arch.services.preferences import UserPreferences


class _FakeRedis:
    """in-memory dict 기반 redis_store 대체.

    실제 redis_store 와 동일 인터페이스 중 본 모듈이 사용하는 것만 구현한다.
    set / get 은 JSON 직렬화/역직렬화를 흉내내어 직렬화 호환성도 함께 검증한다.
    """

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        self.lists: Dict[str, List[str]] = {}
        self.expires: Dict[str, int] = {}

    async def get(self, key: str) -> Any:
        return self.kv.get(key)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        self.kv[key] = json.loads(json.dumps(value))
        if expire is not None:
            self.expires[key] = expire
        return True

    async def delete(self, key: str) -> bool:
        existed = key in self.kv or key in self.lists
        self.kv.pop(key, None)
        self.lists.pop(key, None)
        self.expires.pop(key, None)
        return existed

    async def rpush(self, key: str, *values: Any) -> bool:
        self.lists.setdefault(key, []).extend(json.dumps(v) for v in values)
        return True

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        lst = self.lists.get(key, [])
        end_idx = None if end == -1 else end + 1
        self.lists[key] = lst[start:end_idx]
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        self.expires[key] = seconds
        return True

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        lst = self.lists.get(key, [])
        end_idx = None if end == -1 else end + 1
        return [json.loads(v) for v in lst[start:end_idx]]

    # 자동 압축용 — set_nx / lindex / getdel
    async def set_nx(
        self, key: str, value: Any, expire: Optional[int] = None
    ) -> bool:
        """SET NX 시뮬레이션 — 키 존재 시 False, 미존재 시 set 후 True."""
        if key in self.kv:
            return False
        self.kv[key] = json.loads(json.dumps(value))
        if expire is not None:
            self.expires[key] = expire
        return True

    async def lindex(self, key: str, index: int) -> Any:
        """리스트 인덱스 조회. 범위 밖이면 None."""
        lst = self.lists.get(key, [])
        try:
            return json.loads(lst[index])
        except (IndexError, KeyError):
            return None

    async def getdel(self, key: str) -> Any:
        """GETDEL atomic 시뮬레이션 — 값 반환 후 키 삭제."""
        value = self.kv.pop(key, None)
        self.expires.pop(key, None)
        return value


@pytest.fixture
def fake_redis(monkeypatch):
    """conv_module 의 redis_store 를 in-memory fake 로 교체."""
    fake = _FakeRedis()
    monkeypatch.setattr(conv_module, "redis_store", fake)
    return fake


@pytest.fixture
def no_prefs(monkeypatch):
    """사용자 preferences 가 없는 경우 — 전역 폴백(MAX_HISTORY=20, SESSION_TTL=86400) 사용."""
    async def _none(_sub):
        return None
    monkeypatch.setattr(conv_module, "get_user_preferences", _none)


def _make_prefs(
    user_sub: str = "u-1",
    max_history: int = 20,
    session_ttl_seconds: int = 3600,
) -> UserPreferences:
    """사용자 preferences 객체 헬퍼."""
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    return UserPreferences(
        user_sub=user_sub,
        max_history=max_history,
        session_ttl_seconds=session_ttl_seconds,
        created_at=now,
        updated_at=now,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. append_turn — tool_ctx 저장 / 삭제 / trim
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendTurnToolCtx:
    """append_turn 이 tool_calls 유무에 따라 tool_ctx 키를 저장/삭제하는지 검증."""

    @pytest.mark.asyncio
    async def test_saves_tool_ctx_when_tool_calls_present(self, fake_redis, no_prefs):
        """tool_calls 가 있으면 `{key}:tool_ctx` 키에 JSON 저장."""
        key = "conversation:test-001"
        await conv_module.append_turn(
            key,
            "보고서 보여줘",
            "월간 보고서입니다...",
            tool_calls=[
                {"id": "c1", "name": "get_report", "arguments": {"type": "monthly"}}
            ],
            tool_results=[
                {"tool_call_id": "c1", "name": "get_report", "content": "{...}"}
            ],
        )

        ctx = fake_redis.kv.get(f"{key}:tool_ctx")
        assert ctx is not None
        assert ctx["tool_calls"][0]["name"] == "get_report"
        assert ctx["tool_results"][0]["tool_call_id"] == "c1"
        assert ctx["tool_results"][0]["content"] == "{...}"
        assert "saved_at" in ctx

    @pytest.mark.asyncio
    async def test_preserves_tool_ctx_when_no_tools(self, fake_redis, no_prefs):
        """도구 호출 없는 턴(되물음 등)이 와도 이전 tool_ctx 를 유지하고 TTL 을 갱신한다.

        되묻는 턴이 직전 조회 대상(초점 엔티티)을 지워, 후속 "방금 것" 이 초점을
        잃던 멀티턴 그라운딩 실패를 막기 위함이다.
        """
        key = "conversation:test-002"
        # 사전 데이터 (이전 도구 턴에서 저장된 것처럼)
        fake_redis.kv[f"{key}:tool_ctx"] = {"tool_calls": [{"id": "old"}]}

        await conv_module.append_turn(
            key, "안녕", "안녕하세요!", tool_calls=None, tool_results=None
        )

        # 삭제되지 않고 유지 + TTL 이 history LIST 와 동일(SESSION_TTL)로 갱신됨
        assert fake_redis.kv[f"{key}:tool_ctx"] == {"tool_calls": [{"id": "old"}]}
        assert fake_redis.expires[f"{key}:tool_ctx"] == conv_module.SESSION_TTL

    @pytest.mark.asyncio
    async def test_trims_large_tool_result_content(self, fake_redis, no_prefs):
        """MAX_TOOL_RESULT_CHARS 초과 content 는 trim + TRUNCATE_NOTICE 부착.

        JSON 이 아닌 도구 결과(평문)는 종전대로 문자 절단한다 — 구조가 없어 필드 단위로
        줄일 게 없다. 길이는 **상한 기준으로** 만든다. 상수를 박아 두면 상한을 조정할 때
        "축약이 안 걸리는 입력"이 돼 조용히 무의미해진다(2026-08-25 8,000→20,000에서 겪음).
        """
        key = "conversation:test-003"
        large_content = "x" * (conv_module.MAX_TOOL_RESULT_CHARS + 1)

        await conv_module.append_turn(
            key,
            "q",
            "a",
            tool_calls=[{"id": "c1", "name": "big_tool", "arguments": {}}],
            tool_results=[
                {"tool_call_id": "c1", "name": "big_tool", "content": large_content}
            ],
        )

        ctx = fake_redis.kv.get(f"{key}:tool_ctx")
        saved_content = ctx["tool_results"][0]["content"]
        assert len(saved_content) == conv_module.MAX_TOOL_RESULT_CHARS + len(
            conv_module.TRUNCATE_NOTICE
        )
        assert saved_content.endswith(conv_module.TRUNCATE_NOTICE)

    @pytest.mark.asyncio
    async def test_uses_user_prefs_ttl_for_tool_ctx(self, fake_redis, monkeypatch):
        """tool_ctx 키 TTL 도 사용자 preferences 의 session_ttl_seconds 를 따른다."""
        custom_ttl = 600  # 사용자가 짧게 설정한 케이스
        prefs = _make_prefs(user_sub="u-1", session_ttl_seconds=custom_ttl)

        async def _get_prefs(_sub):
            return prefs
        monkeypatch.setattr(conv_module, "get_user_preferences", _get_prefs)

        key = "conversation:test-004"
        await conv_module.append_turn(
            key,
            "q",
            "a",
            user_sub="u-1",
            tool_calls=[{"id": "c1", "name": "t", "arguments": {}}],
            tool_results=[{"tool_call_id": "c1", "name": "t", "content": "ok"}],
        )

        # history LIST 와 tool_ctx STRING 모두 동일 TTL 적용
        assert fake_redis.expires[key] == custom_ttl
        assert fake_redis.expires[f"{key}:tool_ctx"] == custom_ttl

    @pytest.mark.asyncio
    async def test_falls_back_to_global_ttl_when_no_prefs(self, fake_redis, no_prefs):
        """prefs 가 없으면 SESSION_TTL=86400 폴백을 tool_ctx 에도 적용."""
        key = "conversation:test-005"
        await conv_module.append_turn(
            key,
            "q",
            "a",
            user_sub="u-x",
            tool_calls=[{"id": "c1", "name": "t", "arguments": {}}],
            tool_results=[{"tool_call_id": "c1", "name": "t", "content": "ok"}],
        )

        assert fake_redis.expires[f"{key}:tool_ctx"] == conv_module.SESSION_TTL


# ─────────────────────────────────────────────────────────────────────────────
# 2. load_history — tool_ctx 주입
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadHistoryInjection:
    """load_history 가 tool_ctx 를 마지막 user 직후에 주입하는지 검증."""

    @pytest.mark.asyncio
    async def test_injects_after_last_user(self, fake_redis, no_prefs):
        """직전 user 뒤에 assistant(tool_calls) + tool 메시지 삽입."""
        key = "conversation:test-006"

        await conv_module.append_turn(
            key,
            "보고서 보여줘",
            "월간 보고서 응답",
            tool_calls=[{"id": "c1", "name": "get_report", "arguments": {}}],
            tool_results=[
                {"tool_call_id": "c1", "name": "get_report", "content": "데이터"}
            ],
        )

        history = await conv_module.load_history(key)

        # 예상 순서: [user("보고서..."), assistant(tool_call), tool, assistant("월간 보고서 응답")]
        assert len(history) == 4
        assert history[0].role == "user"
        assert history[0].content == "보고서 보여줘"
        assert history[1].role == "assistant"
        assert history[1].tool_calls is not None
        assert history[1].tool_calls[0]["name"] == "get_report"
        assert history[2].role == "tool"
        assert history[2].content == "데이터"
        assert history[2].name == "get_report"
        assert history[2].tool_call_id == "c1"
        assert history[3].role == "assistant"
        assert history[3].content == "월간 보고서 응답"

    @pytest.mark.asyncio
    async def test_no_injection_when_tool_ctx_absent(self, fake_redis, no_prefs):
        """tool_ctx 키가 없으면 기존 동작 — user/assistant 페어만 반환."""
        key = "conversation:test-007"
        await conv_module.append_turn(key, "안녕", "안녕하세요!")

        history = await conv_module.load_history(key)

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty(self, fake_redis):
        """LIST 가 비어있으면 빈 리스트 반환 (tool_ctx 만 있어도 주입 안 함)."""
        key = "conversation:test-008"
        # tool_ctx 만 있고 LIST 가 없는 비정상 상태
        await fake_redis.set(
            f"{key}:tool_ctx",
            {"tool_calls": [{"id": "c1", "name": "t"}], "tool_results": []},
        )

        history = await conv_module.load_history(key)
        assert history == []

    @pytest.mark.asyncio
    async def test_no_injection_when_tool_calls_empty(self, fake_redis, no_prefs):
        """tool_ctx 가 있지만 tool_calls 가 비어있는 비정상 케이스 — 주입 스킵."""
        key = "conversation:test-009"
        await conv_module.append_turn(key, "안녕", "안녕하세요!")
        # 빈 tool_calls 직접 주입
        await fake_redis.set(f"{key}:tool_ctx", {"tool_calls": [], "tool_results": []})

        history = await conv_module.load_history(key)
        assert len(history) == 2  # user + assistant 만


# ─────────────────────────────────────────────────────────────────────────────
# 3. clear_user_conversations — tool_ctx 키 함께 삭제
# ─────────────────────────────────────────────────────────────────────────────


class TestClearUserConversationsToolCtx:
    """clear_user_conversations 호출 시 tool_ctx 키도 함께 삭제되는지 검증."""

    @pytest.mark.asyncio
    async def test_deletes_tool_ctx_alongside_history(
        self, fake_redis, no_prefs, monkeypatch
    ):
        """본인 대화방 키 + tool_ctx + 자동 압축 부속 키 모두 삭제."""
        # 두 대화방에 history + tool_ctx + 압축 부속 키 셋업
        for cid in ("c-1", "c-2"):
            await conv_module.append_turn(
                f"conversation:{cid}",
                "q",
                "a",
                tool_calls=[{"id": "c1", "name": "t", "arguments": {}}],
                tool_results=[{"tool_call_id": "c1", "name": "t", "content": "x"}],
            )
            await fake_redis.set(
                f"conversation:{cid}:compact_announce",
                {"ts": 1, "n_summarized": 2},
            )
            await fake_redis.set(f"conversation:{cid}:compact_lock", "1")

        # list_conversations 를 monkeypatch — clear_user_conversations 내부 동적 import
        from agent_arch.db import repository as repo_module

        async def _list_convs(*, user_sub, limit, include_archived):
            return [{"id": "c-1"}, {"id": "c-2"}]
        monkeypatch.setattr(repo_module, "list_conversations", _list_convs)

        count = await conv_module.clear_user_conversations("u-1")

        assert count == 2
        # 본체 LIST + 부속 키 3종(tool_ctx/compact_announce/compact_lock) 전부
        # 사라졌는지 확인 — 본체만 지우고 부속 키가 남는 회귀 방지.
        for cid in ("c-1", "c-2"):
            assert f"conversation:{cid}" not in fake_redis.lists
            assert f"conversation:{cid}:tool_ctx" not in fake_redis.kv
            assert f"conversation:{cid}:compact_announce" not in fake_redis.kv
            assert f"conversation:{cid}:compact_lock" not in fake_redis.kv

    @pytest.mark.asyncio
    async def test_returns_pg_row_count_not_deleted_keys(
        self, fake_redis, monkeypatch
    ):
        """반환값은 PG 대화방 수다 — Redis에 키가 하나도 없어도 동일하다.

        "실제 삭제된 키 수"가 아니므로 사용자 노출용 수치로 쓰면 안 된다는
        docstring 계약을 고정한다.
        """
        from agent_arch.db import repository as repo_module

        async def _list_convs(*, user_sub, limit, include_archived):
            return [{"id": "c-1"}, {"id": "c-2"}, {"id": "c-3"}]
        monkeypatch.setattr(repo_module, "list_conversations", _list_convs)

        # Redis는 완전히 빈 상태 — 실삭제 0건이어도 PG 행 수 3을 반환한다.
        count = await conv_module.clear_user_conversations("u-1")
        assert count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3-1. /compact 커맨드 — 사용자 메시지 정합성
# ─────────────────────────────────────────────────────────────────────────────


class TestCompactCommandMessage:
    """/compact 결과 메시지가 실제 동작(컨텍스트 삭제만)을 반영하는지 검증.

    실제 동작은 Redis 컨텍스트 키 삭제뿐이다 — PG 영구 기록은 유지되고
    요약(토큰 기반 자동 압축)도 일어나지 않는다. PG 대화방 행 수가
    "삭제된 개수"처럼 메시지에 노출되던 회귀를 막는다.
    """

    @pytest.mark.asyncio
    async def test_message_has_no_count_and_states_facts(self, monkeypatch):
        """로그인 사용자: 개수 미노출 + '기억 비움/기록 유지' 두 사실 전달."""
        from agent_arch.utils.commands import handle_command

        # PG 대화방 10개를 흉내 — 반환값이 메시지에 노출되면 안 된다.
        clear_mock = AsyncMock(return_value=10)
        monkeypatch.setattr(conv_module, "clear_user_conversations", clear_mock)

        result = await handle_command("/compact", "gemini", user_sub="u-1")

        assert result is not None
        assert result["success"] is True
        clear_mock.assert_awaited_once_with("u-1")

        output = result["output"]
        # "(N개 대화방) 초기화" 같은 수치 노출 회귀 방지.
        assert not any(ch.isdigit() for ch in output)
        assert "기억하지 않고" in output
        assert "유지" in output

    @pytest.mark.asyncio
    async def test_requires_login(self, monkeypatch):
        """비로그인: 삭제 시도 없이 실패 메시지."""
        from agent_arch.utils.commands import handle_command

        clear_mock = AsyncMock(return_value=0)
        monkeypatch.setattr(conv_module, "clear_user_conversations", clear_mock)

        result = await handle_command("/compact", "gemini", user_sub=None)

        assert result is not None
        assert result["success"] is False
        clear_mock.assert_not_awaited()
        assert "로그인" in result["output"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. 토큰 기반 자동 압축
# ─────────────────────────────────────────────────────────────────────────────


class _FakeLLM:
    """LLMFactory.create_with_fallback() 가 반환하는 LLM 의 미니멀 stub.

    invoke 만 사용되며, summary 텍스트(또는 예외) 를 미리 주입한다.
    """

    def __init__(
        self,
        summary_text: str = "이전 대화 요약: 사용자가 질문하고 응답을 받았다.",
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.summary_text = summary_text
        self.raise_exc = raise_exc
        self.invoke_calls: list = []

    async def invoke(self, messages, **kwargs):
        self.invoke_calls.append((list(messages), kwargs))
        if self.raise_exc is not None:
            raise self.raise_exc
        return LLMResponse(
            content=self.summary_text,
            tool_calls=None,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


@pytest.fixture
def fake_llm(monkeypatch):
    """LLMFactory.create_with_fallback 을 FakeLLM 으로 교체."""
    llm = _FakeLLM()

    class _FakeFactory:
        @classmethod
        def create_with_fallback(cls):
            return llm

    # _compact_history 내부에서 동적 import 하므로 LLMFactory 모듈을 직접 monkeypatch.
    from agent_arch.llm import factory as factory_module
    monkeypatch.setattr(factory_module, "LLMFactory", _FakeFactory)
    return llm


def _seed_messages(count: int, content_len: int = 200) -> List[dict]:
    """user/assistant 페어로 count 개의 메시지를 생성한다.

    content_len 으로 1메시지당 글자수 조절 — 토큰 임계값 트리거 테스트용.
    """
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"메시지 {i}: " + "가" * content_len
        msgs.append({"role": role, "content": content})
    return msgs


class TestCompactHistory:
    """자동 압축 동작 검증."""

    @pytest.mark.asyncio
    async def test_no_compact_below_threshold(self, fake_redis, fake_llm):
        """토큰 합 < trigger 면 LIST 무변경, LLM 호출 0회."""
        key = "conversation:compact-001"
        msgs = _seed_messages(6, content_len=50)  # 작은 메시지
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        assert len(fake_llm.invoke_calls) == 0
        # LIST 보존
        assert len(fake_redis.lists[key]) == len(msgs)

    @pytest.mark.asyncio
    async def test_compact_triggers_above_threshold(self, fake_redis, fake_llm):
        """토큰 합 ≥ 6000 일 때 LIST[0] 이 summary 메시지로 치환."""
        key = "conversation:compact-002"
        # 큰 메시지로 6K 토큰 초과 유도 (한국어 1자 ≈ 0.6 토큰 휴리스틱 기준 약 16K char)
        msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        assert len(fake_llm.invoke_calls) == 1
        # LIST head 가 summary 로 치환
        head = json.loads(fake_redis.lists[key][0])
        assert head["role"] == "system"
        assert head["name"] == conv_module.COMPACT_SUMMARY_MARKER
        assert "요약" in head["content"] or len(head["content"]) > 0

    @pytest.mark.asyncio
    async def test_keeps_recent_8_messages_verbatim(self, fake_redis, fake_llm):
        """압축 후 최근 8 메시지 원문 보존."""
        key = "conversation:compact-003"
        msgs = _seed_messages(20, content_len=600)
        original_last_8 = msgs[-8:]
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        # [summary, *last_8] = 9 elements
        assert len(fake_redis.lists[key]) == 9
        stored = [json.loads(v) for v in fake_redis.lists[key][1:]]
        for stored_msg, original in zip(stored, original_last_8):
            assert stored_msg["role"] == original["role"]
            assert stored_msg["content"] == original["content"]

    @pytest.mark.asyncio
    async def test_re_compact_absorbs_previous_summary(self, fake_redis, fake_llm):
        """기존 summary 가 있을 때 새 압축이 그것을 흡수한다."""
        key = "conversation:compact-004"
        existing_summary = {
            "role": "system",
            "name": conv_module.COMPACT_SUMMARY_MARKER,
            "content": "이전 요약: 사용자가 X에 대해 물었음",
        }
        new_msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(existing_summary)] + [json.dumps(m) for m in new_msgs]

        await conv_module._compact_history(key, ttl=3600)

        # 요약 호출 시 previous_summary 가 user dump 에 포함되어야 함
        assert len(fake_llm.invoke_calls) == 1
        invoked_messages, _ = fake_llm.invoke_calls[0]
        # 두 번째 메시지(user)에 "[이전 요약]" 헤더 포함
        user_dump = invoked_messages[1].content
        assert "[이전 요약]" in user_dump
        assert "사용자가 X에 대해 물었음" in user_dump

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_compact(self, fake_redis, fake_llm):
        """락 키가 선점되어 있으면 _compact_history_guarded 즉시 return."""
        key = "conversation:compact-005"
        msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]
        # 락 사전 선점
        fake_redis.kv[f"{key}:compact_lock"] = "1"

        await conv_module._compact_history_guarded(key, ttl=3600)

        # 락 잡혔으므로 LLM 호출 안 됨, LIST 변경 없음
        assert len(fake_llm.invoke_calls) == 0
        assert len(fake_redis.lists[key]) == len(msgs)

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_list_intact(self, fake_redis, monkeypatch):
        """Fake LLM 이 예외 던지면 LIST 무변경, 락 해제."""
        key = "conversation:compact-006"
        msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        failing_llm = _FakeLLM(raise_exc=RuntimeError("LLM down"))

        class _FailFactory:
            @classmethod
            def create_with_fallback(cls):
                return failing_llm

        from agent_arch.llm import factory as factory_module
        monkeypatch.setattr(factory_module, "LLMFactory", _FailFactory)

        await conv_module._compact_history_guarded(key, ttl=3600)

        # LIST 보존
        assert len(fake_redis.lists[key]) == len(msgs)
        # 락 해제됨
        assert f"{key}:compact_lock" not in fake_redis.kv
        # announce 키 미생성
        assert f"{key}:compact_announce" not in fake_redis.kv

    @pytest.mark.asyncio
    async def test_tool_ctx_key_unaffected(self, fake_redis, fake_llm):
        """압축 후 `{key}:tool_ctx` 키 그대로 유지."""
        key = "conversation:compact-007"
        msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]
        fake_redis.kv[f"{key}:tool_ctx"] = {"tool_calls": [{"id": "t1"}], "tool_results": []}

        await conv_module._compact_history(key, ttl=3600)

        # tool_ctx 키 보존
        assert f"{key}:tool_ctx" in fake_redis.kv
        assert fake_redis.kv[f"{key}:tool_ctx"]["tool_calls"][0]["id"] == "t1"

    @pytest.mark.asyncio
    async def test_compact_skipped_for_short_history(self, fake_redis, fake_llm):
        """메시지 수 < KEEP_RECENT + MIN_OLD 면 임계값 넘어도 압축 스킵."""
        key = "conversation:compact-008"
        # 큰 메시지 8개 (= KEEP_RECENT_MESSAGES) → old 가 0 으로 스킵
        msgs = _seed_messages(8, content_len=2000)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        assert len(fake_llm.invoke_calls) == 0
        assert len(fake_redis.lists[key]) == 8

    @pytest.mark.asyncio
    async def test_token_estimator_tiktoken_path(self):
        """tiktoken 정상 로드 경로 — 양수 반환 확인."""
        # 모듈 캐시 초기화
        conv_module._tiktoken_encoder = None
        conv_module._tiktoken_unavailable = False

        msgs = [Message(role="user", content="안녕하세요 반갑습니다 hello world")]
        tokens = conv_module._estimate_tokens(msgs)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_token_estimator_heuristic_fallback(self, monkeypatch):
        """tiktoken 로드 실패 시 휴리스틱(`len * 0.6`)으로 폴백."""
        conv_module._tiktoken_encoder = None
        conv_module._tiktoken_unavailable = True  # 폴백 강제

        text = "가" * 100
        msgs = [Message(role="user", content=text)]
        tokens = conv_module._estimate_tokens(msgs)
        # 0.6 * 100 = 60
        assert tokens == 60

        # 테스트 후 리셋
        conv_module._tiktoken_unavailable = False

    @pytest.mark.asyncio
    async def test_compact_sets_announce_key(self, fake_redis, fake_llm):
        """압축 성공 후 `{key}:compact_announce` 에 ts, n_summarized 가 저장."""
        key = "conversation:compact-009"
        msgs = _seed_messages(20, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        announce = fake_redis.kv.get(f"{key}:compact_announce")
        assert announce is not None
        assert "ts" in announce
        assert "n_summarized" in announce
        assert announce["n_summarized"] == 20 - conv_module.COMPACT_KEEP_RECENT_MESSAGES

    @pytest.mark.asyncio
    async def test_consume_compact_announce_once(self, fake_redis):
        """consume_compact_announce 호출 시 첫 번째만 값 받고 키 삭제."""
        key = "conversation:compact-010"
        fake_redis.kv[f"{key}:compact_announce"] = {"ts": 1234, "n_summarized": 12}

        first = await conv_module.consume_compact_announce(key)
        assert first == {"ts": 1234, "n_summarized": 12}

        second = await conv_module.consume_compact_announce(key)
        assert second is None

    @pytest.mark.asyncio
    async def test_compact_skip_does_not_set_announce(self, fake_redis, fake_llm):
        """압축 스킵(임계값 미달) 시 announce 키 미생성."""
        key = "conversation:compact-011"
        msgs = _seed_messages(6, content_len=50)  # 작은 메시지
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module._compact_history(key, ttl=3600)

        assert f"{key}:compact_announce" not in fake_redis.kv

    @pytest.mark.asyncio
    async def test_append_turn_triggers_compact_when_threshold_met(
        self, fake_redis, no_prefs, fake_llm
    ):
        """append_turn 끝의 fire-and-forget 트리거 검증.

        토큰 합이 임계값을 넘는 상태에서 append_turn 호출 시 압축 태스크가 생성되어
        실행되어야 함. asyncio.sleep(0) 로 백그라운드 태스크에 제어권을 양보.
        """
        import asyncio

        key = "conversation:compact-012"
        # 큰 메시지 18개 미리 시드 — append_turn 한 번에 20 메시지로 늘어남
        msgs = _seed_messages(18, content_len=600)
        fake_redis.lists[key] = [json.dumps(m) for m in msgs]

        await conv_module.append_turn(key, "마지막 질문", "마지막 응답")
        # 백그라운드 압축 태스크 완료 대기
        for _ in range(5):
            await asyncio.sleep(0)

        # 압축이 일어났다면 LIST head 가 summary
        head = json.loads(fake_redis.lists[key][0])
        assert head["role"] == "system"
        assert head["name"] == conv_module.COMPACT_SUMMARY_MARKER


class TestPendingDraft:
    """양식별 pending draft stash — tool_ctx 와 달리 절삭·삭제되지 않아 멀티턴 유실 방지."""

    _TPL = "expense-report"

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, fake_redis):
        body = {"total_amount": "8280", "items": [{"merchant": "가게", "amount": "8280"}]}
        await conv_module.save_pending_draft("c1", self._TPL, body)
        assert await conv_module.load_pending_draft("c1", self._TPL) == body
        assert await conv_module.load_pending_draft_source("c1", self._TPL) == body

    @pytest.mark.asyncio
    async def test_revision_updates_current_but_preserves_source(self, fake_redis):
        source = {
            "total_amount": "30000",
            "items": [{"payment_method": "하나기업", "amount": "30000"}],
        }
        revised = {
            "total_amount": "30000",
            "items": [{"payment_method": "법인카드(0993)", "amount": "30000"}],
        }
        await conv_module.save_pending_draft("c1", self._TPL, source)
        await conv_module.save_pending_draft_revision("c1", self._TPL, revised)

        assert await conv_module.load_pending_draft("c1", self._TPL) == revised
        assert await conv_module.load_pending_draft_source("c1", self._TPL) == source

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, fake_redis):
        assert await conv_module.load_pending_draft("nope", self._TPL) is None

    @pytest.mark.asyncio
    async def test_clear_removes(self, fake_redis):
        await conv_module.save_pending_draft("c1", self._TPL, {"items": []})
        await conv_module.clear_pending_draft("c1", self._TPL)
        assert await conv_module.load_pending_draft("c1", self._TPL) is None
        assert await conv_module.load_pending_draft_source("c1", self._TPL) is None

    @pytest.mark.asyncio
    async def test_none_ids_are_noop(self, fake_redis):
        await conv_module.save_pending_draft(None, self._TPL, {"items": []})
        await conv_module.save_pending_draft("c1", None, {"items": []})
        assert await conv_module.load_pending_draft("c1", self._TPL) is None
        assert fake_redis.kv == {}

    @pytest.mark.asyncio
    async def test_isolated_by_template(self, fake_redis):
        # 양식별로 키가 분리돼 다른 양식 create 에 섞이지 않는다(교차 오염 방지의 핵심).
        await conv_module.save_pending_draft("c1", "expense-report", {"total_amount": "1"})
        assert await conv_module.load_pending_draft("c1", "leave") is None
        assert await conv_module.load_pending_draft("c1", "expense-report") is not None

    @pytest.mark.asyncio
    async def test_survives_no_tool_turn(self, fake_redis, no_prefs):
        # tool_ctx 는 무툴 턴에 삭제되지만(test_deletes_tool_ctx_when_no_tools),
        # pending_draft 는 별도 키(SUFFIX)라 무툴 턴 append_turn 뒤에도 살아남는다 — 핵심 계약.
        cid = "c-multiturn"
        await conv_module.save_pending_draft(cid, self._TPL, {"total_amount": "100", "items": []})
        await conv_module.append_turn(
            f"conversation:{cid}", "결재자 알려줘", "결재자 성함을 알려주세요",
            tool_calls=None, tool_results=None,
        )
        assert await conv_module.load_pending_draft(cid, self._TPL) is not None


# ── 거절된 초안 stash (정정 턴 재발행용) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_rejected_draft_roundtrip_and_ttl(fake_redis):
    """거절 초안 저장 → 조회 → 삭제. TTL 은 정정 창(30분) 상수를 쓴다."""
    args = {"summary": "테스트 회의", "dtstart": "2026-08-13T09:00:00+09:00", "dtend": None}
    await conv_module.save_rejected_draft("c1", "calendar_create_event", args, [{"label": "제목", "value": "테스트 회의"}])

    key = f"conversation:c1:{conv_module.REJECTED_DRAFT_SUFFIX}"
    assert fake_redis.expires[key] == conv_module.REJECTED_DRAFT_TTL

    got = await conv_module.load_rejected_draft("c1")
    assert got["tool_name"] == "calendar_create_event"
    assert got["arguments"] == args          # 인자 원본이 그대로 (표시용 fields 아님)
    assert got["fields"] == [{"label": "제목", "value": "테스트 회의"}]

    await conv_module.clear_rejected_draft("c1")
    assert await conv_module.load_rejected_draft("c1") is None


@pytest.mark.asyncio
async def test_rejected_draft_ignores_incomplete_input(fake_redis):
    """conversation_id·tool_name·arguments 중 하나라도 없으면 저장하지 않는다(빈 stash 로 오염 방지)."""
    await conv_module.save_rejected_draft(None, "calendar_create_event", {"a": 1})
    await conv_module.save_rejected_draft("c1", None, {"a": 1})
    await conv_module.save_rejected_draft("c1", "calendar_create_event", None)
    assert fake_redis.kv == {}
    assert await conv_module.load_rejected_draft("c1") is None
    assert await conv_module.load_rejected_draft(None) is None


class TestPendingDraftEnvelope:
    """pending_draft 봉투(v2) — body 와 수집 상태(slots)를 함께 싣는다.

    slots 가 필요한 이유: 경비청구의 결제수단·카드 4자리·대상자 응답 여부는 body 에서
    되살릴 수 없고(포맷된 payment_method 한 덩어리로 접힌다), tool_ctx 는 다음 도구 턴이
    덮어써 턴을 넘기지 못한다. 배포 시점에 진행 중이던 v1 레코드도 읽혀야 한다.
    """

    @pytest.mark.asyncio
    async def test_saves_body_and_slots_and_reads_back(self, fake_redis):
        body = {"total_amount": "30000", "items": [{"merchant": "카츠안"}]}
        slots = {"payment_type": "법인카드", "card_last4": "1234"}
        await conv_module.save_pending_draft("c1", "expense-report", body, slots=slots)

        assert await conv_module.load_pending_draft("c1", "expense-report") == body
        assert await conv_module.load_pending_draft_slots("c1", "expense-report") == slots
        # source 도 같은 봉투로 저장되고 body 만 꺼내진다.
        assert await conv_module.load_pending_draft_source("c1", "expense-report") == body

    @pytest.mark.asyncio
    async def test_slots_default_to_empty_when_not_given(self, fake_redis):
        # slots 를 쓰지 않는 양식(휴가·기안)은 빈 dict — 호출자가 초기값으로 승격한다.
        await conv_module.save_pending_draft("c1", "leave", {"start_date": "2026-08-20"})
        assert await conv_module.load_pending_draft_slots("c1", "leave") == {}

    @pytest.mark.asyncio
    async def test_reads_legacy_v1_record_as_body(self, fake_redis):
        """봉투 없이 body 만 들어 있던 구 레코드 — 진행 중 대화가 배포로 깨지지 않아야 한다."""
        legacy = {"total_amount": "30000", "items": [{"merchant": "카츠안"}]}
        await fake_redis.set(
            "conversation:c1:pending_draft:expense-report", legacy
        )
        assert await conv_module.load_pending_draft("c1", "expense-report") == legacy
        # slots 는 없다 → 빈 dict (관문은 이 흐름에 개입하지 않는다)
        assert await conv_module.load_pending_draft_slots("c1", "expense-report") == {}

    @pytest.mark.asyncio
    async def test_revision_preserves_slots_when_not_given(self, fake_redis):
        """정정이 수집 상태를 건드리지 않으면 기존 slots 를 이어받는다(같은 질문 반복 방지)."""
        await conv_module.save_pending_draft(
            "c1", "expense-report", {"items": [{"merchant": "A"}]},
            slots={"payment_type": "법인카드", "participants_status": "RESOLVED"},
        )
        await conv_module.save_pending_draft_revision(
            "c1", "expense-report", {"items": [{"merchant": "B"}]},
        )
        assert await conv_module.load_pending_draft("c1", "expense-report") == {
            "items": [{"merchant": "B"}]
        }
        assert await conv_module.load_pending_draft_slots("c1", "expense-report") == {
            "payment_type": "법인카드", "participants_status": "RESOLVED",
        }

    @pytest.mark.asyncio
    async def test_revision_updates_slots_and_keeps_source(self, fake_redis):
        original = {"items": [{"merchant": "A"}]}
        await conv_module.save_pending_draft(
            "c1", "expense-report", original, slots={"payment_type": None},
        )
        await conv_module.save_pending_draft_revision(
            "c1", "expense-report", {"items": [{"merchant": "B"}]},
            slots={"payment_type": "개인카드", "card_last4": "9154"},
        )
        assert await conv_module.load_pending_draft_slots("c1", "expense-report") == {
            "payment_type": "개인카드", "card_last4": "9154",
        }
        # OCR source 는 최초 값 그대로 — 정정 감지(body-diff)의 기준선이다.
        assert await conv_module.load_pending_draft_source("c1", "expense-report") == original

    @pytest.mark.asyncio
    async def test_clear_removes_both_slots_and_body(self, fake_redis):
        await conv_module.save_pending_draft(
            "c1", "expense-report", {"items": []}, slots={"payment_type": "법인카드"},
        )
        await conv_module.clear_pending_draft("c1", "expense-report")
        assert await conv_module.load_pending_draft("c1", "expense-report") is None
        assert await conv_module.load_pending_draft_slots("c1", "expense-report") == {}


class TestCollectingDraftPointer:
    """"되물을 값이 남은 초안 진행 중" 포인터 — 값이 아니라 도구·양식만 담는다.

    redis_store 에 scan/keys 가 없어 "이 대화에 수집 중 stash 가 있나"를 pending_draft 로는
    O(1) 로 물을 수 없어서 별도 포인터를 둔다. supervisor 가 이걸 보고 짧은 답변 턴을
    조회가 아닌 실행으로 태운다.
    """

    @pytest.mark.asyncio
    async def test_save_and_load(self, fake_redis):
        await conv_module.save_collecting_draft(
            "c1", "eapproval_create_document", "expense-report"
        )
        assert await conv_module.load_collecting_draft("c1") == {
            "tool_name": "eapproval_create_document", "template_code": "expense-report",
        }

    @pytest.mark.asyncio
    async def test_ttl_is_short_enough_to_expire_before_the_draft_body(self, fake_redis):
        """포인터 TTL < pending_draft TTL — 하루 뒤 정상 조회가 실행으로 밀리지 않게."""
        await conv_module.save_collecting_draft("c1", "eapproval_create_document")
        key = "conversation:c1:collecting_draft"
        assert fake_redis.expires[key] == conv_module.COLLECTING_DRAFT_TTL
        assert conv_module.COLLECTING_DRAFT_TTL < conv_module.SESSION_TTL

    @pytest.mark.asyncio
    async def test_ignores_incomplete_input(self, fake_redis):
        await conv_module.save_collecting_draft(None, "eapproval_create_document")
        await conv_module.save_collecting_draft("c1", None)
        assert fake_redis.kv == {}
        assert await conv_module.load_collecting_draft("c1") is None
        assert await conv_module.load_collecting_draft(None) is None

    @pytest.mark.asyncio
    async def test_clear(self, fake_redis):
        await conv_module.save_collecting_draft("c1", "eapproval_create_document")
        await conv_module.clear_collecting_draft("c1")
        assert await conv_module.load_collecting_draft("c1") is None

    @pytest.mark.asyncio
    async def test_save_pending_draft_records_pointer_when_resume_tool_given(self, fake_redis):
        await conv_module.save_pending_draft(
            "c1", "expense-report", {"items": []}, slots={"payment_type": None},
            resume_tool="eapproval_create_document",
        )
        pointer = await conv_module.load_collecting_draft("c1")
        assert pointer["tool_name"] == "eapproval_create_document"
        assert pointer["template_code"] == "expense-report"

    @pytest.mark.asyncio
    async def test_draft_title_survives_revision(self, fake_redis):
        """정정 턴이 제목을 다시 짓지 않아도 이어받는다.

        실측 2026-09-23: 제목이 stash 에 없어 정정 턴마다 모델이 다시 지어야 했고,
        빠뜨리는 순간 경비청구 흐름이 끊겼다.
        """
        await conv_module.save_pending_draft(
            "c1", "expense-report", {"items": [{"amount": "68400"}]},
            title="경비청구서",
        )
        assert await conv_module.load_pending_draft_title("c1", "expense-report") == "경비청구서"

        # 정정 저장에서 title 을 안 주면 기존 제목을 잃지 않는다.
        await conv_module.save_pending_draft_revision(
            "c1", "expense-report", {"items": [{"amount": "68400", "purpose": "점심식대"}]},
        )
        assert await conv_module.load_pending_draft_title("c1", "expense-report") == "경비청구서"
        assert (await conv_module.load_pending_draft("c1", "expense-report"))["items"][0][
            "purpose"
        ] == "점심식대"

    @pytest.mark.asyncio
    async def test_turn_attachments_survive_to_a_later_turn(self, fake_redis):
        """증빙은 OCR 을 건너뛴 턴에서도 이어받을 수 있어야 한다.

        실측 2026-09-29: 모델이 영수증 사진을 직접 읽고 OCR 을 건너뛰자 증빙 참조가 아예
        생기지 않아, 상신은 됐는데 문서에 증빙이 비어 나갔다.
        """
        metas = [{"file_id": "fs-1", "filename": "영수증.jpg", "content_type": "image/jpeg"}]
        await conv_module.save_turn_attachments("c1", metas)

        assert await conv_module.load_turn_attachments("c1") == metas

    @pytest.mark.asyncio
    async def test_turn_attachments_empty_when_never_saved(self, fake_redis):
        """첨부 없던 대화는 빈 목록 — 엉뚱한 파일이 증빙으로 붙지 않는다."""
        assert await conv_module.load_turn_attachments("c-none") == []
        await conv_module.save_turn_attachments("c1", None)
        assert await conv_module.load_turn_attachments("c1") == []

    @pytest.mark.asyncio
    async def test_draft_title_absent_on_old_records(self, fake_redis):
        """제목 없이 저장된 옛 초안은 None 을 돌려준다(터지지 않는다)."""
        await conv_module.save_pending_draft("c1", "expense-report", {"items": []})
        assert await conv_module.load_pending_draft_title("c1", "expense-report") is None

    @pytest.mark.asyncio
    async def test_save_pending_draft_without_resume_tool_records_nothing(self, fake_redis):
        # 휴가·기안 등 이 계약을 쓰지 않는 양식은 포인터를 만들지 않는다 → 타 도메인 무영향.
        await conv_module.save_pending_draft("c1", "leave", {"start_date": "2026-08-20"})
        assert await conv_module.load_collecting_draft("c1") is None

    @pytest.mark.asyncio
    async def test_clear_pending_draft_also_clears_pointer(self, fake_redis):
        """상신 완료로 stash 를 지우면 포인터도 함께 사라진다(이후 조회가 실행으로 안 밀리게)."""
        await conv_module.save_pending_draft(
            "c1", "expense-report", {"items": []},
            resume_tool="eapproval_create_document",
        )
        await conv_module.clear_pending_draft("c1", "expense-report")
        assert await conv_module.load_collecting_draft("c1") is None
        assert await conv_module.load_pending_draft("c1", "expense-report") is None


class TestHistoryFallbackToPostgres:
    """Redis 이력은 24시간뿐인데 턴은 Postgres 에 영구히 있다 — 비었으면 되살린다.

    이게 없으면 하루 지난 대화방에서 "어제 그거" 가 통째로 안 됐다. 이력이 사라진 게 아니라
    아무도 안 읽었을 뿐이다(2026-09-22 진단).
    """

    _KEY = "conversation:conv-1"
    _TURNS = [
        {"query": "어제 뭐 했더라", "answer": "정산 자료를 정리했습니다.", "created_at": None},
        {"query": "그거 다시 줘", "answer": "여기 있습니다.", "created_at": None},
    ]

    @pytest.fixture
    def redis(self, monkeypatch):
        fake = _FakeRedis()
        monkeypatch.setattr(conv_module, "redis_store", fake)
        return fake

    def _patch_db(self, monkeypatch, turns, spy=None):
        import agent_arch.db.repository.messages as repo

        async def _fetch(conversation_id, user_sub, limit, since=None):
            if spy is not None:
                spy.append({"cid": conversation_id, "sub": user_sub, "limit": limit, "since": since})
            return turns if user_sub == "u1" else []      # 소유권은 SQL 이 강제한다

        monkeypatch.setattr(repo, "fetch_recent_turns", _fetch)

    async def _load(self, sub="u1"):
        return await conv_module.load_history(
            self._KEY, conversation_id="conv-1", user_sub=sub)

    @pytest.mark.asyncio
    async def test_empty_cache_is_restored_from_db(self, redis, monkeypatch):
        self._patch_db(monkeypatch, self._TURNS)

        assert [(m.role, m.content) for m in await self._load()] == [
            ("user", "어제 뭐 했더라"), ("assistant", "정산 자료를 정리했습니다."),
            ("user", "그거 다시 줘"), ("assistant", "여기 있습니다."),
        ]

    @pytest.mark.asyncio
    async def test_restored_history_is_written_back(self, redis, monkeypatch):
        self._patch_db(monkeypatch, self._TURNS)
        await self._load()

        assert len(redis.lists[self._KEY]) == 4       # 다음 턴은 캐시에서 바로 나온다
        assert redis.expires[self._KEY] == conv_module.SESSION_TTL

    @pytest.mark.asyncio
    async def test_written_back_text_is_raw_not_stripped(self, redis, monkeypatch):
        """Redis 에는 PG 와 같은 원문이 들어가야 한다 — 정화는 읽는 자리에서만 한다."""
        raw = "정리했습니다. " + conv_module._ARTIFACT_PHRASES[0][0] + "."
        self._patch_db(monkeypatch, [{"query": "q", "answer": raw, "created_at": None}])

        history = await self._load()

        assert conv_module._ARTIFACT_PHRASES[0][0] not in history[1].content
        stored = json.loads(redis.lists[self._KEY][1])["content"]
        assert conv_module._ARTIFACT_PHRASES[0][0] in stored

    @pytest.mark.asyncio
    async def test_without_ids_it_behaves_exactly_as_before(self, redis, monkeypatch):
        """게스트 등 인자를 안 넘기는 호출부는 DB 를 건드리지 않는다."""
        spy = []
        self._patch_db(monkeypatch, self._TURNS, spy)

        assert await conv_module.load_history(self._KEY) == []
        assert spy == []

    @pytest.mark.asyncio
    async def test_kill_switch_disables_it(self, redis, monkeypatch):
        spy = []
        self._patch_db(monkeypatch, self._TURNS, spy)
        monkeypatch.setattr(conv_module.settings, "enable_history_hydration", False, raising=False)

        assert await self._load() == [] and spy == []

    @pytest.mark.asyncio
    async def test_other_peoples_conversation_yields_nothing(self, redis, monkeypatch):
        self._patch_db(monkeypatch, self._TURNS)
        assert await self._load(sub="남") == []

    @pytest.mark.asyncio
    async def test_half_turns_are_dropped_whole(self, redis, monkeypatch):
        """답이 없는 턴을 반쪽만 넣으면 user 메시지가 연달아 두 개 생겨 역할 교대가 깨진다."""
        self._patch_db(monkeypatch, [
            {"query": "답이 없는 질문", "answer": "", "created_at": None},
            {"query": "정상 질문", "answer": "정상 답", "created_at": None},
        ])

        roles = [m.role for m in await self._load()]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_preferences_decide_how_much(self, redis, monkeypatch):
        """사용자가 정한 max_history·TTL 을 따른다 — 적재 규칙이 append_turn 과 갈라지면 안 된다."""
        spy = []
        self._patch_db(monkeypatch, self._TURNS, spy)
        monkeypatch.setattr(
            conv_module, "get_user_preferences",
            lambda sub: _async_value(_make_prefs(user_sub="u1", max_history=6,
                                                 session_ttl_seconds=600)))

        await self._load()

        assert spy[0]["limit"] == 3                    # 6 메시지 = 3턴
        assert redis.expires[self._KEY] == 600

    @pytest.mark.asyncio
    async def test_oversized_history_is_trimmed_to_the_compaction_budget(self, redis, monkeypatch):
        """압축 요약은 PG 에 안 남는다 — 원문으로 되살리면 압축 전보다 커질 수 있다."""
        big = "가" * 20000
        self._patch_db(monkeypatch, [{"query": f"질문{i}", "answer": big, "created_at": None}
                                     for i in range(5)])

        history = await self._load()

        # 최신 한 쌍은 예산을 넘겨도 남긴다 — 이번 질문의 직전 맥락까지 버리면 답을 못 한다.
        assert [(m.role, m.content[:4]) for m in history] == [("user", "질문4"), ("assistant", "가가가가")]

    @pytest.mark.asyncio
    async def test_cleared_context_is_not_resurrected(self, redis, monkeypatch):
        """/compact 는 "이제 이전 대화를 기억하지 않는다" 고 약속한다 — 폴백이 그걸 깨면 안 된다."""
        spy = []
        self._patch_db(monkeypatch, self._TURNS, spy)
        await conv_module.clear_history(self._KEY)

        await self._load()

        assert spy[0]["since"] is not None              # 비운 시각 이후만 되살린다

    @pytest.mark.asyncio
    async def test_compaction_or_another_hydration_blocks_the_write(self, redis, monkeypatch):
        """압축은 delete → rpush 라 리스트가 잠깐 빈다. 동시 요청 둘이면 이력이 두 배가 된다."""
        self._patch_db(monkeypatch, self._TURNS)
        redis.kv[conv_module._compact_lock_key(self._KEY)] = "1"

        history = await self._load()

        assert len(history) == 4                      # 이번 턴은 맥락을 얻고
        assert self._KEY not in redis.lists           # 적재는 다음 턴으로 미룬다

    @pytest.mark.asyncio
    async def test_db_failure_is_not_fatal(self, redis, monkeypatch):
        import agent_arch.db.repository.messages as repo

        async def _boom(*a, **k):
            raise RuntimeError("pool down")

        monkeypatch.setattr(repo, "fetch_recent_turns", _boom)
        assert await self._load() == []


async def _async_value(value):
    return value
