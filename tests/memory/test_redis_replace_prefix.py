"""replace_prefix 의 Lua 를 실제 Lua 인터프리터(fakeredis[lua])로 돌려 본다.

대화 압축은 요약한 앞쪽 구간만 원자적으로 바꾼다. 앞쪽이 그사이 바뀌었으면 아무것도 바꾸지 않는다.
"""
import pytest

fakeredis = pytest.importorskip("fakeredis")

from agent_arch.memory.redis_store import RedisStore  # noqa: E402


@pytest.fixture
def store():
    st = RedisStore()
    st.redis = fakeredis.FakeAsyncRedis()
    return st


@pytest.mark.asyncio
async def test_swaps_prefix_and_keeps_tail(store):
    await store.rpush("k", {"a": 1}, {"b": 2}, {"c": 3})
    prefix = await store.lrange_raw("k", 0, 1)
    await store.rpush("k", {"d": 4})                     # 압축 중 붙은 턴

    assert await store.replace_prefix("k", prefix, [{"sum": 1}], 60) is True
    assert await store.lrange("k") == [{"sum": 1}, {"c": 3}, {"d": 4}]
    assert 0 < await store.redis.ttl("k") <= 60


@pytest.mark.asyncio
async def test_refuses_when_prefix_changed(store):
    await store.rpush("k", {"a": 1}, {"b": 2}, {"c": 3})
    prefix = await store.lrange_raw("k", 0, 1)
    await store.rpush("k", {"d": 4})
    await store.ltrim("k", -3, -1)                       # 앞쪽이 트림으로 바뀜(길이 그대로)

    assert await store.replace_prefix("k", prefix, [{"sum": 1}], 60) is False
    assert await store.lrange("k") == [{"b": 2}, {"c": 3}, {"d": 4}]
