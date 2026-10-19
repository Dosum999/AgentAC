"""연결 풀 세션 타임존 고정.

``created_at`` 은 ``DEFAULT CURRENT_TIMESTAMP``(timestamp without time zone)라 세션 타임존 시각이
저장된다. NKS Postgres 가 KST 로 초기화돼 새 대화가 9시간 미래로 저장되고 리뷰 푸시에서 빠졌다
(2026-10-07). 접속이 UTC 를 요구하는지 못 박는다.
"""
from unittest.mock import AsyncMock, patch

import pytest

from agent_arch.db.connection import DBPool


@pytest.mark.asyncio
async def test_pool_sessions_are_pinned_to_utc():
    with patch("agent_arch.db.connection.asyncpg.create_pool", new=AsyncMock(return_value=object())) as create:
        await DBPool().connect()
    assert create.await_args.kwargs["server_settings"] == {"timezone": "UTC"}
