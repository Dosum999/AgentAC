import asyncio

from fastapi import APIRouter

from agent_arch.core.config import settings
from agent_arch.db.connection import db_pool
from agent_arch.memory.redis_store import redis_store

router = APIRouter(prefix="/health", tags=["health"])

# 의존 서비스 확인 한 건의 상한(초). 헬스 체크가 의존 장애에 같이 매달리면 안 된다.
_PROBE_TIMEOUT = 2.0


async def _redis_status() -> str:
    if redis_store.redis is None:
        return "not_connected"
    try:
        await asyncio.wait_for(redis_store.redis.ping(), timeout=_PROBE_TIMEOUT)
        return "healthy"
    except Exception as e:  # noqa: BLE001
        return f"unhealthy: {type(e).__name__}"


@router.get("")
async def health_check():
    """헬스 체크 — 프로세스는 살아 있으면 200, 의존 서비스 상태는 본문으로 알린다.

    의존 서비스가 하나라도 연결돼 있지 않으면 ``status`` 는 ``degraded`` 다. 예전에는 늘 ``healthy`` 를
    돌려 "DB 풀 없이 떠서 대화가 저장되지 않는" 상태를 헬스 체크로 알 수 없었다.
    """
    services = {
        "redis": await _redis_status(),
        "postgres": "healthy" if db_pool.get_pool() is not None else "not_connected",
    }
    status = "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    return {
        "status": status,
        "version": settings.app_version,
        "environment": settings.environment,
        "services": services,
    }
