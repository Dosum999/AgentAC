"""asyncpg 연결 풀 싱글톤 모듈.

앱 시작 시 pool을 생성하고 종료 시 닫는다.
"""
import logging
from typing import Optional
import asyncpg

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)


class DBPool:
    """asyncpg 연결 풀을 래핑하는 싱글톤 클래스."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """asyncpg 연결 풀을 생성한다.

        DATABASE_URL 환경변수의 DSN을 사용한다.
        실패 시 경고 로그만 남기고 앱 시작은 계속된다.

        세션 타임존을 **UTC 로 고정**한다. ``created_at`` 류는 ``timestamp without time zone`` +
        ``DEFAULT CURRENT_TIMESTAMP`` 라 세션 타임존 시각이 그대로 저장되는데, 조회·푸시 코드는 전부
        naive UTC 를 전제한다. stage(compose) Postgres 는 TZ 가 없어 UTC 였지만 NKS Postgres 는
        ``TZ=Asia/Seoul`` 로 떠서 새 대화가 9시간 미래로 저장됐고, 리뷰 푸시(``created_at < 지금 UTC``)
        에서 빠졌다(2026-10-07). DB 설정과 무관하게 같은 의미가 되도록 접속 쪽에서 정한다.
        """
        try:
            # asyncpg는 postgresql:// 또는 postgres:// 스킴 모두 지원
            dsn = settings.database_url
            self._pool = await asyncpg.create_pool(
                dsn=dsn, min_size=2, max_size=10, server_settings={"timezone": "UTC"}
            )
            logger.info("PostgreSQL 연결 풀 생성 완료")
        except Exception:
            logger.warning("PostgreSQL 연결 풀 생성 실패 — 대화 영구 저장 비활성화", exc_info=True)
            self._pool = None

    async def disconnect(self) -> None:
        """연결 풀을 닫는다."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL 연결 풀 종료")

    def get_pool(self) -> Optional[asyncpg.Pool]:
        """현재 연결 풀을 반환한다.

        Returns:
            asyncpg.Pool: 연결 풀. 초기화 실패 시 None.
        """
        return self._pool


# 모듈 수준 싱글톤
db_pool = DBPool()
