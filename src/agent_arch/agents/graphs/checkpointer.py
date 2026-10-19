"""LangGraph checkpointer 팩토리 — 상태 영속 백엔드를 한 곳에서 선택한다.

모든 LangGraph 그래프(HITL, 향후 orchestration/react 서브그래프)는 컴파일 시
``compile(checkpointer=get_checkpointer())`` 로 이 팩토리에서 checkpointer 를 얻는다.
백엔드 전환(memory → postgres)이 **설정 한 줄 + 이 파일**로 끝나게 하는 단일 주입 지점.

백엔드:
    - ``memory`` (기본): ``MemorySaver``. in-process·단일 워커 한정, 재배포 시 상태 소실.
      현재 앱은 단일 uvicorn 워커라 승인 흐름이 완전 동작한다(미승인 초안 소실만 감수).
    - ``postgres``: ``AsyncPostgresSaver``. 멀티워커·재배포 생존. 이미 있는 Postgres 를
      재사용하며 신규 인프라가 필요없다. ``langgraph-checkpoint-postgres`` 패키지가 있어야
      하고, 최초 1회 테이블 셋업(``.setup()``)이 필요하다 — orchestration 이관 시점에 배선.

선택 근거는 project_langgraph_hitl / 마이그레이션 전략(Q3) 참조: 바닐라 Redis 는 공식
langgraph-checkpoint-redis(Redis Stack/RedisVL) 를 못 돌려 제외, Postgres 는 이미 보유.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)


def get_checkpointer(name: Optional[str] = None):
    """설정된 백엔드의 LangGraph checkpointer 인스턴스를 만들어 반환한다.

    호출자(그래프 팩토리)가 컴파일된 그래프를 싱글턴으로 캐시하므로, 이 함수는
    매 호출마다 새 saver 를 만든다(그래프당 1회 호출이 전제).

    Args:
        name: 백엔드 명시 오버라이드("memory"/"postgres"). None 이면 설정값 사용.

    Returns:
        BaseCheckpointSaver 구현체(MemorySaver 등).

    Raises:
        RuntimeError: postgres 백엔드인데 langgraph-checkpoint-postgres 미설치/미배선.
        ValueError: 알 수 없는 백엔드 이름.
    """
    backend = (name or settings.langgraph_checkpointer or "memory").strip().lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("[langgraph] checkpointer=MemorySaver (in-process, 단일 워커 한정)")
        return MemorySaver()

    if backend == "postgres":
        # 멀티워커·재배포 생존이 필요할 때 활성화. 신규 런타임 의존성이라 명시 설치 필요.
        try:
            import langgraph.checkpoint.postgres  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LANGGRAPH_CHECKPOINTER=postgres 인데 'langgraph-checkpoint-postgres' 가 "
                "설치돼 있지 않습니다. requirements 에 추가·설치하고, 연결 풀 수명주기와 "
                "최초 .setup() 배선을 완료한 뒤 사용하세요(orchestration 이관 슬라이스)."
            ) from exc
        # 커넥션 풀/‧setup() 수명주기가 앱 lifespan 과 엮여야 안전하므로, 실제 배선 전까지
        # 오연결을 만들지 않는다. (아직 검증되지 않은 경로를 '동작'하는 척하지 않음.)
        raise RuntimeError(
            "postgres checkpointer 는 아직 배선되지 않았습니다. get_checkpointer 의 postgres "
            "분기에서 AsyncPostgresSaver 커넥션 풀 + .setup() 을 앱 lifespan 과 함께 구성하세요."
        )

    raise ValueError(f"알 수 없는 checkpointer 백엔드: {backend!r} (memory|postgres)")
