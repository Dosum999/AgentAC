"""애플리케이션 기동·종료.

기동 순서: Redis → PostgreSQL → LangSmith → API 레지스트리(도구 색인) → MCP 서버 도구 등록 →
(MCP 등록이 모두 끝난 뒤) 레지스트리 최종 색인 → 장기 메모리 v2 스케줄러 → 오류 이벤트 flush 루프.

**외부 의존이 없거나 실패해도 앱은 뜬다.** 각 단계는 상한 시간 안에서 시도하고 실패하면 경고만
남긴다 — 그 기능이 빠진 채로 동작한다(DB 없으면 대화 영구 저장 없음, Qdrant 없으면 도구 색인 없음).
``STARTUP_CONNECT_EXTERNAL=false`` 면 외부 연결 단계를 통째로 건너뛴다(테스트·로컬 조립 확인용).

업무 도메인 기동(사내 지식베이스 수집·메신저 감시·예약 실행기 등)은 코어가 모른다 — 조립부(create_app 의
setup 훅)가 필요한 백그라운드 작업을 직접 띄운다.
"""
import asyncio
import logging
import os
from collections.abc import Callable

from fastapi import FastAPI

from agent_arch.core.config import settings
from agent_arch.db.connection import db_pool
from agent_arch.memory.redis_store import redis_store
from agent_arch.services.api_registry import api_indexer, tool_loader
from agent_arch.services.monitoring import event_extra, watch_task

logger = logging.getLogger(__name__)

# 외부 연결 한 단계의 상한(초). 넘으면 그 기능 없이 뜬다.
_STEP_TIMEOUT = 10.0


async def _step(label: str, coro) -> bool:
    """기동 한 단계를 상한 시간 안에서 시도한다. 실패·초과면 경고만 남기고 False."""
    try:
        await asyncio.wait_for(coro, timeout=_STEP_TIMEOUT)
        return True
    except Exception as e:  # noqa: BLE001 — 기동을 막지 않는다
        logger.warning("%s 초기화 실패 — 이 기능 없이 시작: %s: %r", label, type(e).__name__, e)
        return False


async def _index_with_retry(
    coro_fn: Callable,
    label: str,
    max_retries: int = 6,
    base_delay: float = 5.0,
    mcp_server: str | None = None,
) -> None:
    """인덱싱 코루틴을 실행하고 실패 시 지수 백오프로 재시도한다.

    Args:
        coro_fn: 인자 없이 호출 가능한 async 코루틴 팩토리 함수.
        label: 로그 출력용 레이블 문자열.
        max_retries: 최대 재시도 횟수.
        base_delay: 기본 대기 시간(초). 재시도 간격은 base_delay * 2**attempt.
        mcp_server: MCP 도구 등록이면 서버 키 — 최종 실패가 오류 이벤트에 ``mcp_tool`` 로
            남는다(등록 실패 = 그 서버 도구가 통째로 없는 상태). 그 밖은 ``background``.
    """
    for attempt in range(max_retries):
        try:
            await coro_fn()
            return
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            # 일부 예외(예: httpx.ReadError(''))는 str(e)가 빈 문자열이라 원인이 가려진다.
            logger.warning(
                "%s 인덱싱 실패 (시도 %d/%d): %s: %r",
                label, attempt + 1, max_retries, type(e).__name__, e,
            )
            if attempt < max_retries - 1:
                logger.info("%s 재시도 대기 중 (%.0f초)...", label, delay)
                await asyncio.sleep(delay)
    logger.error(
        "%s 인덱싱 최대 재시도 초과 - 비활성화", label,
        extra=event_extra("mcp_tool" if mcp_server else "background", mcp_server=mcp_server),
    )


def _init_langsmith() -> None:
    """LangSmith 트레이싱을 활성화한다."""
    if not (settings.enable_langsmith and settings.langsmith_api_key):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)


async def _init_api_registry() -> None:
    """API 레지스트리(Qdrant)를 연결하고 동적 도구 색인을 백그라운드로 시작한다.

    정적 스킬(registry) 색인은 여기서 하지 않는다 — MCP 등록이 끝난 뒤 최종 전체 패스에서 한다
    (_index_registry_after_mcp docstring).
    """
    if not await _step("API Registry(Qdrant)", api_indexer.connect()):
        return
    try:
        tools = tool_loader.load_from_json("data/llm_tools.json")
    except FileNotFoundError:
        logger.warning("data/llm_tools.json not found - 동적 API 도구 비활성화")
        return
    watch_task(asyncio.create_task(
        _index_with_retry(lambda: api_indexer.index_tools(tools), "API Registry")
    ), "api_registry_index")
    logger.info("API Registry: %d개 도구 백그라운드 인덱싱 시작", len(tools))


def _warn_unmapped_tools() -> None:
    """권한 메타가 등록되지 않은 도구를 기동 로그에 드러낸다.

    메타 없는 도구는 게이트에서 전원 deny 되어 **아무에게도 노출되지 않는데**, 사용자에게는
    "그런 기능이 없습니다"로 나간다 — 등록 누락이 기능 부재로 위장된다. 정적·MCP 도구 등록이
    모두 끝난 뒤(_index_registry_after_mcp) 한 번 확인한다.
    """
    from agent_arch.agents.permissions import permission_index
    from agent_arch.agents.tools.registry import ToolRegistry

    unmapped = sorted(
        name for name in ToolRegistry.get_all() if permission_index.get(name) is None
    )
    if unmapped:
        logger.warning(
            "권한 메타 미등록 도구 %d개 — 전원 deny 되어 노출되지 않습니다: %s. "
            "정적 도구는 클래스에 permission_required 를 선언해야 합니다.",
            len(unmapped), ", ".join(unmapped),
        )
    else:
        logger.info(
            "권한 메타 커버리지 정상: 등록 도구 %d개 전부 매핑됨", len(ToolRegistry.get_all())
        )


async def _index_registry_after_mcp(mcp_tasks: list) -> None:
    """모든 MCP 등록 태스크 종료 후 레지스트리 전체(정적+MCP)를 최종 색인한다.

    증분 색인(provider, 부분 목록)은 자기 목록에 없는 registry-origin 포인트를 고아로 판정해
    지운다 — 부팅 초기에 정적 패스를 먼저 돌리면 MCP 증분들이 그 포인트를 도로 지워 검색에서
    사라졌다(부팅마다 "고아 포인트 삭제" x 서버 수). 그래서 부분 패스가 **모두 끝난 마지막**에
    전체 스냅샷 패스를 돌린다: 이때 레지스트리에는 정적+MCP 가 다 있어 고아 판정이 정확하고,
    증분들의 상호 삭제로 뜯긴 포인트도 해시 불일치로 다시 색인된다.
    """
    if mcp_tasks:
        await asyncio.gather(*mcp_tasks, return_exceptions=True)  # 재시도 소진으로 유한 종결
    _warn_unmapped_tools()
    if not settings.unified_tool_indexing:
        return
    from agent_arch.services.api_registry.registry_loader import defs_from_registry

    registry_defs = defs_from_registry()
    await _index_with_retry(
        lambda: api_indexer.index_tools(registry_defs), "Registry 도구 최종 인덱스"
    )
    logger.info("Registry 도구: %d개 최종 인덱싱 완료 (MCP 등록 후 전체 패스)", len(registry_defs))


#: 목록 불일치를 다시 확인하는 간격(초)과 횟수. 배포 순서가 어긋난 경우는 대개 1~2분 안에
#: 상대가 새 버전으로 올라온다. 길게 끌어 봐야 정책 잔재(영구 불일치)에서 소음만 커진다.
_SKEW_RECHECK_DELAY = 20.0
_SKEW_RECHECK_TRIES = 4


async def _register_mcp_server(spec) -> None:
    """MCP 서버 도구를 등록한다. 목록 불일치는 **실패로 치지 않는다.**

    연결 실패는 그대로 올려 호출부의 백오프 재시도를 태운다. 반면 "기대한 도구가 목록에 없다"는
    **등록 자체는 성공한 것**이므로 여기서 삼키고 백그라운드 재확인만 건다. 두 가지를 같은 실패로
    취급하면, 서버가 일부 도구를 영구히 내리지 않을 때 매 기동 재시도가 소진될 때까지(수 분)
    최종 색인이 밀린다.
    """
    from agent_arch.agents.tools.mcp.provider import register_server_tools
    from agent_arch.agents.tools.mcp.skew import MCPToolListSkew

    try:
        await register_server_tools(spec)
    except MCPToolListSkew as skew:
        logger.warning("%s 등록된 도구는 유효 — 목록만 재확인합니다.", skew)
        watch_task(asyncio.create_task(_recheck_mcp_skew(spec)), f"mcp_skew_recheck:{spec.name}")


async def _recheck_mcp_skew(spec) -> None:
    """상대가 새 버전으로 올라오는지 짧게 다시 본다(기동을 막지 않는 백그라운드)."""
    from agent_arch.agents.tools.mcp.provider import register_server_tools
    from agent_arch.agents.tools.mcp.skew import MCPToolListSkew

    last: Exception | None = None
    for attempt in range(_SKEW_RECHECK_TRIES):
        await asyncio.sleep(_SKEW_RECHECK_DELAY)
        try:
            await register_server_tools(spec)
        except MCPToolListSkew as e:
            last = e
            continue
        except Exception as e:  # noqa: BLE001 — 연결 문제면 다음 차례에 다시 본다
            last = e
            logger.warning("[mcp:%s] 목록 재확인 실패(%d/%d): %s: %r",
                           spec.name, attempt + 1, _SKEW_RECHECK_TRIES, type(e).__name__, e)
            continue
        logger.info("[mcp:%s] 목록 불일치 해소 — 기대한 도구가 모두 등록됐습니다.", spec.name)
        return
    missing = getattr(last, "missing", None)
    logger.warning(
        "[mcp:%s] 목록 불일치가 %.0f초 동안 남아 있습니다: %s. "
        "서버가 그 도구를 비활성화했거나(정책표 잔재) 배포가 더 늦어지는 중입니다 — "
        "다음 재기동 때 다시 확인합니다.",
        spec.name, _SKEW_RECHECK_DELAY * _SKEW_RECHECK_TRIES,
        ", ".join(missing) if missing else last,
    )


def _init_mcp_tools() -> None:
    """MCP 서버(들) 도구 등록을 서버별 백그라운드 태스크로 시작한다.

    서버가 백엔드보다 늦게 떠도 백오프 재시도로 등록되며, 한 서버 장애가 다른 서버 등록을 막지
    않는다(태스크 분리). orchestrator 는 요청마다 레지스트리를 읽으므로 등록 즉시 반영된다.
    MCP 서버가 없어도 정적 스킬 색인을 위해 최종 패스는 항상 예약한다.
    """
    from agent_arch.agents.tools.mcp.provider import server_specs

    specs = server_specs()
    tasks: list = []
    if not specs:
        logger.info("설정된 MCP 서버 없음 (URL/TOKEN 미설정) - MCP 도구 비활성화")
    else:
        tasks = [
            watch_task(asyncio.create_task(
                _index_with_retry(
                    lambda s=spec: _register_mcp_server(s), f"{spec.name} MCP 도구",
                    mcp_server=spec.name,
                )
            ), f"mcp_register:{spec.name}")
            for spec in specs
        ]
    watch_task(asyncio.create_task(_index_registry_after_mcp(tasks)), "registry_index_after_mcp")


def _init_error_events():
    """시스템 오류 이벤트 flush 루프(메모리 버퍼 → agent.error_events)를 띄운다.

    수집 자체(로그 핸들러·HTTP 미들웨어)는 기동 전부터 켜져 있고 메모리 버퍼에만 쌓는다.

    Returns:
        asyncio.Task | None. 비활성화면 None.
    """
    if not settings.enable_error_events:
        logger.info("오류 이벤트 수집 비활성화 (ENABLE_ERROR_EVENTS=false)")
        return None
    from agent_arch.services.monitoring import run_error_events_flush_loop

    return watch_task(
        asyncio.create_task(run_error_events_flush_loop()), "error_events_flush", forever=True
    )


def _init_memory_v2_scheduler():
    """장기 메모리 v2 배치 스케줄러(캡처된 턴을 L1 atom 으로 추출)를 띄운다.

    MEMORY_V2_MODE 가 shadow/active 이고 ENABLE_USER_MEMORY 가 켜진 경우에만.

    Returns:
        asyncio.Task | None — 종료부가 묶어서 ``cancel()`` 하므로 태스크가 아니면 안 된다.
    """
    try:
        from agent_arch.services.memory_v2 import enabled as memory_v2_enabled
        if not memory_v2_enabled():
            return None
        from agent_arch.services.memory_v2.scheduler import scheduler
        task = watch_task(scheduler.start(), "memory_v2_scheduler")
        logger.info("Memory v2 스케줄러 시작 (mode=%s)", settings.memory_v2_mode)
        # active 면 회상이 핫 패스에 있다 — Qdrant 연결·임베딩 콜드 스타트를 기동 시 미리 치른다.
        from agent_arch.services.memory_v2 import is_active as memory_v2_active
        if memory_v2_active():
            from agent_arch.services.memory_v2.recall import warm_up
            watch_task(asyncio.create_task(warm_up()), "memory_v2_warm_up")
        return task
    except Exception as e:
        logger.error("Memory v2 스케줄러 초기화 실패: %s", e)
        return None


async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 이벤트를 관리한다."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # 답변 가드(이름 환각·답변 언어·환각 액션)는 LangGraph 경로에만 있다. 레거시 loop 경로로 돌면
    # 가드 없이 도는데 아무 신호가 없으므로 기동 로그에 한 줄 남긴다.
    if not settings.use_langgraph_react and not settings.use_langgraph_supervisor:
        logger.warning(
            "USE_LANGGRAPH_REACT/SUPERVISOR 가 모두 꺼져 있습니다 — 레거시 loop 경로로 동작하며 "
            "답변 가드가 **비활성**입니다."
        )

    tasks: list = []
    connected = bool(settings.startup_connect_external)
    if connected:
        await _step("Redis", redis_store.connect())
        await _step("PostgreSQL", db_pool.connect())
        _init_langsmith()
        await _init_api_registry()
        _init_mcp_tools()
        tasks += [_init_memory_v2_scheduler(), _init_error_events()]
    else:
        logger.info("STARTUP_CONNECT_EXTERNAL=false — 외부 연결 없이 시작")

    yield

    logger.info("Shutting down...")
    for task in tasks:
        if task is not None:
            task.cancel()
    if connected:
        from agent_arch.services.monitoring import flush_buffer

        try:
            # 마지막 버퍼를 DB 가 닫히기 전에 비운다.
            await asyncio.wait_for(flush_buffer(), timeout=3)
        except Exception:  # noqa: BLE001 — 종료를 막지 않는다
            pass
        try:
            await redis_store.disconnect()
        except Exception:  # noqa: BLE001
            pass
        await db_pool.disconnect()
