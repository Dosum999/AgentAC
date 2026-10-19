"""슬래시 커맨드 처리 모듈.

사용자 입력이 LLM에 전달되기 전에 슬래시 커맨드를 감지하고 처리한다.
커맨드가 감지되면 LLM 호출 없이 즉시 결과를 반환한다.
"""
from typing import Optional, Dict, Any
from agent_arch.core.config import settings


# 동기 커맨드: async 가 필요 없는 커맨드
_SYNC_COMMANDS: Dict[str, Dict[str, Any]] = {
    "/status": {
        "action": lambda: None,
        "message": None,  # handle_command에서 동적으로 생성
    },
}

# 비동기 커맨드: Redis 등 async I/O가 필요한 커맨드
_ASYNC_COMMANDS = {"/compact", "/refresh-permissions"}


async def handle_command(
    input: str,
    llm_provider: str,
    user_sub: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """슬래시 커맨드를 처리한다.

    커맨드가 감지되면 시스템 설정을 변경하고 결과 딕셔너리를 반환한다.
    커맨드가 아닌 경우 None을 반환하여 일반 처리 흐름을 계속한다.

    Args:
        input (str): 사용자 입력 문자열
        llm_provider (str): 현재 LLM 공급자 이름 (응답 포맷 구성용)
        user_sub (Optional[str]): 사용자 고유 ID. 사용자별 상태 변경 커맨드에 사용.

    Returns:
        Optional[Dict[str, Any]]: 커맨드 처리 결과. 커맨드가 아니면 None.
    """
    command = input.strip().lower()

    # 동기 커맨드 처리
    spec = _SYNC_COMMANDS.get(command)
    if spec is not None:
        spec["action"]()
        if command == "/status":
            message = (
                f"현재 상태\n"
                f"- LLM 프로바이더: {settings.default_llm_provider}\n"
                f"- 에이전트 모드: {settings.agent_mode}"
            )
        else:
            message = spec["message"]
        return {
            "success": True,
            "output": message,
            "llm_provider": settings.default_llm_provider,
            "tool_calls": [],
            "metadata": {
                "command": command.lstrip("/"),
                "agent_mode": settings.agent_mode,
            },
        }

    # 비동기 커맨드 처리
    if command == "/compact":
        return await _handle_compact(llm_provider, user_sub)
    if command == "/refresh-permissions":
        return await _handle_refresh_permissions(llm_provider, user_sub)

    return None


async def _handle_compact(
    llm_provider: str,
    user_sub: Optional[str],
) -> Dict[str, Any]:
    """/compact 커맨드를 처리한다. 사용자 대화 컨텍스트(Redis)를 비운다.

    LLM이 다음 턴에 참조하는 Redis 컨텍스트 키만 삭제한다. PostgreSQL에
    영구 저장된 대화방·메시지 기록은 건드리지 않는다 (의도된 범위 —
    확장 금지). clear_user_conversations의 반환값은 PG 대화방 행 수라
    "실제 삭제된 키 수"가 아니므로 사용자 메시지에 노출하지 않는다.

    Args:
        llm_provider (str): 현재 LLM 공급자 이름
        user_sub (Optional[str]): 사용자 고유 ID

    Returns:
        Dict[str, Any]: 커맨드 처리 결과
    """
    if user_sub:
        # 한 사용자가 여러 대화방을 가지므로 본인 모든 대화방의 컨텍스트 키를 일괄 삭제.
        from agent_arch.memory.conversation import clear_user_conversations
        await clear_user_conversations(user_sub)
        message = (
            "이제 이전 대화 내용을 기억하지 않고 새로 시작합니다. "
            "저장된 대화 기록은 그대로 유지됩니다."
        )
    else:
        message = "로그인 정보가 없어 이전 대화 내용을 비울 수 없습니다."

    return {
        "success": bool(user_sub),
        "output": message,
        "llm_provider": llm_provider,
        "tool_calls": [],
        "metadata": {"command": "compact"},
    }


async def _handle_refresh_permissions(
    llm_provider: str,
    user_sub: Optional[str],
) -> Dict[str, Any]:
    """/refresh-permissions 커맨드를 처리한다.

    사용자 본인의 Frappe 프로필 Redis 캐시(``user_profile:{sub}``)를 즉시 삭제한다.
    다음 질문에서 ``get_profile()`` 가 캐시 미스를 감지해 DB → Redis backfill 로
    프로필을 재조회한다. 직책·권한 변경 즉시 반영용 자가 갱신 수단.

    Args:
        llm_provider (str): 현재 LLM 공급자 이름
        user_sub (Optional[str]): 사용자 고유 ID

    Returns:
        Dict[str, Any]: 커맨드 처리 결과
    """
    if user_sub:
        # 순환 import 방지를 위해 지연 import
        from agent_arch.services.user_profile import invalidate as invalidate_profile
        await invalidate_profile(user_sub)
        message = "권한 정보를 갱신했습니다. 다음 질문부터 새 권한이 반영됩니다."
    else:
        message = "로그인 정보가 없어 권한 캐시를 갱신할 수 없습니다."

    return {
        "success": bool(user_sub),
        "output": message,
        "llm_provider": llm_provider,
        "tool_calls": [],
        "metadata": {"command": "refresh-permissions"},
    }
