"""권한 요구사항 인덱스.

``tool_name -> PermissionRequirement`` 단일 룩업을 제공한다. 실제로 인덱스를
채우는 경로는 **모두 register() 호출 세 곳**이다:

    1. import 시: ``@register_tool`` 데코레이터가 클래스의
       ``permission_required`` 를 등록 (``app.agents.tools.registry``).
    2. 실행 시: ``DynamicAPITool`` 생성자가 ``llm_tools.json`` 의
       ``required_permissions`` 를 등록 (없으면 ACCESS_ONLY 폴백).
    3. 부팅 시: MCP provider 가 서버별 정책을 등록
       (``app.agents.tools.mcp.provider.register_server_tools``).

메타가 없는 도구는 게이트에서 전원 deny 되어 아무에게도 노출되지 않으므로
(fail-safe), 등록 누락은 "기능이 없음"으로 위장된다 — 부팅 시
``app.core.lifespan._warn_unmapped_tools`` 가 이를 경고하고, 게이트도
``DenyReason.NO_META`` 로 WARNING 을 남긴다.

``load_from_json`` / ``_DEFAULT_SKILL_PATH`` 는 **현재 호출자가 없다**
(``data/skill_permissions.json`` 도 존재하지 않는다). JSON 으로 매핑을 외부화·
핫리로드할 여지를 남겨 보존한 코드이며, 지금은 테스트에서만 쓰인다. 운영에서
권한 매핑을 찾을 때 이 파일을 뒤지지 말 것 — 위 세 경로가 진실원천이다.

게이트는 도구 이름으로만 룩업한다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from agent_arch.agents.permissions.spec import GateMode, PermissionRequirement

logger = logging.getLogger(__name__)

_DEFAULT_SKILL_PATH = "data/skill_permissions.json"


class PermissionIndex:
    """tool_name → PermissionRequirement 인메모리 인덱스 (싱글톤)."""

    def __init__(self) -> None:
        self._index: dict[str, PermissionRequirement] = {}

    def load_from_json(self, path: str = _DEFAULT_SKILL_PATH) -> int:
        """JSON 파일에서 인덱스를 재구성한다.

        **현재 호출자 없음** (모듈 docstring 참조) — 매핑 외부화를 위한 여지로
        보존된 경로이며, 호출하면 register() 로 채워진 기존 인덱스를 **덮어쓴다**.

        파일이 없으면 빈 인덱스로 초기화하고 경고만 남긴다 — 운영 환경에서
        매핑이 아직 배포되지 않은 상태에서도 서버 부팅이 막히지 않도록 한다.
        메타 없는 도구는 게이트에서 deny되므로 fail-safe 동작.

        Args:
            path: 로드할 JSON 파일 경로.

        Returns:
            로드된 항목 수.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(
                "skill_permissions.json not found: %s — 빈 인덱스로 시작",
                file_path,
            )
            self._index = {}
            return 0

        with open(file_path, encoding="utf-8") as f:
            raw: dict = json.load(f)

        new_index: dict[str, PermissionRequirement] = {}
        for tool_name, spec in raw.items():
            # ``_`` prefix 키는 주석·메타용으로 예약 (예: ``"_comment"``).
            if tool_name.startswith("_") or not isinstance(spec, dict):
                continue
            try:
                new_index[tool_name] = self._parse(spec)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "skill_permissions: '%s' 항목 파싱 실패 — 무시 (deny default): %s",
                    tool_name, e,
                )
        self._index = new_index
        logger.info("skill_permissions 로드 완료: %d개", len(new_index))
        return len(new_index)

    @staticmethod
    def _parse(spec: dict) -> PermissionRequirement:
        """단일 항목 dict → PermissionRequirement 변환."""
        return PermissionRequirement.from_dict(spec)

    def register(self, tool_name: str, req: PermissionRequirement) -> None:
        """동적 도구가 런타임에 자기 권한 메타를 등록할 때 사용.

        같은 이름이 이미 있으면 덮어쓴다 (마지막 등록자 우선).
        """
        self._index[tool_name] = req

    def get(self, tool_name: str) -> Optional[PermissionRequirement]:
        """도구 이름으로 요구사항을 조회한다.

        Returns:
            등록된 PermissionRequirement, 없으면 None.
        """
        return self._index.get(tool_name)

    def size(self) -> int:
        """현재 인덱스의 항목 수."""
        return len(self._index)

    def clear(self) -> None:
        """인덱스 비우기 (주로 테스트용)."""
        self._index = {}


permission_index = PermissionIndex()
