"""서비스 이름 정규화 표.

도구 prefix 또는 ``llm_tools.json.external_service`` 값을 Frappe payload의
``permissions`` dict 서비스 키로 통일한다.

예: ``mattermost_skill_*`` 도구는 payload상 ``"talk"`` 서비스에 해당하므로
    ``normalize_service("mattermost") -> "talk"``.

새 매핑은 이 dict 한 군데에서만 관리한다 (단일 진실원천).
"""
from __future__ import annotations

# 도구 prefix / external_service → payload permissions 서비스 키
SERVICE_ALIAS: dict[str, str] = {
    # mattermost 스킬 prefix (정적) → Frappe 서비스 "talk"
    "mattermost": "talk",
    # 동적 API 도구의 external_service 값들
    "mattermost_service": "talk",
    "hr_service": "hr",
    # 레거시 물류 서비스(mespoc_service_*, MES 앱 PoC 데모 전용, 2026-08-24 격리) —
    # 이 한 줄이 (a) required_permissions 누락 시 권한 폴백 키와 (b) 검색 인덱스의
    # 라우팅 도메인(loader.py 가 normalize_service(external_service) 로 도출) 을 동시에
    # 결정한다. "mespoc" 은 Frappe 에 없는 키지만 231건 전부 명시 메타(PUBLIC)라 폴백
    # 실경로가 없고 도메인 축으로만 쓰인다. "logistics" 로 바꾸면 PoC 가 물류 MCP
    # 도메인(일반 채팅)에 새고, "mes" 는 향후 MES 전용 도구 몫이라 충돌 — 유지할 것.
    "logistics_service": "mespoc",
    "mes_service": "mes",
}


def normalize_service(raw: str) -> str:
    """원시 서비스 식별자를 payload permissions 서비스 키로 변환한다.

    매핑이 없으면 입력값을 그대로 반환한다.

    Args:
        raw: 도구 prefix(``"mattermost"``) 또는 ``external_service``(``"hr_service"``)
            등 정규화 전 식별자.

    Returns:
        Frappe payload의 ``permissions`` dict 키와 동일한 형태의 문자열.
    """
    return SERVICE_ALIAS.get(raw, raw)
