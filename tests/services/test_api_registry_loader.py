"""ToolLoader 의 도메인 조회(defs_by_domain) 계약.

채널 스코핑(특정 진입점에서만 한 도메인을 여는 노출)이 인덱스 검색 대신 이 조회로
도메인 전체를 직접 바인딩한다 — 검색은 top_k 근사라 "그 도메인의 전부"를 보장하지
못하기 때문이다 (채널 봇 구성이 그 도메인 도구를 직접 바인딩할 때).
"""
from __future__ import annotations

import json

import pytest

from agent_arch.services.api_registry.loader import ToolLoader


@pytest.fixture
def tools_json(tmp_path):
    """도메인이 섞인 llm_tools.json 을 만든다 (enabled/비enabled 포함)."""
    rows = [
        {
            "tool_name": "mattermost_service_list_channels",
            "description": "작업지시 목록",
            "is_external": 1,
            "external_service": "mattermost_service",
            "http_method": "GET",
            "endpoint_path": "/work-orders",
            "parameters_schema": "{}",
            "enabled": 1,
        },
        {
            "tool_name": "mattermost_service_disabled_one",
            "description": "꺼진 도구",
            "is_external": 1,
            "external_service": "mattermost_service",
            "http_method": "GET",
            "endpoint_path": "/nope",
            "parameters_schema": "{}",
            "enabled": 0,
        },
        {
            "tool_name": "hr_service_list_logs",
            "description": "근태 로그",
            "is_external": 1,
            "external_service": "hr_service",
            "http_method": "GET",
            "endpoint_path": "/logs",
            "parameters_schema": "{}",
            "enabled": 1,
        },
    ]
    path = tmp_path / "llm_tools.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_returns_only_requested_domain(tools_json):
    """요청한 도메인 항목만 돌려준다 — 다른 도메인이 섞이면 채널 격리가 깨진다."""
    loader = ToolLoader()
    loader.load_from_json(tools_json)

    names = {d.tool_name for d in loader.defs_by_domain("talk")}

    assert names == {"mattermost_service_list_channels"}


def test_excludes_disabled_tools(tools_json):
    """enabled 가 아닌 정의는 애초에 캐시에 없다 (인덱싱 대상과 같은 집합)."""
    loader = ToolLoader()
    loader.load_from_json(tools_json)

    names = {d.tool_name for d in loader.defs_by_domain("talk")}

    assert "mattermost_service_disabled_one" not in names


def test_unknown_domain_is_empty(tools_json):
    """없는 도메인은 빈 리스트 — 호출부가 스킬만으로 계속 동작한다."""
    loader = ToolLoader()
    loader.load_from_json(tools_json)

    assert loader.defs_by_domain("nope") == []
