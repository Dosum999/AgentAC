"""사내 메신저 앱 설치 안내 — 고정 사실이 프롬프트에 살아 있는지 검증.

회귀 1 (실측 2026-08-10, 금아금속 "모바일 메신저 사용방법 알려줘"): 모델이 일반 지식으로
"App Store 또는 Google Play 에서 Mattermost 를 검색해 설치하고 서버 주소를 입력하라"고
답했다. 실제 앱은 Unlisted(사내·고객사 전용) 라 스토어 검색으로는 나오지 않는다.

회귀 2 (2026-08-11): 그 수정에서 배포 경로를 iOS 전용 앱스토어 링크로만 잡았는데 틀렸다.
실제 배포처는 **DMP 다운로드 센터**이고 Windows·Android·iOS 셋 다 받을 수 있다 —
"안드로이드는 없다"는 안내는 사실이 아니다.

프롬프트에 박은 고정 사실이라 도구 호출 없이도 성립해야 하고, 리팩터링 중 조용히
사라지면 곧바로 같은 환각으로 되돌아간다. 그래서 문자열 계약으로 잠근다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod

_APP_URL = "https://apps.apple.com/kr/app/chat-app/id6782740913"
_SECTION = "### 사내 메신저 앱 설치 안내"


@pytest.fixture
def production_prompt() -> str:
    """실사용 경로(default 모드)에서 조립되는 시스템 프롬프트."""
    return sys_mod.build_system_prompt(mode="default")


@pytest.fixture
def section(production_prompt: str) -> str:
    """설치 안내 섹션 본문만 잘라낸다(다음 섹션 헤더 전까지)."""
    assert _SECTION in production_prompt
    return production_prompt.split(_SECTION)[1].split("### 실행 요청")[0]


def test_download_center_path_is_present(section: str):
    """다운로드 센터 진입 경로(물음표 아이콘 → 메뉴)가 안내돼야 한다."""
    assert "다운로드 센터" in section
    assert "조직도 아이콘 옆 물음표" in section


def test_download_center_link_is_rendered(section: str):
    """클릭 경로만으로는 손이 많이 간다 — 바로 여는 링크가 함께 있어야 한다."""
    assert f"{sys_mod.settings.dmp_portal_url.rstrip('/')}/main/download" in section
    assert "<<DOWNLOAD_CENTER_URL>>" not in section  # 마커가 그대로 새면 안 됨


def test_download_center_link_follows_portal_setting(monkeypatch):
    """링크 오리진은 DMP_PORTAL_URL 설정을 따라야 한다(dev/sb 는 다른 도메인)."""
    monkeypatch.setattr(sys_mod.settings, "dmp_portal_url", "https://example.com/")
    prompt = sys_mod.build_system_prompt(mode="default")
    assert "https://example.com/main/download" in prompt
    assert "https://example.com//main/download" not in prompt  # 슬래시 중복 방지


def test_all_three_platforms_are_listed(section: str):
    """Windows·Android·iOS 셋 다 받을 수 있다는 사실이 있어야 한다."""
    for os_name in ("Windows", "Android", "iOS"):
        assert os_name in section


def test_ios_app_store_link_is_present(section: str):
    """아이폰 보조 경로인 앱스토어 전용 링크는 그대로 노출돼야 한다."""
    assert _APP_URL in section


def test_unlisted_caveat_is_present(section: str):
    """'스토어 검색으로 안 나온다'는 단서가 빠지면 사용자가 검색하다 포기한다."""
    assert "Unlisted" in section
    assert "검색해도 나오지 않습니다" in section


def test_ios_only_claim_is_gone(production_prompt: str):
    """회귀 2 — 'iOS만 제공'·'안드로이드용 앱은 없습니다' 단정이 남아 있으면 안 된다."""
    assert "아이폰(iOS)만 제공" not in production_prompt
    assert "안드로이드용 앱은 없습니다" not in production_prompt


def test_store_search_guidance_is_forbidden(section: str):
    """스토어 검색 안내 금지 조항이 있어야 한다(회귀 1 답변의 정확한 형태)."""
    assert "금지" in section
    assert "Google Play" in section  # 금지 대상으로 명시
    assert "Mattermost" in section


def test_guidance_survives_tool_section_render(production_prompt: str):
    """도구 섹션 자동 렌더(<<...>>) 치환 뒤에도 섹션이 남아 있어야 한다."""
    assert _SECTION in production_prompt
    assert "<<" not in production_prompt
