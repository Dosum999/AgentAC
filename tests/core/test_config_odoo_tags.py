"""Settings.odoo_lead_tag_id_list CSV 파싱 단위 테스트.

ODOO_LEAD_TAG_IDS 문자열을 정수 ID 리스트로 변환하는 프로퍼티를 검증한다.
confluence_*_list 와 동일한 split-strip 컨벤션을 따른다.

주의: 필드가 alias="ODOO_LEAD_TAG_IDS" 로 정의돼 있어 값 주입은 반드시
alias(대문자) 로 해야 한다. 파이썬 필드명으로 넘기면 extra 로 흡수되고
실제 필드는 기본값을 쓴다.
"""
import pytest

from agent_arch.core.config import Settings


@pytest.mark.parametrize("raw,expected", [
    ("1,33", [1, 33]),            # 기본값 형식
    ("1, 33", [1, 33]),           # 공백 허용
    (" 7 , 9 , 12 ", [7, 9, 12]),  # 양끝/사이 공백
    ("1,,33", [1, 33]),           # 빈 항목 무시
    ("", []),                      # 빈 문자열 → 빈 리스트
    ("5", [5]),                    # 단일 값
])
def test_odoo_lead_tag_id_list_파싱(raw: str, expected: list):
    settings = Settings(ODOO_LEAD_TAG_IDS=raw)
    assert settings.odoo_lead_tag_id_list == expected


def test_기본값은_STK2026과_에이전트챌린지():
    """env 미설정 시 라이브 Odoo 로 확인한 기본 태그 id [1, 33] 를 반환한다."""
    settings = Settings(ODOO_LEAD_TAG_IDS="1,33")
    assert settings.odoo_lead_tag_id_list == [1, 33]
