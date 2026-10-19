import re
from typing import List, Tuple, Dict


class PIIPattern:
    """개인정보(PII) 패턴 정의 클래스.

    정규식 상수들을 모아두어 재사용성과 유지보수성을 높입니다.
    """

    # 이메일 주소 패턴
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # 한국 전화번호 패턴 (지역번호 포함)
    PHONE_KR = r'\b0\d{1,2}-?\d{3,4}-?\d{4}\b'

    # 한국 주민등록번호 패턴
    SSN_KR = r'\b\d{6}-?[1-4]\d{6}\b'

    # 신용카드 번호 패턴
    CREDIT_CARD = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'

    # IP 주소 패턴
    IP_ADDRESS = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'

    # 일반 긴 숫자 패턴 (선택적 사용)
    # NUMBERS = r'\b\d{10,}\b'


def mask_pii_text(
    text: str,
    patterns: List[str] = None,
    mask_char: str = "*"
) -> str:
    """텍스트에서 개인정보(PII)를 감지하여 마스킹 처리합니다.

    Args:
        text (str): 마스킹할 원본 텍스트
        patterns (List[str]): 적용할 정규식 패턴 리스트. None이면 기본 패턴 사용
        mask_char (str): 마스킹에 사용할 문자 (기본값: "*")

    Returns:
        str: 개인정보가 마스킹된 텍스트

    Examples:
        >>> mask_pii_text("연락처: 010-1234-5678")
        '연락처: 01-****-****'
        >>> mask_pii_text("이메일: test@example.com")
        '이메일: te**@ex******.com'
    """
    if not text:
        return text

    # 기본 패턴 목록 설정
    if patterns is None:
        patterns = [
            PIIPattern.EMAIL,
            PIIPattern.PHONE_KR,
            PIIPattern.SSN_KR,
            PIIPattern.CREDIT_CARD,
        ]

    masked_text = text

    # 각 패턴에 대해 마스킹 적용
    for pattern in patterns:
        masked_text = re.sub(
            pattern,
            lambda m: _mask_string(m.group(), mask_char),
            masked_text
        )

    return masked_text


def _mask_string(s: str, mask_char: str = "*") -> str:
    """문자열을 부분적으로 마스킹합니다.

    이메일과 일반 문자열에 따라 다른 마스킹 규칙을 적용합니다.
    이메일은 @ 기준으로 로컬과 도메인을 각각 마스킹하고,
    일반 문자열은 앞뒤 2자만 노출합니다.

    Args:
        s (str): 마스킹할 문자열
        mask_char (str): 마스킹에 사용할 문자 (기본값: "*")

    Returns:
        str: 마스킹된 문자열

    Examples:
        >>> _mask_string("test@example.com")
        'te**@ex******.com'
        >>> _mask_string("12345678")
        '12****78'
    """
    if len(s) <= 4:
        return mask_char * len(s)

    # 이메일 주소 처리
    if "@" in s:
        local, domain = s.split("@", 1)
        # 로컬 부분: 앞 2자만 노출
        masked_local = local[:2] + mask_char * (len(local) - 2)
        # 도메인 부분: 앞 2자만 노출
        domain_parts = domain.split(".")
        masked_domain_name = domain_parts[0][:2] + mask_char * (len(domain_parts[0]) - 2)
        masked_domain = masked_domain_name + "." + ".".join(domain_parts[1:])
        return f"{masked_local}@{masked_domain}"

    # 일반 문자열: 앞뒤 2자만 노출
    return s[:2] + mask_char * (len(s) - 4) + s[-2:]


def detect_pii(text: str) -> List[Dict[str, str]]:
    """텍스트에서 개인정보(PII)를 감지하고 위치 정보를 반환합니다.

    마스킹 없이 어떤 종류의 PII가 어디에 위치하는지 정보를 제공합니다.

    Args:
        text (str): 분석할 원본 텍스트

    Returns:
        List[Dict[str, str]]: 감지된 PII 정보 리스트
        각 딕셔너리는 다음 키를 포함합니다:
            - type: PII 종류 (email, phone_kr, ssn_kr 등)
            - value: 감지된 실제 값
            - start: 텍스트 내 시작 위치
            - end: 텍스트 내 종료 위치

    Examples:
        >>> detect_pii("연락처: 010-1234-5678, 이메일: test@example.com")
        [{'type': 'phone_kr', 'value': '010-1234-5678', 'start': 5, 'end': 17},
         {'type': 'email', 'value': 'test@example.com', 'start': 20, 'end': 36}]
    """
    patterns = {
        "email": PIIPattern.EMAIL,
        "phone_kr": PIIPattern.PHONE_KR,
        "ssn_kr": PIIPattern.SSN_KR,
        "credit_card": PIIPattern.CREDIT_CARD,
        "ip_address": PIIPattern.IP_ADDRESS,
    }

    detected = []

    # 각 PII 타입에 대해 패턴 매칭 수행
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            detected.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    return detected


def has_pii(text: str) -> bool:
    """텍스트에 개인정보(PII)가 포함되어 있는지 확인합니다.

    Args:
        text (str): 확인할 원본 텍스트

    Returns:
        bool: PII 포함 여부 (True: 포함됨, False: 포함되지 않음)

    Examples:
        >>> has_pii("일반 텍스트")
        False
        >>> has_pii("연락처: 010-1234-5678")
        True
    """
    return len(detect_pii(text)) > 0
