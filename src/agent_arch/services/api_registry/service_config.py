from agent_arch.core.config import settings


def get_base_url(service_name: str) -> str:
    """서비스 이름으로 Base URL을 반환한다.

    Args:
        service_name (str): external_service 필드 값 (e.g. "hr_service", "mattermost_service")

    Returns:
        str: 서비스 Base URL 문자열

    Raises:
        ValueError: 알 수 없는 서비스 이름인 경우

    Examples:
        >>> get_base_url("hr_service")
        'http://hr-service'
    """
    mapping = {
        "hr_service": settings.hr_service_base_url,
        "mattermost_service": settings.mattermost_base_url,
    }
    url = mapping.get(service_name)
    if not url:
        raise ValueError(f"Unknown external service: '{service_name}'")
    return url
