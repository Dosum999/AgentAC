"""LangSmith 설정 필드 테스트.

config.py에 추가한 enable_langsmith / langsmith_api_key / langsmith_project
필드가 환경변수에서 올바르게 로드되는지 검증한다.
"""
import pytest
from unittest.mock import patch


class TestLangSmithConfigDefaults:
    """기본값 동작 검증.

    .env 파일에 다른 필수 필드가 존재하므로, 각 키를 명시적으로
    오버라이드하여 on/off 동작을 검증한다.
    """

    def test_enable_langsmith_false_when_overridden(self):
        """`ENABLE_LANGSMITH=false`로 오버라이드하면 False로 읽힌다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"ENABLE_LANGSMITH": "false"}):
            s = Settings()
            assert s.enable_langsmith is False

    def test_enable_langsmith_true_when_overridden(self):
        """`ENABLE_LANGSMITH=true`로 오버라이드하면 True로 읽힌다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"ENABLE_LANGSMITH": "true"}):
            s = Settings()
            assert s.enable_langsmith is True

    def test_langsmith_project_overridable(self):
        """`LANGSMITH_PROJECT`를 오버라이드하면 해당 값으로 읽힌다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"LANGSMITH_PROJECT": "custom-project"}):
            s = Settings()
            assert s.langsmith_project == "custom-project"


class TestLangSmithConfigFromEnv:
    """환경변수 로드 검증."""

    def test_enable_langsmith_true_from_env(self):
        """`ENABLE_LANGSMITH=true` 환경변수를 bool True로 로드한다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"ENABLE_LANGSMITH": "true"}):
            s = Settings()
            assert s.enable_langsmith is True

    def test_langsmith_api_key_from_env(self):
        """`LANGSMITH_API_KEY` 환경변수를 문자열로 로드한다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key-123"}):
            s = Settings()
            assert s.langsmith_api_key == "test-key-123"

    def test_langsmith_project_from_env(self):
        """`LANGSMITH_PROJECT` 환경변수를 문자열로 로드한다."""
        from agent_arch.core.config import Settings

        with patch.dict("os.environ", {"LANGSMITH_PROJECT": "my-project"}):
            s = Settings()
            assert s.langsmith_project == "my-project"
