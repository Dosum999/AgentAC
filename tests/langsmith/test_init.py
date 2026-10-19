"""LangSmith 초기화 로직 테스트.

app.core.lifespan._init_langsmith()가 enable_langsmith=True일 때
LANGSMITH_TRACING 등 환경변수를 올바르게 설정하는지 검증한다.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestLangSmithInit:
    """LangSmith 환경변수 초기화 검증."""

    def _make_settings(self, enable: bool, api_key=None, project="agent-backend"):
        s = MagicMock()
        s.enable_langsmith = enable
        s.langsmith_api_key = api_key
        s.langsmith_project = project
        return s

    def test_env_vars_set_when_enabled(self):
        """enable_langsmith=True이고 api_key가 있으면 환경변수 3개가 설정된다."""
        from agent_arch.core.lifespan import _init_langsmith

        mock_settings = self._make_settings(
            enable=True, api_key="lsv2_test_key", project="my-project"
        )
        with patch("agent_arch.core.lifespan.settings", mock_settings):
            _init_langsmith()

        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") == "lsv2_test_key"
        assert os.environ.get("LANGSMITH_PROJECT") == "my-project"

    def test_env_vars_not_set_when_disabled(self):
        """enable_langsmith=False이면 환경변수를 건드리지 않는다."""
        from agent_arch.core.lifespan import _init_langsmith

        mock_settings = self._make_settings(enable=False, api_key="lsv2_test_key")
        with patch("agent_arch.core.lifespan.settings", mock_settings):
            env_before = os.environ.copy()
            _init_langsmith()

        assert os.environ.get("LANGSMITH_TRACING") == env_before.get("LANGSMITH_TRACING")

    def test_env_vars_not_set_without_api_key(self):
        """api_key가 None이면 enable_langsmith=True여도 환경변수를 설정하지 않는다."""
        from agent_arch.core.lifespan import _init_langsmith

        mock_settings = self._make_settings(enable=True, api_key=None)
        with patch("agent_arch.core.lifespan.settings", mock_settings):
            env_before = os.environ.copy()
            _init_langsmith()

        assert os.environ.get("LANGSMITH_TRACING") == env_before.get("LANGSMITH_TRACING")

    def test_project_name_propagated(self):
        """langsmith_project 값이 LANGSMITH_PROJECT 환경변수에 그대로 전달된다."""
        from agent_arch.core.lifespan import _init_langsmith

        mock_settings = self._make_settings(
            enable=True, api_key="test-key", project="agent-backend-dev"
        )
        with patch("agent_arch.core.lifespan.settings", mock_settings):
            _init_langsmith()

        assert os.environ["LANGSMITH_PROJECT"] == "agent-backend-dev"
