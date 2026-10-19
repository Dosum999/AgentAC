"""app/utils/sanitize.py 단위 테스트.

내부 도구 이름을 LLM 응답 본문에서 마스킹하는 정규식과
스트림 버퍼링(StreamSanitizer)의 동작을 검증한다.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from agent_arch.core.config import settings
from agent_arch.utils import sanitize as sanitize_mod
from agent_arch.utils.sanitize import (
    strip_reference_links_section,
    GUIDE_AGENT_HINT,
    StreamSanitizer,
    collapse_repeated_runs,
    is_wiki_url,
    normalize_html_breaks,
    normalize_math_markup,
    reset_whitelist_cache,
    sanitize_tool_names,
    strip_wiki_links,
)


# ---------------------------------------------------------------------------
# HTML 줄바꿈 노출 회귀 — 표·링크 보존 및 스트리밍 경계
# ---------------------------------------------------------------------------

HTML_BREAKS = ("<br>", "<br/>", "<BR />", r"\<br>", r"\<br\>", "&lt;br&gt;", "&lt;br /&gt;")


@pytest.mark.parametrize("tag", HTML_BREAKS)
def test_html_break_preserves_inquiry_table_and_link(tag, empty_tool_cache):
    before = (
        "| 구분 | 특징 및 방법 |\n| --- | --- |\n"
        "| 회원 | [문의 폼 바로가기](https://example.com/inquiry)"
    )
    after = "플랫폼 우측 상단 **물음표(?) → 문의하기** 메뉴 선택 |\n"
    assert sanitize_tool_names(before + tag + after) == before + " " + after


@pytest.mark.parametrize("tag", HTML_BREAKS)
def test_stream_html_break_at_every_emit_boundary(tag, empty_tool_cache):
    # 태그의 모든 내부 위치에 emit 경계를 걸어 저장본과 화면이 같은지 검증.
    for offset in range(1, len(tag)):
        tail = "나" * (StreamSanitizer._LOOKBEHIND - len(tag) + offset)
        stream = StreamSanitizer()
        assert stream.feed("앞" + tag + tail) + stream.flush() == "앞 " + tail


@pytest.mark.parametrize("size", (1, 2, 7, 81, 200))
def test_stream_html_breaks_match_final_answer(size, empty_tool_cache):
    text = "문의 안내입니다. " * 10 + "\n".join(
        f"| 회원 | [문의 폼 바로가기](https://example.com/inquiry){tag}메뉴 선택 |"
        for tag in HTML_BREAKS
    ) + "\n추가 안내입니다. " * 10
    stream = StreamSanitizer()
    output = "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size))
    output += stream.flush()
    assert output == sanitize_tool_names(text)
    assert "<br" not in output.lower()
    assert "&lt;" not in output


@pytest.mark.parametrize("text", (
    None, "", "일반 문장\n다음 줄", "<browser> <b>강조</b>",
    "[링크](https://example.com/?a=1&b=2) <https://example.com>",
))
def test_html_break_normalization_preserves_other_text(text):
    assert normalize_html_breaks(text) == text


# ---------------------------------------------------------------------------
# 픽스처: 화이트리스트 캐시 격리
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_cache():
    """각 테스트 전후로 화이트리스트 캐시를 초기화한다."""
    reset_whitelist_cache()
    yield
    reset_whitelist_cache()


@contextmanager
def _stub_sources(dynamic=(), registry=(), config=(), sources="dynamic,registry,config"):
    """화이트리스트 소스 **3종을 모두** 스텁한다.

    dynamic 하나만 패치하던 예전 방식으로는 더 이상 격리가 성립하지 않는다:
    `ToolRegistry._tools` 는 클래스 변수(프로세스 전역)라 테스트 세션이
    `app.agents.tools` 를 import 하는 것만으로 정적 스킬 35개가 들어오고,
    `settings.tools_config` 는 실제 yaml 82개를 들고 있다. 셋 다 막아야
    "이 이름들만 존재한다"는 전제가 만들어진다.

    Args:
        dynamic: llm_tools.json 기반 동적 도구 이름들
        registry: ToolRegistry(정적 스킬 + MCP) 도구 이름들
        config: tools_config.yaml 의 tools 키들
        sources: `SANITIZE_TOOL_NAME_SOURCES` 값 (소스 선택 스위치 검증용)
    """
    dyn = {name: object() for name in dynamic}
    with patch("agent_arch.services.api_registry.loader.tool_loader") as mock_loader, patch(
        "agent_arch.agents.tools.registry.ToolRegistry.list_tool_names",
        return_value=list(registry),
    ), patch.object(
        settings, "tools_config", {"tools": {n: {} for n in config}}
    ), patch.object(
        settings, "sanitize_tool_name_sources", sources
    ):
        mock_loader._cache = dyn
        mock_loader.load_from_json.return_value = list(dyn.values())
        reset_whitelist_cache()
        try:
            yield
        finally:
            reset_whitelist_cache()


@pytest.fixture
def empty_tool_cache(isolated_cache):
    """세 소스를 모두 비운다 (안전망 패턴만 동작)."""
    with _stub_sources():
        yield


@pytest.fixture
def stub_tool_cache(isolated_cache):
    """대표 도구 이름들을 동적 소스로 stub한다 (나머지 두 소스는 빈 상태)."""
    stub_names = [
        "mespoc_service_list_defective_performances_quality_i_53aeef98",
        "hr_service_attendance_read",
        "mattermost_skill_find_channels",
        "mespoc_service_assets_list",
    ]
    with _stub_sources(dynamic=stub_names):
        yield stub_names


# ---------------------------------------------------------------------------
# sanitize_tool_names — 화이트리스트 매칭
# ---------------------------------------------------------------------------

class TestWhitelistMatching:
    """tool_loader._cache의 실제 도구명에 대한 정확 매칭."""

    def test_hex_suffix_tool_in_korean_sentence(self, stub_tool_cache):
        text = "앞서 mespoc_service_list_defective_performances_quality_i_53aeef98 도구로 조회한 결과"
        assert sanitize_tool_names(text) == "앞서 내부 도구로 조회한 결과"  # 중복 "도구" 흡수

    def test_no_hex_suffix_tool(self, stub_tool_cache):
        text = "hr_service_attendance_read를 호출했습니다."
        assert sanitize_tool_names(text) == "내부 도구를 호출했습니다."

    def test_skill_tool(self, stub_tool_cache):
        text = "mattermost_skill_find_channels로 채널을 검색합니다."
        assert sanitize_tool_names(text) == "내부 도구로 채널을 검색합니다."

    def test_tool_in_backticks(self, stub_tool_cache):
        """치환어를 감싼 백틱은 함께 흡수한다 — 한국어 명사구를 코드 스팬에 둘 이유가 없다."""
        text = "`mespoc_service_assets_list`를 호출"
        assert sanitize_tool_names(text) == "내부 도구를 호출"

    def test_multiple_tools_in_one_text(self, stub_tool_cache):
        text = (
            "먼저 hr_service_attendance_read를 호출한 뒤 "
            "mattermost_skill_find_channels를 호출했습니다."
        )
        out = sanitize_tool_names(text)
        assert "hr_service_attendance_read" not in out
        assert "mattermost_skill_find_channels" not in out
        assert out.count("내부 도구") == 2


# ---------------------------------------------------------------------------
# sanitize_tool_names — 안전망 패턴 (캐시 미적중 케이스)
# ---------------------------------------------------------------------------

class TestFallbackPattern:
    """캐시에 없는 새 도구 이름도 패턴으로 잡히는지 검증."""

    def test_unknown_service_tool_with_hex(self, empty_tool_cache):
        text = "future_service_some_new_endpoint_abcd1234를 사용했습니다."
        assert sanitize_tool_names(text) == "내부 도구를 사용했습니다."

    def test_unknown_service_tool_without_hex(self, empty_tool_cache):
        text = "newservice_service_query_data로 데이터를 조회"
        assert sanitize_tool_names(text) == "내부 도구로 데이터를 조회"

    def test_unknown_skill_tool(self, empty_tool_cache):
        text = "Slack의 slack_skill_send_message가 호출되었습니다."
        assert sanitize_tool_names(text) == "Slack의 내부 도구가 호출되었습니다."


# ---------------------------------------------------------------------------
# sanitize_tool_names — 일반 텍스트 보존 (오탐 방지)
# ---------------------------------------------------------------------------

class TestNoFalsePositive:
    """평범한 한국어/영어 텍스트는 변경되지 않아야 한다."""

    def test_korean_only(self, empty_tool_cache):
        text = "고객 서비스 안내를 받았습니다."
        assert sanitize_tool_names(text) == text

    def test_english_words(self, empty_tool_cache):
        # `service` 단어가 단독으로 등장할 때는 매칭되지 않아야 한다.
        text = "Customer service is important. Use the API."
        assert sanitize_tool_names(text) == text

    def test_path_with_service_segment(self, empty_tool_cache):
        # URL 경로의 service 세그먼트는 매칭되지 않아야 한다 (대문자 + 슬래시 경계).
        text = "GET /api/Service/v1/data"
        assert sanitize_tool_names(text) == text

    def test_empty_input(self):
        assert sanitize_tool_names("") == ""
        assert sanitize_tool_names(None) is None

    def test_already_sanitized_idempotent(self, stub_tool_cache):
        # 이미 sanitize된 텍스트를 다시 처리해도 변경되지 않아야 한다 (idempotent).
        text = "내부 도구를 사용했습니다."
        assert sanitize_tool_names(text) == text


# ---------------------------------------------------------------------------
# sanitize_tool_names — 한글-ASCII 경계
# ---------------------------------------------------------------------------

class TestKoreanBoundary:
    """한국어 사이에 박힌 ASCII 식별자를 정확히 매칭한다."""

    def test_korean_directly_around_tool(self, stub_tool_cache):
        text = "조회를mespoc_service_assets_list로처리"
        out = sanitize_tool_names(text)
        assert "mespoc_service" not in out
        assert "내부 도구" in out

    def test_punctuation_boundary(self, stub_tool_cache):
        """괄호 안이 도구명뿐이면 괄호째 흡수된다 → 치환어도 남지 않는다."""
        text = "도구(hr_service_attendance_read)로 호출."
        out = sanitize_tool_names(text)
        assert "hr_service_attendance_read" not in out
        assert out == "도구로 호출."


# ---------------------------------------------------------------------------
# StreamSanitizer — 토큰 분할 케이스
# ---------------------------------------------------------------------------

class TestGuideAgentHintMarker:
    """가이드 에이전트-힌트 마커는 사용자 표시 텍스트에서 제거되어야 한다."""

    def test_marker_stripped_by_sanitize(self, empty_tool_cache):
        out = sanitize_tool_names("남은 연차는 HR에서 확인하세요." + GUIDE_AGENT_HINT)
        assert GUIDE_AGENT_HINT not in out
        assert "남은 연차는 HR에서 확인하세요." in out

    def test_marker_not_leaked_in_stream_end(self, empty_tool_cache):
        """마커가 응답 끝에 붙어도 스트리밍 출력에 노출되지 않는다(flush에서 제거)."""
        s = StreamSanitizer()
        text = "메일은 메일 서비스에서 확인하실 수 있어요." + GUIDE_AGENT_HINT
        out = "".join(s.feed(c) for c in text)
        out += s.flush()
        assert GUIDE_AGENT_HINT not in out
        assert "메일 서비스에서 확인" in out


class TestStreamSanitizer:
    """LLM 토큰이 도구 이름 중간에서 잘려도 누락 없이 마스킹되어야 한다."""

    def test_single_chunk(self, stub_tool_cache):
        s = StreamSanitizer()
        out = s.feed("앞서 hr_service_attendance_read를 호출")
        out += s.flush()
        assert "hr_service_attendance_read" not in out
        assert "내부 도구" in out

    def test_split_in_middle_of_tool_name(self, stub_tool_cache):
        s = StreamSanitizer()
        chunks = [
            "앞서 logist",
            "ics_service_",
            "list_defective_performances_quality_i_53aeef98",
            "로 조회",
        ]
        out_parts = [s.feed(c) for c in chunks]
        out_parts.append(s.flush())
        result = "".join(out_parts)
        assert "mespoc_service" not in result
        assert "53aeef98" not in result
        assert "내부 도구" in result

    def test_one_char_at_a_time(self, stub_tool_cache):
        full_text = "앞서 hr_service_attendance_read를 호출했습니다."
        s = StreamSanitizer()
        out_parts = [s.feed(c) for c in full_text]
        out_parts.append(s.flush())
        result = "".join(out_parts)
        assert "hr_service_attendance_read" not in result
        assert "내부 도구" in result

    def test_buffer_emits_after_lookbehind_exceeded(self, stub_tool_cache):
        """버퍼가 _LOOKBEHIND를 초과하기 전에는 yield되지 않는다.

        필러는 **비식별자 문자**(한글)를 쓴다. ASCII 만으로 채우면 버퍼 전체가 하나의
        식별자 런이 되어, 이름을 쪼개지 않으려는 emit 경계 보정이 cut 을 런 시작까지
        당긴다(→ 보류). 이 테스트의 의도는 버퍼 임계 동작이지 식별자 처리가 아니므로
        런이 생기지 않는 입력을 쓴다. 식별자 런 자체는 아래 별도 테스트로 다룬다.
        """
        s = StreamSanitizer()
        small = "가" * (StreamSanitizer._LOOKBEHIND - 1)
        assert s.feed(small) == ""  # 보류
        # 한 글자 더 넣으면 _LOOKBEHIND 초과 → head 일부 emit
        out = s.feed("나")
        # head는 (LOOKBEHIND 직전까지) emit, 나머지는 보류
        # 텍스트가 평범하면 변경 없이 그대로 emit
        assert out == ""  # _LOOKBEHIND 정확히 도달 시 (== 조건은 보류)
        out2 = s.feed("다")
        assert out2 == "가"  # 첫 한 글자가 head로 emit됨
        # flush로 잔여 처리
        rest = s.flush()
        assert out + out2 + rest == small + "나다"

    def test_consistency_with_full_text_sanitize(self, stub_tool_cache):
        """1자씩 feed한 결과가 한 번에 sanitize한 결과와 동일해야 한다."""
        full_text = (
            "먼저 hr_service_attendance_read를 호출하고 "
            "이어서 mattermost_skill_find_channels로 채널을 찾았습니다."
        )
        expected = sanitize_tool_names(full_text)

        s = StreamSanitizer()
        parts = [s.feed(c) for c in full_text]
        parts.append(s.flush())
        actual = "".join(parts)

        assert actual == expected

    def test_math_span_not_split_across_lookbehind(self, empty_tool_cache):
        """80자 경계를 넘는 긴 메시지에서도 $...$ 수식 스팬이 스트리밍 분할로 깨지지 않는다.

        회귀 가드: 초기엔 라이브 스트림이 경계에 걸친 $\\rightarrow$ 를 변환하지 못해
        생으로 노출됐다(_safe_emit_len 도입 전). 1자씩 feed 해도 최종 sanitize 와 동일해야 한다.
        """
        msg = (
            r"협업 워크플로우입니다. 담당자 탐색 $\rightarrow$ 연락하고, "
            r"논의 내용 파악 $\rightarrow$ 공유하며, 시간 조율 $\rightarrow$ 등록하고, "
            r"남은 연차 $\le$ 3일이면 알림. 가격은 $100 와 $200 입니다."
        )
        assert len(msg) > StreamSanitizer._LOOKBEHIND  # 경계 분할이 실제로 발생하는 길이
        expected = sanitize_tool_names(msg)

        s = StreamSanitizer()
        parts = [s.feed(c) for c in msg]  # 1자씩(최악 케이스)
        parts.append(s.flush())
        result = "".join(parts)

        assert result == expected
        assert r"\rightarrow" not in result  # 화살표 전부 변환
        assert "$100" in result and "$200" in result  # 통화는 보존


# ---------------------------------------------------------------------------
# 캐시 빌드 실패 fallback
# ---------------------------------------------------------------------------

class TestCacheLoadFailure:
    """tool_loader 로드 실패 시 안전망 패턴만으로 동작."""

    def test_loader_raises_falls_back_to_pattern(self, isolated_cache):
        with patch("agent_arch.services.api_registry.loader.tool_loader") as mock_loader:
            mock_loader._cache = None
            mock_loader.load_from_json.side_effect = RuntimeError("disk error")
            # 화이트리스트는 못 만들지만 안전망 패턴은 동작해야 한다.
            text = "mespoc_service_assets_list로 조회"
            out = sanitize_tool_names(text)
            assert "mespoc_service" not in out
            assert "내부 도구" in out


# ---------------------------------------------------------------------------
# normalize_math_markup — LaTeX/수식 → 유니코드 평문
# ---------------------------------------------------------------------------

class TestMathNormalization:
    """모델이 뱉은 LaTeX 수식을 프론트가 그릴 수 있는 평문으로 정규화한다."""

    def test_inline_arrow(self):
        # 사용자가 실제 겪은 증상: $\rightarrow$ 가 생으로 노출.
        assert normalize_math_markup(r"내 남은 연차? $\rightarrow$ 잔여 휴가") == "내 남은 연차? → 잔여 휴가"

    def test_bare_command_without_dollar(self):
        assert normalize_math_markup(r"사이드바 \rightarrow 휴가 \rightarrow 신청") == "사이드바 → 휴가 → 신청"

    def test_block_math_frac_and_times(self):
        assert normalize_math_markup(r"계산: $$\frac{a}{b} \times 2$$") == "계산: (a)/(b) × 2"

    def test_bracket_delimiters(self):
        assert normalize_math_markup(r"\(x \le 5\) 이고 \[y \ge 3\]") == "x ≤ 5 이고 y ≥ 3"

    def test_superscript_subscript_braces_stripped(self):
        assert normalize_math_markup(r"$x^{2} + y_{i}$") == "x^2 + y_i"

    def test_currency_two_dollars_preserved(self):
        # 통화 오탐 방지: 한 줄에 $ 2개라도 수식 표시(\,^,_)가 없으면 불변.
        assert normalize_math_markup("가격은 $100 와 $200 입니다") == "가격은 $100 와 $200 입니다"

    def test_single_currency_preserved(self):
        assert normalize_math_markup("단일 $100 입니다") == "단일 $100 입니다"

    def test_plain_text_unchanged(self):
        text = "백슬래시도 달러도 없는 평범한 문장."
        assert normalize_math_markup(text) is text  # fast-path: 동일 객체 반환

    def test_empty_and_none(self):
        assert normalize_math_markup("") == ""
        assert normalize_math_markup(None) is None

    def test_sanitize_tool_names_applies_math(self, empty_tool_cache):
        # chokepoint 통합: 도구명 마스킹과 수식 정규화가 함께 적용된다.
        out = sanitize_tool_names(r"hr_service_q_data 결과 $\rightarrow$ 완료")
        assert out == "내부 도구 결과 → 완료"


# ---------------------------------------------------------------------------
# 소스 3종 확장 — MCP·정적 스킬·설정 파일 커버리지
# ---------------------------------------------------------------------------

class TestMultiSourceWhitelist:
    """동적 API 도구 외의 소스에서 온 이름도 마스킹되어야 한다.

    2026-08 누출 회귀 테스트: `mail_read_message`(mailcow MCP)가 동적 소스에 없어
    "mail_read_message 도구를 사용하면..." 이 그대로 사용자에게 나갔다.
    """

    def test_mcp_tool_from_registry(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names("mail_read_message 도구를 사용하면 됩니다.")
            assert "mail_read_message" not in out
            assert out == "내부 도구를 사용하면 됩니다."  # 중복 "도구" 흡수

    def test_mcp_tool_in_backticks(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"]):
            assert sanitize_tool_names("`mail_read_message` 를 호출") == "내부 도구 를 호출"

    def test_mcp_tool_adjacent_to_korean(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names("mail_read_message를 쓰면 됩니다")
            assert out == "내부 도구를 쓰면 됩니다"

    def test_config_only_name_when_mcp_registration_failed(self, isolated_cache):
        """MCP 서버가 죽어 등록이 안 돼도 yaml 키만으로 마스킹된다.

        MCP 등록은 백오프 재시도가 소진되면 그 프로세스 생애 동안 복구되지 않는데,
        LLM 은 프롬프트·과거 맥락에서 배운 이름을 여전히 뱉을 수 있다.
        """
        with _stub_sources(registry=[], config=["calendar_create_event"]):
            out = sanitize_tool_names("calendar_create_event 로 등록했습니다.")
            assert "calendar_create_event" not in out

    def test_all_three_sources_merged(self, isolated_cache):
        with _stub_sources(
            dynamic=["mespoc_service_assets_list"],
            registry=["mail_read_message"],
            config=["hr_leave_balance"],
        ):
            out = sanitize_tool_names(
                "mespoc_service_assets_list 와 mail_read_message 와 hr_leave_balance"
            )
            assert out.count("내부 도구") == 3


class TestNeverMaskFilter:
    """도구명이자 평범한 영단어인 이름은 화이트리스트에서 제외되어야 한다.

    `search` / `calculator` / `database` 는 실제 등록된 도구 이름이지만, 마스킹하면
    답변 본문의 모든 "search"·"database" 가 "내부 도구"로 치환돼 정상 문장이 깨진다.
    도구명 노출보다 답변 파괴가 더 나쁘다는 판단 — 이 이름들은 프롬프트 규칙에만 맡긴다.
    """

    def test_bare_english_tool_names_preserved(self, isolated_cache):
        with _stub_sources(registry=["search", "calculator", "database"]):
            text = "web search 기능과 database 연동, calculator 를 씁니다"
            assert sanitize_tool_names(text) == text

    def test_underscored_name_still_masked(self, isolated_cache):
        """제외는 언더스코어 없는 이름 한정 — `search_colleagues` 는 그대로 마스킹."""
        with _stub_sources(registry=["search", "search_colleagues"]):
            out = sanitize_tool_names("search 로 search_colleagues 를 호출")
            assert out == "search 로 내부 도구 를 호출"


class TestSourceIsolation:
    """소스 하나가 터져도 나머지로 마스킹이 살아남아야 한다."""

    def test_dynamic_source_failure_does_not_kill_others(self, isolated_cache):
        with patch(
            "agent_arch.agents.tools.registry.ToolRegistry.list_tool_names",
            return_value=["mail_read_message"],
        ), patch.object(settings, "tools_config", {"tools": {}}), patch.object(
            settings, "sanitize_tool_name_sources", "dynamic,registry,config"
        ), patch(
            "agent_arch.services.api_registry.loader.tool_loader"
        ) as mock_loader:
            type(mock_loader)._cache = property(
                lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
            )
            reset_whitelist_cache()
            try:
                out = sanitize_tool_names("mail_read_message 로 조회")
                assert "mail_read_message" not in out
            finally:
                reset_whitelist_cache()


class TestSourceSwitch:
    """SANITIZE_TOOL_NAME_SOURCES 로 소스를 끄면 그 소스 이름은 안 걸린다(후퇴 레버)."""

    def test_dynamic_only_excludes_registry_names(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"], sources="dynamic"):
            text = "mail_read_message 로 조회"
            assert sanitize_tool_names(text) == text

    def test_no_sources_disables_whitelist_but_keeps_fallback(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"], sources=""):
            assert sanitize_tool_names("mail_read_message 로 조회") == "mail_read_message 로 조회"
            # 안전망 패턴(`*_service_*`)은 화이트리스트와 무관하게 계속 동작한다.
            assert sanitize_tool_names("hr_service_attendance_read 호출") == "내부 도구 호출"


class TestRegistryInvalidationHook:
    """ToolRegistry 등록/해제가 화이트리스트 캐시를 무효화해야 한다.

    이 훅이 없으면 lazy 빌드된 첫 스냅샷이 프로세스 생애 동안 고정돼, MCP 등록 전에
    들어온 첫 요청 하나가 그 프로세스 내내 메일·일정 도구명을 못 걸리게 만든다.
    """

    def test_newly_registered_tool_masked_immediately(self, isolated_cache):
        from agent_arch.agents.tools.registry import ToolRegistry

        class _Probe:
            name = "zzz_probe_tool"

        # 등록 전 스냅샷을 먼저 굳혀 둔다 (무효화가 없으면 이 상태가 유지된다).
        assert sanitize_tool_names("zzz_probe_tool 호출") == "zzz_probe_tool 호출"

        ToolRegistry.register(_Probe())
        try:
            assert sanitize_tool_names("zzz_probe_tool 호출") == "내부 도구 호출"
        finally:
            # 프로세스 전역 레지스트리라 teardown 없으면 다른 테스트로 누수된다.
            ToolRegistry.unregister("zzz_probe_tool")

        assert sanitize_tool_names("zzz_probe_tool 호출") == "zzz_probe_tool 호출"


# ---------------------------------------------------------------------------
# StreamSanitizer — emit 경계가 도구명을 쪼개는 케이스
# ---------------------------------------------------------------------------

class TestStreamBoundarySplitsIdentifier:
    """emit 경계(_LOOKBEHIND)가 도구명 한복판에 떨어져도 누출되면 안 된다.

    2026-08-19 실측 누출: 이름이 버퍼 끝에서 정확히 _LOOKBEHIND 근처에 놓이면 앞 head 에
    `mail_list_`, 다음 head 에 `messages` 만 남아 양쪽 다 매칭되지 않았다. 최종
    LoopResult 는 전문을 한 번에 정화하므로 DB 저장본은 멀쩡했고 **화면에만** 도구명이
    그대로 나갔다 — 저장본만 확인해서는 잡히지 않는 종류의 버그.
    """

    def _feed_all(self, text: str) -> str:
        s = StreamSanitizer()
        return s.feed(text) + s.flush()

    def test_name_straddling_emit_boundary(self, isolated_cache):
        name = "mail_list_messages"
        # 뒤쪽을 76자로 두면 cut(=len-80)이 이름 안(시작+14)에 떨어진다.
        text = "가" * 40 + name + "나" * 76
        with _stub_sources(registry=[name]):
            out = self._feed_all(text)
            assert name not in out
            assert "mail_list_" not in out
            assert "내부 도구" in out

    @pytest.mark.parametrize("tail_len", range(60, 100, 3))
    def test_boundary_sweep(self, isolated_cache, tail_len):
        """이름 뒤 길이를 훑어 어떤 정렬에서도 안 새는지 확인한다."""
        name = "mail_read_message"
        text = "가" * 50 + name + "나" * tail_len
        with _stub_sources(registry=[name]):
            out = self._feed_all(text)
            assert name not in out
            assert "mail_read" not in out

    def test_chunked_stream_keeps_masking(self, isolated_cache):
        """실제 스트림처럼 잘게 나눠 먹여도 동일하다."""
        name = "calendar_create_event"
        text = "일정을 등록하려면 " + "가" * 60 + f" `{name}` 도구를 씁니다. " + "나" * 40
        with _stub_sources(registry=[name]):
            s = StreamSanitizer()
            out = "".join(s.feed(text[i:i + 7]) for i in range(0, len(text), 7)) + s.flush()
            assert name not in out
            assert "내부 도구" in out

    def test_long_identifier_run_not_held_forever(self, isolated_cache):
        """식별자 런이 창보다 길면 당기지 않는다(무한 보류 방지)."""
        with _stub_sources(registry=["mail_read_message"]):
            blob = "A" * 200
            out = self._feed_all("데이터: " + blob + " 끝" + "나" * 90)
            assert blob in out

    def test_pure_identifier_run_holds_until_boundary(self, isolated_cache):
        """버퍼가 통째로 하나의 식별자 런이면 경계가 올 때까지 보류한다.

        도구명일 가능성을 배제할 수 없으므로 쪼개서 내보내지 않는다. 공백·문장부호·한글
        어느 것이든 런을 끊으므로 실제 답변에서 오래 묶이지 않고, 런이 창(_LOOKBEHIND)을
        넘어서면 위 test_long_identifier_run_not_held_forever 대로 그냥 내보낸다.
        """
        with _stub_sources():
            s = StreamSanitizer()
            blob = "a" * (StreamSanitizer._LOOKBEHIND + 5)
            assert s.feed(blob) == ""            # 쪼개서 내보내지 않고 전부 보류
            out = s.feed(" 끝" + "가" * 90)      # 런이 창 밖으로 밀려나면 emit 재개
            assert out
            assert out + s.flush() == blob + " 끝" + "가" * 90


# ---------------------------------------------------------------------------
# 치환 잔재 흡수 — 중복 분류어 · 백틱 · 괄호
# ---------------------------------------------------------------------------

class TestAbsorbRedundancy:
    """치환은 이름 단위로만 일어나므로, 모델이 쓴 "식별자 + 분류어" 구조에서 중복이 남는다.

    ``mail_read_message 도구를`` → ``내부 도구 도구를`` 처럼 치환어가 이미 품은 "도구"가
    원문의 "도구"와 겹친다. 흡수 규칙은 **치환이 실제로 일어난 자리 인접**에서만 돈다.
    """

    def test_duplicate_classifier(self, isolated_cache):
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names("`mail_read_message` 도구를 사용하면 됩니다.")
            assert out == "내부 도구를 사용하면 됩니다."

    @pytest.mark.parametrize("classifier", ["도구", "툴", "함수", "API"])
    def test_classifier_variants(self, isolated_cache, classifier):
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names(f"mail_read_message {classifier}로 조회")
            assert out == "내부 도구로 조회"

    @pytest.mark.parametrize("particle", ["를", "로", "가", "는", "와", "의", "도", "에서"])
    def test_classifier_with_particles(self, isolated_cache, particle):
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names(f"mail_read_message 도구{particle} 어쩌고")
            assert out == f"내부 도구{particle} 어쩌고"

    def test_plural_suffix_not_absorbed(self, isolated_cache):
        """"도구들" 은 분류어+조사가 아니므로 흡수하지 않는다(조사 목록에 없는 음절)."""
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names("mail_read_message 도구들을 확인")
            assert out == "내부 도구 도구들을 확인"

    def test_adjacent_duplicates_merged(self, isolated_cache):
        with _stub_sources(registry=["calendar_list_events", "calendar_search_events"]):
            out = sanitize_tool_names(
                "calendar_list_events 또는 calendar_search_events를 사용해"
            )
            assert out == "내부 도구를 사용해"

    def test_paren_of_names_only_dropped(self, isolated_cache):
        """DB 실제 기록 형태 — 괄호 안이 이름·구분자뿐이면 괄호째 버린다."""
        with _stub_sources(registry=["calendar_list_events", "calendar_search_events"]):
            out = sanitize_tool_names(
                "제가 일정을 조회하는 도구(`calendar_list_events` 또는 "
                "`calendar_search_events`)를 사용하여 데이터를 가져오면"
            )
            assert out == "제가 일정을 조회하는 도구를 사용하여 데이터를 가져오면"

    def test_paren_with_space_before(self, isolated_cache):
        with _stub_sources(registry=["search_colleagues"]):
            out = sanitize_tool_names("사내 인사 디렉토리 (`search_colleagues`)에서 조회")
            assert out == "사내 인사 디렉토리에서 조회"

    def test_paren_with_prose_kept(self, isolated_cache):
        """괄호에 산문이 섞이면 삭제하지 않는다 — 사용자 정보를 지우지 않기 위한 유일한 방어선."""
        with _stub_sources(registry=["mail_read_message"]):
            out = sanitize_tool_names("메일 도구(mail_read_message 로 본문 조회)를 씁니다")
            assert out == "메일 도구(내부 도구 로 본문 조회)를 씁니다"


class TestAbsorbNoFalsePositive:
    """치환이 일어나지 않은 자리의 "도구"·백틱·괄호는 그대로여야 한다."""

    @pytest.mark.parametrize("text", [
        "이 도구는 유용합니다.",
        "도구를 사용하면 편리해요. 도구 함수 API 툴",
        "참조자(CC) 목록을 가져옵니다.",
        "`uid` 를 사용하여 조회합니다.",
        "일정을 조회하는 도구(직접 확인 필요)를 사용",
    ])
    def test_untouched_without_substitution(self, isolated_cache, text):
        with _stub_sources(registry=["mail_read_message"]):
            assert sanitize_tool_names(text) == text


class TestAbsorbAcrossStreamBoundary:
    """흡수는 매치 **뒤쪽** 문맥을 봐야 하므로 emit 경계가 그 창을 쪼개면 안 된다.

    쪼개지면 화면에는 "내부 도구 도구를" 이 그대로 나가고 저장본(전문 정화)만 깨끗해진다 —
    `e01f0bc` 에서 고친 "이름 한복판 절단"과 같은 종류의 화면 전용 버그.
    """

    def _stream(self, text: str, size: int | None = None) -> str:
        s = StreamSanitizer()
        if size is None:
            return s.feed(text) + s.flush()
        return "".join(s.feed(text[i:i + size]) for i in range(0, len(text), size)) + s.flush()

    @pytest.mark.parametrize("tail_len", range(60, 100, 3))
    def test_classifier_absorbed_at_any_alignment(self, isolated_cache, tail_len):
        with _stub_sources(registry=["mail_read_message"]):
            text = "가" * 50 + "mail_read_message 도구를 사용하면" + "나" * tail_len
            out = self._stream(text)
            assert "내부 도구 도구를" not in out
            assert "내부 도구를 사용하면" in out

    @pytest.mark.parametrize("size", [1, 3, 7, 40])
    def test_chunked_stream_matches_full_text(self, isolated_cache, size):
        with _stub_sources(registry=["calendar_list_events", "calendar_search_events"]):
            text = (
                "안내드립니다. " + "가" * 60
                + " 제가 일정을 조회하는 도구(`calendar_list_events` 또는 "
                "`calendar_search_events`)를 사용하여 가져옵니다. " + "나" * 40
            )
            assert self._stream(text, size) == sanitize_tool_names(text)


# ── 내부 식별자 노출 차단 (2026-09-15) ───────────────────────────────────────
#
# 실측: "에이전트_요금제_안내.xlsx (file_id: 8paerwxuyb8jfch8z8o14xe97o)" 가 그대로 게시됐다.
# 사용자에게는 의미 없는 문자열이고 파일명만 가린다.

class TestInternalIdMasking:
    def test_file_id_is_removed_with_its_label(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        out = strip_internal_ids(
            "에이전트_요금제_안내.xlsx (file_id: 8paerwxuyb8jfch8z8o14xe97o) 입니다.")

        assert "8paerwxuyb8jfch8z8o14xe97o" not in out
        assert "에이전트_요금제_안내.xlsx" in out      # 파일명은 살아야 한다
        assert "(" not in out                     # 빈 괄호가 남으면 안 된다

    def test_bare_channel_id_is_removed(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        assert "buup8jx6hpg3xrmw1bgxca3qaw" not in strip_internal_ids(
            "채널 buup8jx6hpg3xrmw1bgxca3qaw 에서 확인했습니다")

    def test_numeric_codes_are_left_alone(self):
        """주문번호·사업자번호 같은 순수 숫자는 사용자에게 필요한 값이다."""
        from agent_arch.utils.sanitize import strip_internal_ids

        text = "주문번호 12345678901234567890123456 확인"
        assert strip_internal_ids(text) == text

    def test_ordinary_text_is_untouched(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        text = "이 스레드에서 요약한 보고서.xlsx 파일입니다."
        assert strip_internal_ids(text) == text

    def test_applied_through_the_main_sanitizer(self):
        from agent_arch.utils.sanitize import sanitize_tool_names

        assert "8paerwxuyb8jfch8z8o14xe97o" not in sanitize_tool_names(
            "파일 (file_id: 8paerwxuyb8jfch8z8o14xe97o) 를 공유합니다")


class TestIdsInsideUrlsSurvive:
    """주소 안의 id 는 지우면 안 된다 (2026-09-16 실측 회귀).

    사용자가 "메시지 바로가기"를 눌러도 아무 일이 없었다. 링크는 파랗게 보이는데 주소가
    `https://chat.example.com/team/pl/` 로 끝나 목적지가 없었다 — 26자 id 를 지우는
    규칙이 URL 경로 안까지 지운 것이다. 지어낸 링크로 오판해 `answer_guard` 에 딥링크
    가드를 넣었지만, 그 가드는 정화 **전에** 돌아 온전한 URL 을 통과시키고 증상은 그대로였다.
    """

    PERMALINK = "https://chat.example.com/team/pl/hiozn4yzoibg9nwu1icjkahuzy"

    def test_messenger_permalink_keeps_its_post_id(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        text = f"- [🔗 메시지 바로가기]({self.PERMALINK})"
        assert strip_internal_ids(text) == text

    def test_file_download_url_keeps_its_file_id(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        text = "받으실 곳: https://chat.example.com/api/v4/files/8paerwxuyb8jfch8z8o14xe97o"
        assert strip_internal_ids(text) == text

    def test_scheme_less_and_relative_addresses_are_kept(self):
        """스킴을 생략한 표기·상대 경로도 렌더러가 링크로 만든다 — 목적지를 지우면 안 된다."""
        from agent_arch.utils.sanitize import strip_internal_ids

        for text in (
            "chat.example.com/team/pl/hiozn4yzoibg9nwu1icjkahuzy 참고하세요",
            "[보기](/team/pl/hiozn4yzoibg9nwu1icjkahuzy)",
        ):
            assert strip_internal_ids(text) == text

    def test_bare_id_in_the_same_text_is_still_removed(self):
        """예외는 주소 안일 때뿐이다 — 맨몸으로 적힌 id 는 여전히 지운다."""
        from agent_arch.utils.sanitize import strip_internal_ids

        out = strip_internal_ids(
            f"[바로가기]({self.PERMALINK})\n"
            "- **파일 ID**: `8paerwxuyb8jfch8z8o14xe97o`\n")

        assert self.PERMALINK in out
        assert "8paerwxuyb8jfch8z8o14xe97o" not in out

    def test_link_guard_does_not_fire_after_sanitizing(self):
        """정화를 거친 뒤에도 딥링크 가드가 "지어낸 링크"로 오판하지 않아야 한다."""
        from agent_arch.agents.hitl.answer_guard import find_fabricated_permalinks
        from agent_arch.utils.sanitize import sanitize_tool_names

        assert find_fabricated_permalinks(
            sanitize_tool_names(f"[🔗 메시지 바로가기]({self.PERMALINK})")) == []

    @pytest.mark.parametrize("size", [1, 3, 7, 80, 81, 200])
    def test_streaming_keeps_the_address_whole(self, size):
        """토큰이 잘게 쪼개져도 화면(스트림)과 저장본(전문 정화)이 같아야 한다."""
        from agent_arch.utils.sanitize import StreamSanitizer, sanitize_tool_names

        text = f"관련 메신저 링크는 [🔗 메시지 바로가기]({self.PERMALINK}) 입니다. 확인해 주세요."
        s = StreamSanitizer()
        out = "".join(s.feed(text[i:i + size]) for i in range(0, len(text), size)) + s.flush()

        assert out == sanitize_tool_names(text) == text


class TestEmptyIdLineIsRemoved:
    """id 를 지운 뒤 남는 빈 라벨 줄은 줄 자체를 없앤다 (2026-09-15).

    실측: "- **파일 ID**: ``" 가 백틱만 남은 채 화면에 남았다.
    """

    def test_markdown_bullet_line_disappears(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        out = strip_internal_ids(
            "- **파일명**: 에이전트_요금제_안내.xlsx\n"
            "- **파일 ID**: `8paerwxuyb8jfch8z8o14xe97o`\n"
            "확인해 주세요.")

        assert "파일 ID" not in out
        assert "``" not in out
        assert "에이전트_요금제_안내.xlsx" in out and "확인해 주세요." in out

    def test_plain_label_line_disappears(self):
        from agent_arch.utils.sanitize import strip_internal_ids

        out = strip_internal_ids("file_id: 8paerwxuyb8jfch8z8o14xe97o\n다음 줄")

        assert out.strip() == "다음 줄"

    def test_line_with_other_content_is_kept(self):
        """ID 말고 다른 정보가 같은 줄에 있으면 줄을 지우면 안 된다."""
        from agent_arch.utils.sanitize import strip_internal_ids

        out = strip_internal_ids(
            "- 보고서.xlsx (file_id: 8paerwxuyb8jfch8z8o14xe97o) 37KB")

        assert "보고서.xlsx" in out and "37KB" in out


# ---------------------------------------------------------------------------
# 반복 문자 런 접기 — 모델 반복 루프 잔해(2026-09-15 stg 표 구분선 대시 124,956자)
# ---------------------------------------------------------------------------


class TestCollapseRepeatedRuns:
    def test_table_separator_loop_collapses_but_table_survives(self):
        """stg 실측 재현: 표 구분선 대시가 12만 자 이어져도 접히고, 표 머리·본문은 그대로다."""
        body = (
            "### 서버 설정값\n\n| 항목 | 넣을 값 |\n| :---- | :"
            + "-" * 124_000
            + " |\n| 계정 유형 | IMAP |\n"
        )
        out = collapse_repeated_runs(body)
        assert len(out) < 200
        assert "| 항목 | 넣을 값 |" in out
        assert "| :---- | :--- |" in out
        assert "| 계정 유형 | IMAP |" in out

    def test_normal_markdown_untouched(self):
        """정상 문서의 짧은 구분선·수평선·밑줄(수십 자)은 건드리지 않는다."""
        body = (
            "제목\n" + "=" * 60 + "\n\n" + "-" * 80 + "\n\n"
            "| a | b |\n| :---- | ----: |\n| 1 | 2 |\n" + "." * 100
        )
        assert collapse_repeated_runs(body) == body

    def test_threshold_boundary(self):
        assert collapse_repeated_runs("x" * 119) == "x" * 119
        assert collapse_repeated_runs("x" * 120) == "xxx"

    def test_multiple_runs_and_mixed_chars(self):
        out = collapse_repeated_runs("앞" + "-" * 300 + "중간" + "=" * 500 + "뒤" + "\n" * 200 + "끝")
        assert out == "앞---중간===뒤\n\n\n끝"

    def test_empty_and_none_passthrough(self):
        assert collapse_repeated_runs("") == ""
        assert collapse_repeated_runs(None) is None


# ---------------------------------------------------------------------------
# 가이드 위키 주소 차단
# ---------------------------------------------------------------------------

_WIKI = "https://example.atlassian.net/wiki/spaces/GUIDE/pages/123456789/%ED%9C%B4%EA%B0%80+%EC%8B%A0%EC%B2%AD"
_FORM = "https://example.com/main/support/inquiry/write"


class TestIsWikiUrl:
    def test_subdomain_and_bare_host(self):
        assert is_wiki_url("https://example.atlassian.net/wiki/x")
        assert is_wiki_url("http://atlassian.net/")

    def test_lookalike_hosts_and_garbage_not_wiki(self):
        assert not is_wiki_url("https://evil-atlassian.net/x")  # 접미가 아니라 다른 호스트
        assert not is_wiki_url("https://example.com/atlassian.net")  # 경로에만 있는 경우
        assert not is_wiki_url(_FORM)
        assert not is_wiki_url("")
        assert not is_wiki_url("not a url")


class TestStripWikiLinks:
    def test_markdown_link_keeps_label_text(self):
        text = f"자세한 절차는 [휴가 신청 가이드]({_WIKI})를 참고하세요."
        assert strip_wiki_links(text) == "자세한 절차는 휴가 신청 가이드를 참고하세요."

    def test_service_link_untouched(self):
        text = f"문의는 [문의 폼]({_FORM})에서 접수됩니다."
        assert strip_wiki_links(text) == text

    def test_bare_and_autolink_removed(self):
        assert strip_wiki_links(f"주소: {_WIKI} 입니다") == "주소:  입니다"
        assert strip_wiki_links(f"주소 <{_WIKI}> 끝") == "주소  끝"
        assert strip_wiki_links(f"가이드({_WIKI})를 보세요") == "가이드를 보세요"

    def test_image_link_removed_entirely(self):
        assert strip_wiki_links(f"그림 ![캡처]({_WIKI}/img.png) 참고") == "그림  참고"

    def test_reference_section_wiki_only_items_dropped_with_header(self):
        text = (
            "절차는 위와 같습니다.\n\n"
            "🔗 참고 링크\n"
            f"- [휴가 신청 가이드]({_WIKI})\n"
            f"- [연차 정책]({_WIKI}2)\n"
        )
        assert strip_wiki_links(text) == "절차는 위와 같습니다.\n\n"

    def test_reference_section_keeps_service_items(self):
        text = (
            "절차는 위와 같습니다.\n\n"
            "🔗 참고 링크\n"
            f"- [휴가 신청 가이드]({_WIKI})\n"
            f"- [회원 문의폼]({_FORM})\n"
        )
        assert strip_wiki_links(text) == (
            "절차는 위와 같습니다.\n\n🔗 참고 링크\n" f"- [회원 문의폼]({_FORM})\n"
        )

    def test_no_wiki_returns_same_object(self):
        text = f"[문의 폼]({_FORM}) 과 (빈 괄호 아님) 그리고\n- 항목"
        assert strip_wiki_links(text) is text

    def test_applied_by_sanitize_tool_names(self, empty_tool_cache):
        out = sanitize_tool_names(f"[가이드]({_WIKI})를 보세요")
        assert "atlassian" not in out and out == "가이드를 보세요"

    @pytest.mark.parametrize("size", [1, 3, 7, 80, 81, 200])
    def test_stream_drops_wiki_only_reference_section(self, size, empty_tool_cache):
        """항목이 전부 위키인 "🔗 참고 링크" 는 머리말까지 사라진다 — 스트리밍도 전문 정화와 같아야 한다.
        머리말은 항목을 다 보기 전엔 지울지 알 수 없어, 목록이 끝날 때까지 한 head 에 묶어 둔다."""
        text = (
            "안내드립니다. " * 8
            + "\n\n🔗 참고 링크\n"
            + f"- [휴가 신청 가이드]({_WIKI})\n"
            + f"- [연차 정책]({_WIKI}2)\n"
        )
        expected = sanitize_tool_names(text)
        assert "참고 링크" not in expected
        s = StreamSanitizer()
        parts = [s.feed(text[i : i + size]) for i in range(0, len(text), size)]
        parts.append(s.flush())
        assert "".join(parts) == expected

    @pytest.mark.parametrize("size", [1, 3, 7, 80, 81, 200])
    def test_stream_matches_full_sanitize_for_long_wiki_url(self, size, empty_tool_cache):
        """퍼센트 인코딩된 긴 Confluence 주소가 _LOOKBEHIND(80자) 경계에 걸쳐도 스트리밍 결과가
        전문 정화와 같다 — head 에 `[라벨](https://example.atl` 까지만 들어가 판정을 못 하면 화면에만
        새는 유형(저장본은 멀쩡)을 막는다."""
        text = (
            "안내드립니다. " * 6
            + f"자세한 절차는 [휴가 신청 가이드]({_WIKI})를 참고하시고, "
            + f"문의는 [문의 폼]({_FORM})에서 접수됩니다. "
            + "🔗 참고 링크\n"
            + f"- [휴가 신청 가이드]({_WIKI})\n"
            + f"- [회원 문의폼]({_FORM})\n"
        )
        assert len(_WIKI) > StreamSanitizer._LOOKBEHIND
        expected = sanitize_tool_names(text)
        s = StreamSanitizer()
        parts = [s.feed(text[i : i + size]) for i in range(0, len(text), size)]
        parts.append(s.flush())
        actual = "".join(parts)
        assert actual == expected
        assert "atlassian" not in actual
        assert _FORM in actual


class TestReferenceLinksSection:
    """모델이 쓴 "참고 링크" 섹션(`- [라벨](URL)` 목록)은 코드 소유(guide_links 가 `- URL` 로 다시 붙임)라
    어떤 머리말 변형이든 걷어낸다. 실측(2026-09-16~17): `[제목](None)`, 제목을 영어로 번역한 가짜 경로, 🔗 없는 굵은 머리말."""

    _FAKE = "https://docs.example.com/652vpj8/652vpj8"
    _SCREENSHOT = (
        "고객 문의는 주로 두 가지 방법으로 접수하고 대응할 수 있습니다.\n\n"
        "1. 회사-문의하기 채널\n\n"
        "**참고 링크**\n"
        f"- [메신저 채널 설정 및 운영 - 회사-문의하기 채널]({_FAKE})\n"
        "- [대표메일 담당자 배분·회신 가이드](https://example.com/user-guide/representative-email)\n\n"
        "메신저의 문의 채널을 바로 열어드릴 수 있어요."
    )

    def test_bold_header_section_removed_and_trailing_sentence_kept(self):
        assert strip_reference_links_section(self._SCREENSHOT) == (
            "고객 문의는 주로 두 가지 방법으로 접수하고 대응할 수 있습니다.\n\n"
            "1. 회사-문의하기 채널\n\n"
            "메신저의 문의 채널을 바로 열어드릴 수 있어요."
        )

    @pytest.mark.parametrize("header", ["🔗 참고 링크", "### 🔗 참고 링크:", "**🔗 참고 링크**", "🔗 **참고 링크**", "참고링크:", "__참고 링크__"])
    def test_header_variants(self, header):
        text = f"끝.\n\n{header}\n- [DMP 표시 언어 가이드](None)\n* [메일 언어 설정 가이드](None)\n1. [에이전트 번역 가이드]({_FORM})\n"
        assert strip_reference_links_section(text) == "끝.\n"

    def test_inline_and_midline_forms(self):
        assert strip_reference_links_section("본문.\n**참고 링크**: [A](https://x.com/a), [B](https://x.com/b)\n다음 문장") == "본문.\n다음 문장"
        assert strip_reference_links_section(f"접수됩니다. 🔗 참고 링크\n- [폼]({_FORM})\n") == "접수됩니다. "   # 줄바꿈을 빼먹은 머리말

    def test_code_footer_and_plain_sentences_untouched(self):
        code = f"본문.\n\n🔗 참고 링크\n- {_FORM}\n- https://www.example.com/contact"
        assert strip_reference_links_section(code) is code                    # 코드 각주(`- URL`)는 모델 꼴이 아니다
        keep = "참고 링크는 아래에서 확인하세요.\n- [A](https://x.com/a)"           # 문장 속 '참고 링크' 는 머리말 아님
        assert strip_reference_links_section(keep) is keep
        assert strip_reference_links_section("문의는 [문의 폼](None) 에서, [센터]() 도.") == "문의는 문의 폼 에서, 센터 도."

    def test_applied_by_sanitize_tool_names(self, empty_tool_cache):
        out = sanitize_tool_names(self._SCREENSHOT)
        assert "참고 링크" not in out and self._FAKE not in out and out.endswith("열어드릴 수 있어요.")

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 80, 81, 200])
    @pytest.mark.parametrize("variant", ["screenshot", "emoji_end", "heading", "inline", "midline", "code_footer"])
    def test_stream_matches_full_sanitize(self, size, variant, empty_tool_cache):
        """스트리밍 정화 == 전문 정화. 머리말이 도착 중이거나 목록이 head 경계에 걸쳐도 같아야 한다
        (1자 스트림 실측: 머리말이 버퍼 맨 앞에 놓여 `*|*참고` 사이가 잘리면 섹션이 통째로 화면에 샌다)."""
        base = "안내드립니다. " * 8
        texts = {
            "screenshot": self._SCREENSHOT,
            "emoji_end": base + "\n\n🔗 참고 링크\n- [DMP 표시 언어 가이드](None)\n- [메일 언어 설정 가이드](None)\n",
            "heading": base + "\n\n### 🔗 참고 링크:\n- [A](https://x.com/a)\n- [B](https://x.com/b)\n\n끝 문장입니다.",
            "inline": base + "\n\n**참고 링크**: [A](https://x.com/a), [B](https://x.com/b)\n끝 문장입니다.",
            "midline": base + f"문의는 [문의 폼]({_FORM})에서 접수됩니다. 🔗 참고 링크\n- [휴가 신청 가이드]({_WIKI})\n- [회원 문의폼]({_FORM})\n",
            "code_footer": base + f"\n\n🔗 참고 링크\n- {_FORM}\n- https://www.example.com/contact",
        }
        text = texts[variant]
        expected = sanitize_tool_names(text)
        assert ("참고 링크" in expected) == (variant == "code_footer")
        s = StreamSanitizer()
        parts = [s.feed(text[i : i + size]) for i in range(0, len(text), size)]
        parts.append(s.flush())
        assert "".join(parts) == expected


# ---------------------------------------------------------------------------
# 근거 없는 링크 제거 — 지어낸 가이드 문서 링크가 화면·저장본 어디에도 남지 않는다
# ---------------------------------------------------------------------------

class TestUnallowedLinks:
    """2026-09-21 신고: "더 자세한 내용은 아래 가이드를 참고해 주세요." + 가이드 문서 링크 3개.

    주소는 모델이 지어낸 것이었다(가이드 페이지의 진짜 주소는 LLM 에게 가지 않는다). 판정은
    `agents/react/link_policy.py` 가 하고 여기는 콜백만 받는다 — 허용 목록에 없는 주소만 지운다.
    """

    _ALLOWED = (_FORM, "https://example.com/main/download")
    _FAKE_A = "https://example.com/user-guide/messenger-agent"     # 실측 환각: 진짜와 같은 호스트
    _FAKE_B = "https://example.com/user-guide/agent"

    def _permits(self, url):
        return any(url.rstrip(".,)").startswith(a) for a in self._ALLOWED)

    def _clean(self, text):
        return sanitize_tool_names(text, link_permits=self._permits)

    def test_leadin_line_with_only_fabricated_links_is_removed(self, empty_tool_cache):
        text = ("에이전트 사용 방법입니다.\n\n"
                f"더 자세한 내용은 아래 가이드를 참고해 주세요. [메신저에서 에이전트 부르는 법]({self._FAKE_A}) "
                f"[Agent 사용 심층 가이드]({self._FAKE_B})\n")
        out = self._clean(text)
        assert out == "에이전트 사용 방법입니다.\n\n"          # 유도 문구까지 통째로(라벨만 남기면 말이 안 된다)

    def test_leadin_stays_when_a_link_is_sourced(self, empty_tool_cache):
        text = f"더 자세한 내용은 아래 가이드를 참고해 주세요. [문의 폼]({_FORM})\n"
        assert self._clean(text) == text

    def test_list_block_and_orphan_leadin(self, empty_tool_cache):
        text = ("안내입니다.\n\n자세한 내용은 아래 가이드를 참고해 주세요.\n"
                f"- [DMP 사용 가이드]({self._FAKE_A})\n- [Agent 가이드]({self._FAKE_B})\n\n끝.\n")
        out = self._clean(text)
        assert "가이드를 참고해 주세요" not in out and "DMP 사용 가이드" not in out
        assert out.startswith("안내입니다.") and out.endswith("끝.\n")

    def test_inline_link_keeps_label_and_drops_address(self, empty_tool_cache):
        text = f"설정은 [설정 가이드]({self._FAKE_A})에 있습니다."
        assert self._clean(text) == "설정은 설정 가이드에 있습니다."

    def test_sourced_links_and_ordinary_sentences_untouched(self, empty_tool_cache):
        text = (f"문의는 [문의 폼]({_FORM}) 에서, 앱은 https://example.com/main/download 에서 받으세요.\n"
                "자세한 내용은 담당자에게 문의해 주세요.\n")
        assert self._clean(text) == text

    def test_absolute_path_links_become_plain_labels(self, empty_tool_cache):
        """`[가이드](/wiki/…)` 는 http 가 아니라 기존 정화기 어디에도 안 걸렸다 — 열면 앱 오리진에
        경로가 붙어 없는 페이지로 간다(신고된 "엉뚱한 페이지"의 다른 얼굴)."""
        text = f"문의는 [문의 폼]({_FORM}) 에서, 절차는 [가이드](/wiki/spaces/AcmeCustomer/pages/1) 에.\n"
        assert self._clean(text) == f"문의는 [문의 폼]({_FORM}) 에서, 절차는 가이드 에.\n"
        only_path = "절차는 [가이드](/wiki/spaces/AcmeCustomer/pages/1) 에 있습니다."
        assert self._clean(only_path) == "절차는 가이드 에 있습니다."      # http 가 없어도 돈다

    def test_no_policy_means_no_change(self, empty_tool_cache):
        text = f"[지어낸 가이드]({self._FAKE_A})"
        assert sanitize_tool_names(text) == text          # 가이드 검색을 안 쓴 턴은 현행 그대로

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 80, 81, 200, 600])
    @pytest.mark.parametrize("variant", ["oneline", "list", "inline", "keep", "long"])
    def test_stream_matches_full_sanitize(self, size, variant, empty_tool_cache):
        """스트리밍 정화 == 전문 정화. 유도 문구와 링크가 head 경계로 갈려도 같아야 한다 —
        갈리면 **화면에만** 고아 문장이 남고 저장본만 깨끗해진다(화면 ≠ 히스토리)."""
        base = "안내드립니다. " * 8
        texts = {
            "oneline": base + f"\n\n더 자세한 내용은 아래 가이드를 참고해 주세요. [A]({self._FAKE_A}) [B]({self._FAKE_B})\n",
            "list": base + f"\n\n자세한 내용은 아래 가이드를 참고해 주세요.\n- [A]({self._FAKE_A})\n- [B]({self._FAKE_B})\n\n끝.\n",
            "inline": base + f"설정은 [설정 가이드]({self._FAKE_A})에 있습니다. 끝.\n",
            "keep": base + f"문의는 [문의 폼]({_FORM}) 에서 접수됩니다.\n",
            "long": "가" * 300 + f"\n\n더 자세한 내용은 아래 가이드를 참고해 주세요. [A]({self._FAKE_A})\n",
        }
        text = texts[variant]
        expected = self._clean(text)
        assert (self._FAKE_A in expected) is False
        s = StreamSanitizer(link_permits=self._permits)
        parts = [s.feed(text[i : i + size]) for i in range(0, len(text), size)]
        parts.append(s.flush())
        assert "".join(parts) == expected
