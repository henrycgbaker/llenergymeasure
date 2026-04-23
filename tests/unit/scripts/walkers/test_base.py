"""Tests for :mod:`scripts.walkers._base`.

Covers AST primitives, pattern detectors, filters, confidence scoring, the
helper tracer, and structured error types — all on synthetic AST fixtures so
the tests never depend on a specific library version.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

# Make the top-level ``scripts`` package importable from tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers._base import (  # noqa: E402
    ALL_DETECTORS,
    ConditionalLoggerWarningDetector,
    ConditionalRaiseDetector,
    ConditionalSelfAssignDetector,
    ConditionalWarningsWarnDetector,
    DetectedPattern,
    MinorIssuesDictAssignDetector,
    WalkerLandmarkMissingError,
    WalkerVersionMismatchError,
    call_func_path,
    check_installed_version,
    extract_assign_target,
    extract_condition_fields,
    extract_loop_literal_iterable,
    filter_condition_references_self,
    filter_kwargs_positive_derivable,
    filter_target_is_public_field,
    find_class,
    find_method,
    find_self_helper_calls,
    first_string_arg,
    resolve_local_assign,
    resolve_same_module_helpers,
    score_confidence,
)


def _parse_if(src: str) -> ast.If:
    module = ast.parse(src.strip())
    stmt = module.body[0]
    assert isinstance(stmt, ast.If)
    return stmt


def _parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def test_call_func_path_logger_warning() -> None:
    call = _parse_expr('logger.warning("msg")')
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["logger", "warning"]


def test_call_func_path_warnings_warn() -> None:
    call = _parse_expr('warnings.warn("msg")')
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["warnings", "warn"]


def test_call_func_path_self_method() -> None:
    call = _parse_expr("self.helper(x)")
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["self", "helper"]


def test_call_func_path_opaque_returns_none() -> None:
    # double-call like foo()() is not a pure attr chain
    call = _parse_expr("foo()()")
    assert isinstance(call, ast.Call)
    assert call_func_path(call) is None


def test_first_string_arg_constant() -> None:
    call = _parse_expr('logger.warning("hello")')
    assert isinstance(call, ast.Call)
    assert first_string_arg(call) == "hello"


def test_first_string_arg_fstring() -> None:
    call = _parse_expr('logger.warning(f"value is {x}")')
    assert isinstance(call, ast.Call)
    out = first_string_arg(call)
    assert out is not None and "x" in out


def test_first_string_arg_format_call() -> None:
    call = _parse_expr('logger.warning("val={v}".format(v=1))')
    assert isinstance(call, ast.Call)
    out = first_string_arg(call)
    assert out is not None and "format" in out


def test_extract_condition_fields_simple() -> None:
    expr = _parse_expr("self.temperature < 0.01")
    assert extract_condition_fields(expr) == {"temperature"}


def test_extract_condition_fields_multi() -> None:
    expr = _parse_expr("self.do_sample is False and self.temperature != 1.0")
    assert extract_condition_fields(expr) == {"do_sample", "temperature"}


def test_extract_assign_target_self_attr() -> None:
    stmt = ast.parse("self.temperature = 0.5").body[0]
    assert isinstance(stmt, ast.Assign)
    assert extract_assign_target(stmt) == "temperature"


def test_extract_assign_target_non_self_returns_none() -> None:
    stmt = ast.parse("other.temperature = 0.5").body[0]
    assert isinstance(stmt, ast.Assign)
    assert extract_assign_target(stmt) is None


def test_resolve_local_assign_finds_literal() -> None:
    src = """
def validate(self):
    greedy_msg = "Greedy wrong: {flag}"
    return greedy_msg.format(flag="temperature")
"""
    func = ast.parse(src.strip()).body[0]
    assert isinstance(func, ast.FunctionDef)
    assert resolve_local_assign(func, "greedy_msg") == "Greedy wrong: {flag}"


def test_resolve_local_assign_missing_returns_none() -> None:
    src = "def validate(self):\n    x = 1\n"
    func = ast.parse(src).body[0]
    assert isinstance(func, ast.FunctionDef)
    assert resolve_local_assign(func, "greedy_msg") is None


def test_extract_loop_literal_iterable_list() -> None:
    loop = ast.parse("for arg in ['a', 'b', 'c']: pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) == ["a", "b", "c"]


def test_extract_loop_literal_iterable_tuple() -> None:
    loop = ast.parse("for arg in (1, 2, 3): pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) == [1, 2, 3]


def test_extract_loop_literal_iterable_self_attr_returns_none() -> None:
    # Non-literal iterable (self.<field>) should downgrade detection.
    loop = ast.parse("for arg in self.allowed: pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) is None


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


def test_conditional_raise_detector_positive() -> None:
    detector = ConditionalRaiseDetector()
    node = _parse_if('if self.temperature < 0:\n    raise ValueError("must be non-negative")')
    pattern = detector.detect(node.body[0])
    assert pattern is not None
    assert pattern.severity == "error"
    assert pattern.emission_channel == "none"
    assert pattern.message_template == "must be non-negative"


def test_conditional_raise_detector_negative() -> None:
    detector = ConditionalRaiseDetector()
    stmt = ast.parse("x = 1").body[0]
    assert detector.detect(stmt) is None


def test_conditional_self_assign_detector_positive() -> None:
    detector = ConditionalSelfAssignDetector()
    node = _parse_if("if cond:\n    self.stop = []")
    pattern = detector.detect(node.body[0])
    assert pattern is not None
    assert pattern.severity == "dormant"
    assert pattern.affected_field == "stop"


def test_conditional_self_assign_detector_non_self() -> None:
    detector = ConditionalSelfAssignDetector()
    node = _parse_if("if cond:\n    other.field = 1")
    assert detector.detect(node.body[0]) is None


def test_conditional_warnings_warn_detector() -> None:
    detector = ConditionalWarningsWarnDetector()
    node = _parse_if('if cond:\n    warnings.warn("deprecated")')
    pattern = detector.detect(node.body[0])
    assert pattern is not None
    assert pattern.severity == "warn"
    assert pattern.emission_channel == "warnings_warn"


def test_conditional_logger_warning_detector_once() -> None:
    detector = ConditionalLoggerWarningDetector()
    node = _parse_if('if cond:\n    logger.warning_once("msg")')
    pattern = detector.detect(node.body[0])
    assert pattern is not None
    assert pattern.emission_channel == "logger_warning_once"


def test_conditional_logger_warning_detector_rejects_other_methods() -> None:
    detector = ConditionalLoggerWarningDetector()
    node = _parse_if('if cond:\n    logger.info("informational")')
    assert detector.detect(node.body[0]) is None


def test_minor_issues_dict_assign_detector() -> None:
    detector = MinorIssuesDictAssignDetector()
    node = _parse_if('if self.x:\n    minor_issues["temperature"] = msg.format(x=self.x)')
    pattern = detector.detect(node.body[0])
    assert pattern is not None
    assert pattern.emission_channel == "minor_issues_dict"
    assert pattern.affected_field == "temperature"


def test_all_detectors_registered_and_ordered() -> None:
    # The detector tuple is a fixed public sequence. The ordering matters:
    # raise before self-assign, minor_issues before generic self-assign.
    names = [type(d).__name__ for d in ALL_DETECTORS]
    assert names.index("ConditionalRaiseDetector") < names.index("ConditionalSelfAssignDetector")
    assert names.index("MinorIssuesDictAssignDetector") > names.index(
        "ConditionalSelfAssignDetector"
    )


# ---------------------------------------------------------------------------
# Filters and confidence scoring
# ---------------------------------------------------------------------------


PUBLIC_FIELDS = frozenset({"temperature", "top_p", "do_sample", "stop"})


def test_filter_condition_references_self_positive() -> None:
    expr = _parse_expr("self.temperature < 0.01")
    assert filter_condition_references_self(expr, PUBLIC_FIELDS) is True


def test_filter_condition_references_self_private_field() -> None:
    expr = _parse_expr("self._internal is True")
    # _internal is not in the public field set.
    assert filter_condition_references_self(expr, PUBLIC_FIELDS) is False


def test_filter_condition_references_self_argument_only() -> None:
    expr = _parse_expr("strict")
    assert filter_condition_references_self(expr, PUBLIC_FIELDS) is False


def test_filter_target_is_public_field_positive() -> None:
    pattern = DetectedPattern(
        severity="dormant",
        emission_channel="none",
        affected_field="temperature",
        message_template=None,
        detail="self.temperature = 0",
    )
    assert filter_target_is_public_field(pattern, PUBLIC_FIELDS) is True


def test_filter_target_is_public_field_rejects_private() -> None:
    pattern = DetectedPattern(
        severity="dormant",
        emission_channel="none",
        affected_field="_initialized",
        message_template=None,
        detail="",
    )
    assert filter_target_is_public_field(pattern, PUBLIC_FIELDS) is False


def test_filter_target_is_public_field_neutral_for_non_assign() -> None:
    # Non-assign pattern (no affected_field) passes this filter trivially.
    pattern = DetectedPattern(
        severity="error",
        emission_channel="none",
        affected_field=None,
        message_template=None,
        detail="",
    )
    assert filter_target_is_public_field(pattern, PUBLIC_FIELDS) is True


def test_filter_kwargs_positive_derivable_simple() -> None:
    expr = _parse_expr("self.do_sample is False and self.temperature != 1.0")
    assert filter_kwargs_positive_derivable(expr) is True


def test_filter_kwargs_positive_derivable_rejects_opaque() -> None:
    # Opaque helper call against external state is not derivable.
    expr = _parse_expr("importlib.util.find_spec('scipy')")
    assert filter_kwargs_positive_derivable(expr) is False


def test_filter_kwargs_positive_derivable_accepts_isinstance() -> None:
    expr = _parse_expr("not isinstance(self.top_p, float)")
    assert filter_kwargs_positive_derivable(expr) is True


def test_score_confidence_levels() -> None:
    assert score_confidence(3) == "high"
    assert score_confidence(2) == "medium"
    assert score_confidence(1) == "low"
    assert score_confidence(0) == "low"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


def test_check_installed_version_in_range() -> None:
    check_installed_version("transformers", "4.56.0", SpecifierSet(">=4.50,<5.0"))


def test_check_installed_version_out_of_range() -> None:
    with pytest.raises(WalkerVersionMismatchError) as exc_info:
        check_installed_version("transformers", "5.0.0", SpecifierSet(">=4.50,<5.0"))
    assert "transformers" in str(exc_info.value)
    assert "5.0.0" in str(exc_info.value)


def test_check_installed_version_invalid_version_string() -> None:
    with pytest.raises(WalkerVersionMismatchError):
        check_installed_version("transformers", "not-a-version", SpecifierSet(">=4.50,<5.0"))


def test_walker_landmark_missing_error_carries_detail() -> None:
    exc = WalkerLandmarkMissingError("GenerationConfig.validate", "method removed")
    assert exc.landmark == "GenerationConfig.validate"
    assert "method removed" in str(exc)


# ---------------------------------------------------------------------------
# One-level helper tracer
# ---------------------------------------------------------------------------


_HELPER_MODULE_SRC = """
class Thing:
    def entry(self):
        self.sub_check()

    def sub_check(self):
        if self.temperature < 0:
            raise ValueError("bad")

    def unrelated(self):
        return 1
"""


def test_resolve_same_module_helpers_returns_class_methods() -> None:
    module = ast.parse(_HELPER_MODULE_SRC)
    helpers = resolve_same_module_helpers(module, "Thing")
    assert set(helpers) == {"entry", "sub_check", "unrelated"}


def test_find_self_helper_calls_depth_one() -> None:
    module = ast.parse(_HELPER_MODULE_SRC)
    cls = find_class(module, "Thing")
    assert cls is not None
    entry = find_method(cls, "entry")
    assert entry is not None
    assert find_self_helper_calls(entry) == ["sub_check"]


def test_find_class_and_method_helpers() -> None:
    module = ast.parse(_HELPER_MODULE_SRC)
    assert find_class(module, "NonExistent") is None
    cls = find_class(module, "Thing")
    assert cls is not None
    assert find_method(cls, "entry") is not None
    assert find_method(cls, "does_not_exist") is None
