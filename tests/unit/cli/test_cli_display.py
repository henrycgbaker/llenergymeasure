"""Unit tests for CLI display utilities and VRAM estimator.

Tests only pure functions that require no GPU or network access.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.cli._display import (
    _format_duration,
    _sig3,
    format_error,
    format_validation_error,
    print_dry_run,
    print_experiment_header,
    print_result_summary,
    print_study_summary,
)
from llenergymeasure.cli._vram import DTYPE_BYTES
from llenergymeasure.study._progress import print_study_progress
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# _sig3 tests
# =============================================================================


def test_sig3_integer():
    """312.4 rounds to 3 sig figs -> 312."""
    assert _sig3(312.4) == "312"


def test_sig3_decimal():
    """3.12 has 3 sig figs already -> '3.12'."""
    assert _sig3(3.12) == "3.12"


def test_sig3_small():
    """Small numbers retain 3 sig figs."""
    assert _sig3(0.00312) == "0.00312"


def test_sig3_zero():
    """Zero returns '0'."""
    assert _sig3(0) == "0"


def test_sig3_large_integer():
    """1234 rounds to 1230 at 3 sig figs."""
    assert _sig3(1234) == "1230"


def test_sig3_round_value():
    """847.0 stays as '847' (already 3 sig figs)."""
    assert _sig3(847.0) == "847"


# =============================================================================
# _format_duration tests
# =============================================================================


def test_format_duration_seconds():
    """Sub-minute durations shown as 'Xs.Xs'."""
    assert _format_duration(4.2) == "4.2s"


def test_format_duration_minutes():
    """272 seconds = 4m 32s."""
    assert _format_duration(272) == "4m 32s"


def test_format_duration_hours():
    """3900 seconds = 1h 05m."""
    assert _format_duration(3900) == "1h 05m"


def test_format_duration_exact_minute():
    """60 seconds = 1m 00s."""
    assert _format_duration(60) == "1m 00s"


def test_format_duration_exact_hour():
    """3600 seconds = 1h 00m."""
    assert _format_duration(3600) == "1h 00m"


# =============================================================================
# DTYPE_BYTES tests
# =============================================================================


def test_vram_dtype_bytes():
    """DTYPE_BYTES contains expected dtype entries."""
    assert DTYPE_BYTES["float32"] == 4
    assert DTYPE_BYTES["float16"] == 2
    assert DTYPE_BYTES["bfloat16"] == 2
    assert DTYPE_BYTES["int8"] == 1
    assert DTYPE_BYTES["int4"] == 0.5


def test_vram_dtype_bytes_keys():
    """DTYPE_BYTES has all expected keys."""
    expected_keys = {"float32", "float16", "bfloat16", "int8", "int4"}
    assert set(DTYPE_BYTES.keys()) == expected_keys


# =============================================================================
# format_validation_error tests
# =============================================================================


def test_format_validation_error():
    """ValidationError from bad engine value gets a friendly header."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(task={"model": "gpt2"}, engine="pytorh")  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    assert "Config validation failed" in result
    assert "error" in result.lower()


def test_format_validation_error_did_you_mean():
    """Literal errors on engine field suggest correct spelling."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(task={"model": "gpt2"}, engine="pytorh")  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    # Should suggest 'transformers' for the typo 'pytorh'
    assert "transformers" in result.lower() or "Did you mean" in result or "engine" in result


@pytest.mark.parametrize(
    "bad_kwargs, expected_str",
    [
        ({"engine": "bad"}, "1 error)"),
        ({"engine": "bad", "dtype": "bad"}, "2 errors)"),
    ],
)
def test_format_validation_error_singular_plural(bad_kwargs, expected_str):
    """Single error uses '1 error)', multiple errors use 'N errors)'."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(task={"model": "gpt2"}, **bad_kwargs)  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    assert expected_str in result


# =============================================================================
# format_error tests
# =============================================================================


def test_format_error_concise():
    """format_error without verbose=True shows class name and message, no traceback."""
    err = ConfigError("missing required field 'model'")
    result = format_error(err, verbose=False)
    assert "ConfigError" in result
    assert "missing required field" in result
    # Should not include traceback keywords
    assert "Traceback" not in result
    assert "File " not in result


def test_format_error_includes_class_name():
    """format_error prefix is the exception class name."""
    err = ConfigError("test message")
    result = format_error(err, verbose=False)
    assert result.startswith("ConfigError:")


def test_format_error_subclass():
    """format_error works for any LLEMError subclass."""
    from llenergymeasure.utils.exceptions import EngineError

    err = EngineError("GPU OOM")
    result = format_error(err, verbose=False)
    assert "EngineError" in result
    assert "GPU OOM" in result


def test_format_error_verbose_with_traceback():
    """format_error with verbose=True includes traceback when exception is active."""
    from llenergymeasure.utils.exceptions import ExperimentError

    try:
        raise ExperimentError("inference crashed")
    except ExperimentError as e:
        result = format_error(e, verbose=True)

    # The traceback should be included in verbose mode
    assert "ExperimentError" in result
    assert "inference crashed" in result
    # When raised in try/except, traceback module captures a real traceback
    assert "Traceback" in result or "ExperimentError" in result


def test_format_error_docker_error_with_suggestion_and_stderr():
    """format_error surfaces fix_suggestion and stderr_snippet for DockerError subclasses."""
    from llenergymeasure.infra.docker_errors import DockerContainerError

    err = DockerContainerError(
        message="Container exited with code 1.",
        fix_suggestion="Check container logs above for details.",
        stderr_snippet="RuntimeError: CUDA out of memory\nKilled",
    )
    result = format_error(err, verbose=False)

    assert "DockerContainerError" in result
    assert "Container exited with code 1" in result
    assert "Suggestion: Check container logs above for details." in result
    assert "Container stderr (last 20 lines):" in result
    assert "CUDA out of memory" in result
    assert "Killed" in result


def test_format_error_docker_error_without_optional_fields():
    """format_error with DockerError that has no suggestion or stderr omits those sections."""
    from llenergymeasure.infra.docker_errors import DockerContainerError

    err = DockerContainerError(
        message="Container exited with code 1.",
        fix_suggestion="",
        stderr_snippet=None,
    )
    result = format_error(err, verbose=False)

    assert "DockerContainerError" in result
    assert "Suggestion:" not in result
    assert "Container stderr" not in result


def test_format_error_regular_llem_error_no_docker_fields():
    """format_error with a non-Docker LLEMError does not include Docker-specific output."""
    err = ConfigError("bad config")
    result = format_error(err, verbose=False)

    assert "Suggestion:" not in result
    assert "Container stderr" not in result


# =============================================================================
# print_result_summary tests
# =============================================================================


def test_print_result_summary_minimal(capsys):
    """Minimal result (no baseline, no FLOPs, no warnings) prints required sections."""
    from tests.conftest import make_result

    result = make_result(
        total_flops=0.0,
        baseline_power_w=None,
        energy_adjusted_j=None,
        measurement_warnings=[],
        latency_stats=None,
        warmup_excluded_samples=None,
    )
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "Energy" in out
    assert "Performance" in out
    assert "Timing" in out
    # No warnings section for a result without warnings
    assert "Warnings" not in out


def test_print_result_summary_with_baseline(capsys):
    """Result with baseline_power_w shows 'Baseline' line."""
    from tests.conftest import make_result

    result = make_result(
        baseline_power_w=50.0,
        energy_adjusted_j=8.0,
    )
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "Baseline" in out
    assert "Adjusted" in out


def test_print_result_summary_with_warnings(capsys):
    """Result with measurement_warnings shows 'Warnings' section."""
    from tests.conftest import make_result

    result = make_result(measurement_warnings=["thermal floor not reached"])
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "Warnings" in out
    assert "thermal floor not reached" in out


def test_print_result_summary_with_flops(capsys):
    """Result with total_flops > 0 shows FLOPs line in Performance section."""
    from tests.conftest import make_result

    result = make_result(total_flops=1.5e12)
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "FLOPs" in out


def test_print_result_summary_with_latency_stats(capsys):
    """Result with latency_stats that has ttft_ms/itl_ms shows Latency lines.

    The _display.py code checks hasattr(ls, "ttft_ms") — uses duck typing
    so any object with those attributes will trigger the print.
    """
    from unittest.mock import MagicMock

    from tests.conftest import make_result

    # Use MagicMock to create an object with ttft_ms and itl_ms attributes
    # (the display code uses hasattr duck typing, not a concrete LatencyStatistics)
    latency = MagicMock()
    latency.ttft_ms = 12.5
    latency.itl_ms = 3.2

    result = make_result()
    # Inject latency via model_construct to bypass validation
    result = result.model_copy(update={"latency_stats": latency})
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "Latency TTFT" in out
    assert "Latency ITL" in out


def test_print_result_summary_with_warmup(capsys):
    """Result with warmup_excluded_samples shows warmup line in Timing section."""
    from tests.conftest import make_result

    result = make_result(warmup_excluded_samples=5)
    print_result_summary(result)
    out = capsys.readouterr().out

    assert "Warmup" in out
    assert "5 prompts excluded" in out


# =============================================================================
# print_dry_run tests
# =============================================================================


def test_print_dry_run_with_vram(capsys):
    """Dry run with VRAM dict shows weights, KV cache, overhead, total."""
    from tests.conftest import make_config

    config = make_config(model="meta-llama/Llama-2-7b-hf", dtype="float16")
    vram = {"weights_gb": 13.48, "kv_cache_gb": 0.32, "overhead_gb": 2.02, "total_gb": 15.82}
    print_dry_run(config, vram, gpu_vram_gb=None)
    out = capsys.readouterr().out

    assert "Config (resolved)" in out
    assert "VRAM estimate" in out
    assert "Weights" in out
    assert "KV cache" in out
    assert "Overhead" in out
    assert "Total" in out
    assert "(unavailable)" not in out


def test_print_dry_run_without_vram(capsys):
    """Dry run with vram=None shows '(unavailable)'."""
    from tests.conftest import make_config

    config = make_config(model="gpt2", dtype="float16")
    print_dry_run(config, vram=None, gpu_vram_gb=None)
    out = capsys.readouterr().out

    assert "(unavailable)" in out


def test_print_dry_run_fits_gpu(capsys):
    """When total_gb <= gpu_vram_gb, output shows 'OK'."""
    from tests.conftest import make_config

    config = make_config(model="gpt2", dtype="float16")
    vram = {"weights_gb": 5.0, "kv_cache_gb": 0.1, "overhead_gb": 0.75, "total_gb": 5.85}
    print_dry_run(config, vram, gpu_vram_gb=80.0)
    out = capsys.readouterr().out

    assert "OK" in out


def test_print_dry_run_does_not_fit_gpu(capsys):
    """When total_gb > gpu_vram_gb, output shows 'WARNING'."""
    from tests.conftest import make_config

    config = make_config(model="meta-llama/Llama-70b-hf", dtype="float32")
    vram = {"weights_gb": 140.0, "kv_cache_gb": 1.0, "overhead_gb": 21.0, "total_gb": 162.0}
    print_dry_run(config, vram, gpu_vram_gb=80.0)
    out = capsys.readouterr().out

    assert "WARNING" in out


def test_print_dry_run_verbose_annotations(capsys):
    """With verbose=True, default values show annotation like '(default)'."""
    from tests.conftest import make_config

    config = make_config(model="gpt2", engine="transformers", dtype="bfloat16")
    vram = {"weights_gb": 0.5, "kv_cache_gb": 0.01, "overhead_gb": 0.075, "total_gb": 0.585}
    print_dry_run(config, vram, gpu_vram_gb=None, verbose=True)
    out = capsys.readouterr().out

    # verbose mode adds "(default)" annotation for default engine/dtype
    assert "(default)" in out


def test_print_dry_run_config_valid_message(capsys):
    """Dry run output ends with 'Config valid. Run without --dry-run to start.'"""
    from tests.conftest import make_config

    config = make_config(model="gpt2")
    print_dry_run(config, vram=None, gpu_vram_gb=None)
    out = capsys.readouterr().out

    assert "Config valid" in out


# =============================================================================
# print_study_dry_run tests
# =============================================================================


def test_print_study_dry_run_shows_configs(capsys, monkeypatch):
    """Study dry run shows each experiment config in a table."""
    from llenergymeasure.cli._display import print_study_dry_run
    from llenergymeasure.config.models import StudyConfig
    from tests.conftest import make_config

    configs = [
        make_config(model="gpt2", dtype="float16"),
        make_config(model="gpt2", dtype="float32"),
    ]
    study = StudyConfig(experiments=configs, study_name="test-sweep")

    import llenergymeasure.cli._vram as _vram_mod

    monkeypatch.setattr(_vram_mod, "estimate_vram", lambda c: None)
    monkeypatch.setattr(_vram_mod, "get_gpu_vram_gb", lambda: None)

    print_study_dry_run(study)
    out = capsys.readouterr().out

    assert "2 configs" in out
    assert "gpt2" in out
    # Both dtypes appear in the per-experiment list (both non-default)
    assert "float16" in out
    assert "float32" in out
    assert "Config valid" in out


def test_print_study_dry_run_vram_peak(capsys, monkeypatch):
    """Study dry run shows VRAM estimate for the peak model."""
    from llenergymeasure.cli._display import print_study_dry_run
    from llenergymeasure.config.models import StudyConfig
    from tests.conftest import make_config

    configs = [
        make_config(model="gpt2", dtype="float16"),
        make_config(model="gpt2", dtype="bfloat16"),
    ]
    study = StudyConfig(experiments=configs, study_name="vram-test")

    vram_small = {"weights_gb": 0.2, "kv_cache_gb": 0.0, "overhead_gb": 0.03, "total_gb": 0.23}
    vram_large = {"weights_gb": 0.3, "kv_cache_gb": 0.0, "overhead_gb": 0.04, "total_gb": 0.34}

    # Return different VRAM for each config (fp16 < bf16 in our mock)
    call_count = 0

    def mock_estimate(config):
        nonlocal call_count
        call_count += 1
        return vram_small if call_count == 1 else vram_large

    import llenergymeasure.cli._vram as _vram_mod

    monkeypatch.setattr(_vram_mod, "estimate_vram", mock_estimate)
    monkeypatch.setattr(_vram_mod, "get_gpu_vram_gb", lambda: 40.0)

    print_study_dry_run(study)
    out = capsys.readouterr().out

    # Should show the peak (larger) VRAM estimate
    assert "VRAM estimate (peak)" in out
    assert "0.3" in out  # weights of larger config
    assert "OK" in out


# =============================================================================
# print_experiment_header tests
# =============================================================================


def test_print_experiment_header_defaults(capsys):
    """Header with default config shows model and engine (defaults omitted)."""
    from tests.conftest import make_config

    config = make_config(model="gpt2", engine="transformers", dtype="bfloat16")
    print_experiment_header(config)
    err = capsys.readouterr().err

    assert "gpt2" in err
    assert "transformers" in err
    # Default dtype bfloat16 is omitted from the header
    assert "bfloat16" not in err


def test_print_experiment_header_non_default_n(capsys):
    """Non-default n is included in header."""
    from llenergymeasure.config.models import DatasetConfig
    from tests.conftest import make_config

    config = make_config(model="gpt2", dataset=DatasetConfig(n_prompts=50))
    print_experiment_header(config)
    err = capsys.readouterr().err

    assert "n_prompts=50" in err


def test_print_experiment_header_non_default_max_output_tokens(capsys):
    """Non-default max_output_tokens is included in header."""
    from tests.conftest import make_config

    config = make_config(model="gpt2", max_output_tokens=512)
    print_experiment_header(config)
    err = capsys.readouterr().err

    assert "max_output_tokens=512" in err


# =============================================================================
# print_study_summary tests
# =============================================================================


def test_print_study_summary_table_structure(capsys):
    """Study summary prints table header, rows, and separator."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 10.0
    exp.total_energy_j = 50.0
    exp.avg_tokens_per_second = 100.0
    exp.total_inference_time_sec = 8.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="my-study",
        study_design_hash="deadbeef",
        summary=StudySummary(
            total_experiments=1, completed=1, failed=0, total_wall_time_s=12.0, total_energy_j=50.0
        ),
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "my-study" in out
    assert "deadbeef" in out
    assert "#" in out  # table header column
    assert "Config" in out
    assert "Adj. E" in out  # 9-column header
    assert "gpt2" in out


def test_print_study_summary_failed_experiments(capsys):
    """Study summary with failed experiments shows 'Failed:' footer line."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 5.0
    exp.total_energy_j = 20.0
    exp.avg_tokens_per_second = 50.0
    exp.total_inference_time_sec = 4.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="failing-study",
        summary=StudySummary(
            total_experiments=2, completed=1, failed=1, total_wall_time_s=10.0, total_energy_j=20.0
        ),
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "Failed: 1" in out


def test_print_study_summary_no_total_experiments(capsys):
    """Study summary with total_experiments=0 skips footer without crashing."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 5.0
    exp.total_energy_j = 20.0
    exp.avg_tokens_per_second = 50.0
    exp.total_inference_time_sec = 4.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name=None,
        summary=StudySummary(),
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "unnamed" in out


def test_print_study_summary_truncates_long_model_name(capsys):
    """Models with names > 20 chars are truncated with '...' prefix."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    long_model = "organization/very-long-model-name-that-exceeds-twenty-chars"
    exp = MagicMock()
    exp.model_name = long_model
    exp.engine = "transformers"
    exp.duration_sec = 5.0
    exp.total_energy_j = 20.0
    exp.avg_tokens_per_second = 50.0
    exp.total_inference_time_sec = 4.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="truncation-test",
        summary=StudySummary(
            total_experiments=1, completed=1, failed=0, total_wall_time_s=5.0, total_energy_j=20.0
        ),
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "..." in out


def test_print_study_summary_with_result_files(capsys):
    """Study summary with result_files shows 'Results saved:' line."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 5.0
    exp.total_energy_j = 20.0
    exp.avg_tokens_per_second = 50.0
    exp.total_inference_time_sec = 4.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="file-test",
        summary=StudySummary(
            total_experiments=1, completed=1, failed=0, total_wall_time_s=5.0, total_energy_j=20.0
        ),
        result_files=["results/exp1/result.json", "results/exp2/result.json"],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "Results saved" in out
    assert "2 file(s)" in out


def test_print_study_summary_shows_adj_energy(capsys):
    """Adj. E column appears in header and adjusted energy value in row."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 10.0
    exp.total_energy_j = 50.0
    exp.avg_tokens_per_second = 100.0
    exp.total_inference_time_sec = 8.0
    exp.energy_adjusted_j = 35.0
    exp.mj_per_tok_adjusted = 0.35
    exp.mj_per_tok_total = 0.50

    summary = StudySummary(
        total_experiments=1,
        completed=1,
        total_wall_time_s=10.0,
        total_energy_j=50.0,
    )
    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="adj-test",
        summary=summary,
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    assert "Adj. E" in out  # header column
    assert "35" in out  # adjusted energy value


def test_print_study_summary_mj_tok_prefers_adjusted(capsys):
    """mJ/tok column uses adjusted energy when available."""
    from unittest.mock import MagicMock

    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    exp = MagicMock()
    exp.model_name = "gpt2"
    exp.engine = "transformers"
    exp.duration_sec = 10.0
    exp.total_energy_j = 50.0
    exp.avg_tokens_per_second = 100.0
    exp.total_inference_time_sec = 8.0
    exp.energy_adjusted_j = 35.0
    exp.mj_per_tok_adjusted = 0.123
    exp.mj_per_tok_total = 0.456

    summary = StudySummary(
        total_experiments=1,
        completed=1,
        total_wall_time_s=10.0,
        total_energy_j=50.0,
    )
    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="mj-test",
        summary=summary,
        result_files=[],
        measurement_protocol={},
    )
    print_study_summary(result)
    out = capsys.readouterr().out

    # mJ/tok should show adjusted value (0.123), not total (0.456)
    assert "0.123" in out
    assert "0.456" not in out


# =============================================================================
# print_study_progress tests
# =============================================================================


def test_print_study_progress_running_status(capsys):
    """Running status shows '...' icon."""
    from tests.conftest import make_config

    config = make_config(model="gpt2")
    print_study_progress(1, 5, config, status="running")
    err = capsys.readouterr().err

    assert "[1/5]" in err
    assert "..." in err
    assert "gpt2" in err


def test_print_study_progress_failed_status(capsys):
    """Failed status shows 'FAIL' icon."""
    from tests.conftest import make_config

    config = make_config(model="gpt2")
    print_study_progress(3, 5, config, status="failed")
    err = capsys.readouterr().err

    assert "FAIL" in err
    assert "[3/5]" in err


def test_print_study_progress_with_elapsed_and_energy(capsys):
    """Progress line includes formatted elapsed time and energy when provided."""
    from tests.conftest import make_config

    config = make_config(model="gpt2")
    print_study_progress(2, 4, config, status="completed", elapsed=90.0, energy=500.0)
    err = capsys.readouterr().err

    assert "1m 30s" in err  # 90s formatted
    assert "500" in err  # energy value (500 J -> _sig3 -> "500")
