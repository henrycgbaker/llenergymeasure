"""Tests for the baseline-only container entrypoint.

These tests never call real CUDA, pynvml, or Docker. The CUDA prime helper is
patched at module scope and the underlying measurement primitives are patched
at their source modules (``harness/baseline.py``) because the entrypoint uses
lazy local imports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llenergymeasure.config.ssot import ENV_BASELINE_SPEC_PATH
from llenergymeasure.entrypoints import baseline_measure
from llenergymeasure.harness.baseline import BaselineCache

_PATCH_MEASURE = "llenergymeasure.harness.baseline.measure_baseline_power"
_PATCH_SPOT_CHECK = "llenergymeasure.harness.baseline.measure_spot_check"
_PATCH_PRIME_CUDA = "llenergymeasure.entrypoints.baseline_measure._prime_cuda"


def _write_spec(
    tmp_path: Path,
    mode: str = "measure",
    gpu_indices: list[int] | None = None,
    duration_sec: float = 30.0,
) -> Path:
    spec_path = tmp_path / "baseline_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "gpu_indices": gpu_indices if gpu_indices is not None else [0],
                "duration_sec": duration_sec,
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def _make_cache(power_w: float = 42.59) -> BaselineCache:
    return BaselineCache(
        power_w=power_w,
        timestamp=time.time(),
        gpu_indices=[0],
        sample_count=288,
        duration_sec=30.0,
    )


class TestRunBaselineMeasurement:
    def test_reads_spec_writes_result(self, tmp_path: Path):
        spec_path = _write_spec(tmp_path, mode="measure", gpu_indices=[0, 1])

        with (
            patch(_PATCH_PRIME_CUDA) as mock_prime,
            patch(_PATCH_MEASURE, return_value=_make_cache(power_w=42.6)) as mock_measure,
        ):
            result_path = baseline_measure.run_baseline_measurement(spec_path)

        assert result_path.exists()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["power_w"] == pytest.approx(42.6)
        assert payload["mode"] == "measure"
        assert "gpu_indices" in payload
        mock_prime.assert_called_once_with([0, 1])
        mock_measure.assert_called_once()

    def test_spot_check_mode_uses_spot_check_helper(self, tmp_path: Path):
        spec_path = _write_spec(tmp_path, mode="spot_check", gpu_indices=[0], duration_sec=5.0)

        with (
            patch(_PATCH_PRIME_CUDA),
            patch(_PATCH_SPOT_CHECK, return_value=41.9) as mock_spot,
            patch(_PATCH_MEASURE) as mock_measure,
        ):
            result_path = baseline_measure.run_baseline_measurement(spec_path)

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["mode"] == "spot_check"
        assert payload["power_w"] == pytest.approx(41.9)
        mock_spot.assert_called_once()
        mock_measure.assert_not_called()

    def test_measurement_none_raises(self, tmp_path: Path):
        spec_path = _write_spec(tmp_path, mode="measure")

        with (
            patch(_PATCH_PRIME_CUDA),
            patch(_PATCH_MEASURE, return_value=None),
            pytest.raises(RuntimeError, match="measure_baseline_power returned None"),
        ):
            baseline_measure.run_baseline_measurement(spec_path)

    def test_invalid_mode_raises(self, tmp_path: Path):
        spec_path = _write_spec(tmp_path, mode="bogus")

        with patch(_PATCH_PRIME_CUDA), pytest.raises(ValueError, match="invalid mode"):
            baseline_measure.run_baseline_measurement(spec_path)

    def test_cuda_prime_called_before_measurement(self, tmp_path: Path):
        """_prime_cuda must run before the sampling primitive."""
        spec_path = _write_spec(tmp_path, mode="measure", gpu_indices=[2])
        order: list[str] = []

        def _prime_side_effect(*_args, **_kwargs):
            order.append("prime")

        def _measure_side_effect(*_args, **_kwargs):
            order.append("measure")
            return _make_cache()

        with (
            patch(_PATCH_PRIME_CUDA, side_effect=_prime_side_effect),
            patch(_PATCH_MEASURE, side_effect=_measure_side_effect),
        ):
            baseline_measure.run_baseline_measurement(spec_path)

        assert order == ["prime", "measure"]


class TestMain:
    def test_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv(ENV_BASELINE_SPEC_PATH, raising=False)
        with pytest.raises(RuntimeError, match=ENV_BASELINE_SPEC_PATH):
            baseline_measure.main()

    def test_exception_writes_error_json(self, tmp_path: Path, monkeypatch):
        spec_path = _write_spec(tmp_path, mode="measure")
        monkeypatch.setenv(ENV_BASELINE_SPEC_PATH, str(spec_path))

        with (
            patch(_PATCH_PRIME_CUDA),
            patch(_PATCH_MEASURE, side_effect=RuntimeError("boom")),
            pytest.raises(SystemExit) as exc_info,
        ):
            baseline_measure.main()

        assert exc_info.value.code != 0
        error_path = tmp_path / "baseline_error.json"
        assert error_path.exists()
        payload = json.loads(error_path.read_text(encoding="utf-8"))
        assert payload["type"] == "RuntimeError"
        assert "boom" in payload["message"]
        assert "traceback" in payload


class TestPrimeCuda:
    def test_prime_cuda_tolerates_missing_torch(self, monkeypatch):
        """No torch in sys.modules: _prime_cuda logs and returns."""
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Should not raise
        baseline_measure._prime_cuda([0])

    def test_prime_cuda_tolerates_cuda_unavailable(self, monkeypatch):
        """torch present but cuda unavailable: _prime_cuda logs and returns."""
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: False,
            init=lambda: None,
        )
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        fake_torch.zeros = lambda *a, **k: None  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        # Should not raise
        baseline_measure._prime_cuda([0])
