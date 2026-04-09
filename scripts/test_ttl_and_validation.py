#!/usr/bin/env python3
"""Verification script for PR #242 + #243: host-managed TTL in Docker flow.

Tests the end-to-end behaviour of the baseline cache across the host-container
boundary, verifying that:

  1. Host enforces TTL on in-memory baseline (the fix in #243)
  2. Host reuses in-memory baseline when within TTL
  3. Re-measurement after TTL expiry produces a fresh disk cache
  4. Container accepts a freshly-saved cache from the host
  5. Container rejects an expired cache (no host refresh)
  6. Validated strategy: spot-check + no drift keeps baseline within TTL
  7. Long study simulation: baseline stays valid across many experiments

Run: python scripts/test_ttl_and_validation.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
    else:
        _failed += 1
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Host in-memory TTL enforcement
# ---------------------------------------------------------------------------


def test_host_inmemory_ttl() -> None:
    """Host's _get_baseline rejects expired in-memory baseline, reuses fresh."""
    section("1. Host enforces TTL on in-memory baseline")

    from llenergymeasure.harness.baseline import BaselineCache
    from llenergymeasure.study.runner import StudyRunner

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        runner = StudyRunner.__new__(StudyRunner)
        runner._study_config = MagicMock()
        runner._baseline_cache_path = None
        runner._experiments_since_validation = 0
        runner.study_dir = Path(tmpdir1)

        mock_config = MagicMock()
        mock_config.baseline.strategy = "cached"
        mock_config.baseline.cache_ttl_seconds = 300.0  # 5-minute TTL
        mock_config.baseline.duration_seconds = 30.0
        mock_config.gpu_indices = [0]

        # Case A: expired baseline (10 min old, 5-min TTL) -> should re-measure
        old_baseline = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 600,  # 10 min ago
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        runner._baseline = old_baseline

        fresh_baseline = BaselineCache(
            power_w=52.0,
            timestamp=time.time(),
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )

        with (
            patch(
                "llenergymeasure.harness.baseline.measure_baseline_power",
                return_value=fresh_baseline,
            ) as mock_measure,
            patch(
                "llenergymeasure.device.gpu_info._resolve_gpu_indices",
                return_value=[0],
            ),
            patch("llenergymeasure.harness.baseline.save_baseline_cache"),
            patch("llenergymeasure.harness.baseline.load_baseline_cache", return_value=None),
        ):
            result = runner._get_baseline(mock_config)

        check(
            "expired baseline triggers re-measurement",
            mock_measure.call_count == 1,
            f"measure called {mock_measure.call_count} time(s)",
        )
        check(
            "returns fresh baseline (52W, not stale 50W)",
            result is not None and result.power_w == 52.0,
            f"power={result.power_w if result else None}",
        )

        # Case B: fresh baseline (1 min old, 5-min TTL) -> should reuse
        runner2 = StudyRunner.__new__(StudyRunner)
        runner2._study_config = MagicMock()
        runner2._baseline_cache_path = None
        runner2._experiments_since_validation = 0
        runner2.study_dir = Path(tmpdir2)

        fresh_inmem = BaselineCache(
            power_w=55.0,
            timestamp=time.time() - 60,  # 1 min ago
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        runner2._baseline = fresh_inmem

        with patch(
            "llenergymeasure.harness.baseline.measure_baseline_power",
        ) as mock_measure2:
            result2 = runner2._get_baseline(mock_config)

        check(
            "within-TTL baseline reused (no re-measurement)",
            mock_measure2.call_count == 0,
        )
        check(
            "same object returned",
            result2 is fresh_inmem,
        )


# ---------------------------------------------------------------------------
# 2. Re-measurement produces fresh disk cache
# ---------------------------------------------------------------------------


def test_remeasurement_refreshes_disk() -> None:
    """When host re-measures after TTL expiry, new baseline is saved to disk."""
    section("2. Re-measurement refreshes disk cache")

    from llenergymeasure.harness.baseline import BaselineCache
    from llenergymeasure.study.runner import StudyRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StudyRunner.__new__(StudyRunner)
        runner._study_config = MagicMock()
        runner._baseline_cache_path = None
        runner._experiments_since_validation = 0
        runner.study_dir = Path(tmpdir)

        mock_config = MagicMock()
        mock_config.baseline.strategy = "cached"
        mock_config.baseline.cache_ttl_seconds = 300.0
        mock_config.baseline.duration_seconds = 30.0
        mock_config.gpu_indices = [0]

        # Set up expired in-memory baseline
        runner._baseline = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 600,
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )

        fresh = BaselineCache(
            power_w=52.0,
            timestamp=time.time(),
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )

        with (
            patch(
                "llenergymeasure.harness.baseline.measure_baseline_power",
                return_value=fresh,
            ),
            patch(
                "llenergymeasure.device.gpu_info._resolve_gpu_indices",
                return_value=[0],
            ),
            patch(
                "llenergymeasure.harness.baseline.save_baseline_cache",
            ) as mock_save,
            patch(
                "llenergymeasure.harness.baseline.load_baseline_cache",
                return_value=None,
            ),
        ):
            runner._get_baseline(mock_config)

        check(
            "fresh baseline saved to disk after re-measurement",
            mock_save.call_count == 1,
        )
        # The saved baseline should have a recent timestamp
        saved_baseline = mock_save.call_args[0][1]
        age = time.time() - saved_baseline.timestamp
        check(
            "saved baseline has fresh timestamp",
            age < 5,
            f"age={age:.1f}s",
        )


# ---------------------------------------------------------------------------
# 3. Container loads host-managed cache
# ---------------------------------------------------------------------------


def test_container_ttl_scenarios() -> None:
    """Container correctly accepts/rejects cache based on TTL."""
    section("3. Container-side TTL behaviour")

    from llenergymeasure.harness.baseline import (
        BaselineCache,
        load_baseline_cache,
        save_baseline_cache,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "baseline_cache.json"
        ttl = 3600.0  # 1 hour (new default)

        # Case A: 20-min-old cache -> valid
        recent = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 1200,
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        save_baseline_cache(cache_path, recent)
        loaded = load_baseline_cache(cache_path, ttl=ttl)
        check("20-min-old cache accepted (1h TTL)", loaded is not None)

        # Case B: 50-min-old cache -> valid (within 1h)
        older = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 3000,
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        save_baseline_cache(cache_path, older)
        loaded = load_baseline_cache(cache_path, ttl=ttl)
        check("50-min-old cache accepted (1h TTL)", loaded is not None)

        # Case C: 70-min-old cache -> expired
        expired = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 4200,
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        save_baseline_cache(cache_path, expired)
        loaded = load_baseline_cache(cache_path, ttl=ttl)
        check("70-min-old cache rejected (1h TTL)", loaded is None)


# ---------------------------------------------------------------------------
# 4. Host-to-container flow: TTL stays consistent
# ---------------------------------------------------------------------------


def test_host_to_container_consistency() -> None:
    """When host accepts baseline (within TTL), disk cache is also within TTL
    for the container - because both share the same timestamp."""
    section("4. Host-to-container TTL consistency")

    from llenergymeasure.harness.baseline import (
        BaselineCache,
        load_baseline_cache,
        save_baseline_cache,
    )
    from llenergymeasure.study.runner import StudyRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        study_dir = Path(tmpdir)
        ttl = 3600.0

        runner = StudyRunner.__new__(StudyRunner)
        runner._study_config = MagicMock()
        runner._baseline_cache_path = None
        runner._experiments_since_validation = 0
        runner.study_dir = study_dir

        mock_config = MagicMock()
        mock_config.baseline.strategy = "cached"
        mock_config.baseline.cache_ttl_seconds = ttl
        mock_config.baseline.duration_seconds = 30.0
        mock_config.gpu_indices = [0]

        # Baseline measured 40 min ago (within 1h TTL)
        baseline_40min = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 2400,
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
            method="cached",
        )
        runner._baseline = baseline_40min

        # Host accepts it (within TTL)
        result = runner._get_baseline(mock_config)
        check(
            "host accepts 40-min-old baseline (1h TTL)",
            result is not None and result.power_w == 50.0,
        )

        # Save to disk (simulating what happens at initial measurement)
        cache_path = runner._get_baseline_cache_path()
        save_baseline_cache(cache_path, baseline_40min)

        # Container loads the same file
        container_loaded = load_baseline_cache(cache_path, ttl=ttl)
        check(
            "container also accepts (same timestamp, same TTL)",
            container_loaded is not None,
        )

        # Now test that TTL-expired baseline is caught by host
        baseline_70min = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 4200,  # 70 min, past 1h TTL
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        runner._baseline = baseline_70min

        fresh = BaselineCache(
            power_w=53.0,
            timestamp=time.time(),
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )

        with (
            patch(
                "llenergymeasure.harness.baseline.measure_baseline_power",
                return_value=fresh,
            ) as mock_measure,
            patch(
                "llenergymeasure.device.gpu_info._resolve_gpu_indices",
                return_value=[0],
            ),
            patch("llenergymeasure.harness.baseline.save_baseline_cache") as mock_save,
            patch("llenergymeasure.harness.baseline.load_baseline_cache", return_value=None),
        ):
            result = runner._get_baseline(mock_config)

        check(
            "host rejects 70-min-old baseline, re-measures",
            mock_measure.call_count == 1,
        )
        check(
            "fresh baseline (53W) returned",
            result is not None and result.power_w == 53.0,
        )
        check(
            "fresh baseline saved to disk for containers",
            mock_save.call_count == 1,
        )


# ---------------------------------------------------------------------------
# 5. Validated strategy with TTL
# ---------------------------------------------------------------------------


def test_validated_with_ttl() -> None:
    """Validated strategy also enforces TTL on in-memory baseline."""
    section("5. Validated strategy respects TTL")

    from llenergymeasure.harness.baseline import BaselineCache
    from llenergymeasure.study.runner import StudyRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StudyRunner.__new__(StudyRunner)
        runner._study_config = MagicMock()
        runner._baseline_cache_path = None
        runner.study_dir = Path(tmpdir)

        mock_config = MagicMock()
        mock_config.baseline.strategy = "validated"
        mock_config.baseline.validation_interval = 3
        mock_config.baseline.drift_threshold = 0.10
        mock_config.baseline.cache_ttl_seconds = 3600.0
        mock_config.baseline.duration_seconds = 30.0
        mock_config.gpu_indices = [0]

        # Baseline within TTL
        recent = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 1800,  # 30 min, within 1h TTL
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        runner._baseline = recent
        runner._experiments_since_validation = 2  # next call triggers spot-check

        # Spot-check with no drift
        with (
            patch(
                "llenergymeasure.harness.baseline.measure_spot_check",
                return_value=51.0,  # 2% drift
            ) as mock_spot,
            patch(
                "llenergymeasure.device.gpu_info._resolve_gpu_indices",
                return_value=[0],
            ),
        ):
            result = runner._get_baseline(mock_config)

        check("spot-check triggered", mock_spot.call_count == 1)
        check(
            "baseline kept (within TTL + no drift)",
            result.power_w == 50.0,
        )
        check("method set to validated", result.method == "validated")

        # Now set baseline to expired + validated strategy
        expired = BaselineCache(
            power_w=50.0,
            timestamp=time.time() - 7200,  # 2h, past 1h TTL
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )
        runner._baseline = expired
        runner._experiments_since_validation = 0

        fresh = BaselineCache(
            power_w=55.0,
            timestamp=time.time(),
            gpu_indices=[0],
            sample_count=200,
            duration_sec=30.0,
        )

        with (
            patch(
                "llenergymeasure.harness.baseline.measure_baseline_power",
                return_value=fresh,
            ) as mock_measure,
            patch(
                "llenergymeasure.device.gpu_info._resolve_gpu_indices",
                return_value=[0],
            ),
            patch("llenergymeasure.harness.baseline.save_baseline_cache"),
            patch("llenergymeasure.harness.baseline.load_baseline_cache", return_value=None),
        ):
            result = runner._get_baseline(mock_config)

        check(
            "TTL-expired validated baseline triggers re-measurement",
            mock_measure.call_count == 1,
        )
        check(
            "fresh baseline returned (55W)",
            result is not None and result.power_w == 55.0,
        )


# ---------------------------------------------------------------------------
# 6. Long study simulation with 1h TTL
# ---------------------------------------------------------------------------


def test_long_study_with_new_ttl() -> None:
    """10-experiment study: all experiments within 1h TTL get cached baseline."""
    section("6. Long study: 10 experiments over ~50 min (1h TTL)")

    from llenergymeasure.harness.baseline import (
        BaselineCache,
        load_baseline_cache,
        save_baseline_cache,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "baseline_cache.json"
        ttl = 3600.0  # 1 hour

        results = []
        for i in range(10):
            elapsed_min = (i + 1) * 5  # 5, 10, 15, ... 50 min
            simulated_age = elapsed_min * 60

            # Simulate disk cache with original measurement timestamp
            entry = BaselineCache(
                power_w=50.0,
                timestamp=time.time() - simulated_age,
                gpu_indices=[0],
                sample_count=200,
                duration_sec=30.0,
            )
            save_baseline_cache(cache_path, entry)

            loaded = load_baseline_cache(cache_path, ttl=ttl)
            results.append((elapsed_min, loaded is not None))

        for elapsed_min, valid in results:
            check(
                f"t={elapsed_min}min",
                valid,
                "cache valid" if valid else "EXPIRED",
            )

        all_valid = all(valid for _, valid in results)
        check(
            "all 10 experiments use cached baseline",
            all_valid,
            "no wasted re-measurements",
        )


# ---------------------------------------------------------------------------
# 7. TTL boundary precision
# ---------------------------------------------------------------------------


def test_ttl_boundaries() -> None:
    """Boundary cases for TTL comparison (strict > in load_baseline_cache)."""
    section("7. TTL boundary precision")

    from llenergymeasure.harness.baseline import (
        BaselineCache,
        load_baseline_cache,
        save_baseline_cache,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "baseline_cache.json"
        ttl = 60.0

        # 1s inside TTL
        entry = BaselineCache(
            power_w=100.0,
            timestamp=time.time() - 59.0,
            gpu_indices=[0],
            sample_count=100,
            duration_sec=10.0,
        )
        save_baseline_cache(cache_path, entry)
        check("59s old, 60s TTL -> valid", load_baseline_cache(cache_path, ttl=ttl) is not None)

        # 1s past TTL
        entry2 = BaselineCache(
            power_w=100.0,
            timestamp=time.time() - 61.0,
            gpu_indices=[0],
            sample_count=100,
            duration_sec=10.0,
        )
        save_baseline_cache(cache_path, entry2)
        check("61s old, 60s TTL -> expired", load_baseline_cache(cache_path, ttl=ttl) is None)

        # Sub-second precision
        entry3 = BaselineCache(
            power_w=100.0,
            timestamp=time.time() - 0.5,
            gpu_indices=[0],
            sample_count=100,
            duration_sec=5.0,
        )
        save_baseline_cache(cache_path, entry3)
        check("0.5s old, 1s TTL -> valid", load_baseline_cache(cache_path, ttl=1.0) is not None)


# ---------------------------------------------------------------------------
# 8. Config validation
# ---------------------------------------------------------------------------


def test_config_ttl_validation() -> None:
    """BaselineConfig enforces minimum TTL and valid strategies."""
    section("8. Config model validation")

    from pydantic import ValidationError

    from llenergymeasure.config.models import BaselineConfig

    # New default is 3600s
    cfg = BaselineConfig()
    check("default TTL is 7200s (2 hours)", cfg.cache_ttl_seconds == 7200.0)

    # Minimum TTL enforced
    try:
        BaselineConfig(cache_ttl_seconds=59.0)
        check("TTL < 60s rejected", False)
    except ValidationError:
        check("TTL < 60s rejected", True)

    # All strategies accepted
    for strat in ("cached", "validated", "fresh"):
        c = BaselineConfig(strategy=strat)
        check(f"strategy '{strat}' accepted", c.strategy == strat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("\n" + "=" * 60)
    print("  Host-Managed TTL Verification (PR #242 + #243)")
    print("=" * 60)

    test_host_inmemory_ttl()
    test_remeasurement_refreshes_disk()
    test_container_ttl_scenarios()
    test_host_to_container_consistency()
    test_validated_with_ttl()
    test_long_study_with_new_ttl()
    test_ttl_boundaries()
    test_config_ttl_validation()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print(f"{'=' * 60}\n")

    return 1 if _failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
