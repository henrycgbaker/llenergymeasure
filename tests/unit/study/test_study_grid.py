"""Unit tests for study grid expansion, cycle ordering, hash, and invalid handling.

TDD RED phase: all expand_grid / compute_study_design_hash / apply_cycles tests
must fail until grid.py is implemented. ExecutionConfig and StudyConfig model
tests pass immediately from the models.py changes.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from rich.console import Console

from llenergymeasure.config.grid import (
    CycleOrder,
    SkippedConfig,
    apply_cycles,
    build_preflight_panel,
    compute_study_design_hash,
    expand_grid,
    format_preflight_summary,
)
from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# ExecutionConfig model tests
# =============================================================================


class TestExecutionConfig:
    def test_default_values(self):
        ec = ExecutionConfig()
        assert ec.n_cycles == 1
        assert ec.cycle_order == "sequential"
        assert ec.experiment_gap_seconds is None
        assert ec.cycle_gap_seconds is None
        assert ec.shuffle_seed is None

    def test_n_cycles_zero_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(n_cycles=0)

    def test_n_cycles_negative_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(n_cycles=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(unknown_field=42)

    def test_valid_cycle_orders(self):
        for order in ("sequential", "interleaved", "shuffled"):
            ec = ExecutionConfig(cycle_order=order)
            assert ec.cycle_order == order

    def test_invalid_cycle_order_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(cycle_order="random")

    def test_gap_fields_non_negative(self):
        ec = ExecutionConfig(experiment_gap_seconds=0.0, cycle_gap_seconds=60.5)
        assert ec.experiment_gap_seconds == 0.0
        assert ec.cycle_gap_seconds == 60.5

    def test_gap_fields_negative_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(experiment_gap_seconds=-1.0)

    def test_shuffle_seed_explicit(self):
        ec = ExecutionConfig(shuffle_seed=12345)
        assert ec.shuffle_seed == 12345


# =============================================================================
# StudyConfig model tests
# =============================================================================


class TestStudyConfig:
    def test_accepts_all_fields(self):
        exp = ExperimentConfig(model="gpt2")
        sc = StudyConfig(
            experiments=[exp],
            name="my-study",
            execution=ExecutionConfig(n_cycles=3),
            study_design_hash="abc123def456abcd",
            skipped_configs=[{"raw_config": {}, "reason": "test"}],
        )
        assert sc.name == "my-study"
        assert sc.execution.n_cycles == 3
        assert sc.study_design_hash == "abc123def456abcd"
        assert len(sc.skipped_configs) == 1

    def test_empty_experiments_raises(self):
        with pytest.raises(ValidationError):
            StudyConfig(experiments=[])

    def test_default_execution(self):
        exp = ExperimentConfig(model="gpt2")
        sc = StudyConfig(experiments=[exp])
        assert sc.execution.n_cycles == 1
        assert sc.study_design_hash is None
        assert sc.skipped_configs == []

    def test_extra_fields_forbidden(self):
        exp = ExperimentConfig(model="gpt2")
        with pytest.raises(ValidationError):
            StudyConfig(experiments=[exp], unknown_field="x")


# =============================================================================
# expand_grid() — grid sweep mode
# =============================================================================


class TestExpandGridSweep:
    def test_universal_sweep_cartesian_product(self):
        """2 precisions x 2 n values = 4 configs."""
        raw = {
            "model": "gpt2",
            "backend": "pytorch",
            "sweep": {
                "precision": ["fp16", "bf16"],
                "n": [50, 100],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 4
        assert len(skipped) == 0
        precisions = {c.precision for c in valid}
        ns = {c.n for c in valid}
        assert precisions == {"fp16", "bf16"}
        assert ns == {50, 100}

    def test_backend_scoped_sweep_routes_to_section(self):
        """pytorch.batch_size routes to the pytorch section, not top-level."""
        raw = {
            "model": "gpt2",
            "backend": "pytorch",
            "sweep": {
                "pytorch.batch_size": [1, 8],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert len(skipped) == 0
        batch_sizes = {c.pytorch.batch_size for c in valid}
        assert batch_sizes == {1, 8}

    def test_multi_backend_scoped_sweep(self):
        """Multi-backend with scoped keys: independent grids per backend."""
        raw = {
            "model": "gpt2",
            "backend": ["pytorch", "vllm"],
            "sweep": {
                "precision": ["fp16", "bf16"],
                "pytorch.batch_size": [1, 8],
                "vllm.engine.max_num_seqs": [64, 256],
            },
        }
        valid, skipped = expand_grid(raw)
        # pytorch: 2 precisions x 2 batch_sizes = 4
        # vllm: 2 precisions x 2 max_num_seqs = 4
        # total = 8
        assert len(valid) == 8
        assert len(skipped) == 0
        pytorch_configs = [c for c in valid if c.backend == "pytorch"]
        vllm_configs = [c for c in valid if c.backend == "vllm"]
        assert len(pytorch_configs) == 4
        assert len(vllm_configs) == 4
        # pytorch configs must not have vllm section and vice versa
        for c in pytorch_configs:
            assert c.vllm is None
        for c in vllm_configs:
            assert c.pytorch is None
            assert c.vllm.engine.max_num_seqs in (64, 256)


# =============================================================================
# expand_grid() — explicit experiments mode
# =============================================================================


class TestExpandGridExplicit:
    def test_explicit_experiments_list(self):
        raw = {
            "experiments": [
                {"model": "gpt2", "backend": "pytorch"},
                {"model": "gpt2", "backend": "vllm"},
            ]
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2
        assert valid[0].backend == "pytorch"
        assert valid[1].backend == "vllm"


# =============================================================================
# expand_grid() — combined mode
# =============================================================================


class TestExpandGridCombined:
    def test_sweep_plus_explicit(self):
        """Sweep configs come first, then explicit entries appended."""
        raw = {
            "model": "gpt2",
            "backend": "pytorch",
            "sweep": {
                "precision": ["fp16", "bf16"],
            },
            "experiments": [
                {"model": "gpt2-xl", "backend": "pytorch"},
            ],
        }
        valid, _skipped = expand_grid(raw)
        # 2 sweep + 1 explicit = 3
        assert len(valid) == 3
        # Sweep configs first
        sweep_configs = valid[:2]
        explicit_config = valid[2]
        assert {c.precision for c in sweep_configs} == {"fp16", "bf16"}
        assert explicit_config.model == "gpt2-xl"


# =============================================================================
# expand_grid() — base: resolution
# =============================================================================


class TestExpandGridBase:
    def test_base_loads_relative_to_study_yaml(self, tmp_path: Path):
        base_config = {
            "model": "gpt2",
            "backend": "pytorch",
            "n": 50,
        }
        base_file = tmp_path / "base_experiment.yaml"
        base_file.write_text(yaml.dump(base_config))

        raw = {
            "base": "base_experiment.yaml",
            "sweep": {
                "precision": ["fp16", "bf16"],
            },
        }
        study_yaml = tmp_path / "study.yaml"
        valid, _skipped = expand_grid(raw, study_yaml_path=study_yaml)
        assert len(valid) == 2
        for c in valid:
            assert c.model == "gpt2"
            assert c.n == 50

    def test_base_strips_study_only_keys(self, tmp_path: Path):
        """Study-only keys in base file are stripped before merging."""
        base_config = {
            "model": "gpt2",
            "backend": "pytorch",
            # These should be stripped
            "sweep": {"precision": ["fp32"]},
            "experiments": [{"model": "other"}],
            "execution": {"n_cycles": 5},
            "base": "another.yaml",
            "name": "should-be-stripped",
        }
        base_file = tmp_path / "base_experiment.yaml"
        base_file.write_text(yaml.dump(base_config))

        raw = {
            "base": "base_experiment.yaml",
            "sweep": {
                "precision": ["fp16"],
            },
        }
        study_yaml = tmp_path / "study.yaml"
        valid, _skipped = expand_grid(raw, study_yaml_path=study_yaml)
        assert len(valid) == 1
        assert valid[0].precision == "fp16"
        assert valid[0].model == "gpt2"

    def test_missing_base_file_raises(self, tmp_path: Path):
        raw = {"base": "nonexistent.yaml", "sweep": {"precision": ["fp16"]}}
        study_yaml = tmp_path / "study.yaml"
        with pytest.raises(ConfigError, match="base"):
            expand_grid(raw, study_yaml_path=study_yaml)


# =============================================================================
# expand_grid() — invalid combination handling
# =============================================================================


class TestExpandGridInvalidHandling:
    def test_invalid_configs_collected_as_skipped(self):
        """Invalid configs become SkippedConfig, valid ones are returned."""
        raw = {
            "model": "gpt2",
            "sweep": {
                # fp32 with tensorrt backend is valid, but precision fp32 is accepted
                # Use a truly invalid combo: backend=vllm but pytorch section provided via explicit
                "backend": ["pytorch", "vllm"],
                "precision": ["fp16"],
            },
            "experiments": [
                # This will fail: vllm section + backend=pytorch
                {"model": "gpt2", "backend": "pytorch", "vllm": {"max_num_seqs": 64}},
            ],
        }
        valid, skipped = expand_grid(raw)
        # The two sweep configs are valid; the explicit one fails cross-validation
        assert len(valid) == 2
        assert len(skipped) == 1
        assert "vllm" in skipped[0].reason.lower() or "backend" in skipped[0].reason.lower()

    def test_all_invalid_raises_config_error(self):
        """All invalid configs raises ConfigError with count and reasons."""
        raw = {
            "experiments": [
                # Invalid: pytorch section with vllm backend
                {"model": "gpt2", "backend": "vllm", "pytorch": {"batch_size": 4}},
                # Invalid: vllm section with pytorch backend
                {"model": "gpt2", "backend": "pytorch", "vllm": {"max_num_seqs": 64}},
            ]
        }
        with pytest.raises(ConfigError, match=r"nothing to run|all.*invalid|0.*valid"):
            expand_grid(raw)

    def test_skipped_config_short_label(self):
        sc = SkippedConfig(
            raw_config={"backend": "pytorch", "precision": "fp32"},
            reason="some validation error",
        )
        assert sc.short_label == "pytorch, fp32"

    def test_skipped_config_to_dict(self):
        sc = SkippedConfig(
            raw_config={"backend": "vllm", "precision": "fp16"},
            reason="cross-validation error",
            errors=[{"loc": ["backend"], "msg": "test"}],
        )
        d = sc.to_dict()
        assert d["raw_config"] == {"backend": "vllm", "precision": "fp16"}
        assert d["reason"] == "cross-validation error"
        assert d["short_label"] == "vllm, fp16"
        assert len(d["errors"]) == 1

    def test_no_experiments_raises_config_error(self):
        """A sweep with no model and no experiments raises ConfigError."""
        raw = {"name": "empty-study"}
        with pytest.raises(ConfigError):
            expand_grid(raw)


class TestMultiBackendSectionStripping:
    """Top-level backend sections are stripped for non-matching backends in multi-backend studies."""

    def test_sweep_strips_inherited_backend_sections(self):
        """A top-level tensorrt: section must not leak into pytorch/vllm sweep configs."""
        raw = {
            "model": "gpt2",
            "tensorrt": {"max_input_len": 1024},
            "sweep": {
                "precision": ["bf16"],
                "pytorch.batch_size": [1],
                "tensorrt.max_batch_size": [4],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0, f"Expected 0 skipped, got: {[s.reason for s in skipped]}"
        # One pytorch config, one tensorrt config
        pytorch_configs = [c for c in valid if c.backend == "pytorch"]
        tensorrt_configs = [c for c in valid if c.backend == "tensorrt"]
        assert len(pytorch_configs) == 1
        assert len(tensorrt_configs) == 1
        # Pytorch config must NOT have tensorrt section
        assert pytorch_configs[0].tensorrt is None
        # Tensorrt config inherits the top-level tensorrt section
        assert tensorrt_configs[0].tensorrt is not None
        assert tensorrt_configs[0].tensorrt.max_input_len == 1024

    def test_explicit_experiment_strips_inherited_not_explicit(self):
        """Inherited backend sections are stripped; explicitly written ones still fail."""
        raw = {
            "tensorrt": {"max_input_len": 1024},
            "experiments": [
                # Inherited tensorrt: should be stripped for this pytorch experiment
                {"model": "gpt2", "backend": "pytorch"},
                # Explicit vllm: section with backend=pytorch is a user error — should fail
                {
                    "model": "gpt2",
                    "backend": "pytorch",
                    "vllm": {"engine": {"max_num_seqs": 64}},
                },
            ],
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 1
        assert valid[0].backend == "pytorch"
        assert valid[0].tensorrt is None
        assert len(skipped) == 1
        assert "vllm" in skipped[0].reason.lower()

    def test_sweep_with_all_three_backends(self):
        """Three-backend sweep with a shared tensorrt section produces valid configs for all."""
        raw = {
            "model": "gpt2",
            "tensorrt": {"max_input_len": 512},
            "sweep": {
                "pytorch.batch_size": [1],
                "vllm.engine.max_num_seqs": [64],
                "tensorrt.max_batch_size": [4],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0, f"Unexpected skips: {[s.reason for s in skipped]}"
        backends = sorted(c.backend for c in valid)
        assert backends == ["pytorch", "tensorrt", "vllm"]


# =============================================================================
# compute_study_design_hash() tests
# =============================================================================


class TestComputeStudyDesignHash:
    def test_returns_16_char_hex(self):
        experiments = [ExperimentConfig(model="gpt2")]
        h = compute_study_design_hash(experiments)
        assert len(h) == 16
        int(h, 16)  # must be valid hex

    def test_same_experiments_same_hash(self):
        exps1 = [ExperimentConfig(model="gpt2"), ExperimentConfig(model="gpt2", n=50)]
        exps2 = [ExperimentConfig(model="gpt2"), ExperimentConfig(model="gpt2", n=50)]
        assert compute_study_design_hash(exps1) == compute_study_design_hash(exps2)

    def test_different_experiments_different_hash(self):
        exps1 = [ExperimentConfig(model="gpt2")]
        exps2 = [ExperimentConfig(model="gpt2", n=50)]
        assert compute_study_design_hash(exps1) != compute_study_design_hash(exps2)

    def test_stable_across_calls(self):
        experiments = [ExperimentConfig(model="gpt2"), ExperimentConfig(model="gpt2-xl")]
        h1 = compute_study_design_hash(experiments)
        h2 = compute_study_design_hash(experiments)
        assert h1 == h2

    def test_hash_excludes_order_sensitivity(self):
        """Same experiments in same order produce same hash (order matters for reproducibility)."""
        exps_a = [ExperimentConfig(model="gpt2"), ExperimentConfig(model="gpt2", precision="fp16")]
        exps_b = [ExperimentConfig(model="gpt2"), ExperimentConfig(model="gpt2", precision="fp16")]
        assert compute_study_design_hash(exps_a) == compute_study_design_hash(exps_b)


# =============================================================================
# apply_cycles() tests
# =============================================================================


class TestApplyCycles:
    @pytest.fixture
    def two_experiments(self):
        a = ExperimentConfig(model="gpt2")
        b = ExperimentConfig(model="gpt2", n=50)
        return [a, b]

    @pytest.fixture
    def study_hash(self, two_experiments):
        return compute_study_design_hash(two_experiments)

    def test_sequential_ordering(self, two_experiments, study_hash):
        """sequential with 3 cycles and [A, B] -> [A, A, A, B, B, B]."""
        result = apply_cycles(two_experiments, 3, CycleOrder.SEQUENTIAL, study_hash)
        assert len(result) == 6
        # First 3 should be A (gpt2, n=100)
        assert all(r.n == 100 for r in result[:3])
        # Last 3 should be B (gpt2, n=50)
        assert all(r.n == 50 for r in result[3:])

    def test_interleaved_ordering(self, two_experiments, study_hash):
        """interleaved with 3 cycles and [A, B] -> [A, B, A, B, A, B]."""
        result = apply_cycles(two_experiments, 3, CycleOrder.INTERLEAVED, study_hash)
        assert len(result) == 6
        # Alternating: A, B, A, B, A, B
        for i in range(0, 6, 2):
            assert result[i].n == 100  # A
        for i in range(1, 6, 2):
            assert result[i].n == 50  # B

    def test_shuffled_with_explicit_seed_deterministic(self, two_experiments, study_hash):
        """Shuffled with explicit seed produces deterministic reproducible order."""
        result1 = apply_cycles(two_experiments, 3, CycleOrder.SHUFFLED, study_hash, shuffle_seed=42)
        result2 = apply_cycles(two_experiments, 3, CycleOrder.SHUFFLED, study_hash, shuffle_seed=42)
        assert [r.n for r in result1] == [r.n for r in result2]

    def test_shuffled_with_same_hash_same_order(self, two_experiments, study_hash):
        """Same study_design_hash without explicit seed = same shuffle."""
        result1 = apply_cycles(two_experiments, 3, CycleOrder.SHUFFLED, study_hash)
        result2 = apply_cycles(two_experiments, 3, CycleOrder.SHUFFLED, study_hash)
        assert [r.n for r in result1] == [r.n for r in result2]

    def test_shuffled_different_seeds_different_orders(self, study_hash):
        """Seeds 1 and 999 produce different orderings (verified deterministic)."""
        exps = [ExperimentConfig(model="gpt2", n=i) for i in range(1, 6)]
        result1 = apply_cycles(exps, 2, CycleOrder.SHUFFLED, study_hash, shuffle_seed=1)
        result2 = apply_cycles(exps, 2, CycleOrder.SHUFFLED, study_hash, shuffle_seed=999)
        # Seeds 1 and 999 confirmed to produce distinct orderings for 5 experiments x 2 cycles
        # (seed 1 → [3,4,5,1,2,1,3,2,5,4], seed 999 → [3,5,2,4,1,2,4,5,1,3])
        assert [r.n for r in result1] != [r.n for r in result2]

    def test_n_cycles_one_unchanged(self, two_experiments, study_hash):
        """n_cycles=1 returns the original list unchanged."""
        result = apply_cycles(two_experiments, 1, CycleOrder.SEQUENTIAL, study_hash)
        assert len(result) == 2
        assert result[0].n == two_experiments[0].n
        assert result[1].n == two_experiments[1].n

    def test_shuffled_contains_all_experiments_each_cycle(self, two_experiments, study_hash):
        """Each cycle in shuffled mode contains all experiments exactly once."""
        result = apply_cycles(two_experiments, 3, CycleOrder.SHUFFLED, study_hash)
        assert len(result) == 6
        # Check that each pair of 2 contains both experiments
        for i in range(0, 6, 2):
            pair_ns = {result[i].n, result[i + 1].n}
            assert pair_ns == {100, 50}


# =============================================================================
# format_preflight_summary() tests
# =============================================================================


def _make_study_config(
    n_configs: int = 4,
    n_cycles: int = 3,
    cycle_order: str = "interleaved",
    study_hash: str = "abc123def456abcd",
    skipped_configs: list | None = None,
) -> StudyConfig:
    """Helper: build a StudyConfig with the given parameters."""
    experiments = [ExperimentConfig(model="gpt2", n=i + 1) for i in range(n_configs * n_cycles)]
    return StudyConfig(
        experiments=experiments,
        execution=ExecutionConfig(n_cycles=n_cycles, cycle_order=cycle_order),
        study_design_hash=study_hash,
        skipped_configs=skipped_configs or [],
    )


class TestFormatPreflightSummary:
    def test_basic_format(self):
        """Pre-flight string shows config count, cycle count, total runs, order."""
        sc = _make_study_config(n_configs=4, n_cycles=3, cycle_order="interleaved")
        summary = format_preflight_summary(sc)
        assert "4 configs x 3 cycles = 12 runs" in summary
        assert "Order: interleaved" in summary

    def test_hash_displayed(self):
        """study_design_hash appears in the summary line."""
        sc = _make_study_config(study_hash="deadbeef01234567")
        summary = format_preflight_summary(sc)
        assert "deadbeef01234567" in summary

    def test_no_skipped_no_warning(self):
        """Summary with no skipped configs has no Skipping or WARNING lines."""
        sc = _make_study_config()
        summary = format_preflight_summary(sc)
        assert "Skipping" not in summary
        assert "WARNING" not in summary

    def test_with_skipped_shows_skip_line(self):
        """When skipped_configs populated, Skipping line with reasons appears."""
        skipped = [
            {
                "raw_config": {"backend": "pytorch", "precision": "fp32"},
                "reason": "cross-validation failed",
                "short_label": "pytorch, fp32",
                "errors": [],
            }
        ]
        sc = _make_study_config(n_configs=3, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "Skipping 1/" in summary
        assert "pytorch, fp32" in summary
        assert "cross-validation failed" in summary

    def test_high_skip_rate_warning(self):
        """WARNING shown when >50% of generated configs were skipped."""
        # 1 valid config, 2 skipped → 67% skip rate
        skipped = [
            {
                "raw_config": {"backend": "vllm", "precision": "fp16"},
                "reason": "error A",
                "short_label": "vllm, fp16",
                "errors": [],
            },
            {
                "raw_config": {"backend": "vllm", "precision": "bf16"},
                "reason": "error B",
                "short_label": "vllm, bf16",
                "errors": [],
            },
        ]
        sc = _make_study_config(n_configs=1, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "WARNING" in summary
        assert "67%" in summary or "sweep" in summary.lower()

    def test_low_skip_rate_no_warning(self):
        """No WARNING when <50% of generated configs were skipped."""
        # 4 valid, 1 skipped → 20% skip rate
        skipped = [
            {
                "raw_config": {"backend": "pytorch", "precision": "fp32"},
                "reason": "validation error",
                "short_label": "pytorch, fp32",
                "errors": [],
            }
        ]
        sc = _make_study_config(n_configs=4, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "Skipping 1/" in summary
        assert "WARNING" not in summary

    def test_skipped_list_argument_takes_precedence(self):
        """If skipped list of SkippedConfig passed, uses it instead of skipped_configs."""
        skipped_obj = SkippedConfig(
            raw_config={"backend": "pytorch", "precision": "fp16"},
            reason="via argument",
        )
        # StudyConfig has empty skipped_configs
        sc = _make_study_config(n_configs=2, n_cycles=1)
        summary = format_preflight_summary(sc, skipped=[skipped_obj])
        assert "via argument" in summary

    def test_single_cycle_format(self):
        """1 config x 1 cycle = 1 run shown correctly."""
        sc = _make_study_config(n_configs=1, n_cycles=1, cycle_order="sequential")
        summary = format_preflight_summary(sc)
        assert "1 configs x 1 cycles = 1 runs" in summary
        assert "Order: sequential" in summary


# =============================================================================
# build_preflight_panel() tests
# =============================================================================


def _render_panel(study_config: StudyConfig, width: int = 100) -> str:
    """Helper: render a build_preflight_panel() output to a plain-text string."""
    panel = build_preflight_panel(study_config)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=width)
    console.print(panel)
    return buf.getvalue()


def _make_panel_study_config(
    models: list[str] | None = None,
    backends: list[str] | None = None,
    precisions: list[str] | None = None,
    n_cycles: int = 1,
    cycle_order: str = "sequential",
    study_name: str = "test-study",
    study_hash: str = "abc123def456abcd",
    runners: dict | None = None,
) -> StudyConfig:
    """Build a StudyConfig for panel tests with varying fields per experiment."""
    models = models or ["gpt2"]
    backends = backends or ["pytorch"]
    precisions = precisions or ["bf16"]

    # Build one experiment per combination (then replicate for cycles)
    experiments = []
    for model in models:
        for backend in backends:
            for precision in precisions:
                experiments.append(
                    ExperimentConfig(model=model, backend=backend, precision=precision)
                )

    # Replicate for cycles
    all_exps = experiments * n_cycles
    return StudyConfig(
        experiments=all_exps,
        name=study_name,
        execution=ExecutionConfig(n_cycles=n_cycles, cycle_order=cycle_order),
        study_design_hash=study_hash,
        runners=runners,
    )


class TestBuildPreflightPanel:
    def test_panel_contains_study_name_in_title(self):
        """Panel border title contains the study name."""
        sc = _make_panel_study_config(study_name="test-study")
        output = _render_panel(sc)
        assert "Study: test-study" in output

    def test_panel_metadata_experiments_plural(self):
        """Panel shows n configs x n cycles = n runs (plural form)."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2-xl"], n_cycles=3, cycle_order="interleaved"
        )
        output = _render_panel(sc)
        assert "2 configs x 3 cycles = 6 runs" in output

    def test_panel_metadata_experiments_singular(self):
        """Panel shows 1 config x 1 cycle = 1 run (singular form)."""
        sc = _make_panel_study_config(models=["gpt2"], n_cycles=1, cycle_order="sequential")
        output = _render_panel(sc)
        assert "1 config x 1 cycle = 1 run" in output

    def test_panel_pluralisation_singular(self):
        """1 config x 1 cycle = 1 run (all singular)."""
        sc = _make_panel_study_config(models=["gpt2"], n_cycles=1)
        output = _render_panel(sc)
        assert "1 config x 1 cycle = 1 run" in output

    def test_panel_pluralisation_plural(self):
        """2 configs x 3 cycles = 6 runs (all plural)."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2-xl"], n_cycles=3, cycle_order="sequential"
        )
        output = _render_panel(sc)
        assert "2 configs x 3 cycles = 6 runs" in output

    def test_panel_metadata_order(self):
        """Panel shows cycle order in Order row."""
        sc = _make_panel_study_config(n_cycles=2, cycle_order="interleaved")
        output = _render_panel(sc)
        assert "interleaved" in output

    def test_panel_metadata_backends_with_runners(self):
        """Panel shows backend with runner mode when runners dict is provided."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2"],
            backends=["pytorch", "vllm"],
            runners={"pytorch": "local", "vllm": "docker"},
        )
        output = _render_panel(sc)
        assert "pytorch (local)" in output
        assert "vllm (docker)" in output

    def test_panel_metadata_backends_default_local(self):
        """Panel shows (local) for backends when runners is None."""
        sc = _make_panel_study_config(backends=["pytorch"])
        output = _render_panel(sc)
        assert "pytorch (local)" in output

    def test_panel_metadata_dataset(self):
        """Panel shows dataset name in Dataset row."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "aienergyscore" in output

    def test_panel_metadata_energy(self):
        """Panel shows energy backend in Energy row."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "auto" in output

    def test_panel_sweep_dimensions_model(self):
        """Sweep dimensions section contains model names when multiple models used."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"])
        output = _render_panel(sc)
        assert "gpt2" in output
        assert "gpt2-xl" in output

    def test_panel_sweep_dimensions_nested_decoder(self):
        """Non-default decoder config shows decoder header and sub-fields."""
        from llenergymeasure.config.models import DecoderConfig

        exp1 = ExperimentConfig(
            model="gpt2", decoder=DecoderConfig(temperature=0.0, do_sample=False)
        )
        exp2 = ExperimentConfig(
            model="gpt2", decoder=DecoderConfig(temperature=0.0, do_sample=False)
        )
        sc = StudyConfig(
            experiments=[exp1, exp2],
            name="decoder-test",
            execution=ExecutionConfig(n_cycles=1, cycle_order="sequential"),
            study_design_hash="deadbeef01234567",
        )
        output = _render_panel(sc)
        # decoder sub-config header should appear since temperature and do_sample differ from defaults
        assert "decoder" in output
        # temperature=0.0 is non-default (default is 1.0)
        assert "temperature" in output

    def test_panel_hash_displayed(self):
        """Panel contains the study design hash."""
        sc = _make_panel_study_config(study_hash="deadbeef01234567")
        output = _render_panel(sc)
        assert "deadbeef01234567" in output

    def test_panel_unnamed_study(self):
        """Panel with no study name shows 'unnamed' in title."""
        exps = [ExperimentConfig(model="gpt2")]
        sc = StudyConfig(
            experiments=exps,
            execution=ExecutionConfig(n_cycles=1, cycle_order="sequential"),
        )
        output = _render_panel(sc)
        assert "Study: unnamed" in output

    def test_panel_multiple_backends_sorted(self):
        """Multiple backends are sorted alphabetically."""
        exps = [
            ExperimentConfig(model="gpt2", backend="vllm"),
            ExperimentConfig(model="gpt2", backend="pytorch"),
        ]
        sc = StudyConfig(
            experiments=exps,
            execution=ExecutionConfig(n_cycles=1, cycle_order="sequential"),
        )
        output = _render_panel(sc)
        # Both backends appear
        assert "pytorch" in output
        assert "vllm" in output
