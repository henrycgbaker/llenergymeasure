"""Unit tests for the config YAML loader (llenergymeasure.config.loader).

Tests YAML loading, ConfigError vs ValidationError boundary, CLI override
merging, and the deep_merge utility.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llenergymeasure.config.loader import deep_merge, load_experiment_config, load_study_config
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str, name: str = "config.yaml") -> Path:
    """Write content to a YAML file in tmp_path and return the path."""
    path = tmp_path / name
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Load valid YAML
# ---------------------------------------------------------------------------


def test_load_valid_yaml(tmp_path):
    """Minimal valid YAML loads successfully into ExperimentConfig."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: pytorch\n")
    config = load_experiment_config(path)
    assert isinstance(config, ExperimentConfig)
    assert config.model == "gpt2"
    assert config.backend == "pytorch"


def test_load_yaml_with_precision(tmp_path):
    """YAML with optional fields loads correctly."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: pytorch\nprecision: fp16\n")
    config = load_experiment_config(path)
    assert config.precision == "fp16"


def test_load_yaml_with_pytorch_section(tmp_path):
    """YAML with nested pytorch: section loads correctly."""
    yaml_content = "model: gpt2\nbackend: pytorch\npytorch:\n  batch_size: 4\n"
    path = _write_yaml(tmp_path, yaml_content)
    config = load_experiment_config(path)
    assert config.pytorch is not None
    assert config.pytorch.batch_size == 4


def test_load_yaml_with_version_field_stripped(tmp_path):
    """Optional 'version' field in YAML is stripped before validation (not ExperimentConfig field)."""
    path = _write_yaml(tmp_path, "version: 2.0\nmodel: gpt2\nbackend: pytorch\n")
    config = load_experiment_config(path)
    assert config.model == "gpt2"


# ---------------------------------------------------------------------------
# Error boundary: ConfigError for unknown keys
# ---------------------------------------------------------------------------


def test_load_unknown_yaml_key_raises_config_error(tmp_path):
    """YAML with unknown top-level key raises ConfigError (not ValidationError)."""
    path = _write_yaml(tmp_path, "model: gpt2\nthis_key_does_not_exist: bad\n")
    with pytest.raises(ConfigError, match="Unknown field"):
        load_experiment_config(path)


def test_load_unknown_key_includes_did_you_mean(tmp_path):
    """ConfigError for a close typo includes a 'did you mean' suggestion."""
    # 'modul' is close enough to 'model' to get a suggestion
    path = _write_yaml(tmp_path, "modul: gpt2\n")
    with pytest.raises(ConfigError, match="did you mean"):
        load_experiment_config(path)


def test_load_nonexistent_file_raises_config_error(tmp_path):
    """Loading a path that doesn't exist raises ConfigError."""
    path = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_experiment_config(path)


# ---------------------------------------------------------------------------
# Error boundary: ValidationError passes through unchanged
# ---------------------------------------------------------------------------


def test_pydantic_validation_error_passes_through(tmp_path):
    """YAML with valid keys but invalid values raises ValidationError (not ConfigError).

    n=-1 is a valid field name but has an invalid value (ge=1 constraint).
    """
    path = _write_yaml(tmp_path, "model: gpt2\nn: -1\n")
    with pytest.raises(ValidationError):
        load_experiment_config(path)


def test_pydantic_validation_error_not_wrapped_in_config_error(tmp_path):
    """ValidationError from field value failures is not wrapped in ConfigError."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: totally_invalid_backend\n")
    # backend is a valid field name, but the value is invalid → ValidationError
    with pytest.raises(ValidationError):
        load_experiment_config(path)


# ---------------------------------------------------------------------------
# CLI override merging
# ---------------------------------------------------------------------------


def test_cli_overrides_merged(tmp_path):
    """cli_overrides override YAML values at highest priority."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: pytorch\n")
    config = load_experiment_config(path, cli_overrides={"model": "bert-base"})
    assert config.model == "bert-base"


def test_cli_overrides_none_values_ignored(tmp_path):
    """None values in cli_overrides are ignored (unset CLI flags)."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: pytorch\n")
    config = load_experiment_config(path, cli_overrides={"model": None, "precision": None})
    assert config.model == "gpt2"  # file value retained


def test_cli_overrides_without_file():
    """load_experiment_config with cli_overrides and no file works."""
    config = load_experiment_config(
        path=None,
        cli_overrides={"model": "gpt2", "backend": "pytorch"},
    )
    assert config.model == "gpt2"
    assert config.backend == "pytorch"


def test_cli_override_dotted_key(tmp_path):
    """Dotted CLI override keys are unflattened into nested dicts."""
    path = _write_yaml(tmp_path, "model: gpt2\nbackend: pytorch\n")
    config = load_experiment_config(
        path,
        cli_overrides={"pytorch.batch_size": 8},
    )
    assert config.pytorch is not None
    assert config.pytorch.batch_size == 8


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_overlay_wins():
    """deep_merge: overlay value wins over base value for same key."""
    result = deep_merge({"a": 1}, {"a": 2})
    assert result == {"a": 2}


def test_deep_merge_base_preserved_for_unshared_keys():
    """deep_merge: base keys not in overlay are preserved."""
    result = deep_merge({"a": 1, "b": 2}, {"a": 99})
    assert result == {"a": 99, "b": 2}


def test_deep_merge_nested():
    """deep_merge: nested dicts are merged (not replaced)."""
    result = deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    assert result == {"a": {"b": 1, "c": 2}}


def test_deep_merge_nested_overlay_wins():
    """deep_merge: nested overlay value wins over nested base value."""
    result = deep_merge({"a": {"b": 1}}, {"a": {"b": 99}})
    assert result == {"a": {"b": 99}}


def test_deep_merge_does_not_mutate_inputs():
    """deep_merge returns a new dict — originals are unchanged."""
    base = {"a": {"b": 1}}
    overlay = {"a": {"c": 2}}
    _ = deep_merge(base, overlay)
    assert base == {"a": {"b": 1}}
    assert overlay == {"a": {"c": 2}}


def test_deep_merge_empty_base():
    """deep_merge with empty base returns a copy of overlay."""
    result = deep_merge({}, {"a": 1})
    assert result == {"a": 1}


def test_deep_merge_empty_overlay():
    """deep_merge with empty overlay returns a copy of base."""
    result = deep_merge({"a": 1}, {})
    assert result == {"a": 1}


def test_deep_merge_non_dict_overlay_replaces_nested():
    """deep_merge: non-dict overlay replaces nested dict in base."""
    result = deep_merge({"a": {"b": 1}}, {"a": "scalar"})
    assert result == {"a": "scalar"}


# ---------------------------------------------------------------------------
# load_study_config() — grid sweep mode
# ---------------------------------------------------------------------------


def test_load_study_config_grid_sweep(tmp_path):
    """load_study_config() with sweep block returns StudyConfig with correct experiment count."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "model": "gpt2",
                "backend": "pytorch",
                "sweep": {
                    "precision": ["fp16", "bf16"],
                    "n": [50, 100],
                },
            }
        )
    )
    sc = load_study_config(study_yaml)
    assert isinstance(sc, StudyConfig)
    # 2 precisions x 2 n values = 4 configs, default 1 cycle
    assert len(sc.experiments) == 4
    assert sc.study_design_hash is not None
    assert len(sc.study_design_hash) == 16
    # 16-char hex
    int(sc.study_design_hash, 16)
    # execution defaults applied
    assert sc.execution.n_cycles == 1
    assert sc.execution.cycle_order == "sequential"


def test_load_study_config_explicit_experiments(tmp_path):
    """load_study_config() with explicit experiments: list returns correct count."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "experiments": [
                    {"model": "gpt2", "backend": "pytorch"},
                    {"model": "gpt2", "backend": "pytorch", "precision": "fp16"},
                    {"model": "gpt2", "backend": "pytorch", "precision": "bf16"},
                ],
            }
        )
    )
    sc = load_study_config(study_yaml)
    assert len(sc.experiments) == 3


def test_load_study_config_combined_mode(tmp_path):
    """load_study_config() with both sweep and experiments returns combined count."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "model": "gpt2",
                "backend": "pytorch",
                "sweep": {
                    "precision": ["fp16", "bf16"],
                },
                "experiments": [
                    {"model": "gpt2-xl", "backend": "pytorch"},
                ],
            }
        )
    )
    sc = load_study_config(study_yaml)
    # 2 sweep + 1 explicit = 3
    assert len(sc.experiments) == 3


def test_load_study_config_with_execution_block(tmp_path):
    """load_study_config() with execution block applies n_cycles and cycle_order."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "model": "gpt2",
                "backend": "pytorch",
                "sweep": {
                    "precision": ["fp16", "bf16"],
                },
                "execution": {
                    "n_cycles": 3,
                    "cycle_order": "interleaved",
                },
            }
        )
    )
    sc = load_study_config(study_yaml)
    assert sc.execution.n_cycles == 3
    assert sc.execution.cycle_order == "interleaved"
    # 2 base configs x 3 cycles = 6 runs
    assert len(sc.experiments) == 6


def test_load_study_config_cli_overrides(tmp_path):
    """CLI overrides on execution block are applied correctly."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "model": "gpt2",
                "backend": "pytorch",
                "sweep": {"precision": ["fp16", "bf16"]},
                "execution": {"n_cycles": 1},
            }
        )
    )
    sc = load_study_config(study_yaml, cli_overrides={"execution": {"n_cycles": 5}})
    assert sc.execution.n_cycles == 5
    # 2 configs x 5 cycles = 10
    assert len(sc.experiments) == 10


def test_load_study_config_with_base(tmp_path):
    """load_study_config() with base: inherits experiment fields from base file."""
    base_yaml = tmp_path / "experiment.yaml"
    base_yaml.write_text(
        yaml.dump(
            {
                "model": "gpt2",
                "backend": "pytorch",
                "n": 75,
            }
        )
    )
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "base": "experiment.yaml",
                "sweep": {
                    "precision": ["fp16", "bf16"],
                },
            }
        )
    )
    sc = load_study_config(study_yaml)
    assert len(sc.experiments) == 2
    # All experiments should inherit n=75 from base
    for exp in sc.experiments:
        assert exp.model == "gpt2"
        assert exp.n == 75


def test_load_study_config_empty_study_raises(tmp_path):
    """load_study_config() with no model/sweep/experiments raises ConfigError."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.dump(
            {
                "name": "empty-study",
                "execution": {"n_cycles": 1},
            }
        )
    )
    with pytest.raises(ConfigError):
        load_study_config(study_yaml)


def test_load_study_config_all_invalid_raises(tmp_path):
    """load_study_config() with all invalid configs raises ConfigError."""
    study_yaml = tmp_path / "study.yaml"
    # Use explicit experiments that all fail cross-validation
    study_yaml.write_text(
        yaml.dump(
            {
                "experiments": [
                    # Invalid: pytorch section with vllm backend
                    {"model": "gpt2", "backend": "vllm", "pytorch": {"batch_size": 4}},
                    # Invalid: vllm section with pytorch backend
                    {"model": "gpt2", "backend": "pytorch", "vllm": {"max_num_seqs": 64}},
                ],
            }
        )
    )
    with pytest.raises(ConfigError):
        load_study_config(study_yaml)


def test_load_study_config_file_not_found(tmp_path):
    """load_study_config() with non-existent path raises ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_study_config(tmp_path / "nonexistent.yaml")


def test_load_study_config_hash_excludes_execution(tmp_path):
    """study_design_hash is identical when only execution block differs."""
    study_yaml_a = tmp_path / "study_a.yaml"
    study_yaml_b = tmp_path / "study_b.yaml"
    sweep_content = {
        "model": "gpt2",
        "backend": "pytorch",
        "sweep": {"precision": ["fp16", "bf16"]},
    }
    study_yaml_a.write_text(yaml.dump({**sweep_content, "execution": {"n_cycles": 1}}))
    study_yaml_b.write_text(yaml.dump({**sweep_content, "execution": {"n_cycles": 5}}))

    sc_a = load_study_config(study_yaml_a)
    sc_b = load_study_config(study_yaml_b)
    assert sc_a.study_design_hash == sc_b.study_design_hash
