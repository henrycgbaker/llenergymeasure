"""End-to-end integration test for sweep canonicalisation + H1 dedup.

Exercises the full load path: a study YAML with measurement-equivalent
sweep configs goes through ``load_study_config`` and the resulting
``StudyConfig`` records the pre-run equivalence groups + deduplicated
canonical configs.

Run time: < 1s — no GPU involved, all operations are on Pydantic models
and the vendored-rules loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llenergymeasure.config.loader import load_study_config


def _write_study(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_greedy_temperature_sweep_collapses(tmp_path: Path) -> None:
    """Six-config sweep with dormant sampling fields collapses to four unique."""
    study = {
        "study_name": "dedup_test",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling.do_sample": [True, False],
            "transformers.sampling.temperature": [0.5, 1.0, 1.5],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config(path)

    # Dedup mode default is h1.
    assert study_config.dedup_mode == "h1"
    # 6 declared x 1 cycle -> 4 unique x 1 cycle = 4 experiments.
    assert len(study_config.experiments) == 4
    # Declared count preserved via H1 list.
    assert len(study_config.declared_h1_hashes) == 6
    # At least one group has multiple members (the greedy-family collapse).
    group_sizes = sorted(g["member_count"] for g in study_config.pre_run_equivalence_groups)
    assert max(group_sizes) >= 2
    assert sum(group_sizes) == 6


def test_no_dedup_preserves_all_configs(tmp_path: Path) -> None:
    """With ``deduplicate_equivalent: false`` every declared config runs."""
    study = {
        "study_name": "no_dedup",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling.do_sample": [True, False],
            "transformers.sampling.temperature": [0.5, 1.0, 1.5],
        },
        "study_execution": {"deduplicate_equivalent": False},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config(path)

    assert study_config.dedup_mode == "off"
    # All 6 declared configs run — canonicaliser still populated the groups.
    assert len(study_config.experiments) == 6
    # Groups still computed for the sidecar trail.
    assert sum(g["member_count"] for g in study_config.pre_run_equivalence_groups) == 6


def test_cli_override_no_dedup(tmp_path: Path) -> None:
    """CLI-equivalent override (``study_execution.deduplicate_equivalent: false``)."""
    study = {
        "study_name": "cli_no_dedup",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling.do_sample": [True, False],
            "transformers.sampling.temperature": [0.5, 0.7],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config(
        path,
        cli_overrides={"study_execution": {"deduplicate_equivalent": False}},
    )
    assert study_config.dedup_mode == "off"
    # 2 x 2 = 4 declared configs all run.
    assert len(study_config.experiments) == 4


def test_n_cycles_multiplies_unique_set(tmp_path: Path) -> None:
    """Dedup happens within a cycle; ``n_cycles`` multiplies the deduped set."""
    study = {
        "study_name": "cycles",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling.do_sample": [True, False],
            "transformers.sampling.temperature": [0.5, 0.7],
        },
        "study_execution": {"n_cycles": 3},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config(path)

    # 4 declared -> 3 unique (greedy-0.5 + greedy-0.7 collapse to greedy-1.0,
    # plus 2 sampling variants). 3 unique x 3 cycles = 9 runs.
    assert study_config.dedup_mode == "h1"
    unique = {h for h in study_config.declared_h1_hashes}
    # Two declared configs share the same H1 (both greedy).
    assert len(unique) == 3
    assert len(study_config.experiments) == 9


def test_single_config_sweep_no_dedup(tmp_path: Path) -> None:
    """A sweep with one axis and no equivalence runs normally."""
    study = {
        "study_name": "single",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling.temperature": [0.5, 0.7, 0.9],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config(path)

    # Sampling is default-true; three temps should stay distinct.
    assert len(study_config.experiments) == 3
    group_sizes = sorted(g["member_count"] for g in study_config.pre_run_equivalence_groups)
    assert group_sizes == [1, 1, 1]
