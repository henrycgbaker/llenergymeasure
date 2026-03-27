"""Unit tests for ExperimentConfig Pydantic validation.

Tests v2.0 field renames, extra=forbid, backend composition, cross-validators,
and schema-driven dtype validation using SSOT constants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Minimal valid config
# ---------------------------------------------------------------------------


def test_minimal_valid_config():
    """ExperimentConfig(model='gpt2', backend='pytorch') succeeds."""
    config = ExperimentConfig(model="gpt2", backend="pytorch")
    assert config.model == "gpt2"
    assert config.backend == "pytorch"


def test_model_only_uses_pytorch_default():
    """ExperimentConfig with only model= uses backend='pytorch' default."""
    config = ExperimentConfig(model="gpt2")
    assert config.backend == "pytorch"


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------


def test_extra_fields_forbidden():
    """Unknown top-level fields are rejected with ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", backend="pytorch", unknown_field="x")


def test_multiple_extra_fields_all_rejected():
    """Multiple unknown fields are all rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", backend="pytorch", foo="a", bar="b")


# ---------------------------------------------------------------------------
# v2.0 field renames
# ---------------------------------------------------------------------------


def test_field_name_model():
    """v2.0 'model' field (not 'model_name') is accepted."""
    config = ExperimentConfig(model="gpt2")
    assert config.model == "gpt2"


def test_field_name_dtype():
    """v2.0 'dtype' field is accepted."""
    config = ExperimentConfig(model="gpt2", dtype="float16")
    assert config.dtype == "float16"


def test_field_name_n():
    """v2.0 dataset.n_prompts field (not 'num_input_prompts') is accepted."""
    from llenergymeasure.config.models import DatasetConfig

    config = ExperimentConfig(model="gpt2", dataset=DatasetConfig(n_prompts=50))
    assert config.dataset.n_prompts == 50


def test_v1x_field_model_name_rejected():
    """v1.x 'model_name' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model_name="gpt2")  # type: ignore[call-arg]


def test_v1x_field_fp_precision_rejected():
    """v1.x 'fp_precision' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", fp_dtype="float16")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------


def test_invalid_backend_raises_validation_error():
    """Unknown backend value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", backend="invalid_backend")


def test_default_backend_is_pytorch():
    """Default backend is 'pytorch' when not specified."""
    config = ExperimentConfig(model="gpt2")
    assert config.backend == "pytorch"


def test_vllm_backend_accepted():
    """backend='vllm' is accepted."""
    config = ExperimentConfig(model="gpt2", backend="vllm")
    assert config.backend == "vllm"


def test_tensorrt_backend_accepted():
    """backend='tensorrt' is accepted."""
    config = ExperimentConfig(model="gpt2", backend="tensorrt")
    assert config.backend == "tensorrt"


# ---------------------------------------------------------------------------
# Backend section composition
# ---------------------------------------------------------------------------


def test_pytorch_config_section_composition():
    """config with pytorch={batch_size: 4} backend section is accepted."""
    config = ExperimentConfig(
        model="gpt2",
        backend="pytorch",
        pytorch={"batch_size": 4},
    )
    assert config.pytorch is not None
    assert config.pytorch.batch_size == 4


# ---------------------------------------------------------------------------
# PyTorchConfig.num_processes removal (M3 audit fix)
# ---------------------------------------------------------------------------


def test_pytorch_config_has_no_num_processes_field():
    """PyTorchConfig does not have a num_processes field (removed in M3 audit)."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    assert "num_processes" not in PyTorchConfig.model_fields


def test_pytorch_config_num_processes_not_a_declared_field():
    """num_processes is not a declared field on PyTorchConfig.

    PyTorchConfig uses extra='allow' for HuggingFace passthrough, so passing
    num_processes as an extra kwarg does not raise a ValidationError, but it
    is NOT a typed model field and will not be type-checked or validated.
    """
    from llenergymeasure.config.backend_configs import PyTorchConfig

    # Verify it is absent from the declared model fields
    assert "num_processes" not in PyTorchConfig.model_fields
    # Extra kwargs are accepted (extra='allow') but go into __pydantic_extra__
    config = PyTorchConfig(num_processes=4)  # type: ignore[call-arg]
    # Not a typed field - no attribute access by name on the typed model
    assert "num_processes" not in type(config).model_fields


def test_pytorch_section_with_wrong_backend_rejected():
    """pytorch: section with backend='vllm' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"pytorch.*config section provided.*backend"):
        ExperimentConfig(
            model="gpt2",
            backend="vllm",
            pytorch={"batch_size": 4},
        )


def test_vllm_section_with_pytorch_backend_rejected():
    """vllm: section with backend='pytorch' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"vllm.*config section provided.*backend"):
        ExperimentConfig(
            model="gpt2",
            backend="pytorch",
            vllm={"engine": {"max_num_seqs": 16}},
        )


def test_tensorrt_section_with_wrong_backend_rejected():
    """tensorrt: section with backend='pytorch' raises ValidationError."""
    with pytest.raises(ValidationError, match=r"tensorrt.*config section provided.*backend"):
        ExperimentConfig(
            model="gpt2",
            backend="pytorch",
            tensorrt={"max_batch_size": 8},
        )


# ---------------------------------------------------------------------------
# Precision validation
# ---------------------------------------------------------------------------


def test_invalid_dtype_raises_validation_error():
    """Invalid dtype value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", dtype="fp16")  # old shorthand


def test_valid_dtype_float32():
    """dtype='float32' is valid."""
    config = ExperimentConfig(model="gpt2", dtype="float32")
    assert config.dtype == "float32"


def test_valid_dtype_float16():
    """dtype='float16' is valid."""
    config = ExperimentConfig(model="gpt2", dtype="float16")
    assert config.dtype == "float16"


def test_valid_dtype_bfloat16():
    """dtype='bfloat16' is valid."""
    config = ExperimentConfig(model="gpt2", dtype="bfloat16")
    assert config.dtype == "bfloat16"


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["pytorch"])
def test_all_pytorch_dtypes_valid(dt):
    """Schema-driven: all SSOT DTYPE_SUPPORT['pytorch'] values are valid."""
    config = make_config(dtype=dt)
    assert config.dtype == dt


# ---------------------------------------------------------------------------
# passthrough_kwargs collision cross-validator
# ---------------------------------------------------------------------------


def test_passthrough_kwargs_accepted():
    """passthrough_kwargs with non-colliding keys are accepted."""
    config = ExperimentConfig(
        model="gpt2",
        passthrough_kwargs={"custom_flag": True, "my_special_param": 42},
    )
    assert config.passthrough_kwargs is not None
    assert config.passthrough_kwargs["custom_flag"] is True


def test_passthrough_kwargs_collision_with_top_level_field_rejected():
    """passthrough_kwargs keys colliding with ExperimentConfig fields are rejected."""
    with pytest.raises(ValidationError, match=r"passthrough_kwargs.*collide"):
        ExperimentConfig(
            model="gpt2",
            passthrough_kwargs={"model": "override"},  # 'model' is a top-level field
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def test_make_config_helper_returns_valid_config():
    """make_config() factory from conftest creates a valid ExperimentConfig."""
    config = make_config()
    assert isinstance(config, ExperimentConfig)
    assert config.model == "gpt2"
    assert config.backend == "pytorch"


def test_make_config_override():
    """make_config(**overrides) applies overrides over defaults."""
    config = make_config(model="bert-base", dtype="float32")
    assert config.model == "bert-base"
    assert config.dtype == "float32"


# ---------------------------------------------------------------------------
# EnergyConfig schema tests (moved from test_warmup_v2.py)
# ---------------------------------------------------------------------------


def test_energy_config_default() -> None:
    """EnergyConfig defaults to backend='auto'."""
    from llenergymeasure.config.models import EnergyConfig

    assert EnergyConfig().backend == "auto"


def test_energy_config_null() -> None:
    """EnergyConfig(backend=None) disables energy measurement."""
    from llenergymeasure.config.models import EnergyConfig

    cfg = EnergyConfig(backend=None)
    assert cfg.backend is None


def test_energy_config_valid_backends() -> None:
    """All backend literal values are accepted."""
    from llenergymeasure.config.models import EnergyConfig

    for backend in ("auto", "nvml", "zeus", "codecarbon"):
        cfg = EnergyConfig(backend=backend)
        assert cfg.backend == backend


def test_energy_config_invalid_backend() -> None:
    """Unknown backend values raise ValidationError."""
    from llenergymeasure.config.models import EnergyConfig

    with pytest.raises(ValidationError):
        EnergyConfig(backend="unknown_backend")  # type: ignore[arg-type]


def test_energy_config_extra_forbid() -> None:
    """EnergyConfig: extra fields raise ValidationError."""
    from llenergymeasure.config.models import EnergyConfig

    with pytest.raises(ValidationError):
        EnergyConfig(backend="auto", extra_field=1)  # type: ignore[call-arg]


def test_experiment_config_has_energy() -> None:
    """ExperimentConfig.energy defaults to EnergyConfig with backend='auto'."""
    cfg = ExperimentConfig(model="gpt2")
    assert cfg.energy.backend == "auto"


def test_experiment_config_energy_override() -> None:
    """ExperimentConfig allows overriding energy backend."""
    from llenergymeasure.config.models import EnergyConfig

    cfg = ExperimentConfig(model="gpt2", energy=EnergyConfig(backend="nvml"))
    assert cfg.energy.backend == "nvml"


def test_experiment_config_energy_disabled() -> None:
    """ExperimentConfig allows disabling energy measurement via null."""
    from llenergymeasure.config.models import EnergyConfig

    cfg = ExperimentConfig(model="gpt2", energy=EnergyConfig(backend=None))
    assert cfg.energy.backend is None


# ---------------------------------------------------------------------------
# PyTorchConfig tensor parallelism fields (tp_plan, tp_size)
# ---------------------------------------------------------------------------


def test_pytorch_config_tp_plan_accepts_auto():
    """PyTorchConfig(tp_plan='auto') succeeds."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    cfg = PyTorchConfig(tp_plan="auto")
    assert cfg.tp_plan == "auto"


def test_pytorch_config_tp_plan_rejects_invalid():
    """PyTorchConfig(tp_plan='custom') raises ValidationError (Literal enforcement)."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    with pytest.raises(ValidationError):
        PyTorchConfig(tp_plan="custom")  # type: ignore[arg-type]


def test_pytorch_config_tp_size_accepts_positive():
    """PyTorchConfig(tp_plan='auto', tp_size=4) succeeds."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    cfg = PyTorchConfig(tp_plan="auto", tp_size=4)
    assert cfg.tp_plan == "auto"
    assert cfg.tp_size == 4


def test_pytorch_config_tp_size_rejects_zero():
    """PyTorchConfig(tp_size=0) raises ValidationError (ge=1)."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    with pytest.raises(ValidationError):
        PyTorchConfig(tp_size=0)


def test_pytorch_config_tp_plan_device_map_exclusive():
    """PyTorchConfig(tp_plan='auto', device_map='auto') raises ValidationError."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    with pytest.raises(ValidationError, match="mutually exclusive"):
        PyTorchConfig(tp_plan="auto", device_map="auto")


def test_pytorch_config_tp_plan_without_device_map_ok():
    """PyTorchConfig(tp_plan='auto') succeeds (no conflict)."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    cfg = PyTorchConfig(tp_plan="auto")
    assert cfg.tp_plan == "auto"
    assert cfg.device_map is None


def test_pytorch_config_device_map_without_tp_plan_ok():
    """PyTorchConfig(device_map='auto') succeeds (no conflict)."""
    from llenergymeasure.config.backend_configs import PyTorchConfig

    cfg = PyTorchConfig(device_map="auto")
    assert cfg.device_map == "auto"
    assert cfg.tp_plan is None
