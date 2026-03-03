"""Unit tests for ExperimentConfig Pydantic validation.

Tests v2.0 field renames, extra=forbid, backend composition, cross-validators,
and schema-driven precision validation using SSOT constants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import PRECISION_SUPPORT
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


def test_field_name_precision():
    """v2.0 'precision' field (not 'fp_precision') is accepted."""
    config = ExperimentConfig(model="gpt2", precision="fp16")
    assert config.precision == "fp16"


def test_field_name_n():
    """v2.0 'n' field (not 'num_input_prompts') is accepted."""
    config = ExperimentConfig(model="gpt2", n=50)
    assert config.n == 50


def test_v1x_field_model_name_rejected():
    """v1.x 'model_name' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model_name="gpt2")  # type: ignore[call-arg]


def test_v1x_field_fp_precision_rejected():
    """v1.x 'fp_precision' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", fp_precision="fp16")  # type: ignore[call-arg]


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


def test_invalid_precision_raises_validation_error():
    """Invalid precision value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt2", precision="float16")  # wrong alias


def test_valid_precision_fp32():
    """precision='fp32' is valid."""
    config = ExperimentConfig(model="gpt2", precision="fp32")
    assert config.precision == "fp32"


def test_valid_precision_fp16():
    """precision='fp16' is valid."""
    config = ExperimentConfig(model="gpt2", precision="fp16")
    assert config.precision == "fp16"


def test_valid_precision_bf16():
    """precision='bf16' is valid."""
    config = ExperimentConfig(model="gpt2", precision="bf16")
    assert config.precision == "bf16"


@pytest.mark.parametrize("precision", PRECISION_SUPPORT["pytorch"])
def test_all_pytorch_precisions_valid(precision):
    """Schema-driven: all SSOT PRECISION_SUPPORT['pytorch'] values are valid."""
    config = make_config(precision=precision)
    assert config.precision == precision


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
    config = make_config(model="bert-base", precision="fp32")
    assert config.model == "bert-base"
    assert config.precision == "fp32"
