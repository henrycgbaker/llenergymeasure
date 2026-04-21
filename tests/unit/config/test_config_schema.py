"""Unit tests for ExperimentConfig Pydantic validation.

Tests v2.0 field renames, extra=forbid, engine composition, cross-validators,
and schema-driven dtype validation using SSOT constants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig, OutputConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Minimal valid config
# ---------------------------------------------------------------------------


def test_minimal_valid_config():
    """ExperimentConfig(task={'model': 'gpt2'}, engine='transformers') succeeds."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert config.task.model == "gpt2"
    assert config.engine == "transformers"


def test_model_only_uses_pytorch_default():
    """ExperimentConfig with only model= uses engine='transformers' default."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.engine == "transformers"


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------


def test_extra_fields_forbidden():
    """Unknown top-level fields are rejected with ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="transformers", unknown_field="x")


def test_multiple_extra_fields_all_rejected():
    """Multiple unknown fields are all rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="transformers", foo="a", bar="b")


# ---------------------------------------------------------------------------
# v2.0 field renames
# ---------------------------------------------------------------------------


def test_field_name_model():
    """v2.0 'model' field (not 'model_name') is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.task.model == "gpt2"


def test_field_name_dtype():
    """dtype is per-engine (lives on the active engine's config section)."""
    config = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "float16"}
    )
    assert config.transformers is not None
    assert config.transformers.dtype == "float16"


def test_field_name_n():
    """v2.0 dataset.n_prompts field (not 'num_input_prompts') is accepted."""
    from llenergymeasure.config.models import DatasetConfig

    config = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)})
    assert config.task.dataset.n_prompts == 50


def test_v1x_field_model_name_rejected():
    """v1.x 'model_name' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, model_name="gpt2")  # type: ignore[call-arg]


def test_v1x_field_fp_precision_rejected():
    """v1.x 'fp_precision' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, fp_dtype="float16")  # type: ignore[call-arg]


def test_top_level_dtype_rejected():
    """Top-level dtype is rejected — dtype lives per-engine (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, dtype="float16")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Engine validation
# ---------------------------------------------------------------------------


def test_invalid_engine_raises_validation_error():
    """Unknown engine value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="invalid_backend")


def test_default_engine_is_pytorch():
    """Default engine is 'transformers' when not specified."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.engine == "transformers"


def test_vllm_engine_accepted():
    """engine='vllm' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="vllm")
    assert config.engine == "vllm"


def test_tensorrt_engine_accepted():
    """engine='tensorrt' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt")
    assert config.engine == "tensorrt"


# ---------------------------------------------------------------------------
# Engine section composition
# ---------------------------------------------------------------------------


def test_pytorch_config_section_composition():
    """config with transformers={batch_size: 4} engine section is accepted."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"batch_size": 4},
    )
    assert config.transformers is not None
    assert config.transformers.batch_size == 4


# ---------------------------------------------------------------------------
# TransformersConfig.num_processes removal (M3 audit fix)
# ---------------------------------------------------------------------------


def test_pytorch_config_has_no_num_processes_field():
    """TransformersConfig does not have a num_processes field (removed in M3 audit)."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    assert "num_processes" not in TransformersConfig.model_fields


def test_pytorch_config_num_processes_not_a_declared_field():
    """num_processes is not a declared field on TransformersConfig.

    TransformersConfig uses extra='allow' for HuggingFace passthrough, so passing
    num_processes as an extra kwarg does not raise a ValidationError, but it
    is NOT a typed model field and will not be type-checked or validated.
    """
    from llenergymeasure.config.engine_configs import TransformersConfig

    # Verify it is absent from the declared model fields
    assert "num_processes" not in TransformersConfig.model_fields
    # Extra kwargs are accepted (extra='allow') but go into __pydantic_extra__
    config = TransformersConfig(num_processes=4)  # type: ignore[call-arg]
    # Not a typed field - no attribute access by name on the typed model
    assert "num_processes" not in type(config).model_fields


def test_pytorch_section_with_wrong_engine_rejected():
    """pytorch: section with engine='vllm' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"transformers.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            transformers={"batch_size": 4},
        )


def test_vllm_section_with_pytorch_engine_rejected():
    """vllm: section with engine='transformers' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"vllm.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            vllm={"engine": {"max_num_seqs": 16}},
        )


def test_tensorrt_section_with_wrong_engine_rejected():
    """tensorrt: section with engine='transformers' raises ValidationError."""
    with pytest.raises(ValidationError, match=r"tensorrt.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            tensorrt={"max_batch_size": 8},
        )


# ---------------------------------------------------------------------------
# Precision validation
# ---------------------------------------------------------------------------


def test_invalid_dtype_raises_validation_error():
    """Invalid dtype value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "fp16"}
        )  # old shorthand


def test_valid_dtype_float32():
    """dtype='float32' is valid on TransformersConfig."""
    config = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "float32"}
    )
    assert config.transformers.dtype == "float32"


def test_valid_dtype_float16():
    """dtype='float16' is valid on TransformersConfig."""
    config = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "float16"}
    )
    assert config.transformers.dtype == "float16"


def test_valid_dtype_bfloat16():
    """dtype='bfloat16' is valid on TransformersConfig."""
    config = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "bfloat16"}
    )
    assert config.transformers.dtype == "bfloat16"


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["transformers"])
def test_all_pytorch_dtypes_valid(dt):
    """Schema-driven: all SSOT DTYPE_SUPPORT['transformers'] values are valid."""
    config = make_config(dtype=dt)
    assert config.transformers.dtype == dt


# ---------------------------------------------------------------------------
# passthrough_kwargs collision cross-validator
# ---------------------------------------------------------------------------


def test_passthrough_kwargs_accepted():
    """passthrough_kwargs with non-colliding keys are accepted."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        passthrough_kwargs={"custom_flag": True, "my_special_param": 42},
    )
    assert config.passthrough_kwargs is not None
    assert config.passthrough_kwargs["custom_flag"] is True


def test_passthrough_kwargs_collision_with_top_level_field_rejected():
    """passthrough_kwargs keys colliding with ExperimentConfig fields are rejected."""
    with pytest.raises(ValidationError, match=r"passthrough_kwargs.*collide"):
        ExperimentConfig(
            task={"model": "gpt2"},
            passthrough_kwargs={"task": "override"},  # 'task' is a top-level field
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def test_make_config_helper_returns_valid_config():
    """make_config() factory from conftest creates a valid ExperimentConfig."""
    config = make_config()
    assert isinstance(config, ExperimentConfig)
    assert config.task.model == "gpt2"
    assert config.engine == "transformers"


def test_make_config_override():
    """make_config(**overrides) applies overrides over defaults (dtype -> engine section)."""
    config = make_config(model="bert-base", dtype="float32")
    assert config.task.model == "bert-base"
    assert config.transformers.dtype == "float32"


# ---------------------------------------------------------------------------
# energy_sampler field tests (flat Literal on ExperimentConfig)
# ---------------------------------------------------------------------------


def test_energy_sampler_default() -> None:
    """energy_sampler defaults to 'auto'."""
    cfg = ExperimentConfig(task={"model": "gpt2"})
    assert cfg.measurement.energy_sampler == "auto"


def test_energy_sampler_null_disables() -> None:
    """energy_sampler=None disables energy measurement."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": None})
    assert cfg.measurement.energy_sampler is None


def test_energy_sampler_valid_engines() -> None:
    """All energy_sampler literal values are accepted."""
    for engine in ("auto", "nvml", "zeus", "codecarbon"):
        cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": engine})
        assert cfg.measurement.energy_sampler == engine


def test_energy_sampler_invalid_engine() -> None:
    """Unknown energy_sampler values raise ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": "unknown_backend"})


def test_energy_sampler_override() -> None:
    """ExperimentConfig allows overriding energy_sampler."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": "nvml"})
    assert cfg.measurement.energy_sampler == "nvml"


# ---------------------------------------------------------------------------
# save_timeseries field tests (boolean on OutputConfig)
# ---------------------------------------------------------------------------


def test_save_timeseries_default_true() -> None:
    """save_timeseries defaults to True."""
    cfg = OutputConfig()
    assert cfg.save_timeseries is True


def test_save_timeseries_false_accepted() -> None:
    """save_timeseries=False is accepted."""
    cfg = OutputConfig(save_timeseries=False)
    assert cfg.save_timeseries is False


# ---------------------------------------------------------------------------
# TransformersConfig tensor parallelism fields (tp_plan, tp_size)
# ---------------------------------------------------------------------------


def test_pytorch_config_tp_plan_accepts_auto():
    """TransformersConfig(tp_plan='auto') succeeds."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = TransformersConfig(tp_plan="auto")
    assert cfg.tp_plan == "auto"


def test_pytorch_config_tp_plan_rejects_invalid():
    """TransformersConfig(tp_plan='custom') raises ValidationError (Literal enforcement)."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    with pytest.raises(ValidationError):
        TransformersConfig(tp_plan="custom")  # type: ignore[arg-type]


def test_pytorch_config_tp_size_accepts_positive():
    """TransformersConfig(tp_plan='auto', tp_size=4) succeeds."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = TransformersConfig(tp_plan="auto", tp_size=4)
    assert cfg.tp_plan == "auto"
    assert cfg.tp_size == 4


def test_pytorch_config_tp_size_rejects_zero():
    """TransformersConfig(tp_size=0) raises ValidationError (ge=1)."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    with pytest.raises(ValidationError):
        TransformersConfig(tp_size=0)


def test_pytorch_config_tp_plan_device_map_exclusive():
    """TransformersConfig(tp_plan='auto', device_map='auto') raises ValidationError."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    with pytest.raises(ValidationError, match="mutually exclusive"):
        TransformersConfig(tp_plan="auto", device_map="auto")


def test_pytorch_config_tp_plan_without_device_map_ok():
    """TransformersConfig(tp_plan='auto') succeeds (no conflict)."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = TransformersConfig(tp_plan="auto")
    assert cfg.tp_plan == "auto"
    assert cfg.device_map is None


def test_pytorch_config_device_map_without_tp_plan_ok():
    """TransformersConfig(device_map='auto') succeeds (no conflict)."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = TransformersConfig(device_map="auto")
    assert cfg.device_map == "auto"
    assert cfg.tp_plan is None


# ---------------------------------------------------------------------------
# Bug 1.1 — fp8 quantization + float32 dtype (vLLM)
# ---------------------------------------------------------------------------


def test_vllm_dtype_float32_rejected():
    """VLLMConfig.dtype Literal rejects float32 (vLLM does not support fp32)."""
    from llenergymeasure.config.engine_configs import VLLMConfig

    with pytest.raises(ValidationError):
        VLLMConfig(dtype="float32")  # type: ignore[arg-type]


def test_vllm_fp8_float16_accepted():
    """fp8 quantization with dtype=float16 is accepted."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(dtype="float16", engine=VLLMEngineConfig(quantization="fp8")),
    )
    assert cfg.vllm.dtype == "float16"


def test_vllm_fp8_bfloat16_accepted():
    """fp8 quantization with dtype=bfloat16 is accepted."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(dtype="bfloat16", engine=VLLMEngineConfig(quantization="fp8")),
    )
    assert cfg.vllm.dtype == "bfloat16"


def test_vllm_non_fp8_float16_accepted():
    """Non-fp8 quantization (awq) with dtype=float16 is accepted."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(dtype="float16", engine=VLLMEngineConfig(quantization="awq")),
    )
    assert cfg.vllm.dtype == "float16"


def test_vllm_no_quantization_default_dtype_accepted():
    """No quantization set, no explicit dtype, is accepted (engine default applies)."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(engine=VLLMEngineConfig()),
    )
    assert cfg.vllm.dtype is None


# ---------------------------------------------------------------------------
# Bug 1.2 — max_num_batched_tokens < max_model_len (vLLM engine)
# ---------------------------------------------------------------------------


def test_vllm_batched_tokens_less_than_model_len_rejected():
    """max_num_batched_tokens < max_model_len raises ValidationError at parse time."""
    from llenergymeasure.config.engine_configs import VLLMEngineConfig

    with pytest.raises(ValidationError, match=r"max_num_batched_tokens.*must be >="):
        VLLMEngineConfig(max_num_batched_tokens=512, max_model_len=1024)


def test_vllm_batched_tokens_equal_model_len_accepted():
    """max_num_batched_tokens == max_model_len is accepted."""
    from llenergymeasure.config.engine_configs import VLLMEngineConfig

    cfg = VLLMEngineConfig(max_num_batched_tokens=1024, max_model_len=1024)
    assert cfg.max_num_batched_tokens == 1024
    assert cfg.max_model_len == 1024


def test_vllm_batched_tokens_greater_accepted():
    """max_num_batched_tokens > max_model_len is accepted."""
    from llenergymeasure.config.engine_configs import VLLMEngineConfig

    cfg = VLLMEngineConfig(max_num_batched_tokens=2048, max_model_len=1024)
    assert cfg.max_num_batched_tokens == 2048


def test_vllm_batched_tokens_one_none_accepted():
    """Only one of max_num_batched_tokens / max_model_len set is accepted."""
    from llenergymeasure.config.engine_configs import VLLMEngineConfig

    cfg = VLLMEngineConfig(max_num_batched_tokens=512)
    assert cfg.max_num_batched_tokens == 512
    assert cfg.max_model_len is None


# ---------------------------------------------------------------------------
# Bug 1.3 — flash_attention_2/3 + float32 dtype (PyTorch)
# ---------------------------------------------------------------------------


def test_pytorch_flash_attn2_float32_rejected():
    """flash_attention_2 with dtype=float32 raises ValidationError at parse time."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    with pytest.raises(ValidationError, match=r"flash_attention_2.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(
                dtype="float32", attn_implementation="flash_attention_2"
            ),
        )


def test_pytorch_flash_attn3_float32_rejected():
    """flash_attention_3 with dtype=float32 raises ValidationError at parse time."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    with pytest.raises(ValidationError, match=r"flash_attention_3.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(
                dtype="float32", attn_implementation="flash_attention_3"
            ),
        )


def test_pytorch_flash_attn2_bfloat16_accepted():
    """flash_attention_2 with dtype=bfloat16 is accepted."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="bfloat16", attn_implementation="flash_attention_2"),
    )
    assert cfg.transformers.dtype == "bfloat16"


def test_pytorch_eager_float32_accepted():
    """attn_implementation=eager with dtype=float32 is accepted."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float32", attn_implementation="eager"),
    )
    assert cfg.transformers.dtype == "float32"


def test_pytorch_no_attn_impl_float32_accepted():
    """No attn_implementation set with dtype=float32 is accepted."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float32"),
    )
    assert cfg.transformers.dtype == "float32"


# ---------------------------------------------------------------------------
# TRT FP8+float32 cross-validator
# ---------------------------------------------------------------------------


def test_trt_dtype_float32_rejected() -> None:
    """TensorRTConfig.dtype Literal rejects float32 (TRT-LLM does not support fp32)."""
    from llenergymeasure.config.engine_configs import TensorRTConfig

    with pytest.raises(ValidationError):
        TensorRTConfig(dtype="float32")  # type: ignore[arg-type]


def test_trt_fp8_accepts_float16() -> None:
    """FP8 quantization with dtype=float16 is accepted."""
    from llenergymeasure.config.engine_configs import TensorRTConfig, TensorRTQuantConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(dtype="float16", quant=TensorRTQuantConfig(quant_algo="FP8")),
    )
    assert cfg.tensorrt.dtype == "float16"


def test_trt_fp8_accepts_bfloat16() -> None:
    """FP8 quantization with dtype=bfloat16 is accepted."""
    from llenergymeasure.config.engine_configs import TensorRTConfig, TensorRTQuantConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(dtype="bfloat16", quant=TensorRTQuantConfig(quant_algo="FP8")),
    )
    assert cfg.tensorrt.dtype == "bfloat16"


def test_trt_non_fp8_accepts_float16() -> None:
    """Non-FP8 quantization (INT8) with dtype=float16 is accepted."""
    from llenergymeasure.config.engine_configs import TensorRTConfig, TensorRTQuantConfig

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(dtype="float16", quant=TensorRTQuantConfig(quant_algo="INT8")),
    )
    assert cfg.tensorrt.dtype == "float16"


# ---------------------------------------------------------------------------
# n_prompts default
# ---------------------------------------------------------------------------


def test_n_prompts_default_is_100() -> None:
    """DatasetConfig().n_prompts defaults to 100."""
    from llenergymeasure.config.models import DatasetConfig

    assert DatasetConfig().n_prompts == 100
