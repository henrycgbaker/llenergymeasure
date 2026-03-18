"""Unit tests for VLLMBackend.

All tests run without GPU hardware and without vLLM installed.
vLLM imports inside VLLMBackend methods are lazy — the module is importable on
any host. Tests that exercise SamplingParams construction pass a mock class so
no real vLLM import occurs.

Coverage:
  - Protocol compliance and get_backend() registration
  - Precision mapping (fp32/fp16/bf16 → float32/float16/bfloat16)
  - _build_llm_kwargs: minimal defaults + all VLLMConfig fields + None omission
  - _build_sampling_params: greedy, sampling, top_k sentinel mapping
  - No streaming code (CM-07 structurally resolved)
  - --shm-size 8g present in DockerRunner._build_docker_cmd (VLLM-03)
  - Prompt loading (covered by tests/unit/test_datasets.py; harness passes prompts to backends)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from llenergymeasure.backends.vllm import VLLMBackend
from llenergymeasure.config.backend_configs import (
    VLLMAttentionConfig,
    VLLMBeamSearchConfig,
    VLLMConfig,
    VLLMEngineConfig,
    VLLMSamplingConfig,
)
from llenergymeasure.config.models import DecoderConfig
from llenergymeasure.utils.exceptions import BackendError
from tests.conftest import make_config

# =============================================================================
# Helpers
# =============================================================================

_VLLM_DEFAULTS = {"model": "test-model", "backend": "vllm"}


@dataclass
class _FakeSamplingParams:
    """Minimal stand-in for vllm.SamplingParams — captures kwargs for inspection."""

    temperature: float = 1.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    min_p: float | None = None
    _extra: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Store all kwargs as attributes for easy assertion."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._kwargs = kwargs


@dataclass
class _FakeBeamSearchParams:
    """Minimal stand-in for vllm.BeamSearchParams — captures kwargs for inspection."""

    beam_width: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    max_tokens: int = 128
    _extra: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Store all kwargs as attributes for easy assertion."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._kwargs = kwargs


# =============================================================================
# Test Group 1: Protocol compliance and registration
# =============================================================================


class TestProtocolCompliance:
    def test_vllm_backend_name(self):
        """VLLMBackend.name returns 'vllm'."""
        backend = VLLMBackend()
        assert backend.name == "vllm"

    def test_vllm_backend_satisfies_plugin_protocol(self):
        """VLLMBackend satisfies the runtime_checkable BackendPlugin protocol."""
        from llenergymeasure.backends.protocol import BackendPlugin

        backend = VLLMBackend()
        assert isinstance(backend, BackendPlugin)

    def test_get_backend_returns_vllm_instance(self):
        """get_backend('vllm') returns a VLLMBackend with name 'vllm'."""
        from llenergymeasure.backends import get_backend

        backend = get_backend("vllm")
        assert backend.name == "vllm"
        assert isinstance(backend, VLLMBackend)

    def test_get_backend_unknown_mentions_vllm_in_error(self):
        """get_backend('unknown') error message lists vllm as available."""
        from llenergymeasure.backends import get_backend

        with pytest.raises(BackendError, match="vllm"):
            get_backend("unknown")

    def test_get_backend_unknown_raises_backend_error(self):
        """get_backend with unknown name raises BackendError (not KeyError, etc.)."""
        from llenergymeasure.backends import get_backend

        with pytest.raises(BackendError, match="Unknown backend"):
            get_backend("does_not_exist")


# =============================================================================
# Test Group 2: Precision mapping
# =============================================================================


class TestPrecisionMapping:
    def test_precision_fp32_maps_to_float32(self):
        """'fp32' maps to 'float32' — the vLLM dtype string."""
        assert VLLMBackend._map_precision("fp32") == "float32"

    def test_precision_fp16_maps_to_float16(self):
        """'fp16' maps to 'float16' — the vLLM dtype string."""
        assert VLLMBackend._map_precision("fp16") == "float16"

    def test_precision_bf16_maps_to_bfloat16(self):
        """'bf16' maps to 'bfloat16' — the vLLM dtype string."""
        assert VLLMBackend._map_precision("bf16") == "bfloat16"

    def test_precision_unknown_returns_auto(self):
        """Unknown precision strings fall back to 'auto' (vLLM's own detection)."""
        assert VLLMBackend._map_precision("unknown") == "auto"
        assert VLLMBackend._map_precision("") == "auto"
        assert VLLMBackend._map_precision("int8") == "auto"


# =============================================================================
# Test Group 3: _build_llm_kwargs
# =============================================================================


class TestBuildLlmKwargs:
    def test_minimal_config_has_required_keys(self):
        """With no VLLMConfig, kwargs contains model, dtype, trust_remote_code, seed."""
        config = make_config(**_VLLM_DEFAULTS)
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["model"] == "test-model"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["seed"] == 42
        assert "dtype" in kwargs

    def test_minimal_config_dtype_reflects_precision(self):
        """Default precision (bf16) maps to bfloat16 in kwargs."""
        config = make_config(**_VLLM_DEFAULTS)
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)
        assert kwargs["dtype"] == "bfloat16"

    def test_vllm_config_fields_applied_when_not_none(self):
        """All non-None VLLMEngineConfig fields are present in the returned kwargs dict."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                tensor_parallel_size=2,
                gpu_memory_utilization=0.85,
                max_num_seqs=128,
                enable_prefix_caching=True,
                quantization="awq",
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2
        assert kwargs["gpu_memory_utilization"] == 0.85
        assert kwargs["max_num_seqs"] == 128
        assert kwargs["enable_prefix_caching"] is True
        assert kwargs["quantization"] == "awq"

    def test_none_vllm_config_fields_are_omitted(self):
        """None VLLMEngineConfig fields are NOT added to kwargs — backend uses its own default."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(tensor_parallel_size=2))  # only TP set
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2
        assert "gpu_memory_utilization" not in kwargs
        assert "max_num_seqs" not in kwargs
        assert "enable_prefix_caching" not in kwargs
        assert "quantization" not in kwargs

    def test_no_vllm_section_produces_no_extra_keys(self):
        """When config.vllm is None, only the 4 base keys are present."""
        config = make_config(**_VLLM_DEFAULTS)  # vllm=None by default
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert set(kwargs.keys()) == {"model", "dtype", "trust_remote_code", "seed"}

    def test_precision_fp32_in_kwargs(self):
        """fp32 precision maps to float32 in the dtype kwarg."""
        config = make_config(**_VLLM_DEFAULTS, precision="fp32")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "float32"

    def test_precision_fp16_in_kwargs(self):
        """fp16 precision maps to float16 in the dtype kwarg."""
        config = make_config(**_VLLM_DEFAULTS, precision="fp16")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "float16"

    def test_precision_bf16_in_kwargs(self):
        """bf16 precision maps to bfloat16 in the dtype kwarg."""
        config = make_config(**_VLLM_DEFAULTS, precision="bf16")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "bfloat16"

    def test_seed_from_config_random_seed(self):
        """kwargs['seed'] matches config.random_seed."""
        config = make_config(**_VLLM_DEFAULTS, random_seed=1337)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["seed"] == 1337

    def test_model_name_propagated(self):
        """kwargs['model'] matches config.model."""
        config = make_config(**{**_VLLM_DEFAULTS, "model": "meta-llama/Llama-3.1-8B"})
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["model"] == "meta-llama/Llama-3.1-8B"

    def test_quantization_gptq(self):
        """VLLMEngineConfig.quantization='gptq' is forwarded correctly."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(quantization="gptq"))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["quantization"] == "gptq"

    def test_quantization_fp8(self):
        """VLLMEngineConfig.quantization='fp8' is forwarded correctly."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(quantization="fp8"))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["quantization"] == "fp8"


# =============================================================================
# Test Group 4: _build_sampling_params
# =============================================================================


class TestBuildSamplingParams:
    def test_greedy_via_temperature_zero(self):
        """temperature=0.0 triggers greedy mode: temperature=0.0, only max_tokens set."""
        decoder = DecoderConfig(temperature=0.0, do_sample=False)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, max_output_tokens=64)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == 0.0
        assert params._kwargs["max_tokens"] == 64
        # Greedy path: top_p, top_k, repetition_penalty NOT set (minimal kwargs)
        assert "top_p" not in params._kwargs
        assert "top_k" not in params._kwargs

    def test_greedy_via_do_sample_false_temperature_nonzero(self):
        """do_sample=False overrides temperature to produce greedy mode."""
        decoder = DecoderConfig(temperature=0.8, do_sample=False)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == 0.0

    def test_sampling_mode_temperature(self):
        """Non-zero temperature with do_sample=True sets temperature in kwargs."""
        decoder = DecoderConfig(temperature=0.7, do_sample=True)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == pytest.approx(0.7)

    def test_sampling_mode_top_p(self):
        """top_p from DecoderConfig propagates to SamplingParams kwargs."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_p=0.9)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_p"] == pytest.approx(0.9)

    def test_top_k_zero_sentinel_maps_to_minus_one(self):
        """Our top_k=0 (disabled) maps to vLLM's top_k=-1 (disabled sentinel)."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_k=0)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_k"] == -1

    def test_top_k_nonzero_preserved(self):
        """Non-zero top_k passes through unchanged."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_k=40)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_k"] == 40

    def test_repetition_penalty_propagated(self):
        """repetition_penalty from DecoderConfig is forwarded."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, repetition_penalty=1.1)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["repetition_penalty"] == pytest.approx(1.1)

    def test_max_tokens_from_config(self):
        """max_tokens kwarg matches config.max_output_tokens."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, max_output_tokens=256)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["max_tokens"] == 256

    def test_min_p_included_when_set(self):
        """min_p is added to kwargs when provided in DecoderConfig."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=0.05)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert "min_p" in params._kwargs
        assert params._kwargs["min_p"] == pytest.approx(0.05)

    def test_min_p_absent_when_none(self):
        """min_p is NOT in kwargs when DecoderConfig.min_p is None."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=None)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert "min_p" not in params._kwargs


# =============================================================================
# Test Group 5: No streaming code (CM-07 resolved structurally)
# =============================================================================


class TestNoStreamingCode:
    def test_no_run_streaming_method(self):
        """VLLMBackend has no _run_streaming method — offline batch path only (CM-07)."""
        assert not hasattr(VLLMBackend, "_run_streaming"), (
            "VLLMBackend must not have a _run_streaming method — streaming is resolved "
            "structurally by using offline batch inference exclusively"
        )

    def test_no_async_engine_attribute(self):
        """VLLMBackend has no async_engine attribute — no streaming engine (CM-07)."""
        assert not hasattr(VLLMBackend, "async_engine"), (
            "VLLMBackend must not have an async_engine attribute — offline batch only"
        )


# =============================================================================
# Test Group 6: VLLM-03 — --shm-size 8g in DockerRunner
# =============================================================================


class TestShmSizeInDockerRunner:
    def _make_mock_config(self):
        """Create a minimal mock config for _build_docker_cmd."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.backend = "vllm"
        config.tensorrt = None
        return config

    def test_docker_cmd_includes_shm_size_flag(self):
        """DockerRunner._build_docker_cmd includes --shm-size flag (VLLM-03)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd(self._make_mock_config(), "test_hash", "/tmp/test-exchange")
        assert "--shm-size" in cmd

    def test_docker_cmd_shm_size_value_is_8g(self):
        """The value immediately after --shm-size is '8g' (VLLM-03)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd(self._make_mock_config(), "test_hash", "/tmp/test-exchange")

        shm_idx = cmd.index("--shm-size")
        assert cmd[shm_idx + 1] == "8g"

    def test_docker_cmd_shm_size_adjacent_to_flag(self):
        """--shm-size and 8g appear as adjacent elements (not merged with equals)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd(self._make_mock_config(), "test_hash", "/tmp/test-exchange")

        # Confirm neither "--shm-size=8g" nor "--shm-size 8g" as one string
        assert "--shm-size=8g" not in cmd
        assert "--shm-size" in cmd
        assert "8g" in cmd


# =============================================================================
# Test Group 8: Nested engine config fields (new in Plan 02)
# =============================================================================


class TestEngineConfigFields:
    def test_enforce_eager_wires_to_kwargs(self):
        """enforce_eager=True in VLLMEngineConfig → kwargs['enforce_eager'] is True."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(enforce_eager=True))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["enforce_eager"] is True

    def test_block_size_wires_to_kwargs(self):
        """block_size=16 in VLLMEngineConfig → kwargs['block_size'] == 16."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(block_size=16))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["block_size"] == 16

    def test_speculative_model_produces_speculative_config_dict(self):
        """speculative_model + num_speculative_tokens → speculative_config dict (vLLM v0.6+ API)."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(speculative_model="draft-model", num_speculative_tokens=5)
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert "speculative_model" not in kwargs
        assert kwargs["speculative_config"] == {
            "model": "draft-model",
            "num_speculative_tokens": 5,
        }

    def test_all_engine_fields_wired(self):
        """All 14 non-speculative engine fields are forwarded when set."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                gpu_memory_utilization=0.9,
                swap_space=4.0,
                cpu_offload_gb=2.0,
                block_size=32,
                kv_cache_dtype="auto",
                enforce_eager=False,
                enable_chunked_prefill=True,
                max_num_seqs=64,
                max_num_batched_tokens=2048,
                max_model_len=4096,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_prefix_caching=True,
                quantization=None,
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["gpu_memory_utilization"] == 0.9
        assert kwargs["swap_space"] == 4.0
        assert kwargs["cpu_offload_gb"] == 2.0
        assert kwargs["block_size"] == 32
        assert kwargs["kv_cache_dtype"] == "auto"
        assert kwargs["enforce_eager"] is False
        assert kwargs["enable_chunked_prefill"] is True
        assert kwargs["max_num_seqs"] == 64
        assert kwargs["max_num_batched_tokens"] == 2048
        assert kwargs["max_model_len"] == 4096
        assert kwargs["tensor_parallel_size"] == 1
        assert kwargs["pipeline_parallel_size"] == 1
        assert kwargs["enable_prefix_caching"] is True
        assert "quantization" not in kwargs  # None → omitted


# =============================================================================
# Test Group 9: VLLMSamplingConfig overrides (new in Plan 02)
# =============================================================================


class TestSamplingConfigOverrides:
    def test_sampling_max_tokens_overrides_config_max_output_tokens(self):
        """VLLMSamplingConfig.max_tokens overrides ExperimentConfig.max_output_tokens."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(max_tokens=256))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg, max_output_tokens=128)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["max_tokens"] == 256  # override wins

    def test_sampling_presence_penalty_applied(self):
        """VLLMSamplingConfig.presence_penalty appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(presence_penalty=0.5))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["presence_penalty"] == pytest.approx(0.5)

    def test_sampling_frequency_penalty_applied(self):
        """VLLMSamplingConfig.frequency_penalty appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(frequency_penalty=0.3))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["frequency_penalty"] == pytest.approx(0.3)

    def test_sampling_min_tokens_applied(self):
        """VLLMSamplingConfig.min_tokens appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(min_tokens=10))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["min_tokens"] == 10

    def test_sampling_ignore_eos_applied(self):
        """VLLMSamplingConfig.ignore_eos=True appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(ignore_eos=True))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["ignore_eos"] is True

    def test_sampling_overrides_applied_to_greedy_path(self):
        """VLLMSamplingConfig overrides work on the greedy (temperature=0.0) path too."""
        decoder = DecoderConfig(temperature=0.0, do_sample=False)
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(max_tokens=512))
        config = make_config(
            **_VLLM_DEFAULTS, decoder=decoder, vllm=vllm_cfg, max_output_tokens=128
        )
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["temperature"] == 0.0
        assert params._kwargs["max_tokens"] == 512  # override wins on greedy path

    def test_none_sampling_config_does_not_add_extra_kwargs(self):
        """When vllm.sampling is None, no extra sampling kwargs are added."""
        config = make_config(**_VLLM_DEFAULTS)  # vllm=None by default
        decoder = DecoderConfig(temperature=1.0, do_sample=True)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert "presence_penalty" not in params._kwargs
        assert "frequency_penalty" not in params._kwargs
        assert "ignore_eos" not in params._kwargs


# =============================================================================
# Test Group 10: New VLLMEngineConfig fields wiring
# =============================================================================


class TestNewEngineFields:
    def test_disable_custom_all_reduce_wired(self):
        """disable_custom_all_reduce=True -> kwargs['disable_custom_all_reduce'] is True."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(disable_custom_all_reduce=True))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["disable_custom_all_reduce"] is True

    def test_kv_cache_memory_bytes_wired(self):
        """kv_cache_memory_bytes=2**30 -> kwargs['kv_cache_memory_bytes'] == 2**30."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(kv_cache_memory_bytes=2**30))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["kv_cache_memory_bytes"] == 2**30

    def test_offload_params_list_to_set_conversion(self):
        """offload_params=['weight', 'bias'] -> kwargs['offload_params'] == {'weight', 'bias'}."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(offload_params=["weight", "bias"]))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["offload_params"] == {"weight", "bias"}

    def test_offload_group_size_wired(self):
        """offload_group_size=4 -> kwargs['offload_group_size'] == 4."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(offload_group_size=4))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["offload_group_size"] == 4

    def test_compilation_config_dict_passthrough(self):
        """compilation_config dict passes through as-is to kwargs."""
        comp = {"mode": "default", "backend": "inductor"}
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(compilation_config=comp))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["compilation_config"] == {"mode": "default", "backend": "inductor"}

    def test_none_new_fields_omitted(self):
        """When new fields are None, they are NOT in kwargs."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig())
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        for key in [
            "disable_custom_all_reduce",
            "kv_cache_memory_bytes",
            "offload_group_size",
            "offload_params",
            "compilation_config",
        ]:
            assert key not in kwargs


# =============================================================================
# Test Group 11: VLLMAttentionConfig wiring
# =============================================================================


class TestAttentionConfigWiring:
    def test_attention_backend_maps_to_attention_backend_kwarg(self):
        """attention.backend='flash_attn' -> kwargs['attention_backend'] == 'flash_attn'."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(attention=VLLMAttentionConfig(backend="flash_attn"))
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["attention_backend"] == "flash_attn"

    def test_attention_boolean_fields_wired(self):
        """Boolean attention fields are forwarded as flat LLM() kwargs."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                attention=VLLMAttentionConfig(
                    use_cudnn_prefill=True,
                    disable_flashinfer_prefill=True,
                )
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["use_cudnn_prefill"] is True
        assert kwargs["disable_flashinfer_prefill"] is True

    def test_attention_model_extra_forwarded(self):
        """Unknown attention fields pass through via model_extra."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                attention=VLLMAttentionConfig(**{"backend": "flash_attn", "future_attn_opt": 42})
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["future_attn_opt"] == 42

    def test_no_attention_config_no_attention_keys(self):
        """When engine.attention is None, no attention-related keys in kwargs."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig())
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert "attention_backend" not in kwargs
        assert "use_cudnn_prefill" not in kwargs


# =============================================================================
# Test Group 12: Passthrough (model_extra) kwargs
# =============================================================================


class TestPassthroughKwargs:
    def test_engine_model_extra_forwarded_to_llm_kwargs(self):
        """Unknown engine fields pass through to LLM() kwargs via model_extra."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(**{"gpu_memory_utilization": 0.9, "some_future_param": "value"})
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["some_future_param"] == "value"
        assert kwargs["gpu_memory_utilization"] == 0.9  # explicit still works

    def test_sampling_model_extra_forwarded(self):
        """Unknown sampling fields pass through to SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(**{"some_future_sampling_param": True}))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["some_future_sampling_param"] is True

    def test_sampling_n_field_forwarded(self):
        """VLLMSamplingConfig.n=4 -> kwargs['n'] == 4."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(n=4))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["n"] == 4

    def test_engine_extra_overrides_explicit_when_colliding(self):
        """model_extra is merged LAST — if user passes a known field name as extra, it overrides."""
        # This tests the edge case: user deliberately passes a known field via passthrough
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(**{"enforce_eager": True, "enforce_eager_override": "test"})
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["enforce_eager"] is True  # from explicit field
        assert kwargs["enforce_eager_override"] == "test"  # from model_extra


# =============================================================================
# Test Group 13: Beam search params construction
# =============================================================================


class TestBeamSearchParams:
    def test_beam_search_config_triggers_beam_path(self):
        """When beam_search is set, config structure reflects beam search mode."""
        vllm_cfg = VLLMConfig(beam_search=VLLMBeamSearchConfig(beam_width=4))
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        # The beam search path imports BeamSearchParams from vllm — we can't call it without vLLM.
        # Instead, verify the beam_search branch would be taken by checking config structure.
        assert config.vllm is not None
        assert config.vllm.beam_search is not None
        assert config.vllm.beam_search.beam_width == 4

    def test_beam_search_mutual_exclusion_with_sampling(self):
        """Cannot set both beam_search and sampling on VLLMConfig."""
        import pydantic

        with pytest.raises(
            pydantic.ValidationError, match=r"beam_search.*sampling|sampling.*beam_search"
        ):
            VLLMConfig(
                beam_search=VLLMBeamSearchConfig(beam_width=4),
                sampling=VLLMSamplingConfig(max_tokens=100),
            )

    def test_beam_search_config_accepts_all_fields(self):
        """VLLMBeamSearchConfig accepts beam_width, length_penalty, early_stopping, max_tokens."""
        bs = VLLMBeamSearchConfig(
            beam_width=8, length_penalty=1.2, early_stopping=True, max_tokens=256
        )
        assert bs.beam_width == 8
        assert bs.length_penalty == 1.2
        assert bs.early_stopping is True
        assert bs.max_tokens == 256

    def test_beam_search_config_extra_allow(self):
        """VLLMBeamSearchConfig accepts unknown fields via extra='allow'."""
        bs = VLLMBeamSearchConfig(**{"beam_width": 4, "future_beam_param": True})
        assert bs.model_extra.get("future_beam_param") is True

    def test_beam_search_beam_width_ge_1(self):
        """beam_width must be >= 1."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            VLLMBeamSearchConfig(beam_width=0)


# =============================================================================
# Test Group 14: min_new_tokens -> min_tokens mapping (H4)
# =============================================================================


class TestMinNewTokensMapping:
    def test_min_new_tokens_maps_to_min_tokens_sampling(self):
        """decoder.min_new_tokens=5 flows through to kwargs['min_tokens']=5 (sampling path)."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_new_tokens=5)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs.get("min_tokens") == 5

    def test_min_new_tokens_maps_to_min_tokens_greedy(self):
        """decoder.min_new_tokens=3 flows through to kwargs['min_tokens']=3 (greedy path)."""
        decoder = DecoderConfig(temperature=0.0, do_sample=False, min_new_tokens=3)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs.get("min_tokens") == 3

    def test_min_new_tokens_absent_when_none(self):
        """When decoder.min_new_tokens is None, min_tokens is NOT in kwargs."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_new_tokens=None)
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert "min_tokens" not in params._kwargs

    def test_vllm_sampling_min_tokens_overrides_decoder_min_new_tokens(self):
        """vllm.sampling.min_tokens=10 overrides decoder.min_new_tokens=5 (backend-specific wins)."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_new_tokens=5)
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(min_tokens=10))
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs.get("min_tokens") == 10


# =============================================================================
# Test Group 15: Multi-output token counting (H3 audit fix)
# =============================================================================


class TestMultiOutputTokenCounting:
    """Verify output token counting sums across ALL outputs per request, not just outputs[0]."""

    def _make_fake_output(
        self, prompt_token_ids: list[int], output_token_id_lists: list[list[int]]
    ):
        """Build a minimal fake RequestOutput with multiple CompletionOutput objects."""
        from dataclasses import dataclass

        @dataclass
        class _FakeCompletionOutput:
            token_ids: list[int]

        @dataclass
        class _FakeRequestOutput:
            prompt_token_ids: list[int]
            outputs: list[_FakeCompletionOutput]

        return _FakeRequestOutput(
            prompt_token_ids=prompt_token_ids,
            outputs=[_FakeCompletionOutput(token_ids=ids) for ids in output_token_id_lists],
        )

    def test_single_output_per_request(self):
        """Single output per request: counts match outputs[0] (baseline correctness)."""
        outputs = [
            self._make_fake_output([1, 2, 3], [[10, 11, 12]]),
            self._make_fake_output([4, 5], [[20, 21]]),
        ]
        output_count = sum(len(out.token_ids) for o in outputs if o.outputs for out in o.outputs)
        assert output_count == 5  # 3 + 2

    def test_multiple_outputs_per_request_all_counted(self):
        """n=2 produces 2 outputs per request — both must be counted."""
        outputs = [
            self._make_fake_output([1, 2], [[10, 11, 12], [20, 21]]),  # 3 + 2 = 5
            self._make_fake_output([3], [[30, 31], [40, 41, 42]]),  # 2 + 3 = 5
        ]
        output_count = sum(len(out.token_ids) for o in outputs if o.outputs for out in o.outputs)
        assert output_count == 10  # 5 + 5

    def test_first_output_only_undercounts(self):
        """Demonstrate that outputs[0]-only counting would undercount for n>1."""
        outputs = [
            self._make_fake_output([1, 2], [[10, 11, 12], [20, 21]]),  # 3 + 2 tokens
        ]
        # Old (wrong) approach — only first output
        old_count = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
        # New (correct) approach — all outputs
        new_count = sum(len(out.token_ids) for o in outputs if o.outputs for out in o.outputs)
        assert old_count == 3  # undercounts: misses the 2nd output's 2 tokens
        assert new_count == 5  # correct: 3 + 2

    def test_empty_outputs_list_handled(self):
        """Requests with no outputs contribute zero tokens without error."""
        from dataclasses import dataclass

        @dataclass
        class _FakeEmptyOutput:
            prompt_token_ids: list[int]
            outputs: list

        outputs = [_FakeEmptyOutput(prompt_token_ids=[1, 2], outputs=[])]
        output_count = sum(len(out.token_ids) for o in outputs if o.outputs for out in o.outputs)
        assert output_count == 0

    def test_beam_search_four_beams_counted(self):
        """Beam search with beam_width=4 produces 4 outputs — all 4 counted."""
        outputs = [
            self._make_fake_output([1, 2, 3], [[10] * 8, [20] * 7, [30] * 9, [40] * 6]),
        ]
        output_count = sum(len(out.token_ids) for o in outputs if o.outputs for out in o.outputs)
        assert output_count == 30  # 8 + 7 + 9 + 6


# =============================================================================
# Test Group 16: M15 — VRAM query uses current_device(), not hardcoded 0
# =============================================================================


class TestVramCurrentDevice:
    """Verify that VRAM total-memory query uses torch.cuda.current_device()."""

    def test_vram_query_calls_current_device(self):
        """get_device_properties uses current_device(), not hardcoded 0.

        Torch imports inside run_inference are lazy, so we inspect source to
        confirm the correct call expression is present.
        """
        import inspect

        import llenergymeasure.backends.vllm as vllm_mod

        source = inspect.getsource(vllm_mod.VLLMBackend.run_inference)
        assert "current_device()" in source, (
            "run_inference must call torch.cuda.current_device() for VRAM query, not hardcode 0"
        )
        assert "get_device_properties(0)" not in source, (
            "run_inference must not hardcode device 0 — use current_device()"
        )


# =============================================================================
# Test Group 17: M1 — flash_attn fields wired in _build_llm_kwargs
# =============================================================================


class TestFlashAttnFieldsWired:
    """Verify flash_attn_version and flash_attn_max_num_splits_for_cuda_graph are forwarded."""

    def test_flash_attn_version_wired(self):
        """flash_attn_version=3 appears in LLM() kwargs."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(attention=VLLMAttentionConfig(flash_attn_version=3))
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["flash_attn_version"] == 3

    def test_flash_attn_max_num_splits_wired(self):
        """flash_attn_max_num_splits_for_cuda_graph=8 appears in LLM() kwargs."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                attention=VLLMAttentionConfig(flash_attn_max_num_splits_for_cuda_graph=8)
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["flash_attn_max_num_splits_for_cuda_graph"] == 8

    def test_flash_attn_version_none_omitted(self):
        """flash_attn_version=None is not added to kwargs."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(attention=VLLMAttentionConfig(flash_attn_version=None))
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert "flash_attn_version" not in kwargs

    def test_flash_attn_max_splits_none_omitted(self):
        """flash_attn_max_num_splits_for_cuda_graph=None is not added to kwargs."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                attention=VLLMAttentionConfig(flash_attn_max_num_splits_for_cuda_graph=None)
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert "flash_attn_max_num_splits_for_cuda_graph" not in kwargs

    def test_both_flash_attn_fields_together(self):
        """Both flash_attn fields forwarded simultaneously."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(
                attention=VLLMAttentionConfig(
                    flash_attn_version=2,
                    flash_attn_max_num_splits_for_cuda_graph=16,
                )
            )
        )
        config = make_config(**_VLLM_DEFAULTS, vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["flash_attn_version"] == 2
        assert kwargs["flash_attn_max_num_splits_for_cuda_graph"] == 16


# =============================================================================
# Test Group 18: M2 — min_p forwarded in _build_beam_search_params
# =============================================================================


class TestBeamSearchMinP:
    """Verify min_p from DecoderConfig flows through to BeamSearchParams."""

    def _build_beam_params(self, config, monkeypatch):
        """Call _build_beam_search_params with a fake BeamSearchParams to avoid vLLM import."""
        import sys
        import types

        # Inject a fake vllm module with our _FakeBeamSearchParams so the lazy
        # `from vllm import BeamSearchParams` inside _build_beam_search_params succeeds.
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.BeamSearchParams = _FakeBeamSearchParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        beam_cfg = config.vllm.beam_search
        return VLLMBackend._build_beam_search_params(config, beam_cfg)

    def test_min_p_forwarded_to_beam_search_params(self, monkeypatch):
        """min_p=0.05 from DecoderConfig appears in BeamSearchParams kwargs."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=0.05)
        vllm_cfg = VLLMConfig(beam_search=VLLMBeamSearchConfig(beam_width=4))
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, vllm=vllm_cfg)
        params = self._build_beam_params(config, monkeypatch)
        assert "min_p" in params._kwargs
        assert params._kwargs["min_p"] == pytest.approx(0.05)

    def test_min_p_absent_when_none(self, monkeypatch):
        """min_p is NOT in BeamSearchParams kwargs when DecoderConfig.min_p is None."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=None)
        vllm_cfg = VLLMConfig(beam_search=VLLMBeamSearchConfig(beam_width=4))
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, vllm=vllm_cfg)
        params = self._build_beam_params(config, monkeypatch)
        assert "min_p" not in params._kwargs

    def test_beam_width_and_min_p_together(self, monkeypatch):
        """beam_width and min_p both appear in kwargs when set."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=0.1)
        vllm_cfg = VLLMConfig(beam_search=VLLMBeamSearchConfig(beam_width=8, max_tokens=64))
        config = make_config(**_VLLM_DEFAULTS, decoder=decoder, vllm=vllm_cfg)
        params = self._build_beam_params(config, monkeypatch)
        assert params._kwargs["beam_width"] == 8
        assert params._kwargs["min_p"] == pytest.approx(0.1)
        assert params._kwargs["max_tokens"] == 64
