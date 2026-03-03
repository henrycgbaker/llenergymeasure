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
  - _prepare_prompts returns config.n prompt strings
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest

from llenergymeasure.config.backend_configs import VLLMConfig, VLLMEngineConfig, VLLMSamplingConfig
from llenergymeasure.config.models import DecoderConfig, ExperimentConfig
from llenergymeasure.core.backends.vllm import VLLMBackend
from llenergymeasure.exceptions import BackendError

# =============================================================================
# Helpers
# =============================================================================


def _make_config(**overrides) -> ExperimentConfig:
    """Return a minimal valid ExperimentConfig with backend='vllm'."""
    defaults: dict = {"model": "test-model", "backend": "vllm"}
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


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


# =============================================================================
# Test Group 1: Protocol compliance and registration
# =============================================================================


class TestProtocolCompliance:
    def test_vllm_backend_name(self):
        """VLLMBackend.name returns 'vllm'."""
        backend = VLLMBackend()
        assert backend.name == "vllm"

    def test_vllm_backend_satisfies_protocol(self):
        """VLLMBackend satisfies the runtime_checkable InferenceBackend protocol."""
        from llenergymeasure.core.backends.protocol import InferenceBackend

        backend = VLLMBackend()
        assert isinstance(backend, InferenceBackend)

    def test_get_backend_returns_vllm_instance(self):
        """get_backend('vllm') returns a VLLMBackend with name 'vllm'."""
        from llenergymeasure.core.backends import get_backend

        backend = get_backend("vllm")
        assert backend.name == "vllm"
        assert isinstance(backend, VLLMBackend)

    def test_get_backend_unknown_mentions_vllm_in_error(self):
        """get_backend('unknown') error message lists vllm as available."""
        from llenergymeasure.core.backends import get_backend

        with pytest.raises(BackendError, match="vllm"):
            get_backend("unknown")

    def test_get_backend_unknown_raises_backend_error(self):
        """get_backend with unknown name raises BackendError (not KeyError, etc.)."""
        from llenergymeasure.core.backends import get_backend

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
        config = _make_config()
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["model"] == "test-model"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["seed"] == 42
        assert "dtype" in kwargs

    def test_minimal_config_dtype_reflects_precision(self):
        """Default precision (bf16) maps to bfloat16 in kwargs."""
        config = _make_config()
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
        config = _make_config(vllm=vllm_cfg)
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
        config = _make_config(vllm=vllm_cfg)
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2
        assert "gpu_memory_utilization" not in kwargs
        assert "max_num_seqs" not in kwargs
        assert "enable_prefix_caching" not in kwargs
        assert "quantization" not in kwargs

    def test_no_vllm_section_produces_no_extra_keys(self):
        """When config.vllm is None, only the 4 base keys are present."""
        config = _make_config()  # vllm=None by default
        backend = VLLMBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert set(kwargs.keys()) == {"model", "dtype", "trust_remote_code", "seed"}

    def test_precision_fp32_in_kwargs(self):
        """fp32 precision maps to float32 in the dtype kwarg."""
        config = _make_config(precision="fp32")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "float32"

    def test_precision_fp16_in_kwargs(self):
        """fp16 precision maps to float16 in the dtype kwarg."""
        config = _make_config(precision="fp16")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "float16"

    def test_precision_bf16_in_kwargs(self):
        """bf16 precision maps to bfloat16 in the dtype kwarg."""
        config = _make_config(precision="bf16")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["dtype"] == "bfloat16"

    def test_seed_from_config_random_seed(self):
        """kwargs['seed'] matches config.random_seed."""
        config = _make_config(random_seed=1337)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["seed"] == 1337

    def test_model_name_propagated(self):
        """kwargs['model'] matches config.model."""
        config = _make_config(model="meta-llama/Llama-3.1-8B")
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["model"] == "meta-llama/Llama-3.1-8B"

    def test_quantization_gptq(self):
        """VLLMEngineConfig.quantization='gptq' is forwarded correctly."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(quantization="gptq"))
        config = _make_config(vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["quantization"] == "gptq"

    def test_quantization_fp8(self):
        """VLLMEngineConfig.quantization='fp8' is forwarded correctly."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(quantization="fp8"))
        config = _make_config(vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["quantization"] == "fp8"


# =============================================================================
# Test Group 4: _build_sampling_params
# =============================================================================


class TestBuildSamplingParams:
    def test_greedy_via_temperature_zero(self):
        """temperature=0.0 triggers greedy mode: temperature=0.0, only max_tokens set."""
        decoder = DecoderConfig(temperature=0.0, do_sample=False)
        config = _make_config(decoder=decoder, max_output_tokens=64)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == 0.0
        assert params._kwargs["max_tokens"] == 64
        # Greedy path: top_p, top_k, repetition_penalty NOT set (minimal kwargs)
        assert "top_p" not in params._kwargs
        assert "top_k" not in params._kwargs

    def test_greedy_via_do_sample_false_temperature_nonzero(self):
        """do_sample=False overrides temperature to produce greedy mode."""
        decoder = DecoderConfig(temperature=0.8, do_sample=False)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == 0.0

    def test_sampling_mode_temperature(self):
        """Non-zero temperature with do_sample=True sets temperature in kwargs."""
        decoder = DecoderConfig(temperature=0.7, do_sample=True)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["temperature"] == pytest.approx(0.7)

    def test_sampling_mode_top_p(self):
        """top_p from DecoderConfig propagates to SamplingParams kwargs."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_p=0.9)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_p"] == pytest.approx(0.9)

    def test_top_k_zero_sentinel_maps_to_minus_one(self):
        """Our top_k=0 (disabled) maps to vLLM's top_k=-1 (disabled sentinel)."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_k=0)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_k"] == -1

    def test_top_k_nonzero_preserved(self):
        """Non-zero top_k passes through unchanged."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, top_k=40)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["top_k"] == 40

    def test_repetition_penalty_propagated(self):
        """repetition_penalty from DecoderConfig is forwarded."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, repetition_penalty=1.1)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["repetition_penalty"] == pytest.approx(1.1)

    def test_max_tokens_from_config(self):
        """max_tokens kwarg matches config.max_output_tokens."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True)
        config = _make_config(decoder=decoder, max_output_tokens=256)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert params._kwargs["max_tokens"] == 256

    def test_min_p_included_when_set(self):
        """min_p is added to kwargs when provided in DecoderConfig."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=0.05)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert "min_p" in params._kwargs
        assert params._kwargs["min_p"] == pytest.approx(0.05)

    def test_min_p_absent_when_none(self):
        """min_p is NOT in kwargs when DecoderConfig.min_p is None."""
        decoder = DecoderConfig(temperature=1.0, do_sample=True, min_p=None)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)

        assert "min_p" not in params._kwargs


# =============================================================================
# Test Group 5: No streaming code (CM-07 resolved structurally)
# =============================================================================


class TestNoStreamingCode:
    def test_no_streaming_api_call_in_source(self):
        """vllm.py has no streaming API calls — CM-07 resolved structurally.

        The docstring mentions 'streaming' as context, but no streaming API
        argument (stream=True/False, AsyncEngineArgs, etc.) appears in code.
        """
        source = inspect.getsource(VLLMBackend)
        # No streaming API arguments passed to any vLLM call
        assert "stream=True" not in source
        assert "AsyncEngine" not in source
        assert "async_engine" not in source

    def test_no_run_streaming_method(self):
        """VLLMBackend has no _run_streaming method — offline batch only."""
        source = inspect.getsource(VLLMBackend)
        assert "_run_streaming" not in source

    def test_no_per_prompt_streaming_loop(self):
        """Source uses a single generate() call — not a per-prompt streaming loop."""
        source = inspect.getsource(VLLMBackend)
        # Streaming-based per-prompt loops use AsyncLLMEngine or stream= argument
        assert "stream=True" not in source
        assert "stream=False" not in source


# =============================================================================
# Test Group 6: VLLM-03 — --shm-size 8g in DockerRunner
# =============================================================================


class TestShmSizeInDockerRunner:
    def test_docker_cmd_includes_shm_size_flag(self):
        """DockerRunner._build_docker_cmd includes --shm-size flag (VLLM-03)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd("test_hash", "/tmp/test-exchange")
        assert "--shm-size" in cmd

    def test_docker_cmd_shm_size_value_is_8g(self):
        """The value immediately after --shm-size is '8g' (VLLM-03)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd("test_hash", "/tmp/test-exchange")

        shm_idx = cmd.index("--shm-size")
        assert cmd[shm_idx + 1] == "8g"

    def test_docker_cmd_shm_size_adjacent_to_flag(self):
        """--shm-size and 8g appear as adjacent elements (not merged with equals)."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        runner = DockerRunner(image="test-image")
        cmd = runner._build_docker_cmd("test_hash", "/tmp/test-exchange")

        # Confirm neither "--shm-size=8g" nor "--shm-size 8g" as one string
        assert "--shm-size=8g" not in cmd
        assert "--shm-size" in cmd
        assert "8g" in cmd


# =============================================================================
# Test Group 7: _prepare_prompts
# =============================================================================


class TestPreparePrompts:
    def test_returns_n_prompts(self):
        """_prepare_prompts returns exactly config.n prompt strings."""
        config = _make_config(n=5)
        backend = VLLMBackend()
        prompts = backend._prepare_prompts(config)

        assert len(prompts) == 5

    def test_all_prompts_are_strings(self):
        """Every returned prompt is a non-empty string."""
        config = _make_config(n=3)
        backend = VLLMBackend()
        prompts = backend._prepare_prompts(config)

        assert all(isinstance(p, str) and len(p) > 0 for p in prompts)

    def test_n_equals_one(self):
        """_prepare_prompts handles n=1 correctly."""
        config = _make_config(n=1)
        backend = VLLMBackend()
        prompts = backend._prepare_prompts(config)

        assert len(prompts) == 1
        assert isinstance(prompts[0], str)

    def test_n_equals_100(self):
        """_prepare_prompts handles n=100 (default) correctly."""
        config = _make_config(n=100)
        backend = VLLMBackend()
        prompts = backend._prepare_prompts(config)

        assert len(prompts) == 100

    def test_prompts_reflect_max_input_tokens(self):
        """Prompt length scales with max_input_tokens (M1 placeholder behaviour)."""
        config_short = _make_config(n=1, max_input_tokens=4)
        config_long = _make_config(n=1, max_input_tokens=512)
        backend = VLLMBackend()

        short_prompts = backend._prepare_prompts(config_short)
        long_prompts = backend._prepare_prompts(config_long)

        # Longer max_input_tokens → longer prompt string
        assert len(long_prompts[0]) > len(short_prompts[0])


# =============================================================================
# Test Group 8: Nested engine config fields (new in Plan 02)
# =============================================================================


class TestEngineConfigFields:
    def test_enforce_eager_wires_to_kwargs(self):
        """enforce_eager=True in VLLMEngineConfig → kwargs['enforce_eager'] is True."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(enforce_eager=True))
        config = _make_config(vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["enforce_eager"] is True

    def test_block_size_wires_to_kwargs(self):
        """block_size=16 in VLLMEngineConfig → kwargs['block_size'] == 16."""
        vllm_cfg = VLLMConfig(engine=VLLMEngineConfig(block_size=16))
        config = _make_config(vllm=vllm_cfg)
        kwargs = VLLMBackend()._build_llm_kwargs(config)
        assert kwargs["block_size"] == 16

    def test_speculative_model_produces_speculative_config_dict(self):
        """speculative_model + num_speculative_tokens → speculative_config dict (vLLM v0.6+ API)."""
        vllm_cfg = VLLMConfig(
            engine=VLLMEngineConfig(speculative_model="draft-model", num_speculative_tokens=5)
        )
        config = _make_config(vllm=vllm_cfg)
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
        config = _make_config(vllm=vllm_cfg)
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
        config = _make_config(vllm=vllm_cfg, max_output_tokens=128)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["max_tokens"] == 256  # override wins

    def test_sampling_presence_penalty_applied(self):
        """VLLMSamplingConfig.presence_penalty appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(presence_penalty=0.5))
        config = _make_config(vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["presence_penalty"] == pytest.approx(0.5)

    def test_sampling_frequency_penalty_applied(self):
        """VLLMSamplingConfig.frequency_penalty appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(frequency_penalty=0.3))
        config = _make_config(vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["frequency_penalty"] == pytest.approx(0.3)

    def test_sampling_min_tokens_applied(self):
        """VLLMSamplingConfig.min_tokens appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(min_tokens=10))
        config = _make_config(vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["min_tokens"] == 10

    def test_sampling_ignore_eos_applied(self):
        """VLLMSamplingConfig.ignore_eos=True appears in SamplingParams kwargs."""
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(ignore_eos=True))
        config = _make_config(vllm=vllm_cfg)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["ignore_eos"] is True

    def test_sampling_overrides_applied_to_greedy_path(self):
        """VLLMSamplingConfig overrides work on the greedy (temperature=0.0) path too."""
        decoder = DecoderConfig(temperature=0.0, do_sample=False)
        vllm_cfg = VLLMConfig(sampling=VLLMSamplingConfig(max_tokens=512))
        config = _make_config(decoder=decoder, vllm=vllm_cfg, max_output_tokens=128)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert params._kwargs["temperature"] == 0.0
        assert params._kwargs["max_tokens"] == 512  # override wins on greedy path

    def test_none_sampling_config_does_not_add_extra_kwargs(self):
        """When vllm.sampling is None, no extra sampling kwargs are added."""
        config = _make_config()  # vllm=None by default
        decoder = DecoderConfig(temperature=1.0, do_sample=True)
        config = _make_config(decoder=decoder)
        params = VLLMBackend._build_sampling_params(config, _FakeSamplingParams)
        assert "presence_penalty" not in params._kwargs
        assert "frequency_penalty" not in params._kwargs
        assert "ignore_eos" not in params._kwargs
