"""vLLM inference backend — thin BackendPlugin.

Implements the 4-method BackendPlugin protocol:
  load_model, warmup, run_inference, cleanup

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only vLLM-specific inference: model loading via vllm.LLM(), a minimal
1-prompt warmup, offline batch llm.generate(), and cleanup.

All vLLM and torch imports are lazy so this module can be imported on
hosts without vLLM or CUDA installed.
"""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Callable
from typing import Any

from llenergymeasure.backends.protocol import InferenceOutput
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.utils.exceptions import BackendError

logger = logging.getLogger(__name__)


class VLLMBackend:
    """vLLM inference backend — offline batch mode, thin plugin.

    Implements BackendPlugin:
    - load_model: Load model via vllm.LLM(), build SamplingParams
    - warmup: Minimal 1-prompt warmup with 1-token output
    - run_inference: Single llm.generate() call with ALL prompts, returns InferenceOutput
    - cleanup: Delete LLM instance, gc.collect(), clear CUDA cache
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "vllm"

    # -------------------------------------------------------------------------
    # BackendPlugin: load_model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> tuple[Any, Any]:
        """Load model via vllm.LLM() and build SamplingParams.

        All vLLM imports are lazy so this module can be imported without vLLM.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (llm, sampling_params).

        Raises:
            BackendError: If vLLM is not installed or model loading fails.
        """
        from llenergymeasure.backends._helpers import require_import

        _vllm = require_import("vllm", "vllm")
        LLM = _vllm.LLM
        SamplingParams = _vllm.SamplingParams

        kwargs = self._build_llm_kwargs(config)
        logger.info(
            "Loading model %r with vllm.LLM (kwargs: %s)", config.model, list(kwargs.keys())
        )

        try:
            t0 = _time.perf_counter()
            llm = LLM(**kwargs)
            if on_substep is not None:
                on_substep("vLLM engine loaded", _time.perf_counter() - t0)
        except Exception as e:
            raise BackendError(f"vLLM model loading failed: {e}") from e

        logger.debug("vLLM model loaded successfully")

        # Build SamplingParams or BeamSearchParams depending on config
        if config.vllm is not None and config.vllm.beam_search is not None:
            sampling_params = self._build_beam_search_params(config, config.vllm.beam_search)
        else:
            sampling_params = self._build_sampling_params(config, SamplingParams)
        if on_substep is not None:
            on_substep("sampling params built", 0.0)
        return llm, sampling_params

    # -------------------------------------------------------------------------
    # BackendPlugin: warmup
    # -------------------------------------------------------------------------

    def warmup(self, config: ExperimentConfig, model: Any, prompts: list[str]) -> WarmupResult:
        """Run minimal vLLM warmup: 1 prompt, 1 token.

        thermal_floor_wait_s is NOT set here — MeasurementHarness sets it after
        this method returns.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            WarmupResult. thermal_floor_wait_s is left at default 0.0 (set by harness).
        """
        llm, _sampling_params = model

        if config.warmup.enabled:
            logger.debug("Running vLLM warmup (1 prompt, 1 token)...")
            from vllm import SamplingParams as _SP

            from llenergymeasure.backends._helpers import warmup_single_token

            warmup_single_token(llm, prompts, _SP, max_tokens=1)
            logger.debug("vLLM warmup complete")

        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1 if config.warmup.enabled else 0,
            target_cv=config.warmup.cv_threshold,
            max_prompts=config.warmup.max_prompts,
        )

    # -------------------------------------------------------------------------
    # BackendPlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run offline batch inference over all prompts.

        Single llm.generate() call with ALL prompts — no streaming, no
        one-at-a-time loops.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            BackendError: On OOM or other inference failures.
        """
        import time

        from llenergymeasure.backends._helpers import reset_cuda_peak_memory

        llm, sampling_params = model

        # Reset peak stats before the measurement loop so max_memory_allocated() below
        # captures inference-window peak (KV cache occupancy + activations), not pre-allocation.
        reset_cuda_peak_memory()

        logger.info(
            "Starting vLLM offline batch inference: %d prompts, max_tokens=%d",
            len(prompts),
            config.max_output_tokens,
        )

        try:
            t0 = time.perf_counter()
            # BeamSearchParams uses llm.beam_search(), SamplingParams uses llm.generate().
            # Guard import — BeamSearchParams was added in vLLM >=0.8; older versions
            # (e.g. 0.7.3 in the v0.9.0 container image) don't export it.
            try:
                from vllm import BeamSearchParams as _BSP
            except ImportError:
                _BSP = None  # type: ignore[assignment,misc]

            if _BSP is not None and isinstance(sampling_params, _BSP):
                outputs = llm.beam_search(prompts, sampling_params)
            else:
                outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0

        except Exception as e:
            from llenergymeasure.backends._helpers import raise_backend_error

            raise_backend_error(
                e,
                "vLLM",
                hint="reduce n, use gpu_memory_utilization=0.8, or use a smaller model.",
            )

        # Capture peak memory — torch first, NVML fallback for pre-allocation detection.
        from llenergymeasure.backends._helpers import get_cuda_peak_memory_mb

        peak_mb = get_cuda_peak_memory_mb()

        # Heuristic: if peak matches gpu_memory_utilization * total_vram within 5%,
        # it's likely pre-allocation, not actual usage. Fall back to NVML.
        if peak_mb > 0:
            try:
                import torch

                total_vram = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory / (1024 * 1024)
                vllm_cfg = config.vllm
                gpu_util = 0.9  # vLLM default
                if (
                    vllm_cfg is not None
                    and vllm_cfg.engine is not None
                    and vllm_cfg.engine.gpu_memory_utilization is not None
                ):
                    gpu_util = vllm_cfg.engine.gpu_memory_utilization
                expected_prealloc = total_vram * gpu_util
                if abs(peak_mb - expected_prealloc) / expected_prealloc < 0.05:
                    logger.debug(
                        "torch peak (%.1fMB) matches pre-allocation (%.1fMB), trying NVML",
                        peak_mb,
                        expected_prealloc,
                    )
                    nvml_peak = self._nvml_peak_memory_mb()
                    if nvml_peak is not None:
                        peak_mb = nvml_peak
            except Exception:
                pass  # Stick with torch value

        # Count tokens from RequestOutput objects.
        # Sum across ALL outputs per request (n>1 or beam search produces multiple).
        input_token_count = sum(len(o.prompt_token_ids) for o in outputs)
        output_token_count = sum(
            len(out.token_ids) for o in outputs if o.outputs for out in o.outputs
        )

        logger.debug(
            "vLLM inference complete: %d total tokens (in=%d, out=%d) in %.2fs",
            input_token_count + output_token_count,
            input_token_count,
            output_token_count,
            elapsed,
        )

        # Attempt to expose HuggingFace model for FLOPs estimation.
        # vllm.LLM has llm_engine.model_executor.driver_worker.model_runner.model
        # This is an internal API path — stash in extras, harness will attempt FLOPs.
        hf_model = None
        with contextlib.suppress(Exception):
            hf_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        return InferenceOutput(
            elapsed_time_sec=elapsed,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            peak_memory_mb=peak_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=[elapsed],
            extras={"hf_model": hf_model} if hf_model is not None else {},
        )

    # -------------------------------------------------------------------------
    # BackendPlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release vLLM model from memory and clear CUDA cache.

        Args:
            model: Tuple of (llm, sampling_params) from load_model().
        """
        from llenergymeasure.backends._helpers import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("vLLM model cleanup complete")

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """No hardware validation required for vLLM backend."""
        return []

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for vllm.LLM() constructor."""
        kwargs: dict[str, Any] = {
            "model": config.model,
            "dtype": self._map_precision(config.precision),
            "trust_remote_code": True,
            "seed": config.random_seed,
        }

        # Apply VLLMEngineConfig fields if provided — only set non-None values
        vllm_cfg = config.vllm
        if vllm_cfg is not None and vllm_cfg.engine is not None:
            engine = vllm_cfg.engine

            def _set(k: str, v: Any) -> None:
                if v is not None:
                    kwargs[k] = v

            _set("gpu_memory_utilization", engine.gpu_memory_utilization)
            _set("swap_space", engine.swap_space)
            _set("cpu_offload_gb", engine.cpu_offload_gb)
            _set("block_size", engine.block_size)
            _set("kv_cache_dtype", engine.kv_cache_dtype)
            _set("enforce_eager", engine.enforce_eager)
            _set("enable_chunked_prefill", engine.enable_chunked_prefill)
            _set("max_num_seqs", engine.max_num_seqs)
            _set("max_num_batched_tokens", engine.max_num_batched_tokens)
            _set("max_model_len", engine.max_model_len)
            _set("tensor_parallel_size", engine.tensor_parallel_size)
            _set("pipeline_parallel_size", engine.pipeline_parallel_size)
            _set("enable_prefix_caching", engine.enable_prefix_caching)
            _set("quantization", engine.quantization)

            if engine.speculative_model is not None:
                # vLLM v0.6+ uses speculative_config dict, not direct speculative_model kwarg
                kwargs["speculative_config"] = {
                    "model": engine.speculative_model,
                    "num_speculative_tokens": engine.num_speculative_tokens,
                }

            _set("disable_custom_all_reduce", engine.disable_custom_all_reduce)
            _set("kv_cache_memory_bytes", engine.kv_cache_memory_bytes)
            _set("offload_group_size", engine.offload_group_size)
            _set("offload_num_in_group", engine.offload_num_in_group)
            _set("offload_prefetch_step", engine.offload_prefetch_step)
            _set("compilation_config", engine.compilation_config)

            if engine.offload_params is not None:
                kwargs["offload_params"] = set(engine.offload_params)

            if engine.attention is not None:
                attn = engine.attention
                if attn.backend is not None:
                    kwargs["attention_backend"] = attn.backend
                _set("flash_attn_version", attn.flash_attn_version)
                _set(
                    "flash_attn_max_num_splits_for_cuda_graph",
                    attn.flash_attn_max_num_splits_for_cuda_graph,
                )
                _set("use_prefill_decode_attention", attn.use_prefill_decode_attention)
                _set("use_prefill_query_quantization", attn.use_prefill_query_quantization)
                _set("use_cudnn_prefill", attn.use_cudnn_prefill)
                _set("disable_flashinfer_prefill", attn.disable_flashinfer_prefill)
                _set("disable_flashinfer_q_quantization", attn.disable_flashinfer_q_quantization)
                _set("use_trtllm_attention", attn.use_trtllm_attention)
                _set("use_trtllm_ragged_deepseek_prefill", attn.use_trtllm_ragged_deepseek_prefill)
                if attn.model_extra:
                    kwargs.update(attn.model_extra)

            if engine.model_extra:
                kwargs.update(engine.model_extra)

        return kwargs

    @staticmethod
    def _map_precision(precision: str) -> str:
        """Map precision string to vLLM dtype string."""
        return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(precision, "auto")

    @staticmethod
    def _build_sampling_params(config: ExperimentConfig, sampling_params_cls: Any) -> Any:
        """Build vllm.SamplingParams from DecoderConfig."""
        vllm_cfg = config.vllm
        if vllm_cfg is not None and vllm_cfg.beam_search is not None:
            return VLLMBackend._build_beam_search_params(config, vllm_cfg.beam_search)

        decoder = config.decoder
        is_greedy = decoder.temperature == 0.0 or not decoder.do_sample

        if is_greedy:
            kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": config.max_output_tokens,
            }
        else:
            top_k = decoder.top_k if decoder.top_k != 0 else -1
            kwargs = {
                "temperature": decoder.temperature,
                "top_p": decoder.top_p,
                "top_k": top_k,
                "repetition_penalty": decoder.repetition_penalty,
                "max_tokens": config.max_output_tokens,
            }
            if decoder.min_p is not None:
                kwargs["min_p"] = decoder.min_p

        # Map universal decoder.min_new_tokens to vLLM's min_tokens.
        # This is placed before vllm_cfg overrides so that vllm.sampling.min_tokens
        # can override the universal mapping if both are set.
        if decoder.min_new_tokens is not None:
            kwargs["min_tokens"] = decoder.min_new_tokens

        # Apply vLLM-specific sampling overrides
        if vllm_cfg is not None and vllm_cfg.sampling is not None:
            sampling = vllm_cfg.sampling
            if sampling.max_tokens is not None:
                kwargs["max_tokens"] = sampling.max_tokens
            if sampling.min_tokens is not None:
                kwargs["min_tokens"] = sampling.min_tokens
            if sampling.presence_penalty is not None:
                kwargs["presence_penalty"] = sampling.presence_penalty
            if sampling.frequency_penalty is not None:
                kwargs["frequency_penalty"] = sampling.frequency_penalty
            if sampling.ignore_eos is not None:
                kwargs["ignore_eos"] = sampling.ignore_eos
            if sampling.n is not None:
                kwargs["n"] = sampling.n
            if sampling.model_extra:
                kwargs.update(sampling.model_extra)

        return sampling_params_cls(**kwargs)

    @staticmethod
    def _build_beam_search_params(config: ExperimentConfig, beam_cfg: Any) -> Any:
        """Build vllm.BeamSearchParams from VLLMBeamSearchConfig."""
        try:
            from vllm import BeamSearchParams
        except ImportError:
            raise BackendError(
                "beam_search config requires vllm.BeamSearchParams which is not "
                "available in the installed vLLM version (added in vLLM >=0.8). "
                "Either upgrade vLLM or remove the beam_search section from "
                "vllm config."
            ) from None

        kwargs: dict[str, Any] = {}
        if beam_cfg.beam_width is not None:
            kwargs["beam_width"] = beam_cfg.beam_width
        if beam_cfg.length_penalty is not None:
            kwargs["length_penalty"] = beam_cfg.length_penalty
        if beam_cfg.early_stopping is not None:
            kwargs["early_stopping"] = beam_cfg.early_stopping
        kwargs["max_tokens"] = beam_cfg.max_tokens or config.max_output_tokens
        if config.decoder.min_p is not None:
            kwargs["min_p"] = config.decoder.min_p
        if beam_cfg.model_extra:
            kwargs.update(beam_cfg.model_extra)
        return BeamSearchParams(**kwargs)

    # -------------------------------------------------------------------------
    # Private: inference helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _nvml_peak_memory_mb() -> float | None:
        """Query NVML for current GPU memory used. Returns MB or None on failure."""
        try:
            import pynvml

            from llenergymeasure.device.gpu_info import nvml_context

            mem_mb: float | None = None
            with nvml_context():
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_mb = float(info.used) / (1024 * 1024)
            return mem_mb
        except Exception:
            return None
