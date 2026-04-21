"""HuggingFace Transformers inference engine — thin EnginePlugin.

Implements the 4-method EnginePlugin protocol:
  load_model, warmup, run_inference, cleanup

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only Transformers-specific inference: model loading, warmup via
warmup_until_converged(), model.generate() inference loop, and cleanup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import ConfigProbe, DormantField, InferenceOutput

logger = logging.getLogger(__name__)

# Whether to attempt meta-device model construction during T2 probe.
# Enabled by default (cheap: no weight downloads, no GPU memory), but some
# exotic architectures may not support it — the env var escape hatch lets
# operators disable without editing code.
_TIER2_META_DEVICE_ENABLED = True


class TransformersEngine:
    """HuggingFace Transformers inference engine — thin plugin.

    Implements EnginePlugin:
    - load_model: Load HuggingFace model + tokenizer, apply torch.compile
    - warmup: CV-based warmup via warmup_until_converged()
    - run_inference: Batched model.generate() loop, returns InferenceOutput
    - cleanup: Delete model, clear CUDA cache
    """

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "transformers"

    @property
    def version(self) -> str:
        """Transformers library version string."""
        try:
            import torch

            return torch.__version__
        except Exception:
            return "unknown"

    # -------------------------------------------------------------------------
    # EnginePlugin: load_model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer via from_pretrained().

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (model, tokenizer).
        """
        import time as _time

        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = self._model_load_kwargs(config)
        logger.info("Loading model %r with kwargs: %s", config.task.model, list(kwargs.keys()))

        t0 = _time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            config.task.model, trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if on_substep is not None:
            on_substep("tokenizer loaded", _time.perf_counter() - t0)

        t0 = _time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(config.task.model, **kwargs)
        model.eval()
        if on_substep is not None:
            on_substep("model weights loaded", _time.perf_counter() - t0)

        # Apply allow_tf32 (Ampere+ TF32 toggle)
        if config.transformers is not None and config.transformers.allow_tf32 is not None:
            import torch as _torch

            _torch.backends.cuda.matmul.allow_tf32 = config.transformers.allow_tf32

        # Apply torch.compile post-load (must be AFTER from_pretrained + eval)
        if config.transformers is not None and config.transformers.torch_compile:
            import torch as _torch

            mode = config.transformers.torch_compile_mode or "default"
            backend = config.transformers.torch_compile_backend or "inductor"
            try:
                t0 = _time.perf_counter()
                model = _torch.compile(model, mode=mode, backend=backend)  # type: ignore[assignment]
                logger.debug("torch.compile applied (mode=%s, backend=%s)", mode, backend)
                if on_substep is not None:
                    on_substep(f"torch.compile ({mode})", _time.perf_counter() - t0)
            except Exception as e:
                logger.warning("torch.compile failed (non-fatal, continuing without): %s", e)

        logger.debug("Model loaded successfully")
        return model, tokenizer

    # -------------------------------------------------------------------------
    # EnginePlugin: warmup
    # -------------------------------------------------------------------------

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt and return latency in ms.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().
            prompt: Single warmup prompt text.

        Returns:
            Latency in milliseconds.
        """
        import time

        import torch

        hf_model, tokenizer = model
        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_model.generate(**inputs, max_new_tokens=min(config.task.max_output_tokens or 32, 32))
        return (time.perf_counter() - start) * 1000.0

    # -------------------------------------------------------------------------
    # EnginePlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run the batched measurement loop over all prompts.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            EngineError: On CUDA OOM or other inference failures.
        """
        hf_model, tokenizer = model

        batch_size = 1
        if config.transformers is not None and config.transformers.batch_size is not None:
            batch_size = config.transformers.batch_size
        else:
            logger.debug("Transformers batch_size not set, defaulting to 1")

        # Reset peak stats BEFORE the measurement loop so max_memory_allocated()
        # captures inference-window-only peak (KV cache + activations + batch buffers),
        # NOT model weights already allocated by load_model().
        from llenergymeasure.engines._helpers import reset_cuda_peak_memory

        reset_cuda_peak_memory()

        # Seed PyTorch RNG for reproducible sampling (mirrors vLLM's seed= kwarg).
        # manual_seed seeds both CPU and all CUDA devices since PyTorch 1.12+.
        import torch as _torch

        _torch.manual_seed(config.task.random_seed)

        generate_kwargs = self._build_generate_kwargs(config)
        total_input_tokens = 0
        total_output_tokens = 0
        total_time_sec = 0.0
        batch_times: list[float] = []

        logger.info(
            "Starting measurement: %d prompts, batch_size=%d, max_new_tokens=%s",
            len(prompts),
            batch_size,
            config.task.max_output_tokens or "unlimited",
        )

        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start : batch_start + batch_size]
            try:
                batch_input, batch_output, batch_time = self._run_batch(
                    hf_model, tokenizer, config, batch, generate_kwargs
                )
                total_input_tokens += batch_input
                total_output_tokens += batch_output
                total_time_sec += batch_time
                batch_times.append(batch_time)

                logger.debug(
                    "Batch %d-%d: in=%d out=%d tokens in %.2fs",
                    batch_start,
                    batch_start + len(batch) - 1,
                    batch_input,
                    batch_output,
                    batch_time,
                )
            except Exception as e:
                from llenergymeasure.engines._helpers import raise_engine_error

                raise_engine_error(
                    e,
                    "Transformers",
                    hint="reduce batch_size, use dtype=float16, or use a smaller model.",
                )

        # Track peak GPU memory (inference window only — reset above)
        from llenergymeasure.engines._helpers import get_cuda_peak_memory_mb

        peak_memory_mb = get_cuda_peak_memory_mb()

        # model_memory_mb is queried by the harness after load_model(); we report 0.0 here
        # as the harness captures it before warmup (before this method is called).
        logger.debug(
            "Measurement complete: %d total tokens in %.2fs",
            total_input_tokens + total_output_tokens,
            total_time_sec,
        )

        return InferenceOutput(
            elapsed_time_sec=total_time_sec,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            peak_memory_mb=peak_memory_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=batch_times,
            extras={"hf_model": hf_model},  # For FLOPs estimation in harness
        )

    # -------------------------------------------------------------------------
    # EnginePlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release model from memory and clear CUDA cache.

        Args:
            model: Tuple of (model, tokenizer) from load_model().
        """
        from llenergymeasure.engines._helpers import cleanup_model

        hf_model, _tokenizer = model
        cleanup_model(hf_model, use_gc=False)
        logger.debug("Model cleanup complete")

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """No hardware validation required for Transformers engine."""
        return []

    # -------------------------------------------------------------------------
    # EnginePlugin: probe_config
    # -------------------------------------------------------------------------

    def _declared_sampling_params(self, config: ExperimentConfig) -> dict[str, Any]:
        """Return user-declared sampling kwargs before greedy stripping.

        Reads directly from ``config.transformers.sampling`` — the single source
        of truth in the post-restructure data model — and dumps all non-None
        fields. Used by :meth:`probe_config` to diff against effective kwargs
        and surface dormant sampling fields (e.g. ``temperature=0.9`` that the
        engine will silently strip under ``do_sample=False``).
        """
        pt = config.transformers
        sampling = pt.sampling if pt is not None else None
        if sampling is None:
            return {}
        return sampling.model_dump(exclude_none=True)

    @staticmethod
    def _dormancy_reason(key: str, declared_val: Any, effective_val: Any | None) -> str | None:
        """Explain why a declared sampling field went dormant (for probe output)."""
        if key in {"temperature", "top_k", "top_p", "min_p"} and effective_val is None:
            return "greedy decoding (do_sample=false or temperature=0)"
        if key == "do_sample" and declared_val is True and effective_val is False:
            return "greedy decoding forced do_sample=False"
        return None

    def probe_config(self, config: ExperimentConfig) -> ConfigProbe:
        """Observe what Transformers would do with *config* without loading weights.

        Tier breakdown:
          - T0: build effective from_pretrained + generate kwargs and diff
                the sampling subset against the user-declared sampling params.
          - T1: construct ``GenerationConfig`` from effective kwargs (filtered
                to known fields) to catch invalid sampling combinations early.
          - T2: download the model's ``config.json`` via ``AutoConfig`` and —
                when ``_TIER2_META_DEVICE_ENABLED`` is on — build an empty
                model on the meta device to validate dtype x attn x arch.

        Never raises: every exception is captured into :attr:`ConfigProbe.errors`.
        """
        errors: list[str] = []
        warnings: list[str] = []
        effective_engine: dict[str, Any] = {}
        effective_sampling: dict[str, Any] = {}
        dormant: dict[str, DormantField] = {}

        # --- T0: Build effective kwargs ---
        try:
            effective_engine = self._model_load_kwargs(config)
        except Exception as e:
            errors.append(f"T0 engine kwargs: {type(e).__name__}: {e}")
            return ConfigProbe(effective_engine, effective_sampling, dormant, errors, warnings)

        try:
            full_generate_kwargs = self._build_generate_kwargs(config)
        except Exception as e:
            errors.append(f"T0 sampling kwargs: {type(e).__name__}: {e}")
            return ConfigProbe(effective_engine, effective_sampling, dormant, errors, warnings)

        # Scope effective_sampling to the sampling-related subset so it diffs
        # cleanly against declared (non-sampling transformers.* generate fields
        # — use_cache, num_beams etc — are out of the dormancy scope).
        _SAMPLING_KEYS = {
            "temperature",
            "do_sample",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "min_new_tokens",
        }
        effective_sampling = {k: v for k, v in full_generate_kwargs.items() if k in _SAMPLING_KEYS}

        # --- T0: Dormancy diff ---
        try:
            from llenergymeasure.engines._helpers import compute_dormant_fields

            declared = self._declared_sampling_params(config)
            dormant = compute_dormant_fields(
                declared,
                effective_sampling,
                prefix="transformers.sampling.",
                reason_fn=self._dormancy_reason,
            )
        except Exception as e:
            errors.append(f"T0 dormancy diff: {type(e).__name__}: {e}")

        # --- T1: GenerationConfig construction (filtered to known fields) ---
        try:
            from transformers import GenerationConfig

            allowed = set(GenerationConfig().__dict__.keys())
            filtered = {k: v for k, v in full_generate_kwargs.items() if k in allowed}
            GenerationConfig(**filtered)
        except Exception as e:
            errors.append(f"T1 GenerationConfig: {type(e).__name__}: {e}")

        # --- T2: AutoConfig (model metadata) + optional meta-device build ---
        try:
            from transformers import AutoConfig

            trust_remote_code = bool(effective_engine.get("trust_remote_code", False))
            hf_config = AutoConfig.from_pretrained(
                config.task.model, trust_remote_code=trust_remote_code
            )
        except Exception as e:
            errors.append(f"T2 AutoConfig: {type(e).__name__}: {e}")
            return ConfigProbe(effective_engine, effective_sampling, dormant, errors, warnings)

        import os

        env_flag = os.environ.get("LLEM_PROBE_META_DEVICE_ENABLED")
        if env_flag is None:
            meta_enabled = _TIER2_META_DEVICE_ENABLED
        else:
            meta_enabled = env_flag.lower() in {"1", "true", "yes", "on"}

        if meta_enabled:
            try:
                from accelerate import init_empty_weights
                from transformers import AutoModelForCausalLM

                meta_kwargs: dict[str, Any] = {}
                if "torch_dtype" in effective_engine:
                    meta_kwargs["torch_dtype"] = effective_engine["torch_dtype"]
                if "attn_implementation" in effective_engine:
                    meta_kwargs["attn_implementation"] = effective_engine["attn_implementation"]
                meta_kwargs["trust_remote_code"] = bool(
                    effective_engine.get("trust_remote_code", True)
                )

                with init_empty_weights():
                    AutoModelForCausalLM.from_config(hf_config, **meta_kwargs)
            except Exception as e:
                errors.append(f"T2 meta-device: {type(e).__name__}: {e}")

        return ConfigProbe(effective_engine, effective_sampling, dormant, errors, warnings)

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _model_load_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build the full kwargs dict for AutoModelForCausalLM.from_pretrained().

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs ready for from_pretrained().
        """
        pt = config.transformers
        dtype = pt.dtype if pt is not None else None
        kwargs: dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(dtype or "bfloat16"),
        }

        from llenergymeasure.utils.env_config import default_device_map
        from llenergymeasure.utils.security import trust_remote_code_enabled

        # Device placement / tensor parallelism — mutually exclusive
        if pt is not None and pt.tp_plan is not None:
            # Tensor parallelism: tp_plan replaces device_map entirely
            kwargs["tp_plan"] = pt.tp_plan
            if pt.tp_size is not None:
                kwargs["tp_size"] = pt.tp_size
            # Do NOT set device_map — TP handles device placement
        elif pt is not None and pt.device_map is not None:
            kwargs["device_map"] = pt.device_map
        else:
            dm = default_device_map()
            if dm is not None:
                kwargs["device_map"] = dm

        kwargs["trust_remote_code"] = trust_remote_code_enabled()

        # Apply Transformers-specific config options
        if pt is not None:
            if pt.attn_implementation is not None:
                kwargs["attn_implementation"] = self._resolve_attn_implementation(
                    pt.attn_implementation
                )

            # BitsAndBytes quantization — use BitsAndBytesConfig, not raw kwargs
            if pt.load_in_4bit or pt.load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_kwargs: dict[str, Any] = {}
                if pt.load_in_4bit:
                    bnb_kwargs["load_in_4bit"] = True
                    if pt.bnb_4bit_compute_dtype is not None:
                        import torch as _torch

                        _dtype_map = {
                            "float16": _torch.float16,
                            "bfloat16": _torch.bfloat16,
                            "float32": _torch.float32,
                        }
                        bnb_kwargs["bnb_4bit_compute_dtype"] = _dtype_map[pt.bnb_4bit_compute_dtype]
                    if pt.bnb_4bit_quant_type is not None:
                        bnb_kwargs["bnb_4bit_quant_type"] = pt.bnb_4bit_quant_type
                    if pt.bnb_4bit_use_double_quant is not None:
                        bnb_kwargs["bnb_4bit_use_double_quant"] = pt.bnb_4bit_use_double_quant
                if pt.load_in_8bit:
                    bnb_kwargs["load_in_8bit"] = True
                kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

            # Additional from_pretrained() fields
            # revision dropped as typed field (D1); flows through model_extra if set
            if pt.max_memory is not None:
                kwargs["max_memory"] = pt.max_memory
            if pt.low_cpu_mem_usage is not None:
                kwargs["low_cpu_mem_usage"] = pt.low_cpu_mem_usage

        # Transformers extra="allow" passthrough: forward unknown fields to from_pretrained()
        if pt is not None and pt.model_extra:
            kwargs.update(pt.model_extra)

        # passthrough_kwargs merged LAST so researcher can override any default
        if config.passthrough_kwargs:
            kwargs.update(config.passthrough_kwargs)

        return kwargs

    @staticmethod
    def _resolve_attn_implementation(requested: str) -> str:
        """Validate the requested attention implementation is available.

        If flash_attention_2 is requested but the flash_attn package (or any
        of its transitive dependencies such as einops) cannot be imported,
        falls back to sdpa with a warning rather than crashing at model load
        time.

        A simple ``find_spec`` check is insufficient because flash_attn may
        be installed while its dependencies (e.g. einops) are missing.  We
        therefore attempt a real import of the submodule that transformers
        actually uses (``flash_attn.bert_padding``).

        Args:
            requested: The attention implementation string from config.

        Returns:
            The resolved attention implementation string.
        """
        if requested in ("flash_attention_2", "flash_attention_3"):
            fallback = "sdpa"
            try:
                import flash_attn
                import flash_attn.bert_padding  # noqa: F401
            except Exception as exc:
                logger.warning(
                    "attn_implementation=%r requested but flash_attn is not "
                    "fully usable (%s: %s); falling back to %r. "
                    "Install flash-attn and its dependencies (einops) to use "
                    "FlashAttention.",
                    requested,
                    type(exc).__name__,
                    exc,
                    fallback,
                )
                return fallback
            # FA3 additionally requires the flash_attn_interface module
            # (built separately from the flash-attn repo's hopper/ directory)
            if requested == "flash_attention_3":
                try:
                    import flash_attn_interface  # noqa: F401
                except Exception as exc:
                    logger.warning(
                        "attn_implementation='flash_attention_3' requested but "
                        "flash_attn_interface is not installed (%s: %s); "
                        "falling back to %r. Build flash_attn_3 from the "
                        "flash-attn repo's hopper/ directory, or use the "
                        "Docker runner.",
                        type(exc).__name__,
                        exc,
                        fallback,
                    )
                    return fallback
        return requested

    @staticmethod
    def _resolve_torch_dtype(dtype: str) -> Any:
        """Map dtype string to torch dtype object."""
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    # -------------------------------------------------------------------------
    # Private: inference helpers
    # -------------------------------------------------------------------------

    def _run_batch(
        self,
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        batch: list[str],
        generate_kwargs: dict[str, Any],
    ) -> tuple[int, int, float]:
        """Run a single batch through model.generate() and return (input_tokens, output_tokens, time_sec)."""
        import time

        import torch

        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if config.task.max_input_tokens is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = config.task.max_input_tokens

        inputs = tokenizer(batch, **tokenizer_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_token_count = int(inputs["attention_mask"].sum().item())

        # Determine autocast settings
        from contextlib import nullcontext

        _pt = config.transformers
        if _pt is not None and _pt.autocast_enabled is True and torch.cuda.is_available():
            _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            _amp_ctx = torch.autocast(
                device_type="cuda", dtype=_dtype_map[_pt.autocast_dtype or "bfloat16"]
            )
        else:
            _amp_ctx = nullcontext()  # type: ignore[assignment]

        t0 = time.perf_counter()
        with torch.inference_mode(), _amp_ctx:
            gen_kwargs = {**generate_kwargs}
            if config.task.max_output_tokens is not None:
                gen_kwargs["max_new_tokens"] = config.task.max_output_tokens
            outputs = model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        # Count only the newly generated tokens per sequence (handles padding correctly)
        input_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch,)
        output_token_count = int(
            sum(max(0, outputs.shape[1] - int(inp_len.item())) for inp_len in input_lengths)
        )
        return input_token_count, output_token_count, elapsed

    def _build_generate_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build generation kwargs from TransformersSamplingConfig and TransformersConfig.

        None values mean "use HF's default"; only explicit fields are forwarded.
        Greedy decoding (temperature=0 or do_sample=False) strips sampling params
        and forces do_sample=False, matching HF's own greedy semantics.
        """
        pt = config.transformers
        sampling = pt.sampling if pt is not None else None

        kwargs: dict[str, Any] = (
            sampling.model_dump(exclude_none=True) if sampling is not None else {}
        )

        if pt is not None:
            if pt.use_cache is not None:
                kwargs["use_cache"] = pt.use_cache
            if pt.cache_implementation is not None:
                kwargs["cache_implementation"] = pt.cache_implementation
            if pt.num_beams is not None:
                kwargs["num_beams"] = pt.num_beams
            if pt.early_stopping is not None:
                kwargs["early_stopping"] = pt.early_stopping
            if pt.length_penalty is not None:
                kwargs["length_penalty"] = pt.length_penalty
            if pt.no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = pt.no_repeat_ngram_size
            if pt.prompt_lookup_num_tokens is not None:
                kwargs["prompt_lookup_num_tokens"] = pt.prompt_lookup_num_tokens

        if kwargs.get("temperature") == 0.0 or kwargs.get("do_sample") is False:
            kwargs["do_sample"] = False
            for key in ("temperature", "top_k", "top_p", "min_p"):
                kwargs.pop(key, None)

        return kwargs
