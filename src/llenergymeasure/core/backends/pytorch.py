"""PyTorch/Transformers inference backend — thin BackendPlugin.

Implements the 4-method BackendPlugin protocol:
  load_model, warmup, run_inference, cleanup

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only PyTorch-specific inference: model loading, warmup via
warmup_until_converged(), model.generate() inference loop, and cleanup.
"""

from __future__ import annotations

import logging
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.core.backends.protocol import InferenceOutput
from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.exceptions import BackendError

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """PyTorch/Transformers inference backend — thin plugin.

    Implements BackendPlugin:
    - load_model: Load HuggingFace model + tokenizer, apply torch.compile
    - warmup: CV-based warmup via warmup_until_converged()
    - run_inference: Batched model.generate() loop, returns InferenceOutput
    - cleanup: Delete model, clear CUDA cache
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "pytorch"

    # -------------------------------------------------------------------------
    # BackendPlugin: load_model
    # -------------------------------------------------------------------------

    def load_model(self, config: ExperimentConfig) -> tuple[Any, Any]:
        """Load model and tokenizer directly — no intermediate loader class.

        The P0 fix: kwargs are built by _model_load_kwargs() and passed
        directly to from_pretrained(). The v1.x bug was that _build_model_kwargs()
        built the dict but loader.load(config) ignored it.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = self._model_load_kwargs(config)
        logger.info("Loading model %r with kwargs: %s", config.model, list(kwargs.keys()))

        # trust_remote_code for tokenizer — respects config, defaults True
        trust = True
        if config.pytorch is not None and config.pytorch.trust_remote_code is not None:
            trust = config.pytorch.trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=trust)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        model.eval()

        # Apply torch.compile post-load (must be AFTER from_pretrained + eval)
        if config.pytorch is not None and config.pytorch.torch_compile:
            import torch as _torch

            mode = config.pytorch.torch_compile_mode or "default"
            backend = config.pytorch.torch_compile_backend or "inductor"
            try:
                model = _torch.compile(model, mode=mode, backend=backend)
                logger.debug("torch.compile applied (mode=%s, backend=%s)", mode, backend)
            except Exception as e:
                logger.warning("torch.compile failed (non-fatal, continuing without): %s", e)

        logger.debug("Model loaded successfully")
        return model, tokenizer

    # -------------------------------------------------------------------------
    # BackendPlugin: warmup
    # -------------------------------------------------------------------------

    def warmup(self, config: ExperimentConfig, model: Any) -> WarmupResult:
        """Run warmup using warmup_until_converged() and return WarmupResult.

        thermal_floor_wait_s is NOT set here — MeasurementHarness sets it after
        this method returns.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().

        Returns:
            WarmupResult with convergence status and iteration count.
            thermal_floor_wait_s is left at default 0.0 (set by harness).
        """
        from llenergymeasure.core.warmup import (
            create_warmup_inference_fn,
            warmup_until_converged,
        )

        hf_model, tokenizer = model

        if not config.warmup.enabled:
            logger.debug("Warmup disabled, skipping")
            return WarmupResult(
                converged=True,
                final_cv=0.0,
                iterations_completed=0,
                target_cv=config.warmup.cv_threshold,
                max_prompts=config.warmup.max_prompts,
            )

        # Use first prompt from the standard prompt list for warmup
        words_per_prompt = max(1, config.max_input_tokens // 4)
        warmup_prompt = ("Hello, " * words_per_prompt).strip()

        logger.debug(
            "Running warmup: %d fixed iterations (convergence_detection=%s)",
            config.warmup.n_warmup,
            config.warmup.convergence_detection,
        )

        inference_fn = create_warmup_inference_fn(
            hf_model, tokenizer, warmup_prompt, config.max_output_tokens
        )
        result = warmup_until_converged(inference_fn, config.warmup, show_progress=False)
        logger.debug("Warmup complete: %d iterations", result.iterations_completed)
        return result

    # -------------------------------------------------------------------------
    # BackendPlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(self, config: ExperimentConfig, model: Any) -> InferenceOutput:
        """Run the batched measurement loop over all prompts.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            BackendError: On CUDA OOM or other inference failures.
        """
        import torch

        hf_model, tokenizer = model

        # Prepare prompts (M1 placeholder — generates synthetic prompts)
        prompts = self._prepare_prompts(config)

        batch_size = 1
        if config.pytorch is not None and config.pytorch.batch_size is not None:
            batch_size = config.pytorch.batch_size
        else:
            logger.debug("PyTorch batch_size not set, defaulting to 1")

        # Reset peak stats BEFORE the measurement loop so max_memory_allocated()
        # captures inference-window-only peak (KV cache + activations + batch buffers),
        # NOT model weights already allocated by load_model().
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        generate_kwargs = self._build_generate_kwargs(config)
        total_input_tokens = 0
        total_output_tokens = 0
        total_time_sec = 0.0
        batch_times: list[float] = []

        logger.info(
            "Starting measurement: %d prompts, batch_size=%d, max_new_tokens=%d",
            len(prompts),
            batch_size,
            config.max_output_tokens,
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
                if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                    raise BackendError(
                        f"CUDA out of memory. Try: reduce batch_size, "
                        f"use precision=fp16, or use a smaller model. "
                        f"Original error: {e}"
                    ) from e
                raise BackendError(f"Inference failed: {e}") from e

        # Track peak GPU memory (inference window only — reset above)
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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
    # BackendPlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release model from memory and clear CUDA cache.

        Args:
            model: Tuple of (model, tokenizer) from load_model().
        """
        import torch

        hf_model, _tokenizer = model
        del hf_model
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except Exception:
            logger.debug("CUDA cleanup failed", exc_info=True)
        logger.debug("Model cleanup complete")

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """No hardware validation required for PyTorch backend."""
        return []

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _model_load_kwargs(self, config: ExperimentConfig) -> dict:
        """Build the full kwargs dict for AutoModelForCausalLM.from_pretrained().

        This is the P0 fix location: passthrough_kwargs and pytorch config
        options are ALL collected here and ALL passed to from_pretrained().

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs ready for from_pretrained().
        """
        kwargs: dict = {
            "torch_dtype": self._precision_to_dtype(config.precision),
        }

        pt = config.pytorch

        # Device placement — default "auto" unless researcher overrides
        if pt is not None and pt.device_map is not None:
            kwargs["device_map"] = pt.device_map
        else:
            kwargs["device_map"] = "auto"

        # Trust remote code — default True unless researcher overrides
        if pt is not None and pt.trust_remote_code is not None:
            kwargs["trust_remote_code"] = pt.trust_remote_code
        else:
            kwargs["trust_remote_code"] = True

        # Apply PyTorch-specific config options
        if pt is not None:
            if pt.attn_implementation is not None:
                kwargs["attn_implementation"] = pt.attn_implementation

            # BitsAndBytes quantization — use BitsAndBytesConfig, not raw kwargs
            if pt.load_in_4bit or pt.load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_kwargs: dict = {}
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
            if pt.revision is not None:
                kwargs["revision"] = pt.revision
            if pt.max_memory is not None:
                kwargs["max_memory"] = pt.max_memory

        # PyTorch extra="allow" passthrough: forward unknown fields to from_pretrained()
        if pt is not None and pt.model_extra:
            kwargs.update(pt.model_extra)

        # passthrough_kwargs merged LAST so researcher can override any default
        if config.passthrough_kwargs:
            kwargs.update(config.passthrough_kwargs)

        return kwargs

    @staticmethod
    def _precision_to_dtype(precision: str):
        """Map precision string to torch dtype."""
        import torch

        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]

    # -------------------------------------------------------------------------
    # Private: inference helpers
    # -------------------------------------------------------------------------

    def _prepare_prompts(self, config: ExperimentConfig) -> list[str]:
        """Prepare prompts for inference.

        M1 placeholder: generates simple synthetic prompts.
        """
        words_per_prompt = max(1, config.max_input_tokens // 4)
        base_prompt = ("Hello, " * words_per_prompt).strip()
        prompts = [base_prompt] * config.n
        logger.debug("Prepared %d prompts (M1 placeholder)", config.n)
        return prompts

    def _run_batch(
        self,
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        batch: list[str],
        generate_kwargs: dict,
    ) -> tuple[int, int, float]:
        """Run a single batch through model.generate() and return (input_tokens, output_tokens, time_sec)."""
        import time

        import torch

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_input_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_token_count = int(inputs["attention_mask"].sum().item())

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_output_tokens,
                **generate_kwargs,
            )
        elapsed = time.perf_counter() - t0

        # Count only the newly generated tokens per sequence (handles padding correctly)
        input_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch,)
        output_token_count = int(
            sum(max(0, outputs.shape[1] - int(inp_len.item())) for inp_len in input_lengths)
        )
        return input_token_count, output_token_count, elapsed

    def _build_generate_kwargs(self, config: ExperimentConfig) -> dict:
        """Build generation kwargs from DecoderConfig and PyTorchConfig."""
        decoder = config.decoder
        kwargs: dict = {
            "do_sample": decoder.do_sample,
            "temperature": decoder.temperature,
            "top_k": decoder.top_k,
            "top_p": decoder.top_p,
            "repetition_penalty": decoder.repetition_penalty,
        }

        # DecoderConfig new fields
        if decoder.min_p is not None:
            kwargs["min_p"] = decoder.min_p
        if decoder.min_new_tokens is not None:
            kwargs["min_new_tokens"] = decoder.min_new_tokens

        # PyTorchConfig generate() fields
        pt = config.pytorch
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

        # Greedy decoding: temperature=0 or do_sample=False — strip sampling params
        if decoder.temperature == 0.0 or not decoder.do_sample:
            kwargs["do_sample"] = False
            kwargs.pop("temperature", None)
            kwargs.pop("top_k", None)
            kwargs.pop("top_p", None)
            kwargs.pop("min_p", None)

        return kwargs
