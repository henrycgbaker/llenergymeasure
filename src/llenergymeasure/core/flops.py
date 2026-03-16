"""FLOPs estimation for LLM inference.

v2.0 primary method: PaLM/Chinchilla formula (estimate_flops_palm).
    FLOPs = 2 * N_non_embedding_params * total_tokens
    Split: prefill (input) + decode (output).

Legacy fallback chain (FlopsEstimator class):
1. calflops library — direct measurement (high confidence)
2. Architecture-based — uses model config (medium confidence)
3. Parameter-based — simple 2*P approximation (low confidence)

Key insight: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
computation happens at FP16 after dequantization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llenergymeasure.domain.metrics import FlopsResult

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# v2.0 Primary API — PaLM formula
# =============================================================================


def _count_non_embedding_params(model: Any) -> int:
    """Count non-embedding parameters. Embeddings are memory-bound lookups, not MAC ops."""
    total = 0
    for name, param in model.named_parameters():
        if "embed" not in name.lower():
            total += param.numel()
    return total


def estimate_flops_palm(
    model: Any,
    n_input_tokens: int,
    n_output_tokens: int,
    batch_size: int = 1,
) -> FlopsResult:
    """PaLM/Chinchilla inference FLOPs estimate (v2.0 primary method).

    Formula: FLOPs = 2 * N_non_embedding_params * total_tokens
    Split: prefill (input) + decode (output).
    Caller must exclude warmup tokens from n_input_tokens/n_output_tokens (CM-28).

    Args:
        model: The model (must support named_parameters()).
        n_input_tokens: Number of input/prefill tokens (measurement phase only).
        n_output_tokens: Number of output/decode tokens (measurement phase only).
        batch_size: Batch size multiplier.

    Returns:
        FlopsResult with PaLM formula estimate and high confidence.
    """
    n_params = _count_non_embedding_params(model)
    flops_prefill = 2 * n_params * batch_size * n_input_tokens
    flops_decode = 2 * n_params * batch_size * n_output_tokens
    flops_total = flops_prefill + flops_decode

    return FlopsResult(
        value=float(flops_total),
        method="palm_formula",
        confidence="high",
        precision="n/a",  # precision doesn't affect FLOPs (forward pass)
        notes=(
            f"PaLM formula: 2x{n_params:,}x({n_input_tokens}+{n_output_tokens}) tokens. "
            f"Attention FLOPs omitted (v2.0 limitation, significant only for seq_len>=2048)."
        ),
    )


# =============================================================================
# v2.1 Config-based API — AutoConfig fallback (no model weights required)
# =============================================================================


def _count_params_from_config(model_name: str) -> int | None:
    """Extract non-embedding parameter count from HuggingFace model config.

    Uses AutoConfig.from_pretrained() to read architecture dimensions
    without loading model weights. Returns None on any failure.

    This enables FLOPs estimation for backends that do not expose an hf_model
    object (e.g. vLLM, TensorRT-LLM).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        h = cfg.hidden_size
        layers = cfg.num_hidden_layers
        intermediate = getattr(cfg, "intermediate_size", h * 4)
        # Non-embedding params: attention (Q,K,V,O) + FFN (up,down) per layer
        attn = 4 * h * h * layers
        ffn = 2 * h * intermediate * layers
        return attn + ffn
    except Exception:
        return None


def estimate_flops_palm_from_config(
    model_name: str,
    n_input_tokens: int,
    n_output_tokens: int,
    batch_size: int = 1,
) -> FlopsResult | None:
    """PaLM/Chinchilla inference FLOPs estimate using AutoConfig (no model weights needed).

    Uses _count_params_from_config() to extract architecture dimensions from
    the HuggingFace model config file. Returns None if config cannot be loaded.

    Formula: FLOPs = 2 * N_non_embedding_params * total_tokens
    Split: prefill (input) + decode (output).

    Args:
        model_name: HuggingFace model name or path (passed to AutoConfig.from_pretrained).
        n_input_tokens: Number of input/prefill tokens (measurement phase only).
        n_output_tokens: Number of output/decode tokens (measurement phase only).
        batch_size: Batch size multiplier.

    Returns:
        FlopsResult with config-based estimate and medium confidence, or None on failure.
    """
    n_params = _count_params_from_config(model_name)
    if n_params is None:
        return None

    flops_prefill = 2 * n_params * batch_size * n_input_tokens
    flops_decode = 2 * n_params * batch_size * n_output_tokens
    flops_total = flops_prefill + flops_decode

    return FlopsResult(
        value=float(flops_total),
        method="palm_formula",
        confidence="medium",
        precision="n/a",
        notes=(
            f"PaLM formula via AutoConfig (no weights loaded): "
            f"2x{n_params:,}x({n_input_tokens}+{n_output_tokens}) tokens."
        ),
    )


# =============================================================================
# Legacy fallback chain — FlopsEstimator class
# =============================================================================


class FlopsEstimator:
    """Multi-strategy FLOPs estimation with graceful degradation.

    LEGACY: This class is the v1.x fallback chain. Use estimate_flops_palm()
    for v2.0 measurements.

    Strategy order (highest to lowest confidence):
    1. calflops library — direct measurement, works with most HF models
    2. Architecture-based — uses model config (hidden_size, num_layers, etc.)
    3. Parameter-based — simple 2*P approximation

    Note: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
    computation happens at FP16 after dequantization.
    """

    def __init__(self, timeout_sec: int = 30) -> None:
        """Initialize the FLOPs estimator.

        Args:
            timeout_sec: Timeout for calflops estimation (not currently used,
                         reserved for future async implementation).
        """
        self._timeout_sec = timeout_sec

    def estimate(
        self,
        model: Any,
        input_ids: Any,
        config: ExperimentConfig | None = None,
    ) -> FlopsResult:
        """Estimate FLOPs using the fallback chain.

        Args:
            model: The model to measure (typically a HuggingFace model).
            input_ids: Tokenized input tensor.
            config: Optional experiment config for precision detection.

        Returns:
            FlopsResult with the estimate and provenance information.
        """

        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]

        # Determine actual compute precision (BNB always computes at FP16)
        precision = self._get_compute_precision(config)

        # Strategy 1: calflops library (most accurate)
        result = self._try_calflops(model, seq_len, precision)
        if result is not None:
            return result

        # Strategy 2: Architecture-based calculation
        result = self._try_architecture(model, seq_len, precision)
        if result is not None:
            return result

        # Strategy 3: Parameter-based approximation
        return self._parameter_estimate(model, seq_len, precision)

    def _get_compute_precision(self, config: ExperimentConfig | None) -> str:
        """Determine actual compute precision.

        BitsAndBytes quantization stores weights compressed but computes at FP16.

        Args:
            config: Experiment configuration (may be None).

        Returns:
            Precision string (e.g., "fp16", "fp32").
        """
        if config is None:
            return "fp16"

        # Check for BNB quantization in PyTorch config
        pytorch_cfg = config.pytorch
        if pytorch_cfg and (pytorch_cfg.load_in_4bit or pytorch_cfg.load_in_8bit):
            # BNB always dequantizes to FP16 (or bfloat16 for 4bit compute dtype)
            return pytorch_cfg.bnb_4bit_compute_dtype if pytorch_cfg.load_in_4bit else "fp16"

        # Use the config's precision field
        precision = config.precision.lower()
        if precision in ("float32", "fp32"):
            return "fp32"
        if precision in ("bfloat16", "bf16"):
            return "bf16"
        return "fp16"

    def _try_calflops(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult | None:
        """Try to estimate FLOPs using calflops library.

        Args:
            model: The model to measure.
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult if successful, None otherwise.
        """
        try:
            from calflops import calculate_flops

            # calflops returns (flops, macs, params) or similar
            # The exact return format may vary by version
            result = calculate_flops(
                model=model,
                input_shape=(1, seq_len),
                transformer_tokenizer=None,
                print_results=False,
                output_as_string=False,
            )

            # Handle different return formats
            flops = result[0] if isinstance(result, tuple) else result

            if flops is not None and flops > 0:
                logger.debug("calflops estimation: %.2e FLOPs", flops)
                return FlopsResult(
                    value=float(flops),
                    method="calflops",
                    confidence="high",
                    precision=precision,
                )

        except ImportError:
            logger.debug("calflops not installed, trying architecture-based")
        except Exception as e:
            logger.debug("calflops failed: %s, trying architecture-based", e)

        return None

    def _try_architecture(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult | None:
        """Try to estimate FLOPs from model architecture config.

        Uses the formula for decoder-only transformers:
        - Attention: 4 * seq * hidden (Q,K,V,O projections) + 2 * seq² * head_dim
        - FFN: 8 * hidden² (assuming 4x intermediate)

        Args:
            model: The model (must have .config attribute).
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult if successful, None otherwise.
        """
        try:
            model_config = model.config

            hidden = model_config.hidden_size
            layers = model_config.num_hidden_layers
            # num_attention_heads accessed to verify config exists
            _ = model_config.num_attention_heads

            # Get intermediate size (FFN), default to 4x hidden
            intermediate = getattr(model_config, "intermediate_size", hidden * 4)

            # Per-layer FLOPs (simplified for decoder-only):
            # Attention: Q, K, V, O projections + attention scores
            # QKV projections: 3 * 2 * hidden * hidden = 6 * hidden²
            # Output projection: 2 * hidden * hidden = 2 * hidden²
            # Attention scores: 2 * seq * seq * head_dim * heads = 2 * seq² * hidden
            attn_proj_flops = 8 * hidden * hidden
            attn_score_flops = 2 * seq_len * seq_len * hidden

            # FFN: up projection + down projection (with intermediate size)
            # Up: 2 * hidden * intermediate
            # Down: 2 * intermediate * hidden
            # If using GLU (e.g., LLaMA): add another up projection
            has_glu = getattr(model_config, "hidden_act", "").lower() in ["silu", "swiglu", "gelu"]
            ffn_multiplier = 3 if has_glu else 2
            ffn_flops = ffn_multiplier * 2 * hidden * intermediate

            per_layer = attn_proj_flops + attn_score_flops + ffn_flops
            total = layers * per_layer * seq_len

            model_type = getattr(model_config, "model_type", "unknown")
            logger.debug(
                "Architecture estimation for %s: %.2e FLOPs (%d layers, %d hidden)",
                model_type,
                total,
                layers,
                hidden,
            )

            return FlopsResult(
                value=float(total),
                method="architecture",
                confidence="medium",
                precision=precision,
                notes=f"Based on model config: {model_type}",
            )

        except AttributeError as e:
            logger.debug("Architecture estimation failed (missing attr): %s", e)
        except Exception as e:
            logger.debug("Architecture estimation failed: %s", e)

        return None

    def _parameter_estimate(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult:
        """Estimate FLOPs based on parameter count (2*P approximation).

        This is the fallback when other methods fail. Uses the rule of thumb
        that inference requires ~2 FLOPs per parameter per token.

        Args:
            model: The model.
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult (always succeeds, may return 0 on error).
        """
        try:
            params = sum(p.numel() for p in model.parameters())
            flops = 2 * params * seq_len

            logger.debug(
                "Parameter-based estimation: %.2e FLOPs (2 * %d params * %d tokens)",
                flops,
                params,
                seq_len,
            )

            return FlopsResult(
                value=float(flops),
                method="parameter_estimate",
                confidence="low",
                precision=precision,
                notes=f"Approximation: 2 * {params:,} params * {seq_len} tokens",
            )

        except Exception as e:
            logger.warning("All FLOPs estimation methods failed: %s", e)
            return FlopsResult(
                value=0.0,
                method="parameter_estimate",
                confidence="low",
                precision=precision,
                notes="Could not estimate FLOPs - all methods failed",
            )


# Module-level convenience instance
_default_estimator: FlopsEstimator | None = None


def get_flops_estimator() -> FlopsEstimator:
    """Get or create the default FlopsEstimator instance."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = FlopsEstimator()
    return _default_estimator


def estimate_flops(
    model: Any,
    input_ids: Any,
    config: ExperimentConfig | None = None,
) -> FlopsResult:
    """Convenience function to estimate FLOPs (legacy backward-compat wrapper).

    For new code, prefer estimate_flops_palm() which is the v2.0 primary method.

    Args:
        model: The model to measure.
        input_ids: Tokenized input tensor.
        config: Optional experiment config.

    Returns:
        FlopsResult with the estimate and provenance.
    """
    return get_flops_estimator().estimate(model, input_ids, config)


__all__ = [
    "FlopsEstimator",
    "_count_non_embedding_params",
    "_count_params_from_config",
    "estimate_flops",
    "estimate_flops_palm",
    "estimate_flops_palm_from_config",
    "get_flops_estimator",
]
