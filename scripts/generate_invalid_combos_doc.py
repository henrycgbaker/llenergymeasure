#!/usr/bin/env python3
"""Generate docs/generated/invalid-combos.md from config validators.

This script programmatically documents all invalid parameter combinations
by extracting validation rules from the codebase and test results.

NB: This script is purely a documentation generator
it writes markdown to docs/generated/invalid-combos.md and has no connection to config loading or the runtime.

The script uses the introspection module for streaming constraints (SSOT),
but maintains static lists for validation errors and runtime limitations
that require more context than can be extracted programmatically.

Run: python scripts/generate_invalid_combos_doc.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from llenergymeasure.config.introspection import (
        get_capability_matrix_markdown,
        get_streaming_constraints,
        get_validation_rules,
    )

    USE_INTROSPECTION = True
except ImportError:
    USE_INTROSPECTION = False

# Invalid combinations extracted from validators and test results
# Format: (engine, parameter/combo, reason, resolution)
INVALID_COMBOS: list[tuple[str, str, str, str]] = [
    # Cross-engine validation rules (from ExperimentConfig validator)
    (
        "transformers",
        "parallelism.strategy=pipeline_parallel",
        "PyTorch's generate() requires full model access for autoregressive generation",
        "Use engine='vllm' or engine='tensorrt' for pipeline parallel",
    ),
    (
        "vllm",
        "parallelism.strategy=data_parallel",
        "vLLM manages multi-GPU internally via Ray/tensor parallel",
        "Use parallelism.strategy='tensor_parallel' or 'pipeline_parallel'",
    ),
    (
        "vllm",
        "quantization.load_in_8bit=True",
        "vLLM does not support bitsandbytes 8-bit quantization",
        "Use vllm.quantization_method (awq, gptq, fp8) for quantized inference",
    ),
    (
        "tensorrt",
        "fp_precision=float32",
        "TensorRT-LLM is optimised for lower precision inference",
        "Use fp_precision='float16' or 'bfloat16'",
    ),
    (
        "tensorrt",
        "quantization.load_in_4bit=True",
        "TensorRT does not support bitsandbytes quantization",
        "Use tensorrt.quantization.method (fp8, int8_sq, int4_awq)",
    ),
    (
        "tensorrt",
        "quantization.load_in_8bit=True",
        "TensorRT does not support bitsandbytes quantization",
        "Use tensorrt.quantization.method (fp8, int8_sq, int8_weight_only)",
    ),
    # Shared validation rules (from QuantizationConfig validator)
    (
        "all",
        "quantization.quantization=True (without method)",
        "quantization flag alone doesn't specify which method to use",
        "Set load_in_4bit=True or load_in_8bit=True",
    ),
    (
        "all",
        "quantization.load_in_4bit + load_in_8bit",
        "Cannot use both 4-bit and 8-bit quantization simultaneously",
        "Choose one: load_in_4bit=True OR load_in_8bit=True",
    ),
]

# Streaming mode constraints (params affected by streaming=True)
# Format: (engine, parameter, behaviour with streaming, impact)
STREAMING_CONSTRAINTS: list[tuple[str, str, str, str]] = [
    (
        "all",
        "transformers.batch_size / vllm.max_num_seqs",
        "Ignored - processes 1 request at a time",
        "Reduced throughput but accurate latency",
    ),
    (
        "transformers",
        "transformers.torch_compile",
        "May cause graph-tracing errors",
        "Falls back to non-compiled inference",
    ),
    (
        "transformers",
        "transformers.batching_strategy",
        "Ignored - always sequential",
        "No batching optimisation",
    ),
    (
        "vllm",
        "vllm.enable_chunked_prefill",
        "May interfere with TTFT measurement",
        "Consider disabling for accurate TTFT",
    ),
]

# Known hardware/model limitations (not config validation, but runtime)
RUNTIME_LIMITATIONS: list[tuple[str, str, str, str]] = [
    (
        "transformers",
        "transformers.attn_implementation=flash_attention_2",
        "flash-attn requires Ampere+ GPU (SM80+); fails on older architectures",
        "Use attn_implementation='sdpa' on pre-Ampere GPUs",
    ),
    (
        "transformers",
        "transformers.attn_implementation=flash_attention_3",
        "FA3 requires the flash_attn_3 package (built from flash-attn hopper/ directory) and Ampere+ GPU (SM80+). The Docker PyTorch image includes it pre-built",
        "Install flash_attn_3 from source, or use the Docker runner",
    ),
    (
        "vllm",
        "vllm.kv_cache_dtype=fp8",
        "FP8 KV cache requires Hopper (H100) or newer GPU",
        "Use kv_cache_dtype='auto' for automatic selection",
    ),
    (
        "vllm",
        "vllm.attention.engine=FLASHINFER",
        "FlashInfer requires JIT compilation on first use",
        "Use attention.engine='auto' or 'FLASH_ATTN'",
    ),
    (
        "vllm",
        "vllm.attention.engine=TORCH_SDPA",
        "TORCH_SDPA not registered in vLLM attention backends",
        "Use attention.engine='auto' or 'FLASH_ATTN'",
    ),
    (
        "vllm",
        "vllm.quantization_method=awq/gptq",
        "Requires a pre-quantized model checkpoint",
        "Use a quantized model (e.g., TheBloke/*-AWQ) or omit",
    ),
    (
        "vllm",
        "vllm.load_format=pt",
        "Model checkpoint must have .bin files (not just safetensors)",
        "Use load_format='auto' or 'safetensors'",
    ),
    (
        "tensorrt",
        "tensorrt.quant.quant_algo=FP8",
        "FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises ConfigurationError — no silent emulation or fallback",
        "Use INT8, W4A16_AWQ, W4A16_GPTQ, or W8A16 on A100",
    ),
    (
        "tensorrt",
        "tensorrt.quantization.method=int8_sq",
        "INT8 SmoothQuant requires calibration dataset",
        "Provide tensorrt.quantization.calibration config or use a supported quantization method",
    ),
]


def generate_markdown() -> str:
    """Generate the invalid combinations markdown document."""
    lines = [
        "# Invalid Parameter Combinations",
        "",
        "> Auto-generated from config validators and test results.",
        f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "This document lists parameter combinations that will fail validation or runtime.",
        "The tool validates these at config load time and provides clear error messages.",
        "",
        "## Config Validation Errors",
        "",
        "These combinations are rejected at config load time with a clear error message.",
        "",
        "| Engine | Invalid Combination | Reason | Resolution |",
        "|---------|---------------------|--------|------------|",
    ]

    # Use SSOT validation rules if available, otherwise fall back to static list
    if USE_INTROSPECTION:
        validation_rules = get_validation_rules()
        for rule in validation_rules:
            lines.append(
                f"| {rule['engine']} | `{rule['combination']}` | "
                f"{rule['reason']} | {rule['resolution']} |"
            )
    else:
        for engine, combo, reason, resolution in INVALID_COMBOS:
            lines.append(f"| {engine} | `{combo}` | {reason} | {resolution} |")

    # Streaming constraints section
    lines.extend(
        [
            "",
            "## Streaming Mode Constraints",
            "",
            "When `streaming=True`, certain parameters are ignored or behave differently",
            "because streaming requires sequential per-request processing to measure TTFT/ITL.",
            "",
            "| Engine | Parameter | Behaviour with streaming=True | Impact |",
            "|---------|-----------|------------------------------|--------|",
        ]
    )

    # Use introspection module if available (SSOT), otherwise fall back to static list
    if USE_INTROSPECTION:
        introspection_constraints = get_streaming_constraints()
        for param, explanation in introspection_constraints.items():
            engine = (
                "transformers"
                if param.startswith("transformers.")
                else ("vllm" if param.startswith("vllm.") else "all")
            )
            lines.append(f"| {engine} | `{param}` | {explanation} | See docs |")
    else:
        for engine, param, behaviour, impact in STREAMING_CONSTRAINTS:
            lines.append(f"| {engine} | `{param}` | {behaviour} | {impact} |")

    lines.extend(
        [
            "",
            "**When to use streaming=True:**",
            "- Measuring user-perceived latency (TTFT, ITL)",
            "- Evaluating real-time chat/assistant workloads",
            "- MLPerf inference latency benchmarks",
            "",
            "**When to use streaming=False:**",
            "- Throughput benchmarking",
            "- Batch processing workloads",
            "- torch.compile optimisation testing",
        ]
    )

    lines.extend(
        [
            "",
            "## Runtime Limitations",
            "",
            "These combinations pass config validation but may fail at runtime",
            "due to hardware, model, or package requirements.",
            "",
            "| Engine | Parameter | Limitation | Resolution |",
            "|---------|-----------|------------|------------|",
        ]
    )

    for engine, param, limitation, resolution in RUNTIME_LIMITATIONS:
        lines.append(f"| {engine} | `{param}` | {limitation} | {resolution} |")

    lines.extend(
        [
            "",
            "## Engine Capability Matrix",
            "",
        ]
    )

    # Use SSOT capability matrix if available
    if USE_INTROSPECTION:
        lines.append(get_capability_matrix_markdown())
    else:
        lines.extend(
            [
                "| Feature | PyTorch | vLLM | TensorRT |",
                "|---------|---------|------|----------|",
                "| Tensor Parallel | ✅ | ✅ | ✅ |",
                "| Pipeline Parallel | ❌ | ✅ | ✅ |",
                "| Data Parallel | ✅ | ❌ | ✅ |",
                "| BitsAndBytes (4/8-bit) | ✅ | ❌¹ | ❌ |",
                "| Native Quantization | ❌ | ✅ (AWQ/GPTQ/FP8) | ✅ (FP8/INT8) |",
                "| float32 precision | ✅ | ✅ | ❌ |",
                "| float16 precision | ✅ | ✅ | ✅ |",
                "| bfloat16 precision | ✅ | ✅ | ✅ |",
                "| Streaming (TTFT/ITL) | ✅ | ✅ | ✅ |",
                "| LoRA Adapters | ✅ | ✅ | ✅ |",
                "| Speculative Decoding | ✅ | ✅ | ✅ |",
                "",
                "¹ vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommended Configurations by Use Case",
            "",
            "### Memory-Constrained (Consumer GPU)",
            "```yaml",
            "engine: pytorch",
            "quantization:",
            "  load_in_4bit: true",
            "  bnb_4bit_quant_type: nf4",
            "```",
            "",
            "### High Throughput (Production)",
            "```yaml",
            "engine: vllm",
            "vllm:",
            "  gpu_memory_utilization: 0.9",
            "  enable_prefix_caching: true",
            "```",
            "",
            "### Maximum Performance (Ampere+)",
            "```yaml",
            "engine: tensorrt",
            "fp_precision: float16",
            "tensorrt:",
            "  quantization:",
            "    method: fp8  # Hopper only",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Generate and write the invalid combos documentation."""
    output_path = Path(__file__).parent.parent / "docs" / "generated" / "invalid-combos.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_markdown()
    output_path.write_text(content)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
