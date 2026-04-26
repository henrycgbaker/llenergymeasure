"""Synthesise realistic user-error ExperimentConfigs for probe validation.

The full-suite YAML is deliberately well-formed (sweep: groups prevent
cross-axis invalid combos). Real user configs don't have these guards.
This module generates a corpus of configurations a real user might write,
each annotated with the *class of error* we expect a good probe to catch.

Categories
----------
- sampling_mismatch: conflicting sampling dials (greedy + sampling params)
- library_normalisation: inputs that trigger library post_init clamping
- hardware_incompat: quant/dtype combos the hardware can't support
- cross_field_contradiction: field combinations the library rejects
- schema_preventable: combos Pydantic validators already prevent

Each config comes with an annotation dict describing what the "right"
probe behaviour is — so the PoC can check whether T0/T1/T2/T5 actually
caught the expected class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llenergymeasure.config.engine_configs import (
    TensorRTConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TransformersConfig,
    TransformersSamplingConfig,
    VLLMConfig,
    VLLMEngineConfig,
    VLLMSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig


@dataclass
class SyntheticCase:
    """One synthetic config + what we expect the probe to catch."""

    name: str
    category: str
    expected_tier: str  # "T0" | "T1" | "T2" | "T5" | "schema" | "runtime"
    expected_dormant_fields: list[str]  # empty if none expected
    expect_error: bool
    notes: str
    factory: Any  # callable returning ExperimentConfig, or raising ValidationError


def _tf(sampling: TransformersSamplingConfig | None = None, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "Qwen/Qwen2.5-0.5B"},
        engine="transformers",
        transformers=TransformersConfig(sampling=sampling or TransformersSamplingConfig(), **kw),
    )


def _vllm(
    sampling: VLLMSamplingConfig | None = None, engine: VLLMEngineConfig | None = None, **kw
) -> ExperimentConfig:
    vc = VLLMConfig(sampling=sampling, engine=engine, **kw)
    return ExperimentConfig(
        task={"model": "Qwen/Qwen2.5-0.5B"},
        engine="vllm",
        vllm=vc,
    )


def _trt(sampling: TensorRTSamplingConfig | None = None, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "Qwen/Qwen2.5-0.5B"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(sampling=sampling, **kw),
    )


def build_corpus() -> list[SyntheticCase]:
    """Return the full synthetic-error corpus."""
    cases: list[SyntheticCase] = []

    # --- Category 1: sampling mismatch (the classic footgun) ---

    cases.append(
        SyntheticCase(
            name="tf_greedy_with_sampling",
            category="sampling_mismatch",
            expected_tier="T0",
            expected_dormant_fields=[
                "transformers.sampling.temperature",
                "transformers.sampling.top_p",
                "transformers.sampling.top_k",
                "transformers.sampling.min_p",
            ],
            expect_error=False,
            notes="Classic: user says do_sample=False but sets temp/top_p/top_k/min_p",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(
                    do_sample=False,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=40,
                    min_p=0.1,
                )
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="tf_temp_zero_with_sampling",
            category="sampling_mismatch",
            expected_tier="T0",
            expected_dormant_fields=[
                "transformers.sampling.temperature",
                "transformers.sampling.top_p",
                "transformers.sampling.top_k",
            ],
            expect_error=False,
            notes="do_sample=True but temp=0 (treated as greedy by wrapper)",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(
                    do_sample=True,
                    temperature=0.0,
                    top_p=0.9,
                    top_k=40,
                )
            ),
        )
    )

    # --- Category 2: library normalisation (T1.5 territory) ---

    cases.append(
        SyntheticCase(
            name="vllm_near_zero_temp",
            category="library_normalisation",
            expected_tier="T1.5",
            expected_dormant_fields=["top_p", "top_k"],
            expect_error=False,
            notes="vLLM clamps temperature<0.01 and normalizes top_p/top_k to 1.0/0 under near-zero temp",
            factory=lambda: _vllm(
                sampling=VLLMSamplingConfig(temperature=0.0, top_p=0.95, top_k=50)
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_temp_epsilon",
            category="library_normalisation",
            expected_tier="T1.5",
            expected_dormant_fields=["temperature"],
            expect_error=False,
            notes="vLLM warns and clamps temperature=0.001 to 0.01",
            factory=lambda: _vllm(
                sampling=VLLMSamplingConfig(temperature=0.001, top_p=0.9, top_k=50)
            ),
        )
    )

    # --- Category 3: hardware incompat ---

    cases.append(
        SyntheticCase(
            name="trt_fp8_on_a100",
            category="hardware_incompat",
            expected_tier="T5",
            expected_dormant_fields=[],
            expect_error=True,
            notes="FP8 quant requires SM>=8.9. A100 is SM 8.0.",
            factory=lambda: _trt(quant=TensorRTQuantConfig(quant_algo="FP8")),
        )
    )

    cases.append(
        SyntheticCase(
            name="trt_fp8_kv_on_a100",
            category="hardware_incompat",
            expected_tier="T5",
            expected_dormant_fields=[],
            expect_error=True,
            notes="FP8 kv_cache_quant_algo requires SM>=8.9",
            factory=lambda: _trt(quant=TensorRTQuantConfig(kv_cache_quant_algo="FP8")),
        )
    )

    # --- Category 4: cross-field contradiction (library rejects) ---

    cases.append(
        SyntheticCase(
            name="vllm_batched_tokens_lt_model_len",
            category="cross_field_contradiction",
            expected_tier="T2",
            expected_dormant_fields=[],
            expect_error=True,
            notes="vLLM requires max_num_batched_tokens >= max_model_len",
            factory=lambda: _vllm(
                engine=VLLMEngineConfig(
                    max_num_batched_tokens=2048,
                    max_model_len=8192,
                )
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_float32_dtype_forbidden",
            category="cross_field_contradiction",
            expected_tier="T2",
            expected_dormant_fields=[],
            expect_error=True,
            notes="vLLM rejects dtype=float32 (per YAML comment)",
            factory=lambda: _vllm(dtype="float32"),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_invalid_quant_name",
            category="cross_field_contradiction",
            expected_tier="T1",
            expected_dormant_fields=[],
            expect_error=True,
            notes="vLLM rejects unknown quantization names",
            factory=lambda: _vllm(engine=VLLMEngineConfig(quantization="nonsense_quant")),
        )
    )

    # --- Category 5: schema-preventable (Pydantic catches at config load) ---

    def _beam_and_sampling():
        from llenergymeasure.config.engine_configs import VLLMBeamSearchConfig

        return _vllm(
            sampling=VLLMSamplingConfig(temperature=0.9),
            # NB: this factory is EXPECTED TO RAISE at construction
            **{"beam_search": VLLMBeamSearchConfig(beam_width=4)},
        )

    cases.append(
        SyntheticCase(
            name="vllm_beam_and_sampling_schema",
            category="schema_preventable",
            expected_tier="schema",
            expected_dormant_fields=[],
            expect_error=True,
            notes="VLLMConfig validator rejects beam_search + sampling (Pydantic-enforced)",
            factory=_beam_and_sampling,
        )
    )

    # --- Category 6: passthrough nonsense (user passthrough_kwargs with bad keys) ---

    cases.append(
        SyntheticCase(
            name="vllm_passthrough_nonsense",
            category="cross_field_contradiction",
            expected_tier="T2",
            expected_dormant_fields=[],
            expect_error=True,
            notes="Nonsense passthrough key should be rejected by EngineArgs",
            factory=lambda: ExperimentConfig(
                task={"model": "Qwen/Qwen2.5-0.5B"},
                engine="vllm",
                passthrough_kwargs={"made_up_arg_xyz_42": 1},
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="tf_passthrough_nonsense",
            category="cross_field_contradiction",
            expected_tier="T2",  # meta-device construction should reject
            expected_dormant_fields=[],
            expect_error=True,
            notes="Transformers passthrough: unknown kwarg reaches from_pretrained",
            factory=lambda: ExperimentConfig(
                task={"model": "Qwen/Qwen2.5-0.5B"},
                engine="transformers",
                passthrough_kwargs={"made_up_arg_xyz_42": 1},
            ),
        )
    )

    # --- Category 7: TRT quant × hardware (FP8 kv+activation on A100) ---

    cases.append(
        SyntheticCase(
            name="trt_fp8_both_on_a100",
            category="hardware_incompat",
            expected_tier="T5",
            expected_dormant_fields=[],
            expect_error=True,
            notes="Both quant_algo and kv_cache_quant_algo FP8 — double error",
            factory=lambda: _trt(
                quant=TensorRTQuantConfig(
                    quant_algo="FP8",
                    kv_cache_quant_algo="FP8",
                )
            ),
        )
    )

    # --- Category 8: valid-but-useless (baseline, should pass clean) ---

    cases.append(
        SyntheticCase(
            name="tf_valid_baseline",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="Sampling-active transformers, no greedy conflict",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                )
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_valid_baseline",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="Plain vLLM sampling config, nothing to catch",
            factory=lambda: _vllm(
                sampling=VLLMSamplingConfig(temperature=0.7, top_p=0.9, top_k=50)
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="trt_valid_baseline",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="Plain TRT sampling config, nothing to catch",
            factory=lambda: _trt(
                sampling=TensorRTSamplingConfig(temperature=0.7, top_p=0.9, top_k=50)
            ),
        )
    )

    # --- Expanded corpus: more variations, stress-test probe ---

    # vLLM sampling edge cases — negative top_p, out-of-range values
    cases.append(
        SyntheticCase(
            name="vllm_negative_top_p",
            category="cross_field_contradiction",
            expected_tier="schema",
            expected_dormant_fields=[],
            expect_error=True,
            notes="top_p=-0.5 should fail Pydantic ge=0.0 validator",
            factory=lambda: _vllm(sampling=VLLMSamplingConfig(temperature=0.7, top_p=-0.5)),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_top_p_over_1",
            category="cross_field_contradiction",
            expected_tier="schema",
            expected_dormant_fields=[],
            expect_error=True,
            notes="top_p=1.5 should fail Pydantic le=1.0 validator",
            factory=lambda: _vllm(sampling=VLLMSamplingConfig(temperature=0.7, top_p=1.5)),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_gpu_mem_over_1",
            category="cross_field_contradiction",
            expected_tier="T2",
            expected_dormant_fields=[],
            expect_error=True,
            notes="gpu_memory_utilization > 1.0 — vLLM rejects at create_engine_config",
            factory=lambda: _vllm(engine=VLLMEngineConfig(gpu_memory_utilization=1.5)),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_tp_without_multigpu",
            category="cross_field_contradiction",
            expected_tier="T2",
            expected_dormant_fields=[],
            expect_error=True,
            notes="tensor_parallel_size=8 on single-GPU host",
            factory=lambda: _vllm(engine=VLLMEngineConfig(tensor_parallel_size=8)),
        )
    )

    # Transformers — more sampling mismatch variations
    cases.append(
        SyntheticCase(
            name="tf_repetition_penalty_greedy",
            category="sampling_mismatch",
            expected_tier="T0",
            expected_dormant_fields=[
                "transformers.sampling.temperature",
                "transformers.sampling.top_p",
                "transformers.sampling.top_k",
            ],
            expect_error=False,
            notes="Greedy with repetition_penalty — repetition_penalty IS still active (not stripped)",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(
                    do_sample=False,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.3,
                )
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="tf_greedy_with_min_new_tokens",
            category="sampling_mismatch",
            expected_tier="T0",
            expected_dormant_fields=[
                "transformers.sampling.temperature",
                "transformers.sampling.top_p",
                "transformers.sampling.top_k",
            ],
            expect_error=False,
            notes="Greedy with min_new_tokens — min_new_tokens survives the strip",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(
                    do_sample=False,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=40,
                    min_new_tokens=5,
                )
            ),
        )
    )

    # TRT edge cases
    cases.append(
        SyntheticCase(
            name="trt_valid_int8",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="INT8 quant is allowed on A100",
            factory=lambda: _trt(quant=TensorRTQuantConfig(quant_algo="INT8")),
        )
    )

    cases.append(
        SyntheticCase(
            name="trt_valid_w4a16",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="W4A16_AWQ quant is allowed on A100",
            factory=lambda: _trt(quant=TensorRTQuantConfig(quant_algo="W4A16_AWQ")),
        )
    )

    # Transformers valid but with unusual combos
    cases.append(
        SyntheticCase(
            name="tf_valid_beam_search",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="Beam search with do_sample=False (beam search semantics)",
            factory=lambda: _tf(
                sampling=TransformersSamplingConfig(do_sample=False, temperature=0.0),
                num_beams=4,
                early_stopping=True,
            ),
        )
    )

    cases.append(
        SyntheticCase(
            name="vllm_n_gt_1",
            category="valid_baseline",
            expected_tier="none",
            expected_dormant_fields=[],
            expect_error=False,
            notes="vLLM n>1 is valid (multiple generations per prompt)",
            factory=lambda: _vllm(sampling=VLLMSamplingConfig(temperature=0.7, top_p=0.9, n=4)),
        )
    )

    return cases


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Synthetic corpus: {len(corpus)} cases")
    for c in corpus:
        print(f"  [{c.category}/{c.expected_tier}] {c.name}: {c.notes}")
