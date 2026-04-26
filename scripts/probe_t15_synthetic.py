#!/usr/bin/env python3
"""Run the T1.5 diff against the synthetic user-error corpus.

Specifically answers: does T1.5 catch the 2 library_normalisation cases
that the shipped probe missed in Phase B?

If yes → adopting T1.5 would push synthetic coverage from 56% → ~69%.
"""

from __future__ import annotations

import os

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine
from scripts.probe_t15_poc import t15_tensorrt, t15_transformers, t15_vllm
from scripts.synthetic_bad_configs import build_corpus

T15 = {
    "transformers": (TransformersEngine(), t15_transformers),
    "vllm": (VLLMEngine(), t15_vllm),
    "tensorrt": (TensorRTEngine(), t15_tensorrt),
}


def main() -> int:
    corpus = build_corpus()
    print(f"Running T1.5 against {len(corpus)} synthetic cases\n")

    print("=" * 90)
    print(f"{'CASE':40s}  {'EXP TIER':9s}  {'T1.5 FINDS':40s}")
    print("=" * 90)

    t15_caught = 0
    t15_total_lib_norm = 0
    for case in corpus:
        try:
            cfg = case.factory()
        except Exception:
            print(f"{case.name:40s}  {case.expected_tier:9s}  (schema rejected at construction)")
            continue

        engine, fn = T15[cfg.engine]
        diffs = fn(engine, cfg)
        diff_str = ", ".join(f"{k}({v[0]}→{v[1]})" for k, v in list(diffs.items())[:3])
        print(f"{case.name:40s}  {case.expected_tier:9s}  {diff_str[:40]!s:40s}")

        if case.expected_tier == "T1.5":
            t15_total_lib_norm += 1
            if diffs:
                t15_caught += 1

    print()
    print(f"T1.5 on library_normalisation cases: {t15_caught}/{t15_total_lib_norm} caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
