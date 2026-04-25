#!/usr/bin/env python3
"""vLLM T3 canary — runs inside the llenergymeasure:vllm image.

Reads ExperimentConfig JSON on stdin. Creates vLLM LLM(), runs generate() on
a tiny prompt with max_tokens=2, captures stderr/logs during init+generate,
extracts dormancy hints.

Runs inside the vLLM Docker image; called by scripts/probe_t3_canary_poc.py.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

DORMANCY_PATTERNS = [
    (r"top_p[^\n]*1\.0", "top_p"),
    (r"top_k[^\n]*(-1|0)", "top_k"),
    (r"temperature[^\n]*maxed it out", "temperature"),
    (r"Casting [^\n]* to", "dtype_cast"),
    (r"disabling torch\.compile", "torch_compile"),
    (r"Inductor compilation was disabled", "inductor"),
    (r"is enabled", "__enabled_feature__"),  # chunked_prefill etc.
]


def _extract_dormancy(text: str) -> list[str]:
    hints: set[str] = set()
    for pattern, key in DORMANCY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            hints.add(key)
    return sorted(hints)


def run_canary(cfg_dict: dict) -> dict:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.vllm import VLLMEngine

    cfg = ExperimentConfig(**cfg_dict)
    engine = VLLMEngine()

    # Capture both stderr and vLLM's logger output
    stderr_buf = io.StringIO()
    stdout_buf = io.StringIO()
    log_buf = io.StringIO()
    log_handler = logging.StreamHandler(log_buf)
    log_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(log_handler)

    try:
        with redirect_stderr(stderr_buf), redirect_stdout(stdout_buf):
            t_load_start = time.perf_counter()
            llm, sampling_params = engine.load_model(cfg)
            t_load_sec = time.perf_counter() - t_load_start

            # Run a minimal generate — cap max_tokens to 2
            try:
                sampling_params.max_tokens = 2
            except Exception:
                pass

            t_fwd_start = time.perf_counter()
            llm.generate(["Hi"], sampling_params)
            first_forward_sec = time.perf_counter() - t_fwd_start

            # Cleanup
            engine.cleanup((llm, sampling_params))
    finally:
        logging.root.removeHandler(log_handler)

    stderr_text = stderr_buf.getvalue()
    log_text = log_buf.getvalue()

    # Filter relevant lines (WARNING, info about disabled/enabled features, casts)
    relevant = []
    for line in (stderr_text + "\n" + log_text).splitlines():
        if any(
            m in line for m in ("WARNING", "warning", "Casting", "disabled", "enabled", "maxed")
        ):
            relevant.append(line)

    dormancy_hints = _extract_dormancy("\n".join(relevant))

    return {
        "ok": True,
        "t_load_sec": t_load_sec,
        "first_forward_sec": first_forward_sec,
        "relevant_log_lines": relevant[:100],  # cap to avoid huge output
        "dormancy_hints": dormancy_hints,
    }


def main() -> int:
    try:
        cfg_dict = json.load(sys.stdin)
    except Exception as e:
        json.dump({"ok": False, "error": f"stdin parse: {e}"}, sys.stdout)
        return 1

    try:
        result = run_canary(cfg_dict)
    except Exception as e:
        import traceback

        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=6),
        }

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
