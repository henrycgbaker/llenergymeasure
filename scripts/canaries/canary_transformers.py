#!/usr/bin/env python3
"""Transformers T3 canary — runs inside the llenergymeasure:transformers image.

Reads ExperimentConfig as JSON on stdin. Loads model, runs generate() with a
tiny prompt, captures stderr + warnings emitted during generate.

Outputs JSON on stdout:
    {
      "ok": true|false,
      "error": "...",          # if ok=false
      "warnings": ["...", ...], # captured during load + generate
      "stderr": "...",          # captured stderr (filtered)
      "first_forward_sec": 1.23,
      "dormancy_hints": ["temperature", "top_p"]  # extracted from warnings
    }

Why a separate script (not inline):
  - Isolates the canary process crash boundary.
  - Runs unchanged inside a container (transformers pre-installed).
  - Uses stdin/stdout JSON for clean orchestration.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import warnings as warnings_mod
from contextlib import redirect_stderr

DORMANCY_PATTERNS = [
    (r"temperature[^\n]*no effect", "temperature"),
    (r"top_p[^\n]*no effect", "top_p"),
    (r"top_k[^\n]*no effect", "top_k"),
    (r"min_p[^\n]*no effect", "min_p"),
    (r"do_sample[^\n]*no effect", "do_sample"),
    (r"`temperature`[^\n]*is set", "temperature"),
    (r"`top_p`[^\n]*is set", "top_p"),
    (r"`top_k`[^\n]*is set", "top_k"),
    (r"The following generation flags are not valid and may be ignored: \[([^\]]+)\]", "__list__"),
]


def _extract_dormancy(text: str) -> list[str]:
    hints: set[str] = set()
    for pattern, key in DORMANCY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            continue
        if key == "__list__":
            for m in matches:
                for field in re.findall(r"'([^']+)'", m):
                    hints.add(field)
        else:
            hints.add(key)
    return sorted(hints)


def run_canary(cfg_dict: dict) -> dict:
    """Run a minimal transformers canary on the given config dict."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    cfg = ExperimentConfig(**cfg_dict)
    engine = TransformersEngine()

    stderr_buf = io.StringIO()
    captured_warnings: list[str] = []

    def _warning_handler(message, category, filename, lineno, file=None, line=None):
        captured_warnings.append(f"{category.__name__}: {message}")

    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("always")
        old_handler = warnings_mod.showwarning
        warnings_mod.showwarning = _warning_handler
        try:
            with redirect_stderr(stderr_buf):
                # Load model + tokenizer
                model, tokenizer = engine.load_model(cfg)

                # Tokenize a short prompt
                inputs = tokenizer("Hi", return_tensors="pt")
                if hasattr(model, "device"):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Build generate kwargs from config, cap to 2 tokens for canary
                gen_kwargs = engine._build_generate_kwargs(cfg)
                gen_kwargs["max_new_tokens"] = 2

                t0 = time.perf_counter()
                import torch

                with torch.inference_mode():
                    model.generate(**inputs, **gen_kwargs)
                first_forward_sec = time.perf_counter() - t0

                # Cleanup
                engine.cleanup((model, tokenizer))
        finally:
            warnings_mod.showwarning = old_handler

    stderr_text = stderr_buf.getvalue()
    # Filter stderr to WARNING lines + relevant substrings
    relevant_stderr = [
        line
        for line in stderr_text.splitlines()
        if any(m in line for m in ("WARNING", "UserWarning", "has no effect", "may be ignored"))
    ]

    combined_text = "\n".join(captured_warnings + relevant_stderr)
    dormancy_hints = _extract_dormancy(combined_text)

    return {
        "ok": True,
        "warnings": captured_warnings,
        "stderr_filtered": relevant_stderr,
        "first_forward_sec": first_forward_sec,
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
