#!/usr/bin/env python3
"""Raw transformers canary: bypasses our wrapper's greedy-strip.

Runs inside llenergymeasure:pytorch. Accepts two inputs on stdin:
    {
      "cfg": <ExperimentConfig JSON>,
      "raw_generate_kwargs": {...}   # user's DECLARED kwargs, not wrapper-processed
    }

Calls HF generate() with the raw kwargs (no wrapper stripping), captures
warnings/stderr, reports what HF surfaces about dormancy.

Used by scripts/probe_raw_canary_poc.py to answer:
> If we removed the transformers wrapper's greedy-strip, would HF's runtime
  warnings give us the same dormancy signal T0 currently catches?
"""

from __future__ import annotations

import io
import json
import re
import sys
import warnings as warnings_mod
from contextlib import redirect_stderr

DORMANCY_PATTERNS = [
    (r"`temperature`[^\n]*is set", "temperature"),
    (r"`top_p`[^\n]*is set", "top_p"),
    (r"`top_k`[^\n]*is set", "top_k"),
    (r"`min_p`[^\n]*is set", "min_p"),
    (r"do_sample=False[^\n]*has no effect", "do_sample_redundant"),
    (r"temperature[^\n]*no effect", "temperature"),
    (r"top_p[^\n]*no effect", "top_p"),
    (r"top_k[^\n]*no effect", "top_k"),
    (r"The following generation flags are not valid and may be ignored: \[([^\]]+)\]", "__list__"),
    (r"Unused or unrecognized kwargs: ([^\n]+)", "__unused__"),
]


def _extract_dormancy_hf(text: str) -> list[str]:
    hints: set[str] = set()
    for pattern, key in DORMANCY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            continue
        if key == "__list__":
            for m in matches:
                for field in re.findall(r"'([^']+)'", m):
                    hints.add(field)
        elif key == "__unused__":
            for m in matches:
                for field in re.findall(r"(\w+)", m):
                    if field not in ("and", "or"):
                        hints.add(field)
        else:
            hints.add(key)
    return sorted(hints)


def run_raw_canary(cfg_dict: dict, raw_kwargs: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llenergymeasure.config.models import ExperimentConfig

    cfg = ExperimentConfig(**cfg_dict)

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
                # Minimal load — force GPU, no device_map heuristic tricks.
                tokenizer = AutoTokenizer.from_pretrained(cfg.task.model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                import torch

                model = AutoModelForCausalLM.from_pretrained(
                    cfg.task.model, torch_dtype=torch.bfloat16, device_map="auto"
                )
                model.eval()

                inputs = tokenizer("Hi", return_tensors="pt")
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

                # Raw kwargs — pass through directly without wrapper strip
                gen_kwargs = dict(raw_kwargs)
                gen_kwargs["max_new_tokens"] = 2

                with torch.inference_mode():
                    model.generate(**inputs, **gen_kwargs)

                del model
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            warnings_mod.showwarning = old_handler

    stderr_text = stderr_buf.getvalue()
    relevant = []
    for line in stderr_text.splitlines():
        if any(
            m in line
            for m in (
                "WARNING",
                "Warning",
                "has no effect",
                "not valid",
                "is set to",
                "ignored",
                "Unused",
            )
        ):
            relevant.append(line)

    combined = "\n".join(captured_warnings + relevant)
    dormancy_hints = _extract_dormancy_hf(combined)

    return {
        "ok": True,
        "dormancy_hints": dormancy_hints,
        "captured_warnings": captured_warnings,
        "relevant_stderr": relevant[:50],
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        json.dump({"ok": False, "error": f"stdin parse: {e}"}, sys.stdout)
        return 1

    try:
        result = run_raw_canary(payload["cfg"], payload.get("raw_generate_kwargs", {}))
    except Exception as e:
        import traceback

        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=8),
        }

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
