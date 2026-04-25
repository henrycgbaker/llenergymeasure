"""PoC-D: Does the final design's proposed runtime warning capture actually
work on the specific case the revisions-doc PoC empirically showed failed?

Background
----------
revisions §6c demonstrated that `contextlib.redirect_stderr` + root-logger
handler MISSED vLLM's `sampling_params.py:368` warning emission. The final
design's §4.7 proposes `logging.getLogger(_engine_logger_name(config.engine))`
plus a logging.Handler. This PoC verifies that specific fix on the exact
failure case.

Three capture strategies tested:
  1. Root logger + handler (what previously failed)
  2. Engine-named logger + handler (the design's proposal, expected to work)
  3. Explicit submodule logger (belt-and-braces, strongest proposal)

Plus the HF variant: does `logging.getLogger("transformers")` catch the
`logger.warning_once` that HF uses for GenerationConfig.validate()?

Decision criteria (pre-committed)
---------------------------------
- Strategy 2 (engine-named logger) catches vLLM's epsilon-clamp warning
    -> design §4.7 proposal validated; ship as-is.
- Strategy 2 fails, strategy 3 works
    -> update design §4.7 to use explicit submodule logger name.
- Strategies 2 and 3 both fail
    -> runtime warning capture mechanism is broken; §4.7 needs
       fundamental redesign (subprocess+stderr-pipe). Significantly
       weakens the feedback channel #1 story.

Run
---
  /usr/bin/python3.10 scripts/probe_runtime_warning_capture_v2.py
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field


@dataclass
class CaptureResult:
    strategy_name: str
    target_library: str
    warnings_warn_count: int = 0
    log_record_count: int = 0
    captured_messages: list[str] = field(default_factory=list)


class _ListHandler(logging.Handler):
    """Logging handler that appends record messages to a list."""

    def __init__(self, out: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self.out = out

    def emit(self, record: logging.LogRecord) -> None:
        self.out.append(record.getMessage())


def run_capture(
    strategy_name: str,
    logger_name: str | None,
    provoke_fn,
    target_library: str,
) -> CaptureResult:
    """Run a provoke_fn with the given logger name attached to a ListHandler."""
    result = CaptureResult(strategy_name=strategy_name, target_library=target_library)

    log_records: list[str] = []
    handler = _ListHandler(log_records)

    if logger_name is None:
        # Strategy 1: attach to root
        target_logger = logging.getLogger()
    else:
        target_logger = logging.getLogger(logger_name)

    # Propagate: if True, child loggers propagate to this handler via root;
    # if attached at root, child messages bubble up only if their own
    # level allows and their propagate flag is True (default).
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)

    warnings_buf: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                provoke_fn()
            except Exception as e:
                result.captured_messages.append(f"[EXC] {type(e).__name__}: {e}")
            finally:
                warnings_buf = list(w)
    finally:
        target_logger.removeHandler(handler)

    result.warnings_warn_count = len(warnings_buf)
    result.log_record_count = len(log_records)
    result.captured_messages.extend(str(wm.message) for wm in warnings_buf)
    result.captured_messages.extend(log_records)
    return result


# --- Provocation: replicate revisions §6c's exact failure case ---
def provoke_vllm_epsilon_clamp() -> None:
    from vllm import SamplingParams

    SamplingParams(temperature=0.001)  # epsilon clamp -> 0.01 with WARNING log


def provoke_hf_greedy_dormancy() -> None:
    from transformers import GenerationConfig

    # Greedy + temperature is the canonical transformers case
    GenerationConfig(do_sample=False, temperature=0.9, top_p=0.95)


def main() -> None:
    print("=" * 78)
    print("PoC-D: Runtime warning capture on revisions §6c failure case")
    print("=" * 78)
    print()

    vllm_strategies = [
        ("1: root_logger", None),
        ("2: 'vllm'", "vllm"),
        ("3: 'vllm.sampling_params'", "vllm.sampling_params"),
    ]

    print("--- vLLM: SamplingParams(temperature=0.001) (epsilon clamp) ---")
    print()
    print(f"{'STRATEGY':<32}  {'warnings.warn':<15}  {'log_records':<14}  CAPTURED")
    print("-" * 78)
    for name, logger_name in vllm_strategies:
        try:
            r = run_capture(name, logger_name, provoke_vllm_epsilon_clamp, "vllm")
        except ImportError:
            print(f"{name:<32}  (vllm not importable on host)")
            continue
        captured_preview = "; ".join(m[:40] for m in r.captured_messages[:2])
        caught_clamp = any(
            "0.001" in m or "0.01" in m or "temperature" in m.lower() for m in r.captured_messages
        )
        marker = "HIT" if caught_clamp else "miss"
        print(
            f"{name:<32}  {r.warnings_warn_count:<15}  "
            f"{r.log_record_count:<14}  [{marker}] {captured_preview[:50]}"
        )

    print()

    hf_strategies = [
        ("1: root_logger", None),
        ("2: 'transformers'", "transformers"),
        (
            "3: 'transformers.generation.configuration_utils'",
            "transformers.generation.configuration_utils",
        ),
    ]

    print("--- transformers: GenerationConfig(do_sample=False, temperature=0.9) ---")
    print()
    print(f"{'STRATEGY':<52}  {'warnings.warn':<15}  {'log_records':<14}  CAPTURED")
    print("-" * 78)
    for name, logger_name in hf_strategies:
        try:
            r = run_capture(name, logger_name, provoke_hf_greedy_dormancy, "transformers")
        except ImportError:
            print(f"{name:<52}  (transformers not importable on host)")
            continue
        captured_preview = "; ".join(m[:45] for m in r.captured_messages[:2])
        caught = any(
            "temperature" in m.lower() or "may be ignored" in m or "do_sample" in m.lower()
            for m in r.captured_messages
        )
        marker = "HIT" if caught else "miss"
        print(
            f"{name:<52}  {r.warnings_warn_count:<15}  "
            f"{r.log_record_count:<14}  [{marker}] {captured_preview[:40]}"
        )

    print()
    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    print("Look at the strategy-2 row (engine-named logger) for each library:")
    print("  HIT  -> feedback channel #1 mechanism works as designed in §4.7.")
    print("  miss -> need to fall back to strategy 3 (explicit submodule logger),")
    print("          or, if strategy 3 also misses, redesign the capture using")
    print("          subprocess-level stderr piping.")
    print()
    print("Raw signal for human review:")
    print("  - The revisions §6c failure specifically used redirect_stderr +")
    print("    root logger. Strategies 2 and 3 here avoid redirect_stderr")
    print("    entirely, which should be the fix.")
    print("  - If 'HIT' appears for strategy 2 on vLLM, the feedback-channel-#1")
    print("    design is correct and §4.7's proposal ships as-is.")


if __name__ == "__main__":
    main()
