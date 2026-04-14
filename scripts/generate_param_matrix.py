#!/usr/bin/env python3
"""Generate docs/generated/parameter-support-matrix.md from test results.

This script reads test results JSON files and generates a parameter support
matrix showing which parameters work with which engines.

Run: python scripts/generate_param_matrix.py [--results-dir results/]

The script expects test results in the format output by tests/runtime/test_all_params.py:
- results/test_results_pytorch.json
- results/test_results_vllm.json
- results/test_results_tensorrt.json

Generate test results with:
    python -m tests.runtime.test_all_params --engine pytorch --output results/test_results_pytorch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llenergymeasure.config.ssot import Engine


def load_test_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load test results from JSON files."""
    results: dict[str, list[dict[str, Any]]] = {}

    for engine in Engine:
        result_file = results_dir / f"test_results_{engine}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                results[engine] = data.get("results", [])
        else:
            results[engine] = []

    return results


def extract_parameter_status(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract parameter status from test results.

    Returns:
        Dict mapping parameter -> engine -> {status, error, validation_status}
    """
    params: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for engine, tests in results.items():
        for test in tests:
            # Handle test_all_params.py output format
            param = test.get("parameter_varied", "")
            value = test.get("parameter_value", "")
            param_key = f"{param}={value}" if value else param

            # Skip baseline tests
            if not param or "baseline" in test.get("config_name", "").lower():
                continue

            passed = test.get("status") == "passed"
            error = test.get("error_summary", "") or ""
            validation_data = test.get("validation", {})
            validation = (
                validation_data.get("validation_status", "UNKNOWN")
                if validation_data
                else "UNKNOWN"
            )

            params[param_key][engine] = {
                "passed": passed,
                "error": error[:100] if error else "",
                "validation": validation,
            }

    return dict(params)


def categorise_parameters(
    params: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Categorise parameters by type (shared vs engine-specific)."""
    categories: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "Core Settings": {},
        "Batching": {},
        "Decoder/Generation": {},
        "Parallelism": {},
        "Quantization": {},
        "Streaming & Simulation": {},
        "PyTorch-specific": {},
        "vLLM-specific": {},
        "TensorRT-specific": {},
    }

    for param, engines_map in params.items():
        param_lower = param.lower()

        if param_lower.startswith("transformers."):
            categories["PyTorch-specific"][param] = engines_map
        elif param_lower.startswith("vllm."):
            categories["vLLM-specific"][param] = engines_map
        elif param_lower.startswith("tensorrt."):
            categories["TensorRT-specific"][param] = engines_map
        elif "batching" in param_lower or "batch" in param_lower:
            categories["Batching"][param] = engines_map
        elif "decoder" in param_lower or "beam" in param_lower:
            categories["Decoder/Generation"][param] = engines_map
        elif "parallel" in param_lower:
            categories["Parallelism"][param] = engines_map
        elif "quantization" in param_lower or "load_in" in param_lower:
            categories["Quantization"][param] = engines_map
        elif "streaming" in param_lower or "traffic" in param_lower:
            categories["Streaming & Simulation"][param] = engines_map
        else:
            categories["Core Settings"][param] = engines_map

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def generate_markdown(
    categories: dict[str, dict[str, dict[str, dict[str, Any]]]],
    results: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate the markdown document."""
    lines = [
        "# Parameter Support Matrix",
        "",
        "> Auto-generated from test results. Run `python scripts/generate_param_matrix.py` to update.",
        f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
    ]

    # Summary stats
    for engine in Engine:
        tests = results.get(engine, [])
        if tests:
            # Note: test_all_params.py uses "status" field with values "passed"/"failed"/"skipped"
            passed = sum(1 for t in tests if t.get("status") == "passed")
            failed = sum(1 for t in tests if t.get("status") == "failed")
            skipped = sum(1 for t in tests if t.get("status") == "skipped")
            total = len(tests)
            pct = (passed / total * 100) if total > 0 else 0
            lines.append(
                f"- **{engine.upper()}**: {passed}/{total} ({pct:.1f}%) [failed: {failed}, skipped: {skipped}]"
            )

    lines.extend(
        [
            "",
            "## Legend",
            "",
            "| Symbol | Meaning |",
            "|--------|---------|",
            "| ✅ | Passed - parameter works correctly |",
            "| ❌ | Failed - parameter not supported or error |",
            "| ⚠️ | Passed but validation uncertain |",
            "| ➖ | Not tested for this engine |",  # noqa: RUF001
            "",
        ]
    )

    # Generate tables for each category
    for category, params in categories.items():
        if not params:
            continue

        lines.extend(
            [
                f"## {category}",
                "",
                "| Parameter | PyTorch | vLLM | TensorRT | Notes |",
                "|-----------|---------|------|----------|-------|",
            ]
        )

        for param, engines_map in sorted(params.items()):
            row = [f"`{param}`"]

            notes = []
            for engine in Engine:
                if engine in engines_map:
                    status = engines_map[engine]
                    if status["passed"]:
                        if status["validation"] == "VERIFIED":
                            row.append("✅")
                        else:
                            row.append("⚠️")
                    else:
                        row.append("❌")
                        if status["error"]:
                            notes.append(f"{engine}: {status['error']}")
                else:
                    row.append("➖")  # noqa: RUF001

            notes_str = "; ".join(notes[:2])  # Limit notes
            if len(notes_str) > 80:
                notes_str = notes_str[:77] + "..."
            row.append(notes_str)

            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # Add recommendations section
    lines.extend(
        [
            "## Engine Recommendations",
            "",
            "### By Use Case",
            "",
            "| Use Case | Recommended Engine | Reason |",
            "|----------|---------------------|--------|",
            "| Memory-constrained | PyTorch | BitsAndBytes 4/8-bit quantization |",
            "| High throughput | vLLM | Continuous batching, PagedAttention |",
            "| Maximum performance | TensorRT | Compiled engine, FP8 quantization |",
            "| Multi-GPU inference | vLLM | Best tensor/pipeline parallel |",
            "| Development/debugging | PyTorch | Most flexible, familiar API |",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Generate parameter support matrix from test results."""
    parser = argparse.ArgumentParser(description="Generate parameter support matrix")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing test result JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/parameter-support-matrix.md"),
        help="Output markdown file",
    )
    args = parser.parse_args()

    # Load test results
    results = load_test_results(args.results_dir)

    if not any(results.values()):
        print(f"No test results found in {args.results_dir}")
        print("Run test_all_params.py first to generate results.")
        return

    # Extract and categorise parameters
    params = extract_parameter_status(results)
    categories = categorise_parameters(params)

    # Generate markdown
    content = generate_markdown(categories, results)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
