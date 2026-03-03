"""Generate the bundled aienergyscore.jsonl dataset file.

Downloads 1,000 prompts from AIEnergyScore/text_generation (HuggingFace) and
writes them to src/llenergymeasure/datasets/builtin/aienergyscore.jsonl with
a provenance header.

Usage:
    python scripts/generate_aienergyscore_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "llenergymeasure"
    / "datasets"
    / "builtin"
    / "aienergyscore.jsonl"
)
N_PROMPTS = 1000
DATASET_ID = "AIEnergyScore/text_generation"
SPLIT = "train"


def get_dataset_commit_sha() -> str:
    """Retrieve the commit SHA of the dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(DATASET_ID)
        sha = getattr(info, "sha", None) or getattr(info, "id", "unknown")
        return str(sha)
    except Exception:
        return "unknown"


def main() -> None:
    from datasets import load_dataset

    print(f"Downloading {DATASET_ID} split={SPLIT}...")
    ds = load_dataset(DATASET_ID, split=SPLIT)
    print(f"Dataset loaded: {len(ds)} rows")

    # Retrieve commit SHA
    commit_sha = get_dataset_commit_sha()
    print(f"Dataset commit SHA: {commit_sha}")

    # Determine available columns
    columns = ds.column_names
    print(f"Columns: {columns}")

    # AIEnergyScore/text_generation has a single 'text' column
    # Use 'text' as the prompt field; 'source' if present
    prompt_col = "text" if "text" in columns else columns[0]
    source_col = "source" if "source" in columns else None

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        # Write provenance header
        provenance = {
            "_provenance": DATASET_ID,
            "_commit": commit_sha,
            "_license": "apache-2.0",
            "_description": "1000 prompts from WikiText, OSCAR, UltraChat",
        }
        f.write(json.dumps(provenance, ensure_ascii=False) + "\n")

        written = 0
        for row in ds:
            if written >= N_PROMPTS:
                break

            text = row.get(prompt_col, "")
            if not isinstance(text, str) or not text.strip():
                continue

            record: dict = {"prompt": text.strip()}
            if source_col and source_col in row:
                record["source"] = str(row[source_col])

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if written < N_PROMPTS:
        raise ValueError(f"Only {written} prompts available in {DATASET_ID}, expected {N_PROMPTS}.")

    print(f"Written {written} prompts to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
