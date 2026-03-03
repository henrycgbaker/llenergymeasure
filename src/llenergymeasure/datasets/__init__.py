"""Built-in prompt datasets for LLM efficiency measurement.

Provides the bundled aienergyscore dataset and utilities for loading
prompts from built-in or custom JSONL dataset files.

Example usage::

    from llenergymeasure.datasets import aienergyscore
    # aienergyscore is a Path to the bundled JSONL file

    from llenergymeasure.datasets import load_prompts, BUILTIN_DATASETS
"""

from pathlib import Path

from llenergymeasure.datasets.loader import BUILTIN_DATASETS, load_prompts

aienergyscore: Path = BUILTIN_DATASETS["aienergyscore"]

__all__ = ["BUILTIN_DATASETS", "aienergyscore", "load_prompts"]
