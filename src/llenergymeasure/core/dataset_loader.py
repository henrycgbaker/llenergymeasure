"""HuggingFace dataset loading for prompts.

Provides utilities for loading prompts from HuggingFace datasets,
with support for built-in curated datasets and custom user datasets.

Note: FilePromptSource, HuggingFacePromptSource, and PromptSourceConfig are
defined here as lightweight dataclasses. They were originally planned for
config.models but are maintained here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from llenergymeasure.datasets.loader import AUTO_DETECT_COLUMNS, BUILTIN_DATASETS
from llenergymeasure.exceptions import ConfigurationError

if TYPE_CHECKING:
    from datasets import Dataset


# ---------------------------------------------------------------------------
# Source configuration types
# ---------------------------------------------------------------------------


@dataclass
class FilePromptSource:
    """Prompt source backed by a local file (one prompt per line)."""

    path: str | Path


@dataclass
class HuggingFacePromptSource:
    """Prompt source backed by a HuggingFace dataset."""

    dataset: str
    split: str = "train"
    subset: str | None = None
    column: str | None = None
    shuffle: bool = False
    seed: int = 42
    sample_size: int | None = None


# Union type alias for prompt sources
PromptSourceConfig = FilePromptSource | HuggingFacePromptSource


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_prompts_from_source(source: PromptSourceConfig) -> list[str]:
    """Load prompts from configured source.

    Args:
        source: Prompt source configuration (file or huggingface).

    Returns:
        List of prompt strings.

    Raises:
        ConfigurationError: If source is invalid or loading fails.
    """
    if isinstance(source, FilePromptSource):
        return load_prompts_from_file(source.path)
    elif isinstance(source, HuggingFacePromptSource):
        return load_prompts_from_hf(source)
    else:
        raise ConfigurationError(f"Unknown prompt source type: {type(source)}")


def load_prompts_from_file(path: str | Path) -> list[str]:
    """Load prompts from text file (one per line).

    Args:
        path: Path to prompts file.

    Returns:
        List of non-empty prompt strings.

    Raises:
        ConfigurationError: If file doesn't exist or is empty.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError(f"Prompts file not found: {path}")

    prompts = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]

    if not prompts:
        raise ConfigurationError(f"No prompts found in file: {path}")

    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def load_prompts_from_hf(config: HuggingFacePromptSource) -> list[str]:
    """Load prompts from HuggingFace dataset.

    Args:
        config: HuggingFace prompt source configuration.

    Returns:
        List of prompt strings.

    Raises:
        ConfigurationError: If dataset loading fails or column not found.
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading dataset: {config.dataset} (split={config.split})")

        # Load dataset
        ds: Dataset = load_dataset(
            config.dataset,
            config.subset,
            split=config.split,
        )

        # Determine column
        column = config.column or _auto_detect_column(ds.column_names)
        if column not in ds.column_names:
            raise ConfigurationError(
                f"Column '{column}' not found in dataset. Available: {ds.column_names}"
            )

        # Shuffle if requested
        if config.shuffle:
            ds = ds.shuffle(seed=config.seed)

        # Sample if requested
        if config.sample_size is not None:
            ds = ds.select(range(min(config.sample_size, len(ds))))

        # Extract prompts
        prompts = _extract_prompts(ds, column)

        logger.info(f"Loaded {len(prompts)} prompts from {config.dataset}[{column}]")
        return prompts

    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to load HF dataset '{config.dataset}': {e}") from e


def _auto_detect_column(columns: list[str]) -> str:
    """Auto-detect the prompt column from dataset columns.

    Args:
        columns: List of column names in the dataset.

    Returns:
        Detected column name.

    Raises:
        ConfigurationError: If no suitable column found.
    """
    for candidate in AUTO_DETECT_COLUMNS:
        if candidate in columns:
            logger.debug(f"Auto-detected prompt column: {candidate}")
            return candidate

    raise ConfigurationError(
        f"Could not auto-detect prompt column. "
        f"Available columns: {columns}. "
        f"Please specify 'column' explicitly."
    )


def _extract_prompts(dataset: Dataset, column: str) -> list[str]:
    """Extract prompt strings from dataset column.

    Handles both simple string columns and structured columns
    (e.g., ShareGPT conversations format).

    Args:
        dataset: HuggingFace dataset.
        column: Column name to extract.

    Returns:
        List of prompt strings.
    """
    prompts = []

    for row in dataset:
        value = row[column]

        if isinstance(value, str):
            # Simple string column
            if value.strip():
                prompts.append(value.strip())
        elif isinstance(value, list):
            # Conversation format (e.g., ShareGPT)
            # Take first human message
            prompt = _extract_from_conversation(value)
            if prompt:
                prompts.append(prompt)
        elif isinstance(value, dict):
            # Dict with content field
            content = value.get("content", value.get("text", ""))
            if content and content.strip():
                prompts.append(content.strip())

    return prompts


def _extract_from_conversation(messages: list[Any]) -> str | None:
    """Extract first human message from conversation format.

    Args:
        messages: List of conversation messages.

    Returns:
        First human message content, or None if not found.
    """
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("from") or msg.get("role") or ""
            if isinstance(role, str) and role.lower() in ("human", "user"):
                content = msg.get("value") or msg.get("content") or ""
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def list_builtin_datasets() -> dict[str, Any]:
    """List available built-in dataset aliases.

    Returns:
        Dictionary of alias -> path mapping.
    """
    return dict(BUILTIN_DATASETS)
