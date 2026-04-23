"""Library-warning message normalisation for the feedback loop.

Design: ``.product/designs/config-deduplication-dormancy/runtime-config-validation.md``
§10.Q3.

Goal: strip configuration-specific values (numbers, paths, line numbers,
ISO timestamps, ANSI sequences) from a captured log / warning message so two
emissions produced by different configs with the same underlying rule hash to
the same canonical template.

The pipeline is deliberately conservative (documented substitution order, no
fuzzy matching) — it's easy to add new regex cases for real-world false
matches, hard to remove them once consumers depend on permissive behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "NormalisedMessage",
    "build_template_regex",
    "normalise",
]


@dataclass(frozen=True)
class NormalisedMessage:
    """Result of normalising a single library-emitted message.

    Attributes:
        template: The normalised, placeholder-substituted template. Stable
            across emissions that only differ by numeric values / paths /
            timestamps.
        match_regex: A compiled-ready regex pattern that matches the template
            back against future raw messages. Written into corpus rules as
            ``observed_messages_regex`` so the generic validator can
            re-identify the same rule firing later.
        original: The pre-normalisation message — kept for evidence in draft
            PR bodies.
    """

    template: str
    match_regex: str
    original: str


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
# Line numbers at end of path-like tokens ("…sampling_params.py:368"). We
# strip them before path normalisation so the path itself doesn't swallow
# them.
_LINE_NUMBER_RE = re.compile(r"(?<=\.py):\d+")
# Absolute and relative path-like tokens. Kept intentionally narrow — the
# walker emits short human messages, not shell output.
_PATH_RE = re.compile(r"(?:(?<=\s)|^)(?:/[^\s:()\[\]]+|[A-Za-z]:\\[^\s:]+)")
# Hex digests / fingerprints (sha256: prefixes and loose 16+ hex runs).
_HEX_RE = re.compile(r"\b(?:sha256:)?[0-9a-fA-F]{16,}\b")
# Numeric literals: integers, decimals, scientific, signed values.
# Guarded with alphanumeric-aware lookarounds so embedded digits inside
# identifiers (sha256, float32, fp16) aren't split apart.
_NUMBER_RE = re.compile(r"(?<![A-Za-z_0-9])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_0-9])")
# Multiple whitespace collapses to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_ANSI_RE, ""),
    (_ISO_TIMESTAMP_RE, "<TIMESTAMP>"),
    (_LINE_NUMBER_RE, ":<LINENO>"),
    (_PATH_RE, " <PATH>"),
    (_HEX_RE, "<HEX>"),
    (_NUMBER_RE, "<NUM>"),
)


def normalise(message: str) -> NormalisedMessage:
    """Normalise a single message.

    Substitution order (documented; callers should not assume any other):

    1. Strip ANSI escape sequences.
    2. Replace ISO-8601 timestamps with ``<TIMESTAMP>``.
    3. Replace ``.py:NNN`` line numbers with ``.py:<LINENO>``.
    4. Replace absolute / Windows / relative path tokens with ``<PATH>``.
    5. Replace hex digests / long hex runs with ``<HEX>``.
    6. Replace numeric literals with ``<NUM>``.
    7. Collapse repeated whitespace.

    ``build_template_regex`` produces the corresponding match regex —
    placeholders become loose character classes so the template matches
    future raw emissions.
    """
    original = message
    normalised = message
    for pattern, replacement in _SUBSTITUTIONS:
        normalised = pattern.sub(replacement, normalised)
    template = _WHITESPACE_RE.sub(" ", normalised).strip()
    match_regex = build_template_regex(template)
    return NormalisedMessage(template=template, match_regex=match_regex, original=original)


_PLACEHOLDER_REGEXES: dict[str, str] = {
    "<TIMESTAMP>": r"\\S+",
    "<LINENO>": r"\\d+",
    "<PATH>": r"\\S+",
    "<HEX>": r"\\S+",
    "<NUM>": r"-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?",
}


def build_template_regex(template: str) -> str:
    """Return a regex that matches raw messages yielding ``template`` under :func:`normalise`.

    Placeholders are expanded to permissive character classes; the rest of the
    template is escaped. The result is anchored with ``\\A`` / ``\\Z`` so the
    corpus consumer can do an exact match without accidentally aliasing with
    longer emissions.
    """
    # Split on placeholders so we can escape each literal chunk but keep
    # the class regex unescaped.
    tokens = re.split(r"(<TIMESTAMP>|<LINENO>|<PATH>|<HEX>|<NUM>)", template)
    parts: list[str] = []
    for tok in tokens:
        if tok in _PLACEHOLDER_REGEXES:
            parts.append(_PLACEHOLDER_REGEXES[tok].replace("\\\\", "\\"))
        else:
            parts.append(re.escape(tok))
    return r"\A" + "".join(parts) + r"\Z"
