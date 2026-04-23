"""Tests for message normalisation used by report-gaps candidate synthesis."""

from __future__ import annotations

import re

from llenergymeasure.study.message_normalise import normalise


class TestNumericSubstitution:
    def test_integer(self):
        assert normalise("count=42").template == "count=<NUM>"

    def test_float(self):
        assert normalise("temperature=0.001").template == "temperature=<NUM>"

    def test_scientific(self):
        assert normalise("val 1e-5").template == "val <NUM>"

    def test_negative(self):
        assert normalise("offset -12").template == "offset <NUM>"

    def test_preserves_surrounding_identifiers(self):
        # Don't eat "sha256" in "sha256_abc123".
        assert "sha256_abc" in normalise("sha256_abc NUM 42 here").template


class TestPathSubstitution:
    def test_absolute_unix(self):
        assert "<PATH>" in normalise("error in /usr/lib/foo.py:12").template

    def test_windows_drive(self):
        t = normalise("error in C:\\Program Files\\foo.py").template
        assert "<PATH>" in t


class TestTimestampSubstitution:
    def test_iso_with_z(self):
        assert normalise("at 2026-04-23T12:34:56Z").template == "at <TIMESTAMP>"

    def test_iso_with_offset(self):
        assert normalise("at 2026-04-23T12:34:56+00:00").template == "at <TIMESTAMP>"


class TestLineNumberSubstitution:
    def test_py_line_number(self):
        # Path matched first then line number after .py:
        t = normalise("foo.py:368 warning").template
        assert "<LINENO>" in t


class TestHexSubstitution:
    def test_sha256_prefixed(self):
        t = normalise("hash sha256:deadbeefdeadbeef").template
        assert "<HEX>" in t

    def test_bare_long_hex(self):
        t = normalise("token abcdef0123456789deadbeef").template
        assert "<HEX>" in t


class TestAnsiStripping:
    def test_removes_color_codes(self):
        t = normalise("\x1b[31merror\x1b[0m here").template
        assert t == "error here"


class TestWhitespaceCollapse:
    def test_multiple_spaces(self):
        t = normalise("foo   bar   baz").template
        assert t == "foo bar baz"


class TestTemplateRegex:
    def test_num_placeholder_matches_number(self):
        nm = normalise("temperature=0.001")
        assert re.match(nm.match_regex, "temperature=0.5")
        assert re.match(nm.match_regex, "temperature=1e-3")

    def test_path_placeholder_matches_path(self):
        nm = normalise("error in /abs/path.py")
        assert re.match(nm.match_regex, "error in /different/path.py")

    def test_regex_is_anchored(self):
        nm2 = normalise("count=42")
        # Anchored so exact match passes.
        assert re.match(nm2.match_regex, "count=42")
        # Sanity: extra trailing content breaks the match.
        assert not re.match(nm2.match_regex, "count=42 and more")


class TestEndToEndNormalisationEquivalence:
    def test_two_configs_produce_same_template(self):
        a = normalise("vllm clamping temperature=0.001 to 0.01").template
        b = normalise("vllm clamping temperature=0.005 to 0.01").template
        assert a == b

    def test_different_messages_stay_distinct(self):
        a = normalise("temperature out of range").template
        b = normalise("unknown attention implementation").template
        assert a != b
