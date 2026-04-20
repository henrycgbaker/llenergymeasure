"""Verify that the CLI module auto-loads `.env` at import time."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Chdir to a tmp dir, drop CLI modules from sys.modules for fresh reimport.

    Uses ``monkeypatch.delitem`` so each touched ``sys.modules`` entry is
    restored at teardown — a raw ``del sys.modules[...]`` would leak the
    reimport across subsequent tests and cause test-pollution failures
    (e.g. preflight spec lookups returning None in unrelated CLI tests).

    Also saves/restores the ``llenergymeasure.cli`` attribute on the parent
    package: Python's import machinery sets ``parent.child = module`` as a
    side effect when importing a subpackage, and ``monkeypatch.delitem`` on
    ``sys.modules`` alone does not undo that.  Without this, pytest's
    ``resolve()`` (used by string-target ``monkeypatch.setattr``) follows the
    stale parent attribute and patches a ghost module that nothing actually
    calls, making unrelated CLI tests flaky under xdist.
    """
    monkeypatch.chdir(tmp_path)
    # Save the parent-package attribute so teardown undoes the reimport side effect.
    import llenergymeasure

    if hasattr(llenergymeasure, "cli"):
        monkeypatch.setattr(llenergymeasure, "cli", llenergymeasure.cli)
    for modname in list(sys.modules):
        if modname == "llenergymeasure.cli" or modname.startswith("llenergymeasure.cli."):
            monkeypatch.delitem(sys.modules, modname)
    return tmp_path


def test_cli_import_loads_dotenv_from_cwd(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.env` in CWD should be loaded when the CLI module is imported."""
    (isolated_env / ".env").write_text("LLEM_TEST_DOTENV_SENTINEL=loaded-from-file\n")
    monkeypatch.delenv("LLEM_TEST_DOTENV_SENTINEL", raising=False)

    importlib.import_module("llenergymeasure.cli")

    assert os.environ.get("LLEM_TEST_DOTENV_SENTINEL") == "loaded-from-file"


def test_shell_env_wins_over_dotenv_file(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """override=False means a pre-existing shell env var must win."""
    (isolated_env / ".env").write_text("LLEM_TEST_DOTENV_OVERRIDE=from-file\n")
    monkeypatch.setenv("LLEM_TEST_DOTENV_OVERRIDE", "from-shell")

    importlib.import_module("llenergymeasure.cli")

    assert os.environ["LLEM_TEST_DOTENV_OVERRIDE"] == "from-shell"


def test_no_dotenv_file_does_not_raise(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing `.env` must not fail CLI import."""
    assert not (isolated_env / ".env").exists()
    monkeypatch.delenv("LLEM_TEST_DOTENV_MISSING", raising=False)

    importlib.import_module("llenergymeasure.cli")

    assert "LLEM_TEST_DOTENV_MISSING" not in os.environ
