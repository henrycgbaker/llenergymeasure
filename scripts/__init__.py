"""Project tooling scripts.

Marked as a regular package so it wins over any ``scripts`` package that
some transitive dependency may install into site-packages (vLLM's full
install graph triggered this — see PR #418 commit log for details).
"""
