---
created: 2026-03-03T17:55:24.561Z
title: Add PyPI distribution phase
area: infrastructure
files: []
---

## Problem

llenergymeasure is not published to PyPI. The release workflow only creates GitHub Releases with built artifacts. This means Docker images must COPY source + `pip install .` rather than the cleaner `pip install llenergymeasure` from PyPI.

## Solution

Add a future phase/milestone for PyPI distribution:
- Publish package to PyPI on release tag
- Once live, reconsider Docker image install strategy (switch from COPY+install to `pip install llenergymeasure=={version}` from PyPI)
- Update release.yml to include `twine upload` or `gh-action-pypi-publish`
