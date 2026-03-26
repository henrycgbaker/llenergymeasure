- [ ] **Audit & set up a robust SemVer CI pipeline**: Currently version is hardcoded `"2.0.0"` in pyproject.toml. Need to: (a) audit what the correct current version should be (likely 1.x.y), (b) set up automatic version bumping from conventional commits (e.g. python-semantic-release or similar), (c) configure GitHub Actions workflow for automated release tagging + PyPI publish, (d) set pyproject.toml to use dynamic versioning (e.g. `poetry-dynamic-versioning` or `setuptools-scm`). Goal: commits to main with `feat:` / `fix:` / `feat!:` automatically bump MINOR/PATCH/MAJOR, tag, and optionally publish.

- [ ] **Rename CLI command `lem` → `llem`**: Rename the entry point across pyproject.toml `[tool.poetry.scripts]`, all docs (README, quickstart, backends, deployment, CLI reference), CLAUDE.md files, Makefile, docker-compose.yml, Dockerfiles, example configs, and tests. This is a breaking change — best done (a) after SemVer CI is set up so it auto-bumps MAJOR, and (b) before PyPI publish so there's no `lem` package in the wild to confuse users. Could also keep `lem` as a deprecated alias temporarily.

- [ ] **Hybrid campaign: grid + config files in one campaign**: Currently grid and config-file sources are either/or (grid takes priority). Allow campaigns to merge grid-generated experiments with explicit config files in a single execution plan. Pydantic model already accepts both; runner needs to merge the execution orders.

- [ ] in io currently havee results path: `io.results_dir` => add same for configs / state / other volumes etc?
- [ ] 'docs/generated/parameter-support-matrix.md supposedly "auto-generated from test results. Run `python scripts/generate_param_matrix.py` to update." but I which results? i think it needs to have runtime tests results to feed in? if so a) write this, b) check which/how params are added as this doesn't seem to be ALL params (i.e. it deviates from the `config-reference.md` doc that is also auto-generated? e.g. if choose to add baseline or not as per our new feature, I see that appear in `config-reference.md` but not `parameter-support-matrix.md`?)
- [ ] should we check the new baseline feature: I thought it was supposed to take 15/30% off, but it seems to take most off?

- [ ] "In the `[project.optional-dependencies]` section, add:
        ```
        campaign = ["python-on-whales>=0.70"]
        ```

        Also add it to the `dev` extras group so it's available during development.

        Do NOT add it as a core dependency — it's only needed when orchestrating Docker containers from the host. Users running inside containers don't need it."
        - i don't think thjis is right, I think the default should be that users are orchestrating docker containers from host.... i think this should be core
        - go through and properly identify what is / is not in core dependencies in pyproject.toml.... I want this to work a certain way out of the box
