## Git Workflow

All merges to main via squash-merge PRs. Branch protection requires CI to pass.

1. Work on feature/phase branch
2. Push branch, create PR: `gh pr create --base main`
3. Wait for CI: `gh pr checks --watch`
4. Squash merge: `gh pr merge --squash`

Use `/squash-merge` to automate this flow.

## Commit Messages (on main)

`type(scope): description` — types: feat, fix, docs, refactor, test
- Never include phase numbers, milestone IDs, or GSD references
- Domain scopes: `feat(config):`, `fix(energy):`, `docs(product):` etc.

## Versioning

Semver 0.x (pre-1.0). Manual version management (no commitizen/bump tools).

- Phase PRs: never touch version. Code lands on main at current version.
- Milestone completion: bump minor in pyproject.toml + __init__.py, update CHANGELOG,
  commit "chore: release 0.X.0", create git tag v0.X.0, push tag → CI publishes.
- Version sources: pyproject.toml (build) + __init__.py (runtime). Must stay in sync.
- 1.0.0 reserved for production-ready release.

Current: 0.8.0 (M2). Next: 0.9.0 (M3).
