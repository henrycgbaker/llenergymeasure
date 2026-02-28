# Phase 12: Integration (CLI Study Wiring) - Research

**Researched:** 2026-02-28
**Domain:** CLI study integration, multi-experiment orchestration, result aggregation
**Confidence:** MEDIUM-HIGH
**Type:** Retroactive (phase is implemented; research validates design choices)

## Summary

Phase 12 wired `llem run study.yaml` end-to-end: YAML-based study detection, `_run()` dispatcher (single in-process vs multi via `StudyRunner`), CLI study flags (`--cycles`, `--order`, `--no-gaps`), `StudyResult` assembly with `measurement_protocol` and `StudySummary`, and a multi-backend pre-flight hard error. This research validates these implementation choices against peer ML CLI tools and industry patterns.

The implementation aligns well with established patterns. CLI-over-config precedence matches Hydra and lm-eval-harness. Study detection via top-level YAML keys is a simpler and arguably cleaner approach than the flag-based detection used by Hydra (`--multirun`). The `measurement_protocol` dict captures the same class of metadata as MLPerf's TestSettings logs and pytest-benchmark's rounds/iterations/warmup fields. The pre-flight validation pattern follows the universal "defensive early exit" pattern documented across W&B, Optuna, and MLPerf submission tooling.

**Primary finding:** The implementation follows peer-standard patterns. The one notable divergence -- plain-text display instead of Rich progress bars -- is deliberate (no Rich dependency) and adequate for the current UX requirements.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Top-level key check: if YAML has `sweep:` or `experiments:` keys, it's a study; otherwise single experiment
- CLI always routes through `_run(StudyConfig)` internally
- `_run()` dispatch: if `len(experiments) == 1` and `n_cycles == 1`, run in-process; otherwise delegate to `StudyRunner`
- CLI always wins (Hydra-style): `--model X` narrows the sweep grid to just that model
- Warning to stderr when CLI flags narrow a sweep
- Warning also propagated into `StudyResult.summary.warnings` list for traceability
- Standard merge: CLI flags override matching `execution:` block fields
- CLI effective defaults: `n_cycles=3`, `cycle_order="shuffled"` when neither YAML nor CLI specifies
- Pydantic defaults remain conservative (`n_cycles=1`, `cycle_order="sequential"`)
- Rename `config_gap_seconds` to `experiment_gap_seconds`
- Compact status line per experiment, scrolling up as completed
- `--quiet` suppresses progress lines and countdowns but still shows final summary table
- Rich table showing all experiments with columns: Config, Cycle, Time, Energy, tok/s
- `measurement_protocol`: flat dict from `ExecutionConfig`
- `study_design_hash`: carried from `StudyConfig`
- `result_files`: list of paths to per-experiment result files (RES-15)
- `summary`: `StudySummary` with totals and warnings list
- Multi-backend study raises `PreFlightError` at pre-flight with message directing to Docker

### Claude's Discretion
- Exact implementation of sweep narrowing logic
- How `load_study_config()` integrates with existing loader patterns
- Rich table formatting details and colour choices
- How failed experiment results are structured
- Integration test structure

### Deferred Ideas (OUT OF SCOPE)
- Document CLI flag + study YAML interaction in user-facing docs -- Phase 13
- `--resume` flag for interrupted studies -- M4
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RES-13 | StudyResult: study_design_hash, measurement_protocol, result_files, summary | Validated by MLPerf log metadata, pytest-benchmark schema, and lm-eval result patterns |
| CM-10 | Multi-backend study without Docker -> hard error at pre-flight | Aligned with universal defensive early exit pattern (Optuna TrialPruned, W&B skip, MLPerf compliance checker) |
| LA-02 | `run_study(config: str \| Path \| StudyConfig) -> StudyResult` | Three-form API matches lm-eval's `simple_evaluate()` accepting multiple input types |
| LA-05 | `run_study()` always writes manifest to disk | Aligns with MLPerf mandatory log files and optimum-benchmark's always-write-results pattern |
| STU-NEW-01 | `_run()` body implemented -- dispatches single vs study | Single-path dispatcher is cleaner than lm-eval's separate code paths for single vs grouped tasks |
| RES-15 | `result_files` contains paths, not embedded results | Matches optimum-benchmark file-per-run pattern and MLPerf per-scenario file structure |
| CLI-05 | Study-mode flags: --cycles, --no-gaps, --order | Analogous to Hydra's override operators and lm-eval's --num_fewshot style per-run overrides |
| CLI-11 | Thermal gap countdown display during inter-experiment pauses | Novel to energy measurement domain; no direct peer equivalent. Gap countdown aids measurement reproducibility |
</phase_requirements>

## Peer Analysis: Study Detection Patterns

### How Peers Detect Single vs Multi-Experiment Configs

| Tool | Detection Mechanism | Strengths | Weaknesses |
|------|-------------------|-----------|------------|
| **lm-eval-harness** | Comma-separated `--tasks` flag; groups defined in YAML with `group:` key | Explicit, no ambiguity | No config-level detection -- always CLI-driven |
| **optimum-benchmark** | Hydra `--multirun` / `-m` flag enables sweep mode | Clear opt-in signal | Requires explicit flag; config alone cannot trigger multirun |
| **Hydra** | `--multirun` flag with override syntax `param=a,b,c` | Powerful, composable | Complex syntax; new users confused by comma semantics |
| **W&B Sweeps** | Separate `wandb sweep sweep.yaml` command (not same as `wandb run`) | Completely separate workflow | No unified single/multi command |
| **llenergymeasure** | Top-level YAML key check (`sweep:` or `experiments:`) | Zero-flag, automatic, config-driven | Potential edge case if user accidentally includes `sweep:` key |

**Assessment (MEDIUM-HIGH confidence):** The llenergymeasure approach of detecting study mode from YAML structure is unique among peers. Most peers use an explicit flag or separate command. The YAML key approach is arguably better for a config-first tool -- the config file is self-describing, so `llem run study.yaml` just works without needing `--multirun`. The risk of false positives (a user accidentally having a `sweep:` key) is negligible since `sweep:` is not a valid `ExperimentConfig` field and would cause a config error anyway.

**Source:** [lm-eval-harness docs/interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [Hydra multirun docs](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark)

## Peer Analysis: CLI Override Semantics

### How Peers Handle CLI vs Config Precedence

| Tool | Precedence | Override Syntax | Notes |
|------|-----------|----------------|-------|
| **Hydra** | CLI > config file > defaults list | `key=value`, `++key=value` (force add), `~key` (remove) | Three override operators; most powerful but complex |
| **lm-eval-harness** | CLI > config file | Standard argparse; `--config` loads YAML, then CLI args override | Simple, explicit. Documented: "CLI arguments override config file values" |
| **optimum-benchmark** | CLI > Hydra config | Hydra override syntax (inherits Hydra patterns) | Hydra-derived |
| **W&B Sweeps** | Sweep controller > agent config; no user CLI override during sweep | N/A during sweep execution | Different paradigm -- sweep controller owns config |
| **Typer/Click ecosystem** | CLI > env > config file > default | `typer-config` library; `is_eager=True` for config loading | Standard Python CLI pattern |
| **llenergymeasure** | CLI > YAML > Pydantic defaults | `--model X` narrows sweep, `--cycles N` overrides execution | CLI effective defaults layer adds a third precedence level |

**Assessment (HIGH confidence):** "CLI always wins" is the universal standard. Hydra, lm-eval, and the Typer ecosystem all follow this pattern. The implementation is correct. The "CLI effective defaults" pattern (where CLI defaults of `n_cycles=3` differ from Pydantic defaults of `n_cycles=1`) is less common but justified -- it ensures the CLI UX is optimised for scientific rigour while the library API remains conservative.

**Source:** [Hydra override docs](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/), [lm-eval interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [typer-config PyPI](https://pypi.org/project/typer-config/0.1.1/)

## Peer Analysis: Progress Display Patterns

### How Peers Display Multi-Experiment Progress

| Tool | Progress Mechanism | Library | Description |
|------|-------------------|---------|-------------|
| **lm-eval-harness** | tqdm progress bars | tqdm | Standard tqdm during task evaluation; no per-task status lines documented |
| **optimum-benchmark** | Hydra logging | Python logging | Per-run log output; no custom progress display documented for sweeps |
| **W&B Sweeps** | W&B dashboard (web UI) | wandb | Real-time web dashboard; CLI shows basic agent status |
| **pytest-benchmark** | Terminal table | Built-in | Summary table after all runs complete; no real-time progress during execution |
| **MLPerf LoadGen** | Log files | Custom C++ logging | No interactive progress; writes to `mlperf_log_detail.txt` |
| **llenergymeasure** | Plain-text status lines | None (print) | `[3/12] OK model backend precision -- 4m 32s (147 J)` |

**Assessment (MEDIUM confidence):** The plain-text approach is pragmatic. Rich progress bars (Rich library) would be nicer but add a dependency. tqdm is already in the dependency tree (used for single-experiment progress) but is poorly suited for multi-line status displays. The current implementation -- scrolling status lines with OK/FAIL/... icons -- is adequate for M2 and avoids the complexity of managing a Rich `Progress` context across subprocess boundaries.

**Known gap:** The verification report notes that `print_study_progress()` exists in `_display.py` but the progress consumer thread in `runner.py` only forwards events for the Docker path, not the subprocess path. This was acknowledged as an incomplete wiring that does not block any requirement.

**Source:** [tqdm GitHub](https://github.com/tqdm/tqdm), [Rich progress docs](https://github.com/Textualize/rich/discussions/2272)

## Peer Analysis: Result Aggregation and Schema

### How Peers Assemble Results from Multiple Experiments

| Tool | Result Schema | Metadata Captured | Aggregation |
|------|--------------|-------------------|-------------|
| **lm-eval-harness** | JSON with per-task metrics + `results_agg` | Task groups, metric_list, aggregation methods | `aggregate_metric_list` with mean (micro/macro averaging) |
| **pytest-benchmark** | JSON with per-run stats + `extra_info` | rounds, iterations, warmup, min/max/mean/stddev/median/iqr | Statistical aggregation over rounds; comparison across saved runs |
| **MLPerf** | `mlperf_log_summary.txt` + `system_desc_id.json` | Scenario, QPS, latency percentiles (p50/p90/p95/p99/p99.9), TestSettings | Per-scenario logs; submission checker validates completeness |
| **optimum-benchmark** | `benchmark_report.json` + `benchmark_config.json` | Full config, latency, memory, energy, throughput | Per-run files; no built-in cross-run aggregation |
| **llenergymeasure** | `StudyResult` with `measurement_protocol` dict, `StudySummary`, `result_files` | n_cycles, cycle_order, gap seconds, shuffle_seed, design hash | `StudySummary` with totals (completed, failed, wall_time, energy, warnings) |

**Assessment (HIGH confidence):** The `measurement_protocol` dict maps directly to what peers capture:

| llenergymeasure field | pytest-benchmark equivalent | MLPerf equivalent |
|----------------------|---------------------------|-------------------|
| `n_cycles` | `rounds` | Scenario repetitions |
| `cycle_order` | N/A (always sequential) | N/A |
| `experiment_gap_seconds` | N/A | N/A (energy-specific) |
| `shuffle_seed` | N/A | N/A |
| `study_design_hash` | N/A | `system_desc_id` (identity hash) |

The `measurement_protocol` captures energy-measurement-specific metadata (gap seconds, shuffle seed) that has no direct peer equivalent. This is appropriate -- no peer tool measures inter-experiment thermal dynamics. The `study_design_hash` serves the same function as MLPerf's system_desc_id: reproducibility verification.

**Source:** [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/en/latest/usage.html), [MLPerf submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc), [lm-eval task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

## Peer Analysis: Pre-Flight Validation

### How Peers Validate Before Running

| Tool | Pre-flight Pattern | Scope | Failure Mode |
|------|-------------------|-------|-------------|
| **lm-eval-harness** | `lm-eval validate` subcommand + `--check_integrity` flag | Task existence, config syntax, dataset access, metric definitions | Separate validate step; errors before any model loading |
| **MLPerf** | Submission checker (post-hoc); LoadGen warns but does not enforce | Compliance rules, file completeness, log integrity | Warns users: "Choose your TestSettings carefully!" -- validation is user responsibility |
| **Optuna** | `raise TrialPruned()` at trial start for invalid combos | Per-trial categorical constraint checking | Pruned trials excluded from sampler learning |
| **W&B Sweeps** | Early exit pattern: `wandb.finish(exit_code=0)` before GPU init | Per-agent config validity | Clean exit, config counted as "finished" not "failed" |
| **Cloudera ML** | Aggregated preflight checks with pass/fail summary | Resource availability, environment configuration | Single aggregated error with all individual check results |
| **llenergymeasure** | `run_study_preflight()` at `_run()` entry; `run_preflight()` per experiment | Multi-backend guard (CM-10), CUDA, backend installed, model accessible | `PreFlightError` with all failures collected before raising |

**Assessment (HIGH confidence):** The implementation follows the strongest available pattern: collect-all-errors-then-raise (matching Cloudera's aggregated preflight) combined with defence-in-depth (study-level preflight + per-experiment preflight). The multi-backend guard is a hard error with a clear message directing to Docker -- this is the correct UX for the M2 constraint (Docker not yet wired). lm-eval's separate `validate` subcommand is a more complex approach; the inline validation on `_run()` entry is simpler and ensures validation always runs.

**Source:** [lm-eval interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [Existing research: 10-sweep-validation-patterns.md](.product/research/10-sweep-validation-patterns.md), [W&B sweep docs](https://docs.wandb.ai/models/sweeps/define-sweep-configuration)

## Architecture Patterns Validated

### Pattern 1: Single Dispatcher (`_run()`)
**What:** All execution flows through `_run(StudyConfig) -> StudyResult`. Single experiments are wrapped in a degenerate `StudyConfig`.
**Peer precedent:** lm-eval-harness has a single `evaluate()` entry point that handles both single-task and multi-task cases. Hydra always runs through its configured launcher regardless of single vs multirun.
**Assessment:** Correct. A single dispatcher eliminates divergent code paths and ensures manifest/result-file writing always happens.

### Pattern 2: YAML-Structural Detection
**What:** Study mode detected by checking for `sweep:` or `experiments:` top-level YAML keys.
**Peer precedent:** No direct peer equivalent. lm-eval uses CLI flags; Hydra uses `--multirun`. The closest analogy is Kubernetes YAML, where resource type is determined by the `kind:` field.
**Assessment:** Novel but sound. Self-describing configs reduce user error (no forgotten `--multirun` flag).

### Pattern 3: CLI Effective Defaults vs Library Defaults
**What:** CLI applies `n_cycles=3` and `cycle_order="shuffled"` when neither YAML nor CLI specifies, while Pydantic defaults are conservative (`n_cycles=1`, `cycle_order="sequential"`).
**Peer precedent:** lm-eval has different defaults for CLI vs API (e.g., `--log_samples` defaults to False on CLI but the Python API has no such concept). Hydra's `defaults:` list provides a similar layered-defaults mechanism.
**Assessment:** Justified separation of concerns. Library users get predictable minimal defaults; CLI users get scientifically rigorous defaults. The two layers are clearly documented in the code.

### Anti-Patterns Avoided
- **Separate single/multi code paths:** Avoided by routing everything through `_run(StudyConfig)`.
- **Config mutation:** CLI overrides are applied via `deep_merge()` on raw dicts before Pydantic construction, not by mutating constructed config objects.
- **Swallowed errors:** Pre-flight collects all failures before raising; result-save failures produce warnings but don't abort the study.

## Common Pitfalls (Validated Against Implementation)

### Pitfall 1: Double Cycle Application
**What goes wrong:** Calling `apply_cycles()` in both `load_study_config()` and `StudyRunner.run()`, multiplying the experiment count by `n_cycles^2`.
**How it was avoided:** `runner.py` line 273 comment explicitly documents: "study.experiments is already cycled by load_study_config(); use as-is." The runner does not call `apply_cycles()`.
**Warning signs:** Experiment count in manifest being N^2 instead of N.

### Pitfall 2: SIGINT Handling in Subprocess Contexts
**What goes wrong:** Both parent and child trying to handle SIGINT, causing race conditions or double cleanup.
**How it was avoided:** Child subprocess installs `signal.SIG_IGN` for SIGINT (line 71), parent owns all SIGINT handling. Two-stage escalation: SIGTERM first, 2s grace, then SIGKILL.
**Peer precedent:** This is a well-known pattern in multiprocessing. vLLM's engine process also ignores SIGINT and lets the parent decide.

### Pitfall 3: Config-File-Wins Precedence
**What goes wrong:** Accidentally giving config file values higher priority than CLI flags, causing user confusion.
**How it was avoided:** `load_study_config()` applies `cli_overrides` via `deep_merge(raw, cli_overrides)` after loading YAML, ensuring CLI wins.
**Peer validation:** Every peer (Hydra, lm-eval, Click/Typer ecosystem) follows CLI > config file. This is the universal expectation.

### Pitfall 4: Silent Failure on Result Persistence
**What goes wrong:** A save failure discards a completed measurement that consumed GPU time.
**How it was avoided:** Both `_run_in_process()` and `StudyRunner._run_one()` catch save exceptions and continue: `manifest.mark_completed(config_hash, cycle, result_file="")` with a warning.
**Peer precedent:** optimum-benchmark writes results per-run; failures are logged but don't abort the sweep.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config merging | Custom dict merge with edge-case handling | `deep_merge()` as implemented (or OmegaConf if Hydra adopted) | Nested dict merge is deceptively tricky; `deepcopy` + recursive merge is the standard pattern |
| YAML parsing | Custom parser for study vs experiment detection | `yaml.safe_load()` + key check | Standard library handles anchors, aliases, all YAML edge cases |
| Progress bars (future) | Custom terminal control sequences | Rich `Progress` or tqdm | Terminal control is platform-dependent; libraries handle edge cases |
| Process isolation | Manual `os.fork()` | `multiprocessing.get_context("spawn")` | spawn is CUDA-safe; fork causes silent CUDA corruption |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fork-based multiprocessing | Spawn context for CUDA safety | Python 3.12+ default, widespread by 2024 | Prevents silent GPU memory corruption |
| Separate CLI commands for single/multi | Unified `run` command with auto-detection | Hydra popularised this ~2020; lm-eval adopted `--config` in Dec 2025 | Better UX, fewer commands to learn |
| Fire-and-forget sweep execution | Pre-flight validation before GPU allocation | Universal by 2024-2025 (Optuna, W&B, MLPerf) | Saves GPU time, clearer errors |
| Embedded results in parent schema | File paths (RES-15 pattern) | Standard since MLPerf v1.0+ | Scalable; avoids memory issues with large result sets |

## Open Questions

1. **Progress wiring gap**
   - What we know: `print_study_progress()` exists but is not called from the subprocess path's progress consumer
   - What's unclear: Whether this matters for M2 UX (Docker path does call it)
   - Recommendation: Addressed in Phase 15 (M2 tech debt). Non-blocking.

2. **CLI narrowing warning format**
   - What we know: CONTEXT.md specifies `"Warning: --model gpt2 narrows sweep from 6 to 4 experiments"`
   - What's unclear: Whether this warning was fully implemented in the loader/CLI layer
   - Recommendation: Verify in Phase 13 (documentation) or Phase 15 (tech debt).

## Sources

### Primary (HIGH confidence)
- [lm-eval-harness docs/interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) -- CLI precedence, multi-task handling
- [lm-eval-harness docs/task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) -- Result aggregation, task groups
- [Hydra override documentation](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/) -- CLI override semantics
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/en/latest/usage.html) -- Measurement protocol metadata, result schema
- [MLPerf submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc) -- Result structure, metadata requirements
- [Existing research: .product/research/10-sweep-validation-patterns.md](../../.product/research/10-sweep-validation-patterns.md) -- Pre-flight validation patterns
- [Existing research: .product/research/11-peer-cli-patterns.md](../../.product/research/11-peer-cli-patterns.md) -- Peer CLI patterns

### Secondary (MEDIUM confidence)
- [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark) -- Multi-run patterns, Hydra integration
- [EleutherAI evaluation architecture blog](https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html) -- lm-eval internals
- [MLPerf inference documentation](https://docs.mlcommons.org/inference/submission/) -- Submission structure
- [typer-config PyPI](https://pypi.org/project/typer-config/0.1.1/) -- Typer/Click config file patterns

### Tertiary (LOW confidence)
- [W&B sweep configuration](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) -- CLI/config interaction (limited detail)
- [Rich progress discussions](https://github.com/Textualize/rich/discussions/2272) -- Multi-bar progress patterns

## Metadata

**Confidence breakdown:**
- CLI override semantics: HIGH -- universal pattern confirmed across multiple authoritative sources
- Study detection: MEDIUM-HIGH -- novel approach, no direct peer equivalent but well-justified
- Result schema: HIGH -- measurement_protocol maps to established pytest-benchmark and MLPerf patterns
- Pre-flight validation: HIGH -- universal defensive early exit pattern, well-documented across ecosystem
- Progress display: MEDIUM -- peer implementations vary widely; no clear "best practice" for subprocess progress

**Research date:** 2026-02-28
**Valid until:** 2026-03-28 (stable domain; patterns change slowly)
