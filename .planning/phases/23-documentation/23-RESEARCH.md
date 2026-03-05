# Phase 23: Documentation - Research

**Researched:** 2026-03-05
**Domain:** Technical documentation authoring, Markdown, Typer CLI doc generation, Pydantic schema introspection
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Doc structure:**
- Fresh start - delete all existing docs (stale: old CLI name `lem`, old command structure, old concepts like "campaigns")
- Flat `docs/` folder - no subdirectories, no nesting. Matches Zeus and lm-eval scale (~13 files, not vLLM's 80+)
- Plain GitHub-rendered Markdown - no MkDocs build system for now (todo captured for future)
- README = overview + links - project description, feature highlights, quick install snippet, then links into docs/. Not inline content.
- Generated content inlined into parent docs (config reference into study-config.md, param matrix into backends.md, invalid combos into troubleshooting.md). Not in a separate generated/ folder.

**File set (13 files):**

Researcher path (9 files):
- `installation.md` - system requirements, pip with extras, from source (dev), Docker install path (link to docker-setup.md)
- `getting-started.md` - two-track: Quick start (local PyTorch) and Recommended start (Docker). Both use GPT-2 (124M). Linear flow: install → verify (`llem config`) → run → read results (annotated output). Ends with linear "next" pointer.
- `docker-setup.md` - full NVIDIA Container Toolkit walkthrough (install Docker, install drivers, install toolkit, verify). vLLM-focused with mention of future backends (TensorRT-LLM, SGLang). Includes Docker-specific troubleshooting section.
- `backends.md` - PyTorch (local) vs vLLM (Docker) switching. Parameter support matrix (auto-generated) inlined.
- `study-config.md` - YAML reference for experiments and studies/sweeps. Config reference (auto-generated) inlined.
- `cli-reference.md` - `llem run` and `llem config` (or `llem init` after rename) flags and options. Auto-generated from CLI source.
- `energy-measurement.md` - what we measure (energy, throughput, FLOPs) and how. Technical depth for researchers.
- `methodology.md` - statistical approach, measurement methodology, confidence intervals.
- `troubleshooting.md` - common pitfalls (like lm-eval's footguns.md). Invalid combos (auto-generated) inlined.

Policy maker path (4 files):
- `guide-what-we-measure.md` - plain-language explanation of energy, throughput, FLOPs. Why AI energy measurement matters for policy.
- `guide-interpreting-results.md` - how to read llem output. What numbers mean in practical terms.
- `guide-getting-started.md` - minimal path to running a measurement with hand-holding. Assumes basic terminal knowledge only.
- `guide-comparison-context.md` - how llem results relate to other benchmarks (MLPerf, AI Energy Score). Framing for reports and publications.

**Docker positioning:**
- Docker is the recommended/default path - not an advanced or optional setup
- Local PyTorch is positioned as the quick/lightweight alternative for avoiding dependencies
- Getting-started shows both tracks side-by-side, with Docker clearly marked as recommended
- Docker guide is a full walkthrough from zero (not just "install NVIDIA toolkit" link)

**Audience & tone:**
- Dual audience: ML researchers (primary, 9 docs) and policy makers (companion, 4 docs)
- Researcher tone: imperative-practical ("Install with pip", "Run your first experiment"). Direct instructions, not tutorial narrative.
- Assumed knowledge (researchers): Python, pip, HuggingFace models, basic GPU concepts, CLI comfort. Does NOT assume Docker knowledge or energy measurement concepts.
- Policy maker tone: accessible, explanatory, no assumed ML knowledge

**Getting started flow:**
- Two-track: Quick start (local PyTorch, no Docker) alongside Recommended start (Docker)
- Example model: GPT-2 (124M) - tiny, downloads fast, runs anywhere
- Linear 4-step: Install → Verify (`llem config`) → Run first experiment → Read results
- Annotated output: real terminal output with inline annotations explaining each metric
- Linear next pointer: ends with "Next: [next doc]" rather than branching paths
- Verification step: `llem config` as install checkpoint (like Zeus's `show_env`)

**Installation coverage:**
- pip with all extras documented (`[pytorch]`, `[vllm]`, `[tensorrt]`, `[zeus]`, `[codecarbon]`)
- From source (dev setup)
- System requirements (Python version, OS, CUDA compatibility)
- Docker install path (brief, links to docker-setup.md)

**Auto-generation pipeline:**
- Use established libraries where they fit, custom scripts where unique:
  - `typer utils docs` for CLI reference (built-in, zero custom code needed)
  - Pydantic `model_json_schema()` + thin Markdown renderer for config/study reference
  - Keep `generate_param_matrix.py` (unique - generates from test results)
  - Keep `generate_invalid_combos_doc.py` (unique - validation rule documentation)
- Research best-in-class patterns during phase research
- Inline generated content into parent topic files
- CI freshness check: new CI step that regenerates and fails if output differs from committed
- General principle: use libraries rather than reinventing the wheel. Custom scripts only when no existing tool covers the use case.

### Claude's Discretion
- Exact section ordering within each doc file
- Prose transitions between sections
- How much Docker troubleshooting detail vs linking to external resources
- Policy maker doc depth and specific analogies used
- Whether `llem config` references update to `llem init` (depends on rename todo timing)

### Deferred Ideas (OUT OF SCOPE)
- MkDocs Material docs site
- Rename `llem config` to `llem init`
- API reference auto-generation (mkdocstrings-style)
</user_constraints>

<phase_requirements>
## Phase Requirements

The REQUIREMENTS.md does not exist at `.planning/REQUIREMENTS.md` (404). Requirements are inferred from CONTEXT.md, ROADMAP.md, and the phase description.

| ID | Description | Research Support |
|----|-------------|-----------------|
| DOCS-01 | Installation and getting started guide allows `llem run` success on first attempt | CLI source confirms exact flag names; gpt2 model confirmed tiny and fast |
| DOCS-02 | Docker setup guide enables NVIDIA Container Toolkit install and vLLM experiment | Preflight checks confirm toolkit detection steps; Dockerfile.vllm confirms image tag |
| DOCS-03 | Backend config guide explains PyTorch (local) vs vLLM (Docker) switching | runner_resolution.py + image_registry.py confirm runner syntax; backends confirmed |
| DOCS-04 | Study YAML reference covers all sweep grammar with working copy-paste examples | grid.py sweep implementation fully audited; all syntax patterns confirmed |
</phase_requirements>

---

## Summary

Phase 23 is a documentation-only phase. There is no new code to write — all implementation was completed in Phases 16-22. The phase produces 13 Markdown files (9 researcher path + 4 policy maker path) and updates README.md. It also produces or updates 3 auto-generation scripts and adds a CI freshness check.

The most important research finding is that the **existing docs/ directory contains entirely stale content** that references the old CLI (`lem`), old commands (`lem campaign`, `lem experiment`, `lem init`), old concepts (`campaigns`, `num_cycles: 1-10 cycles`), and old config schema (v1.x field names like `model_name`, `fp_precision`, `num_input_prompts`). Every file in `docs/` must be deleted and replaced. The existing `scripts/generate_*.py` files must also be audited — `generate_config_docs.py` uses v1.x introspection patterns that won't work correctly with v2.0 models.

The auto-generation pipeline decision is: **use `typer utils docs` for CLI reference** (zero custom code), **replace `generate_config_docs.py` with Pydantic `model_json_schema()`-based renderer** (more robust with nested models), and **keep `generate_param_matrix.py` and `generate_invalid_combos_doc.py`** for their unique logic.

**Primary recommendation:** Write all 13 doc files from scratch against the actual v2.0 CLI, config schema, and sweep grammar. Use `typer utils docs` for CLI reference generation. Rewrite `generate_config_docs.py` to use `model_json_schema()`. Add CI freshness step.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Markdown (GitHub-rendered) | N/A | Doc format | Locked decision — no MkDocs |
| Pydantic `model_json_schema()` | 2.0+ | Config schema extraction | Official Pydantic v2 API; replaces custom introspection |
| `typer utils docs` | 0.9+ | CLI reference generation | Built-in Typer feature; zero custom code |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python 3.10+ | 3.10+ | Running generate scripts | All scripts already use this |
| `scripts/generate_param_matrix.py` | project | Parameter support matrix | Keep as-is; unique logic |
| `scripts/generate_invalid_combos_doc.py` | project | Invalid combo docs | Keep as-is; unique logic |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom introspection in generate_config_docs.py | `model_json_schema()` | json_schema is official Pydantic v2 API, handles nested models correctly, requires thin renderer |
| `typer utils docs` | Custom click introspection | typer built-in is zero maintenance; custom would need updates with every CLI change |
| Flat docs/ | Nested docs/ with subdirs | Flat matches project scale; nested is MkDocs territory (deferred) |

**Installation:** No new dependencies. All scripts run against the existing project install.

---

## Architecture Patterns

### Recommended Project Structure

```
docs/
├── installation.md          # System requirements, pip extras, dev install
├── getting-started.md       # Two-track: PyTorch quick + Docker recommended
├── docker-setup.md          # NVIDIA CT walkthrough, host requirements
├── backends.md              # PyTorch vs vLLM, runner config, param matrix (inlined)
├── study-config.md          # YAML reference, sweep grammar, config reference (inlined)
├── cli-reference.md         # llem run + llem config flags (auto-generated)
├── energy-measurement.md    # What we measure and how
├── methodology.md           # Statistical approach, warmup, thermal stabilisation
├── troubleshooting.md       # Common pitfalls, invalid combos (inlined)
├── guide-what-we-measure.md       # Policy maker: energy/throughput/FLOPs plain language
├── guide-interpreting-results.md  # Policy maker: reading llem output
├── guide-getting-started.md       # Policy maker: minimal path to running
└── guide-comparison-context.md    # Policy maker: MLPerf / AI Energy Score context

scripts/
├── generate_config_docs.py        # REWRITE: Pydantic model_json_schema() renderer
├── generate_param_matrix.py       # KEEP: unique test-results based logic
└── generate_invalid_combos_doc.py # KEEP: unique validation rule documentation

.github/workflows/ci.yml           # ADD: freshness check step
README.md                          # REWRITE: overview + links structure
```

### Pattern 1: Typer Built-in Doc Generation

Typer 0.9+ ships `typer.utils.docs` which renders a Typer app to Markdown. The CLI app is defined in `src/llenergymeasure/cli/__init__.py`.

**What:** Import the app and call `get_docs_for_typer_app()` to get Markdown.
**When to use:** CLI reference generation only.

```python
# Source: Typer docs + direct codebase inspection
# scripts/generate_cli_reference.py (NEW)
import typer.utils
from llenergymeasure.cli import app

markdown = typer.utils.get_docs_for_typer_app(app)
Path("docs/cli-reference.md").write_text(markdown)
```

Caveat: `typer.utils.get_docs_for_typer_app` is the stable public API as of Typer 0.9+. Verify at generation time that output format is suitable — may need a header/footer wrapper.

### Pattern 2: Pydantic model_json_schema() for Config Reference

The existing `generate_config_docs.py` uses a custom recursive field extractor that doesn't handle Optional nested models correctly (e.g., `vllm: VLLMConfig | None`). Replace with `model_json_schema()`.

**What:** Extract schema from ExperimentConfig and StudyConfig, render to Markdown tables.
**When to use:** study-config.md config reference section.

```python
# Source: Pydantic v2 docs — model_json_schema()
from llenergymeasure.config.models import ExperimentConfig

schema = ExperimentConfig.model_json_schema()
# schema is a dict with 'properties', '$defs', etc.
# Write thin renderer: iterate schema['properties'], resolve $refs from $defs
```

The schema handles nested models via `$defs` — the renderer needs to follow `$ref` links. This is more robust than the current hand-rolled recursive extractor.

### Pattern 3: CI Freshness Check

Add a job or step to `ci.yml` that regenerates all auto-generated content and fails if the output differs from the committed version.

```yaml
# .github/workflows/ci.yml addition
- name: Docs freshness check
  run: |
    uv run python scripts/generate_config_docs.py
    uv run python scripts/generate_param_matrix.py
    uv run python scripts/generate_invalid_combos_doc.py
    git diff --exit-code docs/
```

**Note:** The param matrix requires test result JSON files as input (`results/test_results_*.json`). In CI without GPU, these files won't exist. The freshness check for param matrix should use a pre-committed matrix or skip on missing inputs.

### Pattern 4: Getting Started Annotated Output

The user decisions specify "annotated output" — real terminal output with inline annotations explaining each metric. This pattern requires:
1. Running `llem run --model gpt2 --backend pytorch` locally to capture real output.
2. Annotating each line in the Markdown with comments or a table.
3. The annotated output must be current against the actual CLI display code (`src/llenergymeasure/cli/_display.py`).

### Anti-Patterns to Avoid

- **Copying stale quickstart.md content**: The existing `docs/quickstart.md` uses `lem init`, `lem experiment`, `lem campaign` — none of these exist. Do not reference or copy from existing docs.
- **Using old field names in YAML examples**: Old schema used `model_name`, `fp_precision`, `num_input_prompts`. Current schema uses `model`, `precision`, `n`.
- **Generating separate generated/ folder**: CONTEXT.md explicitly requires generated content inlined into parent topic files.
- **MkDocs or any build system**: Locked decision — plain GitHub-rendered Markdown only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI reference docs | Custom click/typer introspection | `typer.utils.get_docs_for_typer_app()` | Built-in, zero maintenance, stays current with CLI |
| Config schema extraction | Custom recursive model field walker | `ExperimentConfig.model_json_schema()` | Official Pydantic v2 API, handles $refs, nested models, Optional correctly |

**Key insight:** The existing `generate_config_docs.py` is a custom recursive field extractor written before Pydantic v2's `model_json_schema()` was understood. Replace it. The CLI reference generation should use Typer's built-in — the current `scripts/` directory has no CLI reference generator, meaning this is a new script.

---

## Common Pitfalls

### Pitfall 1: Stale CLI Name

**What goes wrong:** Docs say `lem` instead of `llem`. Every existing doc in `docs/` uses the old CLI name.
**Why it happens:** The rename from `lem` to `llem` happened at v2.0. The old docs were never updated.
**How to avoid:** Delete all existing docs. Search project for `llem` usage in CLI source (`src/llenergymeasure/cli/`) to confirm current command names.
**Warning signs:** Any reference to `lem init`, `lem experiment`, `lem campaign`, `lem doctor`, `lem resume`.

### Pitfall 2: Old Config Field Names

**What goes wrong:** YAML examples use `model_name:` instead of `model:`, `fp_precision:` instead of `precision:`, `num_input_prompts:` instead of `n:`.
**Why it happens:** v2.0 renamed these fields (documented in `config/models.py` docstring).
**How to avoid:** All YAML examples must be derived from actual `ExperimentConfig` model fields, not from old docs. The rename mapping is in `models.py`:
  - `model_name` → `model`
  - `fp_precision` → `precision`
  - `num_input_prompts` → `n`
  - `extra_metadata` → `passthrough_kwargs`
**Warning signs:** `model_name:`, `fp_precision:`, `num_input_prompts:`, `config_name:`, `schema_version:`.

### Pitfall 3: Old Commands

**What goes wrong:** Docs reference commands that don't exist: `llem init`, `llem campaign`, `llem experiment`, `llem results`, `llem datasets`, `llem doctor`, `llem aggregate`.
**Why it happens:** v2.0 reduced to 2 commands + 1 flag (`llem run`, `llem config`, `llem --version`). The old multi-command CLI was v1.x.
**How to avoid:** The CLI entrypoint is `src/llenergymeasure/cli/__init__.py`. Only `run` and `config_command` are registered. Check this file as source of truth.
**Warning signs:** Any command other than `llem run`, `llem config`, and `llem --version`.

### Pitfall 4: Old Study Concept Names

**What goes wrong:** Docs use "campaigns" (old term) instead of "studies" (current term).
**Why it happens:** v1.x called multi-experiment runs "campaigns". v2.0 renamed to "studies" throughout.
**How to avoid:** Use "study" / "sweep" consistently. The YAML keys are `experiments:`, `sweep:`, `execution:`.
**Warning signs:** "campaign", `lem campaign`, `_extends:` (old inheritance — now `base:` in study YAML).

### Pitfall 5: Typer docs API Naming

**What goes wrong:** `typer.utils.get_docs_for_typer_app` may not exist in older Typer versions, or may have changed signature.
**Why it happens:** Typer evolves its internal API.
**How to avoid:** Check `typer.__version__` at script runtime. The project uses `typer>=0.9`. Verify `get_docs_for_typer_app` exists at that version.

### Pitfall 6: param_matrix CI Gap

**What goes wrong:** CI freshness check for param matrix fails because test result JSONs don't exist without GPU.
**Why it happens:** `generate_param_matrix.py` reads `results/test_results_{backend}.json` which requires actual GPU test runs.
**How to avoid:** Skip param matrix freshness check in CI (or check only that the script runs without error given missing inputs). Commit a reference matrix that is manually updated.

### Pitfall 7: vllm: section Nesting

**What goes wrong:** YAML examples show flat `vllm:` section, but the actual schema uses nested `vllm.engine:` and `vllm.sampling:`.
**Why it happens:** Phase 19.1 introduced the nested `VLLMEngineConfig` / `VLLMSamplingConfig` split, which post-dates the old docs.
**How to avoid:** All vLLM YAML examples must use the nested structure:
  ```yaml
  backend: vllm
  vllm:
    engine:
      enforce_eager: false
      gpu_memory_utilization: 0.9
    sampling:
      max_tokens: 512
  ```

---

## Code Examples

Verified against actual source:

### Single Experiment YAML (minimal)
```yaml
# Source: config/models.py ExperimentConfig defaults
model: gpt2
backend: pytorch
n: 100
```

### Single Experiment YAML (PyTorch with extras)
```yaml
# Source: config/models.py + config/backend_configs.py PyTorchConfig
model: gpt2
backend: pytorch
n: 100
precision: bf16
max_input_tokens: 512
max_output_tokens: 128
decoder:
  preset: deterministic
pytorch:
  batch_size: 4
  attn_implementation: sdpa
warmup:
  enabled: false   # disable for quick testing
```

### Single Experiment YAML (vLLM via Docker)
```yaml
# Source: config/backend_configs.py VLLMConfig + VLLMEngineConfig + VLLMSamplingConfig
model: gpt2
backend: vllm
n: 100
runners:
  vllm: docker
vllm:
  engine:
    enforce_eager: false
    gpu_memory_utilization: 0.9
    kv_cache_dtype: auto
  sampling:
    max_tokens: 128
```

### Study YAML (sweep grammar)
```yaml
# Source: study/grid.py _expand_sweep() + loader.py load_study_config()
name: batch-size-sweep

model: gpt2
backend: pytorch

sweep:
  pytorch.batch_size: [1, 2, 4, 8]
  precision: [fp16, bf16]

execution:
  n_cycles: 3
  cycle_order: shuffled
```

This produces 4 × 2 = 8 configs × 3 cycles = 24 runs.

### Study YAML (explicit experiments list)
```yaml
# Source: study/grid.py explicit_entries path
name: compare-backends

experiments:
  - model: gpt2
    backend: pytorch
    n: 50
  - model: gpt2
    backend: vllm
    n: 50
    runners:
      vllm: docker

execution:
  n_cycles: 3
  cycle_order: interleaved
```

### Study YAML (base + sweep)
```yaml
# Source: study/grid.py _load_base() + _extract_fixed()
name: precision-sweep

base: base-experiment.yaml   # path relative to study YAML

sweep:
  precision: [fp32, fp16, bf16]

execution:
  n_cycles: 3
```

### Study YAML (vLLM scoped sweep — three-segment path)
```yaml
# Source: study/grid.py _expand_sweep() three-segment handling (Phase 19.2)
name: kv-cache-sweep

model: gpt2
backend: vllm

sweep:
  vllm.engine.block_size: [8, 16, 32]
  vllm.engine.kv_cache_dtype: [auto, fp8]

runners:
  vllm: docker

execution:
  n_cycles: 3
```

### llem run CLI invocations
```bash
# Source: cli/run.py + cli/__init__.py
# Single experiment via flags
llem run --model gpt2 --backend pytorch

# Single experiment via YAML
llem run experiment.yaml

# Dry run (validate + estimate VRAM)
llem run experiment.yaml --dry-run

# Study (auto-detected from YAML sweep: or experiments: keys)
llem run study.yaml

# Study with CLI overrides
llem run study.yaml --cycles 5 --order interleaved

# Suppress thermal gaps (testing only)
llem run study.yaml --no-gaps

# Skip Docker preflight
llem run study.yaml --skip-preflight

# Environment check
llem config
llem config --verbose
```

### User config file path and schema
```yaml
# Path: ~/.config/llenergymeasure/config.yaml (XDG via platformdirs)
# Source: config/user_config.py UserConfig
output:
  results_dir: ./results
  model_cache_dir: ~/.cache/huggingface
runners:
  pytorch: local
  vllm: docker         # auto-elevate vLLM to Docker
  tensorrt: local
measurement:
  datacenter_pue: 1.0
  grid_carbon_intensity: 0.233
```

### Runner config syntax (YAML study + user config)
```yaml
# Per-study runner config (in study YAML)
runners:
  vllm: docker                          # use built-in image
  vllm: docker:my-registry/llem:custom  # explicit image override

# Per-backend env var override
LLEM_RUNNER_VLLM=docker:custom/image:tag llem run study.yaml
```

---

## Codebase-Sourced Content for Docs

### What llem config displays

Sourced from `src/llenergymeasure/cli/config_cmd.py`:

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Backends
  pytorch: installed  (2.5.1+cu124)
  vllm: not installed  (pip install llenergymeasure[vllm])
  tensorrt: not installed  (pip install llenergymeasure[tensorrt])
Energy
  Energy: nvml
Config
  Path: /home/user/.config/llenergymeasure/config.yaml
  Status: using defaults (no config file)
Python
  3.12.0
```

### All `llem run` flags (from `cli/run.py`)

| Flag | Short | Type | Default | Purpose |
|------|-------|------|---------|---------|
| `config` | positional | Path \| None | None | Path to experiment or study YAML |
| `--model` | `-m` | str | None | Model name or HuggingFace path |
| `--backend` | `-b` | str | None | Inference backend (pytorch, vllm, tensorrt) |
| `--dataset` | `-d` | str | None | Dataset name |
| `-n` | | int | None | Number of prompts |
| `--batch-size` | | int | None | Batch size (maps to pytorch.batch_size) |
| `--precision` | `-p` | str | None | Floating point precision |
| `--output` | `-o` | str | None | Output directory |
| `--dry-run` | | bool | False | Validate config, estimate VRAM, no execution |
| `--quiet` | `-q` | bool | False | Suppress progress bars |
| `--verbose` | `-v` | bool | False | Show detailed output and tracebacks |
| `--cycles` | | int | None | Number of cycles (study mode) |
| `--order` | | str | None | Cycle ordering: sequential, interleaved, shuffled |
| `--no-gaps` | | bool | False | Disable thermal gaps between experiments |
| `--skip-preflight` | | bool | False | Skip Docker pre-flight checks |

CLI effective defaults for study mode (applied when neither YAML nor `--cycles`/`--order` set):
- `n_cycles = 3` (Pydantic default is 1; CLI overrides to 3)
- `cycle_order = shuffled` (Pydantic default is sequential; CLI overrides to shuffled)

### Installation extras
```bash
# Source: pyproject.toml
pip install llenergymeasure                          # base only (no inference backend)
pip install "llenergymeasure[pytorch]"               # PyTorch backend
pip install "llenergymeasure[vllm]"                  # vLLM backend
pip install "llenergymeasure[tensorrt]"              # TensorRT-LLM backend
pip install "llenergymeasure[zeus]"                  # Zeus energy measurement
pip install "llenergymeasure[codecarbon]"            # CodeCarbon energy measurement
pip install "llenergymeasure[pytorch,zeus]"          # Combined extras
```

### System requirements
- Python 3.10+ (hard requirement: TensorRT-LLM 3.10 compat)
- Linux (required for vLLM and TensorRT-LLM)
- NVIDIA GPU with CUDA (required for all inference backends)
- Docker + NVIDIA Container Toolkit (required for vLLM and TensorRT-LLM)
- CUDA 12.x on host (for container compatibility with image tags)

### Docker image registry
```
ghcr.io/henrycgbaker/llenergymeasure/vllm:{version}-cuda{cuda_major}
# e.g. ghcr.io/henrycgbaker/llenergymeasure/vllm:0.8.0-cuda12
```
Source: `infra/image_registry.py DEFAULT_IMAGE_TEMPLATE`

### Docker pre-flight checks (what they verify)
Source: `infra/docker_preflight.py`
1. Docker CLI on PATH
2. NVIDIA Container Toolkit binary on PATH (nvidia-container-runtime, nvidia-ctk, or nvidia-container-cli)
3. Host nvidia-smi present (warn only — supports remote daemon)
4. GPU visibility inside container (`docker run --gpus all nvidia-smi`)
5. CUDA/driver compatibility (parsed from container probe output)

### Sweep grammar (complete)
Source: `study/grid.py _expand_sweep()`

| Feature | Syntax | Example |
|---------|--------|---------|
| Single dimension | `key: [v1, v2]` | `precision: [fp16, bf16]` |
| Multiple dimensions | multiple keys, Cartesian product | `precision: [fp16, bf16]` + `n: [50, 100]` = 4 configs |
| Backend-scoped (2-segment) | `backend.param: [v1, v2]` | `pytorch.batch_size: [1, 4, 8]` |
| Backend-scoped (3-segment) | `backend.section.param: [v1, v2]` | `vllm.engine.block_size: [8, 16, 32]` |
| Explicit experiments | `experiments:` list | explicit list, merged with base fixed fields |
| Base inheritance | `base: path.yaml` | strips study-only keys from base file |
| Mixed sweep + explicit | `sweep:` + `experiments:` | sweep configs appended with explicit entries |

Cycle ordering semantics:
- `sequential`: [A, A, A, B, B, B] — all cycles of each experiment together
- `interleaved`: [A, B, A, B, A, B] — one cycle of each experiment, repeated
- `shuffled`: random per-cycle order, seeded from study_design_hash (reproducible)

### ExperimentConfig top-level fields (v2.0)
Source: `config/models.py ExperimentConfig`

Required:
- `model: str` — HuggingFace model ID or local path

Optional top-level:
- `backend: pytorch | vllm | tensorrt` (default: pytorch)
- `n: int` (default: 100) — number of prompts
- `dataset: str | SyntheticDatasetConfig` (default: "aienergyscore")
- `dataset_order: interleaved | grouped | shuffled` (default: interleaved)
- `precision: fp32 | fp16 | bf16` (default: bf16)
- `random_seed: int` (default: 42)
- `max_input_tokens: int` (default: 512)
- `max_output_tokens: int` (default: 128)

Sub-configs:
- `decoder: DecoderConfig` — temperature, top_k, top_p, repetition_penalty, preset
- `warmup: WarmupConfig` — enabled, n_warmup, thermal_floor_seconds
- `baseline: BaselineConfig` — enabled, duration_seconds
- `energy: EnergyConfig` — backend: auto|nvml|zeus|codecarbon|null

Backend-specific sections (match backend field):
- `pytorch: PyTorchConfig` — batch_size, attn_implementation, torch_compile, load_in_4bit, etc.
- `vllm: VLLMConfig` — engine: VLLMEngineConfig, sampling: VLLMSamplingConfig, beam_search: VLLMBeamSearchConfig
- `tensorrt: TensorRTConfig` — max_batch_size, tp_size, quantization, engine_path

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `lem` CLI (multi-command) | `llem` CLI (2 commands + 1 flag) | v2.0 (M1) | All doc examples must use `llem run`, `llem config` |
| `model_name:`, `fp_precision:` fields | `model:`, `precision:`, `n:` | v2.0 (M1) | All YAML examples must use v2.0 field names |
| Flat `vllm:` config | Nested `vllm.engine:` + `vllm.sampling:` | v2.0 Phase 19.1 | vLLM YAML examples need nested structure |
| "campaigns" (multi-config runs) | "studies" / "sweeps" | v2.0 (M2) | All references to "campaign" must change to "study" |
| `lem campaign` command | `llem run study.yaml` (auto-detected) | v2.0 (M2) | No separate study command — `llem run` detects sweep: or experiments: keys |
| `generate_config_docs.py` (custom recursive) | `model_json_schema()` + thin renderer | Phase 23 | More robust; handles Optional nested models |
| No CLI reference generator | `typer.utils.get_docs_for_typer_app` | Phase 23 | Zero-maintenance CLI docs |
| `docs/generated/` subfolder | Inline in parent topic files | Phase 23 (decision) | Simpler structure, no cross-file nav needed |

**Deprecated/outdated in existing docs:**
- `lem init`: Was a v1.x setup wizard command. Does not exist in v2.0.
- `lem doctor`: Was a v1.x environment check. Replaced by `llem config`.
- `lem campaign`: Was a v1.x multi-config command. Replaced by `llem run study.yaml`.
- `lem experiment`: Was a v1.x single experiment command. Replaced by `llem run`.
- `lem results list/show`: Was a v1.x results browser. Removed in v2.0.
- `_extends:` in YAML: Was v1.x config inheritance. Now `base:` in study YAML.
- `docs/generated/` as a separate folder: Per CONTEXT.md decision, generated content goes inline.

---

## Open Questions

1. **typer utils docs output quality**
   - What we know: `typer.utils.get_docs_for_typer_app()` exists in Typer 0.9+. The project uses `typer>=0.9`.
   - What's unclear: Whether the raw output needs a header/footer wrapper to fit into `cli-reference.md`. Output format not verified against current Typer version.
   - Recommendation: Prototype the generation in Wave 0 / early task. If output format is poor, fall back to manually documenting the flags from the verified table above.

2. **Param matrix CI freshness**
   - What we know: `generate_param_matrix.py` requires `results/test_results_{backend}.json` from GPU test runs. CI has no GPU.
   - What's unclear: Whether to skip param matrix check in CI entirely or provide a stub mechanism.
   - Recommendation: Add freshness check for config_docs and invalid_combos (both GPU-free). Document param matrix as "manually updated on GPU hardware runs". Skip its freshness check in CI.

3. **`llem config` vs `llem init` rename**
   - What we know: The todo exists in `.planning/todos/pending/2026-03-05-rename-llem-config-subcommand-to-llem-init.md`. It is deferred — not in Phase 23 scope.
   - What's unclear: Whether the rename will happen before or after Phase 23 merges.
   - Recommendation: Use `llem config` throughout (current name). CONTEXT.md delegates this to Claude's discretion — since rename is deferred, use current name.

4. **Annotated terminal output**
   - What we know: Getting-started requires "annotated output: real terminal output with inline annotations".
   - What's unclear: The exact terminal output format — requires running `llem run --model gpt2 --backend pytorch` locally. Host has GPU visible via pynvml but `torch.cuda.is_available()` returns False outside containers.
   - Recommendation: Run in the Docker container to capture real output, or use known format from `cli/_display.py` to construct a realistic annotated example. The display code is readable and predictable — constructing a realistic example from source is acceptable.

---

## Validation Architecture

`nyquist_validation: true` in `.planning/config.json` — include this section.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` (if present) |
| Quick run command | `uv run pytest tests/unit/ -m "not gpu and not docker" -n auto -v --tb=short` |
| Full suite command | `uv run pytest tests/ -m "not gpu and not docker" -n auto -v --tb=short` |

### Phase Requirements → Test Map

Documentation phases are predominantly manual verification. The success criteria are:
1. A user can run `llem run` successfully following the guide — **manual verification**
2. A user can configure Docker/NVIDIA CT following the guide — **manual verification**
3. Backend config guide covers PyTorch vs vLLM switching — **manual verification**
4. Study YAML reference covers all sweep grammar — **automated: syntax validated by existing grid tests**

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOCS-01 | YAML examples in docs are valid | unit | `uv run pytest tests/unit/config/test_config_loader.py -x` | ✅ |
| DOCS-01 | llem run command flags match source | manual | visual comparison with cli/run.py | N/A |
| DOCS-02 | Docker setup steps are accurate | manual | follow guide on host with Docker | N/A |
| DOCS-03 | Backend switching syntax examples are valid | unit | `uv run pytest tests/unit/config/ -x` | ✅ |
| DOCS-04 | Sweep YAML examples are parseable | unit | `uv run pytest tests/unit/study/test_study_grid.py -x` | ✅ |
| DOCS-04 | CI freshness check regenerates docs | ci | Added step in `.github/workflows/ci.yml` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -m "not gpu and not docker" -n auto --tb=short -q`
- **Per wave merge:** same as above (docs phase has no new Python code to test)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `scripts/generate_cli_reference.py` — new script (no test; script correctness verified by CI freshness check)
- [ ] `.github/workflows/ci.yml` — add docs freshness step (covers DOCS-04 CI validation)

No new unit test files needed — all YAML example validation is covered by existing config loader and grid tests. The primary gap is the CI freshness check step.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/llenergymeasure/cli/run.py` — all CLI flags verified
- Direct codebase inspection: `src/llenergymeasure/cli/config_cmd.py` — `llem config` output format
- Direct codebase inspection: `src/llenergymeasure/config/models.py` — ExperimentConfig, StudyConfig, ExecutionConfig
- Direct codebase inspection: `src/llenergymeasure/config/backend_configs.py` — PyTorchConfig, VLLMConfig hierarchy
- Direct codebase inspection: `src/llenergymeasure/study/grid.py` — sweep grammar, cycle ordering
- Direct codebase inspection: `src/llenergymeasure/infra/image_registry.py` — Docker image template
- Direct codebase inspection: `src/llenergymeasure/infra/docker_preflight.py` — preflight check tiers
- Direct codebase inspection: `src/llenergymeasure/config/user_config.py` — user config schema
- Direct codebase inspection: `pyproject.toml` — package extras, Python requirement, entry point
- Direct codebase inspection: `src/llenergymeasure/__init__.py` — public API, version

### Secondary (MEDIUM confidence)
- `scripts/generate_config_docs.py` — existing script patterns (audited; to be replaced)
- `tests/fixtures/sigint_study.yaml` — confirmed working study YAML example
- `.planning/phases/23-documentation/23-CONTEXT.md` — user decisions (primary constraint source)
- `.planning/STATE.md` — accumulated project decisions, phase history

### Tertiary (LOW confidence)
- Typer `utils.get_docs_for_typer_app()` API — training knowledge (Typer 0.9+); verify at script execution time

---

## Metadata

**Confidence breakdown:**
- Doc content accuracy: HIGH — all field names, CLI flags, YAML syntax verified against live source
- Auto-generation approach: HIGH — `model_json_schema()` is official Pydantic v2 API; Typer built-in is documented feature
- Docker walkthrough steps: MEDIUM — steps sourced from `docker_preflight.py` error messages and NVIDIA CT docs URL; execution not tested on clean host
- Policy maker docs: MEDIUM — content decisions delegated to Claude's discretion per CONTEXT.md

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain; Pydantic/Typer APIs unlikely to change)
