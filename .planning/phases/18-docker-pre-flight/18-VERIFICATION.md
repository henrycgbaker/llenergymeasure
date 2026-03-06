---
phase: 18-docker-pre-flight
status: passed
verified: 2026-03-05T22:27:00Z
retroactive: true
requirements_verified:
  - DOCK-07
  - DOCK-08
  - DOCK-09
evidence_source: "Retroactive verification against codebase on main (PR #28)"
---

# Phase 18 Verification: Docker Pre-flight Checks

Retroactive verification of DOCK-07, DOCK-08, and DOCK-09 against the codebase on
`main`. Evidence gathered by direct inspection of source files on 2026-03-05.

---

## DOCK-07 — NVIDIA Container Toolkit Detection

**Requirement:** The pre-flight check must detect whether the NVIDIA Container Toolkit
is installed on the host before launching any GPU container.

**Status: PASS**

### Evidence

**Function definition** — `src/llenergymeasure/infra/docker_preflight.py`, lines 64-71:

```python
def _check_nvidia_toolkit() -> str | None:
    """Return an error string if no NVIDIA Container Toolkit binary is on PATH, else None."""
    if not any(shutil.which(tool) is not None for tool in _NVIDIA_TOOLKIT_BINS):
        return (
            "NVIDIA Container Toolkit not found\n"
            f"     Fix: Install NVIDIA Container Toolkit — {_NVIDIA_TOOLKIT_INSTALL_URL}"
        )
    return None
```

**Toolkit binaries checked** — `src/llenergymeasure/infra/docker_preflight.py`, lines 45-49:

```python
_NVIDIA_TOOLKIT_BINS = (
    "nvidia-container-runtime",
    "nvidia-ctk",
    "nvidia-container-cli",
)
```

**Call site in `run_docker_preflight()`** — lines 249-251:

```python
toolkit_error = _check_nvidia_toolkit()
if toolkit_error is not None:
    tier1_failures.append(toolkit_error)
```

**Raises `DockerPreFlightError` on failure** — lines 256-259:

```python
if tier1_failures:
    n = len(tier1_failures)
    numbered = "\n".join(f"  {i + 1}. {msg}" for i, msg in enumerate(tier1_failures))
    raise DockerPreFlightError(f"Docker pre-flight failed: {n} issue(s) found\n{numbered}")
```

`DockerPreFlightError` imported from `llenergymeasure.exceptions` (line 27).

---

## DOCK-08 — GPU Visibility Check Inside Container

**Requirement:** The pre-flight check must verify that a GPU is accessible from inside
a Docker container before running experiments.

**Status: PASS**

### Evidence

**Function definition** — `src/llenergymeasure/infra/docker_preflight.py`, lines 117-208:

```python
def _probe_container_gpu(host_driver_version: str | None) -> list[str]:
    """Run a lightweight container probe to validate GPU visibility and CUDA compat.

    Combines GPU name and driver version queries into a single docker run invocation.
    Returns a list of error strings (empty = all OK).
    """
```

**`docker run --gpus all` invocation** — lines 126-141:

```python
result = subprocess.run(
    [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        _PROBE_IMAGE,
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    timeout=_PROBE_TIMEOUT,
)
```

`_PROBE_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"` (line 37).

**GPU not accessible error path** — lines 183-187:

```python
else:
    errors.append(
        "GPU not accessible inside Docker container\n"
        "     Possible cause: NVIDIA Container Toolkit not configured correctly.\n"
        f"     Fix: {_NVIDIA_TOOLKIT_INSTALL_URL}"
    )
```

**Call site in `run_docker_preflight()`** — line 264:

```python
tier2_failures = _probe_container_gpu(host_driver_version)
```

The function is called in Tier 2 — only reached after Tier 1 (host checks) passes.

---

## DOCK-09 — CUDA/Driver Version Compatibility Check

**Requirement:** The pre-flight check must detect CUDA/driver version incompatibilities
and produce a descriptive error message.

**Status: PASS**

### Evidence

**CUDA compat detection patterns** — `src/llenergymeasure/infra/docker_preflight.py`, lines 159-170:

```python
_is_cuda_compat_error = (
    (
        "cuda" in stderr_lower
        and ("version" in stderr_lower or "incompatible" in stderr_lower)
    )
    or "driver/library version mismatch" in stderr_lower
    or ("nvml" in stderr_lower and "driver" in stderr_lower)
    or ("initialize nvml" in stderr_lower and "driver" in stderr_lower)
)
```

Four patterns matched:
1. `cuda` + `version` — "forward compat CUDA X requires driver Y" messages
2. `cuda` + `incompatible` — "CUDA driver version is insufficient for CUDA runtime version"
3. `driver/library version mismatch` — NVML mismatch messages
4. `nvml` + `driver` / `initialize nvml` + `driver` — NVML initialisation failures

**Descriptive error message with host driver info** — lines 172-181:

```python
if _is_cuda_compat_error:
    host_info = (
        f"Host driver: {host_driver_version}"
        if host_driver_version
        else "Host driver: unknown"
    )
    errors.append(
        f"CUDA/driver compatibility error inside container. {host_info}\n"
        "     The container CUDA version may require a newer host driver.\n"
        f"     See: {_CUDA_COMPAT_URL}"
    )
```

The error message includes the host driver version and a link to CUDA compatibility docs.

**Design note (from SUMMARY.md):** The detection patterns were deliberately tightened to
avoid false positives — the phrase "device driver" alone (from generic GPU access errors)
does not trigger the CUDA compat path. Only explicit CUDA version/incompatible/NVML mismatch
patterns match.

---

## Summary

| Requirement | Description                             | Status |
| ----------- | --------------------------------------- | ------ |
| DOCK-07     | NVIDIA Container Toolkit detection      | PASS   |
| DOCK-08     | GPU visibility check inside container   | PASS   |
| DOCK-09     | CUDA/driver version compatibility check | PASS   |

All three requirements are wired end-to-end in
`src/llenergymeasure/infra/docker_preflight.py` and exposed via `run_docker_preflight()`.
Phase 18 code was merged to main in PR #28 (commit 765a251, 2757b94).
