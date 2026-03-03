# Phase 11: Subprocess Isolation & StudyRunner - Research (Retroactive)

**Researched:** 2026-02-28
**Domain:** Python multiprocessing, CUDA process isolation, IPC, signal handling, experiment scheduling
**Confidence:** HIGH (implementation already verified; research validates design choices against peers)
**Status:** Retroactive audit -- phase already implemented and verified (13/13 truths)

## Summary

This retroactive research validates the Phase 11 implementation against peer ML benchmarking tools and Python multiprocessing best practices. The core finding is that the implementation closely follows the patterns established by optimum-benchmark (HuggingFace) -- the most directly comparable tool -- and aligns with PyTorch's official CUDA multiprocessing guidance.

Five specific areas were investigated: (1) start method choice (`spawn` vs `fork` vs `forkserver`), (2) SIGINT handling with SIG_IGN in children, (3) IPC via Pipe for result transfer, (4) timeout and kill strategies, and (5) experiment scheduling/ordering patterns. The implementation is sound in all five areas. Two potential gaps were identified: the absence of a file-based IPC fallback for large results (documented in the design but not implemented), and the use of `multiprocessing.Queue` rather than `SimpleQueue` for progress events.

**Primary finding:** The implementation is well-aligned with peer practice. No critical gaps found.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `spawn` + `Pipe` + `Queue` IPC mechanism
- First Ctrl+C: SIGTERM + grace period, second: SIGKILL
- Children install SIG_IGN for SIGINT -- parent owns signal
- "interrupted" manifest status distinct from "failed"
- Cycle ordering: sequential, interleaved, shuffled
- Timeout heuristic: max(n*2, 600)
- Gap countdown with Enter-to-skip

### Claude's Discretion
- Exact SIGTERM grace period duration (2-3s range)
- Rich display integration details for countdown
- Internal implementation of Enter-to-skip (threading, select, etc.)
- How the seeded shuffle integrates with existing random_seed field

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

---

## Research Question 1: How Do Peers Isolate GPU Experiments?

### Findings (HIGH confidence)

| Tool | Isolation Mechanism | Fresh Process per Experiment? | Start Method |
|------|---------------------|-------------------------------|--------------|
| **optimum-benchmark** | `multiprocessing.Process` + `Pipe` | Yes -- core design principle | `spawn` (implicit via `mp.Process`) |
| **Ray Tune** | Actor (Ray worker process); `reuse_actors=False` for isolation | Configurable; `reuse_actors=False` gives fresh workers | Not applicable (Ray manages workers) |
| **Optuna** | No built-in process isolation; trials run in-process or user-managed | No -- relies on external orchestration | Not applicable |
| **lm-eval-harness** | No subprocess isolation; single-process evaluation | No -- one model per invocation | Not applicable |
| **vLLM bench sweep** | `subprocess.Popen` for server process | Yes per serve_comb; server reused within | `fork` default, `spawn` when CUDA initialised |
| **MLPerf** | Thread-based SUT, one-shot invocation | N/A (no multi-experiment orchestration) | Not applicable |
| **AIEnergyScore** | `subprocess.run(docker_script)` per experiment | Yes -- ephemeral container per experiment | Not applicable (Docker) |

**Key insight:** Only optimum-benchmark and LLenergyMeasure use `multiprocessing.Process` per experiment with Pipe IPC. This is the correct pattern for energy measurement where GPU state isolation is a hard requirement. Ray Tune offers similar isolation via `reuse_actors=False` but at much higher complexity. lm-eval, Optuna, and MLPerf do not provide experiment-level process isolation at all.

**Assessment of implementation:** The `StudyRunner` pattern directly mirrors optimum-benchmark's process launcher. This is the right choice.

**Source:** optimum-benchmark `launchers/process/launcher.py` (GitHub, verified via existing research doc at `.product/research/13-execution-isolation-patterns.md`)

---

## Research Question 2: Is `spawn` the Right Start Method for CUDA?

### Findings (HIGH confidence)

**PyTorch official documentation states clearly:**

> "CUDA runtime does not support `fork`. Use `spawn` or `forkserver`."

Source: [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)

**`spawn` vs `forkserver` comparison:**

| Property | `spawn` | `forkserver` |
|----------|---------|--------------|
| CUDA safety | Safe -- fresh interpreter per child | Safe -- server process forks before CUDA init |
| Startup speed | Slower (full interpreter + module re-import) | Slightly faster (fork from clean server) |
| Library compatibility | Full -- no shared state from parent | Same `__main__` guard problem as `spawn` |
| Complexity | Simple -- one call to `get_context("spawn")` | More complex setup; server process lifecycle |
| Platform | Works on Linux, macOS, Windows | Linux/macOS only |

**vLLM's approach (documented in their design doc):** vLLM defaults to `fork` for performance, but forces `spawn` when CUDA has already been initialised. They considered `forkserver` but rejected it because it has the same `__main__` guard problem as `spawn` -- when vLLM is used as a library, unguarded code gets re-executed. Quote from vLLM docs:

> "The server process is created as a spawned new process, which will re-execute code not protected by a `__main__` guard."

Source: [vLLM Python Multiprocessing Design](https://docs.vllm.ai/en/stable/design/multiprocessing/)

**LLenergyMeasure's choice of `spawn` over `forkserver` is correct because:**

1. `spawn` is simpler (one line: `get_context("spawn")`) with no server process to manage
2. `forkserver` provides no practical benefit -- the startup overhead difference is negligible compared to model loading time (seconds vs tens of seconds/minutes)
3. `spawn` is cross-platform; `forkserver` is not available on Windows
4. `spawn` avoids any risk of inheriting CUDA state from the parent, which `forkserver` also avoids but with added complexity

**Note on `get_context()` vs `set_start_method()`:** The implementation correctly uses `get_context("spawn")` (scoped) rather than `set_start_method("spawn")` (global). This is best practice -- it does not affect other multiprocessing usage elsewhere in the process.

**Assessment of implementation:** Correct. No change needed.

---

## Research Question 3: How Do Peers Handle SIGINT in Multiprocessing?

### Findings (MEDIUM confidence)

**The `SIG_IGN` in children pattern is the standard approach.** Multiple authoritative sources confirm:

1. **Python docs / community consensus:** "You can assign a signal handler specific to a process by placing it in the worker method, such as: `signal.signal(signal.SIGINT, signal.SIG_IGN)` to make that process ignore SIGINT." The goal is to have the parent decide how children are stopped.

2. **The pattern:** Save original handler -> set SIG_IGN before/in children -> parent owns signal via custom handler that sets a threading.Event -> after loop, restore original handler.

3. **optimum-benchmark:** Does NOT install SIG_IGN in child processes. The process launcher has no explicit signal handling at all (verified via GitHub source fetch). It relies on the default behaviour: child inherits parent's signal disposition, and if the parent is killed, the child becomes an orphan (handled by `daemon=False` + `psutil.Process(main_process_pid)` monitoring).

4. **Ray Tune:** Handles signals internally within its actor framework. Not directly comparable.

**LLenergyMeasure's approach -- SIG_IGN in the worker, parent owns SIGINT via custom handler + threading.Event -- is the textbook Python multiprocessing pattern.** This is actually more robust than optimum-benchmark's approach, which lacks explicit signal handling.

**Two-stage escalation (SIGTERM -> 2s grace -> SIGKILL):** This is a common pattern for graceful shutdown. The implementation correctly:
- First Ctrl+C: calls `p.terminate()` (SIGTERM) on the active child
- Grace period: `p.join(timeout=2)` after SIGTERM
- Second Ctrl+C (or grace expired): calls `p.kill()` (SIGKILL)

**Assessment of implementation:** Correct and more robust than the primary peer (optimum-benchmark).

Source: [Python Multiprocessing Graceful Shutdown](https://www.peterspython.com/en/blog/python-multiprocessing-graceful-shutdown-in-the-proper-order), [SIGINT Handling with SIG_IGN](https://archive.zhimingwang.org/blog/2015-05-05-graceful-handling-of-sigint-when-using-pythons-multiprocessingprocess.html)

---

## Research Question 4: What IPC Patterns Do Peers Use for Returning Results?

### Findings (HIGH confidence)

| Tool | IPC Mechanism | Large Result Handling |
|------|---------------|----------------------|
| **optimum-benchmark** | `multiprocessing.Pipe` | File-based fallback for results >1MB (env var `FILE_BASED_COMM_THRESHOLD`) |
| **vLLM bench sweep** | HTTP REST API | N/A (client-server model) |
| **AIEnergyScore** | Mounted volume (Docker) | N/A (filesystem IPC) |
| **MLPerf** | In-process (no IPC needed) | N/A |

**Key difference from optimum-benchmark:** The LLenergyMeasure design doc (`.product/designs/experiment-isolation.md`) specifies a file-based IPC fallback for results >1MB, mirroring optimum-benchmark's `FILE_BASED_COMM_THRESHOLD` pattern. However, the implementation does NOT include this fallback.

**Is this a problem?** Probably not for the current scope:
- `ExperimentResult` in v1.18.0 contains summary metrics (energy, throughput, timing), not raw time-series data
- These results are well under 64KB, let alone 1MB
- The Pipe buffer on Linux is typically 64KB; the OS handles buffering transparently for messages up to this size
- Pickle serialisation of an `ExperimentResult` is likely a few KB

**Potential future risk:** If `ExperimentResult` grows to include time-series data (per-token latencies, GPU power traces), the Pipe could deadlock. The pattern is: child calls `conn.send(large_result)` which blocks because the OS pipe buffer is full, while the parent is blocked at `p.join()` waiting for the child to exit. Neither can proceed.

**Mitigation options (if needed later):**
1. File-based fallback (as designed) -- write to temp file, send path via Pipe
2. Reader thread -- drain Pipe in a background thread concurrent with `p.join()`
3. Use `Queue` instead of `Pipe` (Queue has its own background thread for flushing)

**PyTorch docs note:** "Use `SimpleQueue` over `multiprocessing.Queue` when possible, as the latter spawns internal threads that can cause lock contention." The current implementation uses `mp_ctx.Queue()` for progress events. This is fine for the low-volume progress event use case, but `SimpleQueue` would be marginally safer.

**Assessment of implementation:** Correct for current scope. File-based IPC fallback is not yet needed. Document the risk for future.

---

## Research Question 5: How Do Peers Handle Experiment Timeouts?

### Findings (HIGH confidence)

| Tool | Timeout Mechanism | Kill Strategy |
|------|-------------------|---------------|
| **optimum-benchmark** | **No timeout** -- busy-wait loop `while is_alive() and not poll(): pass` | No kill mechanism |
| **vLLM bench sweep** | `server_ready_timeout` for server startup; no per-experiment timeout | `os.killpg(pgid, SIGKILL)` -- kills entire process group |
| **Ray Tune** | Configurable per-trial timeout | Managed by Ray's actor lifecycle |
| **AIEnergyScore** | `subprocess.run` blocks indefinitely (no timeout) | Container-level Docker timeout |

**Key finding:** optimum-benchmark -- the primary peer -- has NO timeout mechanism at all. A hung CUDA call would block the benchmark indefinitely. LLenergyMeasure's `p.join(timeout=...)` + SIGKILL is strictly more robust.

**SIGKILL vs SIGTERM for hung CUDA processes:** The implementation correctly uses SIGKILL (`p.kill()`) for timeouts rather than SIGTERM (`p.terminate()`). SIGTERM may be ignored by a process stuck in a CUDA kernel call. CUDA's signal handling is not re-entrant, so a process deep in a CUDA operation may deadlock during SIGTERM handler execution. SIGKILL is the only guaranteed termination mechanism.

**The timeout heuristic `max(n*2, 600)` is generous:** 2 seconds per prompt with a 10-minute minimum. This is reasonable -- model loading alone can take 1-5 minutes for large models. The escape hatch (`experiment_timeout_seconds` in YAML) allows users to override.

**vLLM's `os.killpg()` pattern is notable:** vLLM kills the entire process group (`start_new_session=True` + `os.killpg(pgid, SIGKILL)`). This ensures all child processes of the server (tensor parallel workers, etc.) are killed. LLenergyMeasure's `p.kill()` only kills the direct child process. If the child has spawned sub-processes (e.g., vLLM's internal workers), those would become orphans. This is a **potential gap** for the Docker/vLLM phase but not for the current subprocess isolation phase, where the child runs a single backend directly.

**Assessment of implementation:** Correct and more robust than the primary peer (optimum-benchmark).

---

## Research Question 6: What Cycle Ordering / Experiment Scheduling Patterns Exist?

### Findings (MEDIUM confidence)

**Peer tools generally do not offer configurable experiment ordering:**

| Tool | Ordering | User-configurable? |
|------|----------|-------------------|
| **optimum-benchmark** | Sequential (Hydra `--multirun` Cartesian product) | No -- fixed order |
| **vLLM bench sweep** | Nested loops (serve_comb x bench_comb x num_runs) | No -- fixed order |
| **AIEnergyScore** | Sequential iteration over model list | No -- fixed order |
| **Ray Tune** | Scheduler-dependent (FIFO, ASHA, PBT, etc.) | Yes, but focused on hyperparameter search, not measurement rigour |

**Scientific methodology context:** Run order randomisation is a well-established experimental design principle. From DOE (Design of Experiments) methodology:

> "By randomizing the order in which experimental runs are done, you reduce the chance that differences in experimental materials or conditions strongly bias results."

Source: [Randomized and Random Run Order Experiments (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0378375804001119)

**Thermal autocorrelation is a real concern for energy measurement:** GPU thermal state carries over between experiments. Running A,A,A,B,B,B means the later cycles of A run on a warmer GPU than the first, and B starts from A's thermal plateau. Interleaved (A,B,A,B) reduces this systematic bias. Shuffled provides the strongest protection against order effects.

**LLenergyMeasure is unique among ML benchmarking tools in offering configurable cycle ordering for measurement rigour.** This is a genuine differentiator. The three modes (sequential, interleaved, shuffled) with deterministic seeded randomisation are scientifically sound.

**Assessment of implementation:** Correct and ahead of peers.

---

## Potential Gaps Identified

### Gap 1: File-Based IPC Fallback Not Implemented (LOW risk)

**What:** The design doc specifies a `_send_result()` function with a 1MB threshold for file-based IPC, mirroring optimum-benchmark. The implementation sends all results directly via Pipe.

**Risk:** If `ExperimentResult` ever exceeds ~64KB (the OS pipe buffer), `conn.send()` in the child will block, deadlocking with the parent at `p.join()`.

**Current impact:** None. `ExperimentResult` is a few KB at most.

**Recommendation:** Document this as a known limitation. Implement the fallback when adding per-token latency arrays or power trace data to results.

### Gap 2: `Queue` vs `SimpleQueue` for Progress Events (VERY LOW risk)

**What:** PyTorch docs recommend `SimpleQueue` over `Queue` to avoid internal thread contention. The implementation uses `mp_ctx.Queue()`.

**Risk:** Marginal. `Queue` works correctly for the low-volume progress event use case. The internal feeder thread is a concern mainly for high-throughput scenarios.

**Current impact:** None observed.

**Recommendation:** No action needed. If progress events become more frequent (e.g., per-token streaming), consider switching to `SimpleQueue`.

### Gap 3: No Process Group Kill for Sub-Subprocess Cleanup (LOW risk for current phase)

**What:** vLLM's bench sweep uses `os.killpg(pgid, SIGKILL)` with `start_new_session=True` to kill the entire process group. LLenergyMeasure's `p.kill()` only kills the direct child.

**Risk:** When running vLLM as a backend (Phase 19+), the spawned child process will host a vLLM engine that spawns its own tensor parallel workers. `p.kill()` on the child will not kill these grandchild workers. They become orphans holding GPU memory.

**Current impact:** None for Phase 11 (PyTorch backend runs in-process within the child, no sub-processes).

**Recommendation:** When implementing vLLM backend activation (Phase 19), consider:
1. Setting `start_new_session=True` on the Process and using `os.killpg()` for cleanup
2. Or relying on the Docker isolation path (container stop kills all processes)

### Gap 4: Manifest `mark_interrupted()` Does Not Downgrade Running Entries (VERY LOW risk)

**What:** When SIGINT fires, `mark_interrupted()` sets the top-level manifest status to "interrupted" but does not change the status of any experiment entry currently marked "running" back to "pending" or "interrupted".

**Risk:** A manifest could show `status: "interrupted"` at the study level but have an experiment entry stuck in `status: "running"` that never completed.

**Current impact:** Cosmetic only. The `--resume` feature (deferred to M4) would need to handle this.

**Recommendation:** When implementing `--resume`, ensure it treats "running" entries in an "interrupted" manifest as "pending" (re-runnable).

---

## Alignment with Existing Research

The existing research document (`.product/research/13-execution-isolation-patterns.md`, researched 2026-02-18) covered Docker execution patterns extensively but had minimal coverage of:

1. **SIGINT handling patterns** -- not covered at all
2. **`spawn` vs `forkserver` tradeoffs** -- not covered
3. **Pipe buffer deadlock risk** -- not covered
4. **Cycle ordering / experiment scheduling** -- not covered
5. **Timeout mechanisms in peers** -- partially covered (noted optimum-benchmark has no timeout)

This retroactive research fills those gaps. The existing research's coverage of container lifecycle, HTTP vs Docker exec, and dependency isolation remains accurate and relevant.

---

## Summary of Peer Alignment

| Aspect | LLenergyMeasure | Closest Peer | Alignment |
|--------|----------------|--------------|-----------|
| Process isolation | `mp.Process` per experiment | optimum-benchmark | Identical |
| Start method | `spawn` via `get_context()` | optimum-benchmark, PyTorch docs | Correct |
| IPC (results) | `Pipe` (no file fallback) | optimum-benchmark (Pipe + file fallback) | Mostly aligned; file fallback deferred |
| IPC (progress) | `Queue` + daemon consumer thread | optimum-benchmark (sync checkpoints via Pipe) | Different approach, both valid |
| Signal handling | SIG_IGN in child, parent handler + Event | optimum-benchmark (no explicit handling) | More robust than peer |
| Timeout | `p.join(timeout=...)` + SIGKILL | optimum-benchmark (no timeout) | More robust than peer |
| Experiment ordering | sequential / interleaved / shuffled | No peer equivalent | Unique differentiator |
| daemon mode | `daemon=False` | optimum-benchmark (`daemon=False`) | Identical |

---

## Sources

### Primary (HIGH confidence)
- [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) -- CUDA start method requirements
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html) -- Pipe buffer, Queue, daemon, spawn
- [vLLM Python Multiprocessing Design](https://docs.vllm.ai/en/stable/design/multiprocessing/) -- spawn vs fork vs forkserver tradeoffs
- `.product/research/13-execution-isolation-patterns.md` -- peer tool survey (optimum-benchmark, MLPerf, vLLM, AIEnergyScore)
- optimum-benchmark `launchers/process/launcher.py` (GitHub) -- Pipe IPC, file fallback, no signal handling, no timeout

### Secondary (MEDIUM confidence)
- [SIGINT handling with SIG_IGN](https://archive.zhimingwang.org/blog/2015-05-05-graceful-handling-of-sigint-when-using-pythons-multiprocessingprocess.html) -- standard pattern documentation
- [Python Multiprocessing Graceful Shutdown](https://www.peterspython.com/en/blog/python-multiprocessing-graceful-shutdown-in-the-proper-order) -- shutdown ordering
- [Ray Tune reuse_actors documentation](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html) -- GPU memory isolation via actor restart
- [Randomized Run Order Experiments](https://www.sciencedirect.com/science/article/abs/pii/S0378375804001119) -- scientific methodology for experiment ordering

### Tertiary (LOW confidence)
- Various StackOverflow / community discussions on Pipe deadlock patterns (corroborated by Python official docs)

## Metadata

**Confidence breakdown:**
- Start method (spawn vs fork): HIGH -- PyTorch docs are definitive
- SIGINT handling: MEDIUM -- community consensus, no single authoritative standard
- IPC patterns: HIGH -- code-verified against optimum-benchmark source
- Timeout strategies: HIGH -- well-understood OS-level behaviour
- Experiment ordering: MEDIUM -- DOE methodology is well-established; no ML-specific peer precedent

**Research date:** 2026-02-28
**Valid until:** Indefinite (multiprocessing fundamentals are stable; PyTorch CUDA requirements unlikely to change)
