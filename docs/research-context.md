# Research Context: Invariant Mining for ML Engine Configs

This document positions the LLenergyMeasure invariant miner pipeline relative to the academic literature. It is written for paper readers, peer reviewers, and researchers interested in the contribution beyond the practical tooling.

**For end users or extenders**, the [architecture-overview.md](architecture-overview.md), [miner-pipeline.md](miner-pipeline.md), and [extending-miners.md](extending-miners.md) are more directly useful.

---

## What we are doing (in one sentence)

We mine configuration constraint invariants from ML inference engine config classes by combining static AST analysis of validator method bodies with dynamic combinatorial constructor probing, and we emit a versioned corpus of structured constraints that is re-validated against the live library on every library version bump.

---

## The nearest prior art

### 1. Daikon (Ernst et al., 2001) - canonical dynamic invariant detection

Daikon is the direct ancestor of our approach at the predicate-inference stage. It runs an instrumented program, observes variable values at function entry/exit and loop heads, and matches observations against a fixed template library (`x > 0`, `x % y == 0`, `x in {a,b,c}`, etc.).

**Where we converge with Daikon:**
- Same template-based predicate inference loop (we use seven templates; Daikon uses dozens).
- Same recall-first strategy: emit all matching candidates, prune false positives via an external checker.
- Same splitter-predicate structure: `condition ⇒ invariant` (HF's dormancy mode gates like "if do_sample=False then temperature is dormant" are structurally identical to Daikon splitter predicates).

**Where we diverge:**
- Our "execution traces" are constructor-call outcomes (raises/no-raise), not variable value traces at function entry/exit. Daikon instruments a running program; we probe a class constructor.
- Our checker is the live vendor library (authoritative ground truth). Daikon's checker is a theorem prover (Simplify/ESC/Java). Ours is more accurate because the library is the spec; no SMT approximation.
- We deliberately do not implement Daikon's statistical confidence scoring. Daikon computes `P(invariant arises by chance)` and applies a threshold. We remove the `confidence` field entirely: the vendor-CI gate is binary (rule passes or is quarantined). A confidence score that informs no downstream decision is dead weight in the corpus.

**Citation:** Ernst, M.D., Cockrell, J., Griswold, W.G., and Notkin, D. (2001). *Dynamically discovering likely program invariants to support program evolution.* IEEE TSE 27(2):1-25.

### 2. Houdini (Flanagan & Leino, FME 2001) - the canonical guess-and-check pattern

Houdini generates a large set of candidate annotations from heuristic templates (e.g. `f != null` for every reference field, `i ≥ 0` for every int field), then asks a checker (ESC/Java + SMT) which survive verification. It iterates to a fixpoint: remove refuted candidates, recheck, repeat.

This is exactly our pipeline:
- **Houdini guess** ≡ our miner emit (recall-first, all plausible candidates).
- **ESC/Java check** ≡ our vendor-CI prune (replay against live library).
- **Fixpoint** ≡ `_fixpoint_test.py` (asserts dormant rules converge to a stable normalisation state).

The Houdini correctness theorem proves the inferred annotation set is the *unique maximal valid subset* of the candidate set, modulo soundness of the checker. Our corpus is the maximal valid subset of miner candidates modulo the vendor-CI gate's soundness.

Our `_fixpoint_test.py` pins the gate-soundness contract in place: it synthesises malformed rules and asserts the gate fails loudly on each, so that a future refactor cannot silently weaken the gate without breaking CI.

**Citation:** Flanagan, C. and Leino, K.R.M. (2001). *Houdini, an annotation assistant for ESC/Java.* FME 2001, LNCS 2021, pp. 500-517.

### 3. NeuRI (FSE 2023) - the closest ML-library framing

NeuRI mines constraints for neural network operator APIs by generating valid and invalid traces, then using inductive synthesis to infer input constraints. It is the most direct analogue to our approach in the ML-library literature.

**Mapping to our pipeline:**
- NeuRI's "valid + invalid trace pairs" ≡ our `(kwargs_positive, kwargs_negative)` pairs in each corpus rule.
- NeuRI's inductive synthesis step ≡ our predicate-inference templates (we use a simpler template-matching approach; NeuRI uses full inductive synthesis which generalises better to quantified constraints but costs more).
- NeuRI operates on operator kernel APIs (shapes, dtypes, device constraints). We operate on config-class constructor APIs (field ranges, cross-field relationships, mode gating).

**Where we are novel relative to NeuRI:** vLLM and TRT-LLM are uncharted at the config layer. The PyTorch-fuzz lineage (NeuRI, NNSmith, DocTer, FreeFuzz, DeepREL, TitanFuzz, DeepConstr, ConFL) targets operator kernels. We target the config classes that sit above those kernels and govern how a deployment is parameterised - a distinct and previously unmapped constraint space.

**Citation:** Luo, Y., Hu, S., Shi, X., et al. (2023). *NeuRI: Diversifying DNN Generation via Inductive Rule Inference.* ESEC/FSE 2023.

### 4. Configuration constraint mining (Hadoop, Linux kernel corpus studies)

The "configuration mining" literature (e.g. Rabkin & Katz, ASE 2011; Jin et al., FSE 2014) mines real-world deployed config files to discover constraints that users satisfy in practice. Our approach is distinct:

- We mine **library-source validation logic**, not deployed config files.
- We produce a **constraint corpus** (formal predicates with test cases), not a frequency analysis.
- Our constraints are derived from the library's enforcement mechanism, not from statistical patterns in user configurations.

The closest label for our approach: **configuration constraint mining at the API layer**, or **API configuration constraint inference**.

### 5. DocTer (ISSTA 2022) - doc-guided constraint mining

DocTer extracts input constraints for deep learning APIs by parsing API documentation and natural language descriptions. It uses a dependency parser + constraint pattern matcher on docstrings.

**Comparison:** DocTer mines documentation; we mine source code. Documentation describes intent; source code describes enforcement. For config classes where the enforcement is in validator method bodies (all three of our target engines), source mining is more accurate. DocTer's approach is complementary for cases where constraints are documented but not enforced in parseable code.

**Citation:** Xie, D., Li, Y., Pham, N., et al. (2022). *DocTer: Documentation-Guided Fuzzing for Testing Deep Learning API Functions.* ISSTA 2022.

---

## What we are NOT doing

### Not differential testing

Differential testing compares two implementations to observe behavioural divergence. We probe a single implementation. Our `kwargs_positive` / `kwargs_negative` pairs are evidence for one rule's predicate, not a comparison between two systems.

### Not metamorphic testing

Metamorphic testing requires a stated input/output relation (e.g. `f(x+1) ≥ f(x)` for a monotone function). We do not define metamorphic relations. Our predicate inference is inductive from observed behaviour, not deductive from a stated relation.

### Not static analysis of tensor shapes

PyTea (PLDI 2022) translates Python tensor-shape preconditions into Z3 constraints for static verification. Our static miner also does AST-to-constraint translation, but for scalar config values (ranges, allowlists, cross-field arithmetic) not tensor shapes. The Z3-backed path exhaustion approach PyTea takes is not applicable to our setting: our constraints involve Pydantic v2 internals and Rust-backed validators that Z3 cannot model.

### Not LLM-assisted mining

DeepConstr (ISSTA 2024) uses an LLM as a constraint hypothesiser: it generates candidate constraints in natural language, which are then validated against the library. We explicitly do not use LLMs anywhere in the pipeline. Reasons:

1. The corpus build runs in CI, which may be air-gapped or have no API access.
2. LLM providers deprecate model versions; a model fingerprint is not permanent. A Renovate-driven library bump that fires in CI three months from now must produce a reproducible corpus diff.
3. The pipeline must be a pure function of (library SHA, library version, miner code, probe seed). An LLM call introduces a dependency on a third-party API whose output is not reproducible under this definition.

---

## The novel contribution at the type-system level

The lift modules (`_pydantic_lift.py`, `_msgspec_lift.py`, `_dataclass_lift.py`) represent a contribution not present in the prior-art literature: extracting configuration constraints directly from the Python type-system metadata of ML library config classes.

```
  Type-system metadata → constraint corpus

  Pydantic v2 FieldInfo.metadata:
    Gt(0), Le(1), MultipleOf(64)
    → corpus rules at operator granularity
    → aligns operator vocabulary with annotated-types standard (Gt, Le, etc.)

  msgspec Meta(ge=, le=, ...):
    → same corpus operator vocabulary
    → unified with Pydantic lift output

  stdlib Literal[...] annotations:
    → value-allowlist corpus rules
    → no probing required; purely type-structure derived
```

The prior-art tools (NeuRI, DocTer, FreeFuzz, etc.) all operate at the runtime-trace level. None of them exploit the type-system metadata that modern Python ML libraries expose through Pydantic v2, msgspec, and `annotated-types`. This is a Tier-1 adoption opportunity that our pipeline is the first (to our knowledge) to realise in the config-constraint-mining context.

The lift output is deterministic: no probing, no randomness, pure function of the class definition. Rules derived from type-system metadata are the most reliable in the corpus because the type system itself is the spec.

---

## The fail-loud import contract as a design contribution

The Haiku-era TRT-LLM extractor (PRs #415-#417, reverted in #423) demonstrated a failure mode not discussed in the prior-art literature: a miner that silently degraded on import errors. When `LlmConfig` (a class that does not exist in TRT-LLM 0.21.0) failed to import, the extractor caught the `ImportError` and returned `[]`. The empty return was indistinguishable from "no rules found for this engine".

The fail-loud import contract - `TESTED_AGAINST_VERSIONS` + `check_installed_version` + `MinerLandmarkMissingError` - makes silent coverage loss impossible. This is a design contribution specifically for the setting where the miner is a CI artefact that runs against a pinned library version: the version pin and the landmark check together guarantee that a miner either mines at the version it was tested against or fails loudly.

---

## The fixpoint contract

`_fixpoint_test.py` implements what we call a **gate-soundness fixpoint**: a structural test that synthesises malformed rules and asserts the vendor-CI gate records a divergence for each. The three checks it pins:

1. `positive_raises` - `kwargs_positive` must trigger the rule.
2. `message_template_match` - the raised message must contain the template fragment.
3. `negative_does_not_raise` - `kwargs_negative` must not trigger.

This is inspired by the Houdini fixpoint concept but applied to the soundness of the checker itself rather than the convergence of the annotation set. Without the gate-soundness fixpoint, a future maintainer could remove one of the three checks "just to get CI green" without understanding the consequences.

The dormant-rule fixpoint (also in `_fixpoint_test.py`) asserts that applying the corpus's dormant rules to a config state converges to a stable fixed point: idempotent, order-independent, cycle-free. This is a necessary condition for the library-resolution mechanism in the runtime pipeline to be well-defined.

---

## Positioning statement

Our approach is best described as: **API configuration constraint mining via static AST analysis and dynamic constructor probing, with type-system lifting and a live-library vendor-CI gate, in the Houdini guess-and-check tradition.**

The nearest citation cluster: Daikon (inference), Houdini (guess-and-check architecture), NeuRI (ML-library framing). The key novelties:

1. Type-system lifting at the `pydantic` / `msgspec` / `dataclass` level (no analogues in prior art).
2. The fail-loud import contract for pinned-version miners in CI pipelines.
3. The gate-soundness fixpoint contract as a structural regression test.
4. Application to **config-class APIs** (vLLM `SamplingParams`, TRT-LLM `TrtLlmArgs`) rather than operator kernel APIs - a previously uncharted constraint space.

---

## Citation shortlist for any write-up

| Reference | Why it matters |
|-----------|----------------|
| Ernst et al. (2001), Daikon TSE | Predicate-template inference ancestor |
| Flanagan & Leino (2001), Houdini FME | Recall-first emit + checker-prune architecture |
| Luo et al. (2023), NeuRI FSE | ML-library framing; valid+invalid trace pairing |
| Xie et al. (2022), DocTer ISSTA | Documentation-guided constraint mining (contrast point) |
| Dwarakanath et al. (2018), PreInfer DSN | Quantified precondition inference (adjacent; we don't need this yet) |
| DeepConstr (ISSTA 2024) | LLM-assisted constraint hypothesiser (contrast point: we don't use LLMs) |
| PyTea (PLDI 2022) | Static tensor-shape analysis with Z3 (contrast point: different domain) |

---

## See also

- [architecture-overview.md](architecture-overview.md) - system overview
- [miner-pipeline.md](miner-pipeline.md) - implementation details
- [validation-rule-corpus.md](validation-rule-corpus.md) - corpus format
- `.product/research/miners-redo/research-prior-art-miners-2026-04-26.md` - full prior-art survey (internal)
- `.product/designs/invariant-miner-design-2026-04-26.md` - locked design (internal)
