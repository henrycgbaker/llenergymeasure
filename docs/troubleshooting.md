# Troubleshooting

Common issues and solutions for llenergymeasure.

---

## Common Issues

### No GPU detected

**Symptom:** `llem config` shows no GPU, or measurement fails with a CUDA error.

**Cause:** NVIDIA drivers are not installed, the device is not visible in the current
environment, or the system is CPU-only.

**Fix:**
1. Run `nvidia-smi` to verify the GPU is visible on the host.
2. Run `llem config` to see what the tool detects.
3. If `nvidia-smi` works but the engine does not, you may be running outside a container
   that has CUDA — for vLLM and TensorRT-LLM, use `llem run study.yaml` with Docker runners
   (see [docker-setup.md](docker-setup.md)).
4. If `nvidia-smi` fails, install or reinstall the NVIDIA drivers for your OS.

---

### Engine not installed

**Symptom:** `llem run -e vllm ...` fails immediately with an import error.

**Cause:** The required engine package is not installed.

**Fix:** Install the matching extra:

```bash
pip install "llenergymeasure[transformers]"    # Transformers engine
pip install "llenergymeasure[vllm]"       # vLLM engine
pip install "llenergymeasure[tensorrt]"   # TensorRT-LLM engine
```

Run `llem config` to see the current status of each engine.

For vLLM and TensorRT-LLM, the recommended approach is to use the Docker runner rather
than a host install — see [docker-setup.md](docker-setup.md).

---

### Docker pre-flight failed

**Symptom:** `llem run study.yaml` exits early with a pre-flight error about Docker.

**Cause:** One of the Docker pre-flight checks failed. Pre-flight checks verify:
1. Docker CLI is on PATH.
2. NVIDIA Container Toolkit binary is on PATH (`nvidia-container-runtime`, `nvidia-ctk`,
   or `nvidia-container-cli`).
3. Host `nvidia-smi` present (warn only — remote Docker daemon is supported).
4. GPU is visible inside a container (`docker run --gpus all nvidia-smi`).
5. CUDA/driver compatibility (checked from container probe output).

**Fix:** Read the error message — it identifies which check failed.

- `Docker not found`: install Docker Engine ([docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)).
- `NVIDIA Container Toolkit not found`: follow [docker-setup.md](docker-setup.md).
- `GPU not visible inside container`: check that `--gpus all` works with `docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`.
- `CUDA/driver mismatch`: update the host NVIDIA driver to be compatible with the container's CUDA version.

Skip pre-flight checks for testing or remote daemon setups:

```bash
llem run study.yaml --skip-preflight
```

---

### Out of memory (OOM)

**Symptom:** The run crashes with an OOM error (CUDA out of memory).

**Cause:** The model is too large for the available GPU VRAM at the current configuration.

**Fix (try in order):**
1. Use a smaller model.
2. Reduce `transformers.batch_size` (default is 1 — already minimal for Transformers).
3. Switch to lower dtype: `dtype: float16` or `dtype: bfloat16`.
4. Enable BitsAndBytes quantization: `pytorch: { load_in_4bit: true }`.
5. For vLLM: reduce `vllm.engine.gpu_memory_utilization` (e.g. 0.7 instead of 0.9).
6. For vLLM: reduce `vllm.engine.max_model_len` to cap KV cache allocation.

---

### Permission denied (Docker)

**Symptom:** `docker run` fails with permission denied.

**Cause:** The current user is not in the `docker` group.

**Fix:**

```bash
sudo usermod -aG docker $USER
newgrp docker          # activate without logout
```

Or prefix Docker commands with `sudo`. Permanent fix requires re-login.

---

### Study failed partially

**Symptom:** A study run produces some results but not all. Some experiments are missing
from the output directory.

**Cause:** Individual experiments may fail while the study continues. llenergymeasure
uses skip-and-continue: a failed experiment is recorded as an error in the study manifest,
and execution continues with the remaining experiments.

**Fix:**
1. Check the study manifest in `results/` for per-experiment status and error messages.
2. The manifest records the failure reason for each skipped config.
3. Fix the failing config (see the error message) and re-run the specific experiment
   separately before merging results.

---

### Container crashes with "context canceled" or ValidationError

**Symptom:** Experiments using Docker runners fail immediately. The container log shows
`context canceled` or a Pydantic `ValidationError` mentioning unknown fields (e.g.
`dtype: Extra inputs are not permitted` or `dataset: Input should be a valid string`).

**Cause:** The Docker image was built from an older version of the source code. The host
sends config JSON using the current schema, but the container rejects it because its
bundled code expects the old schema.

**Fix:** Rebuild the Docker images from the current source:

```bash
docker build -f docker/Dockerfile.transformers -t ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0 .
docker build -f docker/Dockerfile.vllm -t ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0 .
```

Replace `v0.9.0` with your installed version (`llem --version`). See
[Installation - Building Docker Images](installation.md#building-docker-images-from-source)
for full instructions.

---

### Results look wrong / energy is 0

**Symptom:** `inference_energy_joules` is 0.0, or energy values seem too low.

**Cause:** Energy measurement requires NVML (pynvml) and a supported NVIDIA GPU. If NVML
is unavailable, energy measurement falls back gracefully to zero rather than crashing.

**Fix:**
1. Check `llem config` — it shows the active energy sampler under `Energy:`.
2. Verify pynvml can access the GPU: run `python -c "import pynvml; pynvml.nvmlInit(); print('OK')"`.
3. Check your config. Setting `energy_sampler: null` explicitly disables
   energy measurement (throughput-only mode).
4. If `baseline.enabled: true` (default), ensure the baseline measurement is completing.
   A failed baseline causes `adjusted_j` to be null.
5. For very short inference runs (< 200ms), NVML polling at 100ms intervals may not collect
   enough samples for accurate integration. Use larger `n` values.

Zeus and CodeCarbon are optional extras. If they are not installed, the tool falls back
to NVML. See [energy-measurement.md](energy-measurement.md) for backend details.

---

### Warmup takes too long

**Symptom:** Experiments take much longer than expected. Progress stalls for 1-2 minutes
before measurement begins.

**Cause:** Warmup is enabled by default. It runs `n_warmup=5` warmup inferences, then
waits `thermal_floor_seconds=60.0` seconds for GPU temperature to stabilise before
measuring.

**Fix for quick testing:**

```yaml
warmup:
  enabled: false
```

Or on the CLI:

```bash
llem run experiment.yaml  # add to YAML for testing
```

For publication-quality measurements, leave warmup enabled. See
[methodology.md](methodology.md) for why warmup matters.

---

## Invalid Parameter Combinations

<!-- BEGIN INVALID COMBOS — Auto-generated by scripts/generate_invalid_combos_doc.py -->

### Config Validation Errors

These combinations are rejected at config load time with a clear error message.

| Engine | Invalid Combination | Reason | Resolution |
|---------|---------------------|--------|------------|
| pytorch | `load_in_4bit=True + load_in_8bit=True` | Cannot use both 4-bit and 8-bit quantization simultaneously | Choose one: `transformers.load_in_4bit: true` OR `transformers.load_in_8bit: true` |
| pytorch | `torch_compile_mode without torch_compile=True` | torch_compile_mode/torch_compile_backend only take effect when torch_compile=True | Set `transformers.torch_compile: true` when using `torch_compile_mode` or `torch_compile_backend` |
| pytorch | `bnb_4bit_* without load_in_4bit=True` | BitsAndBytes 4-bit options require 4-bit quantization to be enabled | Set `transformers.load_in_4bit: true` when using `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, or `bnb_4bit_use_double_quant` |
| pytorch | `cache_implementation with use_cache=False` | Cannot specify a cache strategy when caching is explicitly disabled | Remove `use_cache: false` or remove `cache_implementation` |
| all | `engine section mismatch` | Engine section must match the `engine:` field | Ensure `pytorch:` / `vllm:` / `tensorrt:` section matches `engine:` field |
| all | `passthrough_kwargs key collision` | `passthrough_kwargs` keys must not collide with ExperimentConfig fields | Use named fields directly instead of `passthrough_kwargs` |
| tensorrt | `dtype: float32` | TensorRT-LLM is optimised for lower precision inference | Use `dtype: float16` or `dtype: bfloat16` |
| vllm | `transformers.load_in_4bit or pytorch.load_in_8bit` | vLLM does not support bitsandbytes quantization | Use `vllm.engine.quantization` (awq, gptq, fp8) for quantized inference |

### Runtime Limitations

These combinations pass config validation but may fail at runtime due to hardware,
model, or package requirements.

| Engine | Parameter | Limitation | Resolution |
|---------|-----------|------------|------------|
| pytorch | `transformers.attn_implementation: flash_attention_2` | flash-attn requires Ampere+ GPU; may fail on older architectures | Use `attn_implementation: sdpa` on pre-Ampere GPUs |
| pytorch | `transformers.attn_implementation: flash_attention_3` | FA3 requires the `flash_attn_3` package (built from flash-attn `hopper/` directory) and Ampere+ GPU (SM80+). Included in the Docker image by default | Install locally from source if not using Docker. See [Installation - FA3](installation.md#flashattention-3) |
| vllm | `vllm.engine.kv_cache_dtype: fp8` | FP8 KV cache requires Hopper (H100) or newer GPU | Use `kv_cache_dtype: auto` for automatic selection |
| vllm | `vllm.engine.attention.backend: FLASHINFER` | FlashInfer requires JIT compilation on first use | Use `attention.backend: auto` or `FLASH_ATTN` |
| vllm | `vllm.engine.attention.backend: TORCH_SDPA` | TORCH_SDPA not registered in vLLM attention backends | Use `attention.backend: auto` or `FLASH_ATTN` |
| vllm | `vllm.engine.quantization: awq` or `gptq` | Requires a pre-quantized model checkpoint | Use a quantized model (e.g. `TheBloke/*-AWQ`) or omit |
| tensorrt | `tensorrt.quant_config.quant_algo: FP8` | FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises a `ConfigurationError` — no silent emulation or fallback | Use `INT8`, `W4A16_AWQ`, `W4A16_GPTQ`, or `W8A16` on A100 |
| tensorrt | `tensorrt.quantization: int8_sq` | INT8 SmoothQuant requires a calibration dataset | Provide calibration config or use a supported quantization method |

### Engine Capability Matrix

| Feature | Transformers | vLLM | TensorRT |
|---------|---------|------|----------|
| Tensor Parallel | Yes (HF native) | Yes | Yes |
| Data Parallel | Yes | No | No |
| BitsAndBytes (4-bit) | Yes | No | No |
| BitsAndBytes (8-bit) | Yes | No | No |
| Native Quantization | No | AWQ / GPTQ / FP8 | INT8 / W4A16 (AWQ/GPTQ) / FP8 |
| float32 precision | Yes | No | No |
| float16 precision | Yes | Yes | Yes |
| bfloat16 precision | Yes | Yes | Yes |
| Prefix Caching | No | Yes | No |
| LoRA Adapters | Yes | No | No |
| torch.compile | Yes | No | No |
| Beam Search | Yes | Yes | No |
| Speculative Decoding | Yes | Yes | No |
| Static KV Cache | Yes | No | No |

Notes:
- Transformers Tensor Parallel uses HF native TP via `tp_plan`/`tp_size` (requires Transformers >= 4.50 and `torchrun` launch).
- vLLM does not support FP32 precision. Use FP16 or BF16.
- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes.
- TensorRT-LLM is optimised for FP16/BF16/INT8 precision, not FP32.

<!-- END INVALID COMBOS — Auto-generated by scripts/generate_invalid_combos_doc.py -->

---

## Docker rebuild is slow / recompiling flash-attn

**Symptom:** `make docker-build-transformers` takes 15-20 minutes and the post-build
summary line reports `⚠ no GHCR cache imported (cold build)` (or BuildKit
output shows `flash-attn` source downloads and nvcc compilation for every
build).

**Cause:** BuildKit's `cache_from` registry pull was skipped. In rough order
of likelihood:

(a) **`BUILDX_BUILDER` is unset or pointing at the default `docker` driver.**
    The default driver cannot import registry caches at all — `cache_from`
    entries are silently ignored. Confirm with `docker buildx ls`: the row
    marked with `*` (current builder) must show driver `docker-container`,
    not `docker`. Fix by adding `BUILDX_BUILDER=llem-builder` to your `.env`
    (it ships in `.env.example`) and re-running `make docker-builder-setup`
    if the builder doesn't exist yet.
(b) You are on a fresh buildx builder with no local cache (this is normal
    on the very first build — first-pull cost is paid once).
(c) You are offline or GHCR is unreachable.
(d) Your `LLEM_PKG_VERSION` does not match any published tag (cache_from
    resolves to `:v${LLEM_PKG_VERSION}` and falls through to `:latest` —
    if neither has usable layers, BuildKit silently cold-builds).

The full BuildKit log for the most recent attempt is at
`/tmp/llem-build-{engine}.log` — grep it for `importing cache manifest` to
see whether the registry was even reached.

**Fix:**

0. Confirm the builder driver: `docker buildx ls`. The active builder
   (marked `*`) must be `docker-container`. If it's `docker`, run
   `make docker-builder-setup` and ensure `BUILDX_BUILDER=llem-builder` is
   in your `.env` (or exported in the shell).
1. Inspect the builder cache: `docker buildx du --builder llem-builder`. If
   it's near-empty, BuildKit has nothing to reuse locally and will pull from
   the registry.
2. Verify network: `curl -I https://ghcr.io/v2/henrycgbaker/llenergymeasure/transformers/manifests/latest`
   should return 200 or 401 (both fine; 000/timeout means no connectivity).
3. If you recently bumped version but CI hasn't published yet, fall back to
   `:latest` by unsetting `LLEM_PKG_VERSION` for the build:
   `LLEM_PKG_VERSION= docker compose build transformers`.
4. If the cache is corrupt, recreate the builder:
   `make docker-builder-rm && make docker-builder-setup`. Note this discards
   all local layer cache; the first subsequent build will repopulate from
   GHCR.
5. Offline is expected-slow. BuildKit degrades gracefully to a cold build —
   no errors, just minutes.

**CI can't build the Transformers image (FA3 compile OOM / heartbeat loss):**
The FA3 Hopper compile requires ~8-16 GB RAM and multiple hours on a 4-core
runner. Seed the GHCR cache once from a developer machine with more resources:

```bash
docker login ghcr.io           # needs write:packages scope
make docker-seed-transformers  # builds + pushes cache to ghcr.io (~minutes if locally cached)
```

After seeding, CI warm-rebuilds from the GHCR cache in <5 min.

---

## Schema skew between host and Docker image

**Symptom:** `llem run study.yaml` aborts before any experiment with a message
like:

```
Docker image 'llenergymeasure:transformers' was built from llenergymeasure 0.9.0
(schema 9988776655ff) but the host is running 0.9.0 (schema a1b2c3d4e5f6).
The container will reject ExperimentConfig fields added on the host after
the image was built.
```

Or, without the handshake catching it first, a container stack trace full of
`extra_forbidden` Pydantic errors (often with URLs mixing
`errors.pydantic.dev/2.10/…` and `errors.pydantic.dev/2.12/…`, a second tell
for version skew).

**Cause:** the host's `ExperimentConfig` (or a nested model like
`BaselineConfig`, `WarmupConfig`, etc.) gained fields after the Docker image
was built. `llem` stamps every image at build time with a
`llem.expconf.schema.fingerprint` label computed from
`ExperimentConfig.model_json_schema()`. `StudyRunner._prepare_images` compares
that label against the host fingerprint before any experiment starts.

**Fix:** rebuild the affected engine image. One of:

```bash
make docker-build-pytorch        # or -vllm / -tensorrt, local build
make docker-pull                 # pull the newest published tagged release
```

Verify with:

```bash
llem doctor                      # exits 1 on mismatch, 0 when every engine is OK
```

**Bypass (last resort):** set `LLEM_SKIP_IMAGE_CHECK=1` to skip the handshake.
Only safe when you're confident the new field is optional and the container
silently ignores it. The container will still hard-fail on any required field
it doesn't know about.

---

## Getting Help

Run `llem config --verbose` to capture full environment details (Python version, installed
engines, GPU info, energy sampler status, config file path). Include this output when
filing a bug report.

File issues at: [github.com/henrycgbaker/llenergymeasure/issues](https://github.com/henrycgbaker/llenergymeasure/issues)
