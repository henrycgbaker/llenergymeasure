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
3. If `nvidia-smi` works but the backend does not, you may be running outside a container
   that has CUDA — for vLLM and TensorRT-LLM, use `llem run study.yaml` with Docker runners
   (see [docker-setup.md](docker-setup.md)).
4. If `nvidia-smi` fails, install or reinstall the NVIDIA drivers for your OS.

---

### Backend not installed

**Symptom:** `llem run --backend vllm ...` fails immediately with an import error.

**Cause:** The required backend package is not installed.

**Fix:** Install the matching extra:

```bash
pip install "llenergymeasure[pytorch]"    # PyTorch backend
pip install "llenergymeasure[vllm]"       # vLLM backend
pip install "llenergymeasure[tensorrt]"   # TensorRT-LLM backend
```

Run `llem config` to see the current status of each backend.

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
2. Reduce `pytorch.batch_size` (default is 1 — already minimal for PyTorch).
3. Switch to lower precision: `precision: fp16` or `precision: bf16`.
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

### Results look wrong / energy is 0

**Symptom:** `inference_energy_joules` is 0.0, or energy values seem too low.

**Cause:** Energy measurement requires NVML (pynvml) and a supported NVIDIA GPU. If NVML
is unavailable, energy measurement falls back gracefully to zero rather than crashing.

**Fix:**
1. Check `llem config` — it shows the active energy backend under `Energy:`.
2. Verify pynvml can access the GPU: run `python -c "import pynvml; pynvml.nvmlInit(); print('OK')"`.
3. Check your `energy:` config. Setting `energy: { backend: null }` explicitly disables
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

| Backend | Invalid Combination | Reason | Resolution |
|---------|---------------------|--------|------------|
| pytorch | `load_in_4bit=True + load_in_8bit=True` | Cannot use both 4-bit and 8-bit quantization simultaneously | Choose one: `pytorch.load_in_4bit: true` OR `pytorch.load_in_8bit: true` |
| pytorch | `torch_compile_mode without torch_compile=True` | torch_compile_mode/torch_compile_backend only take effect when torch_compile=True | Set `pytorch.torch_compile: true` when using `torch_compile_mode` or `torch_compile_backend` |
| pytorch | `bnb_4bit_* without load_in_4bit=True` | BitsAndBytes 4-bit options require 4-bit quantization to be enabled | Set `pytorch.load_in_4bit: true` when using `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, or `bnb_4bit_use_double_quant` |
| pytorch | `cache_implementation with use_cache=False` | Cannot specify a cache strategy when caching is explicitly disabled | Remove `use_cache: false` or remove `cache_implementation` |
| all | `backend section mismatch` | Backend section must match the `backend:` field | Ensure `pytorch:` / `vllm:` / `tensorrt:` section matches `backend:` field |
| all | `passthrough_kwargs key collision` | `passthrough_kwargs` keys must not collide with ExperimentConfig fields | Use named fields directly instead of `passthrough_kwargs` |
| tensorrt | `precision: fp32` | TensorRT-LLM is optimised for lower precision inference | Use `precision: fp16` or `precision: bf16` |
| vllm | `pytorch.load_in_4bit or pytorch.load_in_8bit` | vLLM does not support bitsandbytes quantization | Use `vllm.engine.quantization` (awq, gptq, fp8) for quantized inference |

### Runtime Limitations

These combinations pass config validation but may fail at runtime due to hardware,
model, or package requirements.

| Backend | Parameter | Limitation | Resolution |
|---------|-----------|------------|------------|
| pytorch | `pytorch.attn_implementation: flash_attention_2` | flash-attn package not installed in Docker image | Install flash-attn or use `attn_implementation: sdpa` |
| vllm | `vllm.engine.kv_cache_dtype: fp8` | FP8 KV cache requires Hopper (H100) or newer GPU | Use `kv_cache_dtype: auto` for automatic selection |
| vllm | `vllm.engine.attention.backend: FLASHINFER` | FlashInfer requires JIT compilation on first use | Use `attention.backend: auto` or `FLASH_ATTN` |
| vllm | `vllm.engine.attention.backend: TORCH_SDPA` | TORCH_SDPA not registered in vLLM attention backends | Use `attention.backend: auto` or `FLASH_ATTN` |
| vllm | `vllm.engine.quantization: awq` or `gptq` | Requires a pre-quantized model checkpoint | Use a quantized model (e.g. `TheBloke/*-AWQ`) or omit |
| tensorrt | `tensorrt.quantization: int8_sq` | INT8 SmoothQuant requires a calibration dataset | Provide calibration config or use `fp8` |

### Backend Capability Matrix

| Feature | PyTorch | vLLM | TensorRT |
|---------|---------|------|----------|
| Tensor Parallel | No | Yes | Yes |
| Data Parallel | Yes | No | No |
| BitsAndBytes (4-bit) | Yes | No | No |
| BitsAndBytes (8-bit) | Yes | No | No |
| Native Quantization | No | AWQ / GPTQ / FP8 | INT8 / INT4 / FP8 |
| float32 precision | Yes | Yes | No |
| float16 precision | Yes | Yes | Yes |
| bfloat16 precision | Yes | Yes | Yes |
| Prefix Caching | No | Yes | No |
| LoRA Adapters | Yes | No | No |
| torch.compile | Yes | No | No |
| Beam Search | Yes | Yes | No |
| Speculative Decoding | Yes | Yes | No |
| Static KV Cache | Yes | No | No |

Notes:
- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes.
- TensorRT-LLM is optimised for FP16/BF16/INT8 precision, not FP32.

<!-- END INVALID COMBOS — Auto-generated by scripts/generate_invalid_combos_doc.py -->

---

## Getting Help

Run `llem config --verbose` to capture full environment details (Python version, installed
backends, GPU info, energy backend status, config file path). Include this output when
filing a bug report.

File issues at: [github.com/henrycgbaker/llenergymeasure/issues](https://github.com/henrycgbaker/llenergymeasure/issues)
