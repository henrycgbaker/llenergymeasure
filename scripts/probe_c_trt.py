"""PoC-C TRT-LLM analogue.

Mirrors the transformers + vLLM PoC-C runs (2026-04-23) against TensorRT-LLM's
native config classes. Confirms three things:

1. `extract_effective_params` feasibility: what kind of object is each config
   type and does it expose a uniform dump API?
2. Private-field leakage: enumerate every `_`-prefixed field surfaced via
   whichever dump mechanism works.
3. Byte-stability: two constructions with identical kwargs dump to identical
   bytes.

Must be run inside the `llenergymeasure:tensorrt` container (NGC 0.21.0) with
`LD_PRELOAD=/usr/local/cuda/compat/lib.real/libcuda.so.1` — host driver 535
ships CUDA 12.2 libcuda which is missing `cuKernelGetName`; forward-compat
libcuda from the container satisfies it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from typing import Any


def _is_pydantic(obj: Any) -> bool:
    return hasattr(obj, "model_dump") and hasattr(type(obj), "model_fields")


def _dump(obj: Any) -> tuple[str, dict[str, Any]]:
    """Return (mechanism, dump_dict). Tries model_dump, then asdict, then
    __slots__ iteration, then __dict__. Normalises non-JSON-serialisable
    values to ``repr``."""
    if _is_pydantic(obj):
        try:
            d = obj.model_dump(mode="python")
            return "pydantic.model_dump", d
        except Exception as exc:  # pragma: no cover — diagnostic
            return f"pydantic.model_dump FAILED: {exc!r}", {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            d = dataclasses.asdict(obj)
            return "dataclasses.asdict", d
        except Exception as exc:
            # Some dataclasses reference non-copyable runtime objects; fall
            # back to shallow iteration over fields.
            try:
                d = {f.name: getattr(obj, f.name, "<unreadable>") for f in dataclasses.fields(obj)}
                return f"dataclasses.fields (asdict raised {type(exc).__name__})", d
            except Exception as exc2:
                return f"dataclasses.* FAILED: {exc2!r}", {}
    if hasattr(obj, "__slots__"):
        try:
            slots = obj.__slots__
            if isinstance(slots, str):
                slots = (slots,)
            d = {s: getattr(obj, s, "<unset>") for s in slots}
            return "__slots__ iteration", d
        except Exception as exc:
            return f"__slots__ FAILED: {exc!r}", {}
    if hasattr(obj, "__dict__"):
        return "__dict__", dict(vars(obj))
    return "NONE", {}


def _private_fields(dump: dict[str, Any]) -> dict[str, str]:
    return {k: type(v).__name__ for k, v in dump.items() if k.startswith("_")}


def _json_safe(obj: Any) -> Any:
    """Produce something JSON-serialisable. Non-serialisable values become
    ``<repr>`` strings so we can still compare byte-stability across two
    construction runs."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return f"<repr:{type(obj).__name__}:{obj!r}>"


def _bytes_canonical(dump: dict[str, Any]) -> bytes:
    """Canonical JSON byte-string for dump comparison."""
    safe = _json_safe(dump)
    return json.dumps(safe, sort_keys=True, default=str).encode("utf-8")


# -- construction helpers ----------------------------------------------------


def make_llm_args():
    import tensorrt_llm as t

    # Minimal: LlmArgs requires `model`. Use a known hub id to avoid needing
    # local weights; we never call .build() so no weights are fetched.
    return t.LlmArgs(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def make_build_config():
    import tensorrt_llm as t

    return t.BuildConfig(max_input_len=512, max_seq_len=1024, max_batch_size=4)


def make_sampling_params():
    import tensorrt_llm as t

    return t.SamplingParams(max_tokens=32, temperature=0.7, top_p=0.9)


FACTORIES = {
    "LlmArgs": make_llm_args,
    "BuildConfig": make_build_config,
    "SamplingParams": make_sampling_params,
}


def probe_class(name: str, factory) -> dict[str, Any]:
    report: dict[str, Any] = {"name": name}
    try:
        obj1 = factory()
    except Exception as exc:
        report["construction_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        return report

    cls = type(obj1)
    report["module"] = cls.__module__
    report["qualname"] = cls.__qualname__
    report["mro"] = [c.__name__ for c in cls.__mro__]
    report["is_pydantic"] = _is_pydantic(obj1)
    report["is_dataclass"] = dataclasses.is_dataclass(cls)
    report["has_slots"] = hasattr(cls, "__slots__")

    mechanism, dump = _dump(obj1)
    report["dump_mechanism"] = mechanism
    report["field_count"] = len(dump)
    report["private_fields"] = _private_fields(dump)

    # byte-stability
    try:
        obj2 = factory()
        _, dump2 = _dump(obj2)
        b1 = _bytes_canonical(dump)
        b2 = _bytes_canonical(dump2)
        report["byte_stable"] = b1 == b2
        if b1 != b2:
            differing = []
            keys = set(dump) | set(dump2)
            for k in sorted(keys):
                v1 = dump.get(k, "<missing>")
                v2 = dump2.get(k, "<missing>")
                if _bytes_canonical({"_": v1}) != _bytes_canonical({"_": v2}):
                    differing.append(
                        {
                            "field": k,
                            "run1": _json_safe(v1),
                            "run2": _json_safe(v2),
                        }
                    )
            report["differing_fields"] = differing[:20]
    except Exception as exc:
        report["byte_stability_error"] = f"{type(exc).__name__}: {exc}"

    # Sample of public field values (trimmed) for human eyeballing
    sample = {}
    for k, v in list(dump.items())[:15]:
        sample[k] = (
            _json_safe(v) if len(repr(v)) < 200 else f"<{type(v).__name__} len={len(repr(v))}>"
        )
    report["sample_fields"] = sample

    return report


def main() -> int:
    import tensorrt_llm

    top = {
        "tensorrt_llm_version": tensorrt_llm.__version__,
        "python": sys.version.split()[0],
        "classes": {},
    }

    for name, factory in FACTORIES.items():
        top["classes"][name] = probe_class(name, factory)

    # Summary: aggregated private-field leak table + mechanism matrix.
    summary = {
        "mechanism_by_class": {n: r.get("dump_mechanism") for n, r in top["classes"].items()},
        "private_field_count_by_class": {
            n: len(r.get("private_fields", {})) for n, r in top["classes"].items()
        },
        "byte_stable_by_class": {n: r.get("byte_stable") for n, r in top["classes"].items()},
        "uniform_dump_api": len(
            {
                r.get("dump_mechanism")
                for r in top["classes"].values()
                if "construction_error" not in r
            }
        )
        == 1,
    }
    top["summary"] = summary

    print(json.dumps(top, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
