"""Compile the W8A16 Triton kernel in a fresh cache and inspect PTX/SASS."""
from __future__ import annotations
import argparse, json, os, platform, re, shutil, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import sys

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    out = Path(args.output) if args.output else repo / "docs/benchmarks" / f"w8a16_codegen_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out = out if out.is_absolute() else repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error", "kernel_path": "unsupported",
        "ptx_paths": [], "ptx_path": None,
        "ptx_contains_mma_sync": False, "ptx_contains_fp16_mma": False,
        "ptx_contains_fma_rn_f32": False,
        "dot_operand_type_summary": {
            "source_operands": "fp16/fp16", "source_accumulator": "fp32",
            "compiled_operands": "unknown", "compiled_accumulator": "unknown"
        },
        "sass_paths": [], "sass_contains_hmma": False,
        "compiler_error": None, "run_error": None,
    }
    try:
        import torch
        import triton
        data["pytorch_version"] = torch.__version__
        data["cuda_version"] = torch.version.cuda
        data["triton_version"] = triton.__version__
        data["python_version"] = platform.python_version()
        data["gpu_name"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        data["compute_capability"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        data["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not torch.cuda.is_available():
            data["compiler_error"] = "CUDA is unavailable"
        else:
            with tempfile.TemporaryDirectory(prefix="nano-vllm-triton-cache-") as cache:
                os.environ["TRITON_CACHE_DIR"] = cache
                from nanovllm.kernels.w8a16 import w8a16_linear
                x = torch.randn(32, 1024, device="cuda", dtype=torch.float16)
                wi = torch.randint(-127, 128, (32, 1024), device="cuda", dtype=torch.int8)
                scale = torch.ones(32, device="cuda", dtype=torch.float16)
                w8a16_linear(x, wi, scale, None)
                torch.cuda.synchronize()
                files = list(Path(cache).rglob("*"))
                ptx = [f for f in files if f.is_file() and f.suffix == ".ptx"]
                artifact_dir = out.with_suffix("") .with_name(out.stem + "_files")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                persisted_ptx = []
                for i, f in enumerate(ptx):
                    target = artifact_dir / f"kernel_{i}.ptx"
                    shutil.copy2(f, target)
                    persisted_ptx.append(str(target))
                data["ptx_paths"] = persisted_ptx
                data["ptx_path"] = persisted_ptx[0] if persisted_ptx else None
                ptx_text = "\n".join(f.read_text(errors="ignore") for f in ptx)
                data["ptx_contains_mma_sync"] = "mma.sync" in ptx_text
                data["ptx_contains_fp16_mma"] = bool(re.search(r"mma\.sync\.aligned\.[^\s;]*f16\.f16\.f32", ptx_text) or re.search(r"mma\.sync\.aligned\.[^\s;]*f32\.f16\.f16\.f32", ptx_text))
                data["ptx_contains_fma_rn_f32"] = "fma.rn.f32" in ptx_text
                mma_types = re.findall(r"mma\.sync\.aligned\.([^;\n]+)", ptx_text)
                data["dot_operand_type_summary"] = {
                    "source_operands": "fp16/fp16", "source_accumulator": "fp32",
                    "compiled_operands": "fp16/fp16" if data["ptx_contains_fp16_mma"] else ("unknown (no mma.sync)" if not mma_types else "; ".join(mma_types[:4])),
                    "compiled_accumulator": "fp32" if data["ptx_contains_fp16_mma"] or data["ptx_contains_fma_rn_f32"] else "unknown",
                    "mma_signatures": mma_types[:4],
                    "fma_signature": "f32" if data["ptx_contains_fma_rn_f32"] else None,
                }
                cubins = [f for f in files if f.is_file() and f.suffix in (".cubin", ".fatbin")]
                cuobjdump = "/usr/local/cuda/bin/cuobjdump"
                sass_text = ""
                if os.path.exists(cuobjdump):
                    for i, cubin in enumerate(cubins):
                        try:
                            text = subprocess.check_output([cuobjdump, "--dump-sass", str(cubin)], text=True, stderr=subprocess.STDOUT)
                            sass_text += text
                            target = artifact_dir / f"kernel_{i}{cubin.suffix}"
                            shutil.copy2(cubin, target)
                            data["sass_paths"].append(str(target))
                        except Exception:
                            pass
                data["sass_contains_hmma"] = bool(re.search(r"HMMA\.[^\n]*F16", sass_text) or re.search(r"HMMA\.[^\n]*F16", sass_text, re.I))
                if data["ptx_contains_fp16_mma"] or data["sass_contains_hmma"]:
                    data["kernel_path"] = "fp16_mma_verified"
                    data["status"] = "passed"
                elif data["ptx_contains_fma_rn_f32"]:
                    data["kernel_path"] = "fp32_fma_fallback"
                    data["status"] = "failed"
                else:
                    data["status"] = "failed"
                    data["run_error"] = "compiled kernel had neither recognized FP16 MMA nor fma.rn.f32 evidence"
    except Exception as exc:
        data["status"] = "error"
        data["compiler_error"] = f"{type(exc).__name__}: {exc}"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(data, indent=2))
    print(f"JSON: {out}")
    return 0 if data["kernel_path"] == "fp16_mma_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
