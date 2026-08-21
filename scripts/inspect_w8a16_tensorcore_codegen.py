"""Run the prebuilt WMMA extension and inspect its PTX/SASS evidence."""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = repo / "docs/benchmarks" / f"w8a16_tensorcore_codegen_{stamp}.json"
    files_dir = out.with_suffix("").with_name(out.stem + "_files")
    files_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error", "kernel_path": "unsupported",
        "extension_build": "unknown", "extension_path": None,
        "ptx_path": None, "sass_path": None,
        "ptx_contains_mma_sync": False, "ptx_contains_fp16_mma": False,
        "sass_contains_hmma": False, "sass_hmma_signatures": [],
        "runtime_path": None, "path_results": {}, "compiler_error": None, "run_error": None,
    }
    try:
        import torch
        import triton
        data.update({
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "triton_version": triton.__version__, "python_version": platform.python_version(),
            "nvcc_version": subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], text=True,
                                            capture_output=True).stdout.strip(),
        })
        so_files = sorted((repo / "nanovllm/kernels").glob("_w8a16_tensorcore*.so"))
        if not so_files:
            data["extension_build"] = "missing"
            data["compiler_error"] = "prebuilt WMMA extension not found; run: python setup.py build_ext --inplace"
        elif not torch.cuda.is_available():
            data["extension_build"] = "present_but_cuda_unavailable"
            data["extension_path"] = str(so_files[-1])
        else:
            data["extension_build"] = "present"
            data["extension_path"] = str(so_files[-1])
            from nanovllm.kernels.w8a16_tensorcore import linear, linear_decode_wide_candidate
            for label, m, n in (("decode_legacy", 1, 128),
                                ("decode_wide_candidate", 1, 1024),
                                ("prefill_legacy", 64, 1024),
                                ("prefill_cta", 256, 1024)):
                x = torch.randn(m, 1024, device="cuda", dtype=torch.float16)
                wi = torch.randint(-127, 128, (n, 1024), device="cuda", dtype=torch.int8)
                scale = torch.ones(n, device="cuda", dtype=torch.float16)
                (linear_decode_wide_candidate if label == "decode_wide_candidate" else linear)(x, wi, scale, None)
                torch.cuda.synchronize()
                data["path_results"][label] = {
                    "shape": {"M": m, "N": n, "K": 1024},
                    "runtime_path": ("cuda_fp16_tensorcore_wmma_decode_wide_candidate" if label == "decode_wide_candidate" else
                                     "cuda_fp16_tensorcore_wmma_decode_legacy" if m <= 16 else
                                     "cuda_fp16_tensorcore_wmma_prefill_cta" if m >= 256 else
                                     "cuda_fp16_tensorcore_wmma_prefill_legacy"),
                }
            data["runtime_path"] = "decode+prefill"
            cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
            sass = subprocess.run([cuobjdump, "--dump-sass", str(so_files[-1])], text=True,
                                  capture_output=True)
            sass_text = sass.stdout + sass.stderr
            sass_file = files_dir / "w8a16_tensorcore.sass"
            sass_file.write_text(sass_text)
            data["sass_path"] = str(sass_file)
            ptx = subprocess.run([cuobjdump, "--dump-ptx", str(so_files[-1])], text=True,
                                 capture_output=True)
            ptx_text = ptx.stdout + ptx.stderr
            ptx_file = files_dir / "w8a16_tensorcore.ptx"
            ptx_file.write_text(ptx_text)
            data["ptx_path"] = str(ptx_file)
            data["ptx_contains_mma_sync"] = "mma.sync.aligned" in ptx_text
            data["ptx_contains_fp16_mma"] = bool(re.search(
                r"mma\.sync\.aligned\.[^;\n]*f16[^;\n]*f16[^;\n]*f32", ptx_text))
            data["sass_hmma_signatures"] = sorted(set(re.findall(r"HMMA\.[A-Z0-9_.]+", sass_text)))
            data["sass_contains_hmma"] = bool(data["sass_hmma_signatures"])
            if data["ptx_contains_fp16_mma"] or data["sass_contains_hmma"]:
                data["kernel_path"] = "cuda_fp16_tensorcore_wmma"
                for result in data["path_results"].values():
                    result["codegen_status"] = "passed"
                    result["sass_path"] = data["sass_path"]
                    result["ptx_path"] = data["ptx_path"]
                    result["sass_contains_hmma"] = data["sass_contains_hmma"]
                data["status"] = "passed"
            else:
                data["kernel_path"] = "fp32_fma_fallback"
                data["status"] = "failed"
                data["run_error"] = "extension executed, but no PTX FP16 MMA or SASS HMMA was found"
                for result in data["path_results"].values():
                    result["codegen_status"] = "failed"
    except Exception as exc:
        data["status"] = "error"
        data["compiler_error"] = f"{type(exc).__name__}: {exc}"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(data, indent=2))
    print(f"JSON: {out}")
    return 0 if data["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
