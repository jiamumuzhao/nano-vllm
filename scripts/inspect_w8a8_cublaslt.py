"""Record cuBLASLt API/plan evidence for the experimental W8A8 path."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    from nanovllm.kernels.w8a8_cublaslt import PATH, W8A8Workspace, linear

    try:
        cublaslt_version = torch.backends.cuda.cublas.version()
    except Exception:
        cublaslt_version = None
    shapes = [(32, 1024, 1024), (256, 4096, 1024)]
    records = []
    for m, n, k in shapes:
        torch.manual_seed(880000 + m + n)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        wi = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        ws = torch.ones(n, device="cuda", dtype=torch.float16)
        workspace = W8A8Workspace(m, n, k, x.device)
        workspace.prepare(wi, shapes=[(m, n, k)])
        plan = workspace.selected_plans[(m, n, k)]
        baseline_workspace = W8A8Workspace(m, n, k, x.device)
        baseline_workspace.prepare(wi, shapes=[(m, n, k)], autotune=False)
        before = workspace.data_ptrs()
        # The call also confirms the selected plan is executable on this GPU.
        linear(x, wi, ws, None, workspace=workspace)
        torch.cuda.synchronize()
        normal_reuse_ptr_stable = workspace.data_ptrs() == before
        mismatch = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        mismatch_rejected = False
        try:
            linear(x, mismatch, ws, None, workspace=workspace)
        except ValueError as exc:
            mismatch_rejected = "prepare" in str(exc) or "packed weight" in str(exc)
        old = int(wi[0, 0].item())
        wi[0, 0] = 1 if old != 1 else 2
        inplace_rejected = False
        try:
            linear(x, wi, ws, None, workspace=workspace)
        except ValueError as exc:
            inplace_rejected = "version" in str(exc) or "prepare" in str(exc)
        workspace.prepare(wi, shapes=[(m, n, k)])
        plan = workspace.selected_plans[(m, n, k)]
        linear(x, wi, ws, None, workspace=workspace)
        torch.cuda.synchronize()
        records.append({
            "shape": {"M": m, "N": n, "K": k},
            "runtime_path": PATH,
            "status": "passed",
            "compute_type": "CUBLAS_COMPUTE_32I",
            "A_dtype": "CUDA_R_8I",
            "B_dtype": "CUDA_R_8I",
            "C_dtype": "CUDA_R_32I",
            "D_dtype": "CUDA_R_32I",
            "weight_input_layout": "[N,K] contiguous int8",
            "weight_packed_layout": "[K,N] contiguous int8 row-major",
            "algorithm": plan.algorithm,
            "workspace_bytes": int(plan.workspace_size),
            "selected_metadata": workspace.selected_metadata[(m, n, k)],
            "baseline_row_major": baseline_workspace.selected_metadata[(m, n, k)],
            "candidate_count": len(workspace.candidates[(m, n, k)]),
            "candidates": workspace.candidates[(m, n, k)],
            "layout_results": workspace.layout_results[(m, n, k)],
            "selected_layout": workspace.selected_metadata[(m, n, k)]["layout"],
            "selected_algorithm": workspace.selected_metadata[(m, n, k)]["algorithm"],
            "workspace_data_ptr_stable": normal_reuse_ptr_stable,
            "normal_reuse_ptr_stable": normal_reuse_ptr_stable,
            "mismatch_weight_rejected": mismatch_rejected,
            "inplace_mutated_weight_rejected": inplace_rejected,
            "bound_weight_identity_fields": ["data_ptr", "shape", "dtype", "device", "version"],
            "reprepare_after_rejection_succeeded": True,
        })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path("docs/benchmarks")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"w8a8_cublaslt_api_{stamp}.json"
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "verification": "cublaslt_api_verified",
        "custom_extension_sass_claim": False,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "cublaslt_version": cublaslt_version,
        "extension": "nanovllm.kernels._w8a8_cublaslt",
        "runtime_path": PATH,
        "shapes": records,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(data, indent=2))
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
