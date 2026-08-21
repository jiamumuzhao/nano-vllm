"""Operator-level W8A8 break-even benchmark.

M is the flattened batch_size * query_tokens.  This is intentionally not a
full-model prefill benchmark: W8A8 is not connected to ModelRunner.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048)
NS = (1024, 4096)
K = 1024
WARMUP = 30
ITERS = 100
W8A8_IMPLS = ("auto", "legacy", "cta16x64", "cta32x64", "cublaslt")
WMMA_IMPLS = ("auto", "legacy", "cta16x64", "cta32x64")


def summary(values):
    q = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(q.quantile(0.50)),
        "p95_ms": float(q.quantile(0.95)),
        "raw_ms": values,
    }


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return summary(values)


def make_case(m, n, k):
    torch.manual_seed(910000 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    wf = torch.randn(n, k, device="cuda", dtype=torch.float16)
    ws = wf.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi = torch.round(wf.float() / ws.float()[:, None]).clamp(-127, 127).to(torch.int8)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    xq = torch.empty((m, k), device="cuda", dtype=torch.int8)
    xs = torch.empty((m,), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    return x, wf, wi, ws, bias, xq, xs, out


def path_for(implementation):
    return {
        "auto": "w8a8_experimental_int8_mma",
        "legacy": "w8a8_experimental_int8_mma",
        "cta16x64": "w8a8_experimental_int8_mma_cta16x64",
        "cta32x64": "w8a8_experimental_int8_mma_cta32x64",
        "cublaslt": "w8a8_experimental_cublaslt_int8_i32",
    }[implementation]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    from nanovllm.kernels import _w8a8_tensorcore as w8
    from nanovllm.kernels import _w8a16_tensorcore as w16
    from nanovllm.kernels.w8a8_tensorcore import linear as w8a8_linear
    from nanovllm.kernels.w8a8_cublaslt import W8A8Workspace, linear as cublaslt_linear
    from nanovllm.kernels import _w8a8_cublaslt as cublaslt_ext

    codegen_files = sorted(Path("docs/benchmarks").glob("w8a8_tensorcore_codegen_*.json"))
    codegen = codegen_files[-1] if codegen_files else None
    codegen_data = json.loads(codegen.read_text()) if codegen else {}
    results = []

    for m in MS:
        for n in NS:
            x, wf, wi, ws, bias, xq, xs, out = make_case(m, n, K)
            cublaslt_workspace = W8A8Workspace(m, n, K, x.device)
            cublaslt_workspace.prepare(wi, shapes=[(m, n, K)])
            cublaslt_plan = cublaslt_workspace.selected_plans[(m, n, K)]
            cublaslt_baseline_workspace = W8A8Workspace(m, n, K, x.device)
            cublaslt_baseline_workspace.prepare(wi, shapes=[(m, n, K)], autotune=False)
            cublaslt_baseline_plan = cublaslt_baseline_workspace.selected_plans[(m, n, K)]
            cublaslt_baseline_ptrs_before = cublaslt_baseline_workspace.data_ptrs()
            cublaslt_ptrs_before = cublaslt_workspace.data_ptrs()
            specs = [
                ("fp16", lambda: torch.nn.functional.linear(x, wf, bias), None),
                ("w8a16_legacy", lambda: w16.linear_cuda_legacy(x, wi, ws, bias, out), "cuda_fp16_tensorcore_wmma_decode_legacy"),
                ("w8a8_quantization", lambda: w8.quantize_cuda(x, xq, xs), "w8a8_experimental_activation_quant"),
            ]
            for implementation in WMMA_IMPLS:
                def gemm_fn(impl=implementation):
                    if impl in ("auto", "legacy"):
                        return w8.gemm_cuda(xq, xs, wi, ws, bias, out)
                    if impl == "cta16x64":
                        return w8.gemm_cuda_cta16(xq, xs, wi, ws, bias, out)
                    return w8.gemm_cuda_cta32(xq, xs, wi, ws, bias, out)

                specs.append((
                    f"w8a8_gemm_{implementation}",
                    gemm_fn,
                    path_for(implementation),
                ))
            specs.extend([
                ("w8a8_cublaslt_quantization", lambda: w8.quantize_cuda(x, cublaslt_workspace.xq, cublaslt_workspace.xs), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_cublaslt_gemm", lambda: cublaslt_plan.run(cublaslt_workspace.xq, cublaslt_workspace.packed_weight, cublaslt_workspace.acc, cublaslt_workspace.cublas_workspace), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_cublaslt_epilogue", lambda: cublaslt_ext.epilogue_cuda(cublaslt_workspace.acc, cublaslt_workspace.xs, ws, bias, cublaslt_workspace.out, m, n, cublaslt_workspace.max_n, cublaslt_workspace.max_n), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_operator_cublaslt_autotuned", lambda: cublaslt_linear(x, wi, ws, bias, workspace=cublaslt_workspace), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_cublaslt_row_major_baseline_quantization", lambda: w8.quantize_cuda(x, cublaslt_baseline_workspace.xq, cublaslt_baseline_workspace.xs), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_cublaslt_row_major_baseline_gemm", lambda: cublaslt_baseline_plan.run(cublaslt_baseline_workspace.xq, cublaslt_baseline_workspace.packed_weight, cublaslt_baseline_workspace.acc, cublaslt_baseline_workspace.cublas_workspace), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_cublaslt_row_major_baseline_epilogue", lambda: cublaslt_ext.epilogue_cuda(cublaslt_baseline_workspace.acc, cublaslt_baseline_workspace.xs, ws, bias, cublaslt_baseline_workspace.out, m, n, cublaslt_baseline_workspace.max_n, cublaslt_baseline_workspace.max_n), "w8a8_experimental_cublaslt_int8_i32"),
                ("w8a8_operator_cublaslt_row_major_baseline", lambda: cublaslt_linear(x, wi, ws, bias, workspace=cublaslt_baseline_workspace), "w8a8_experimental_cublaslt_int8_i32"),
            ])
            for implementation in WMMA_IMPLS:
                specs.append((
                    f"w8a8_operator_{implementation}",
                    lambda impl=implementation: w8a8_linear(x, wi, ws, bias, implementation=impl),
                    path_for(implementation),
                ))

            for mode, fn, path in specs:
                latency = timed(fn)
                results.append({
                    "M": m, "N": n, "K": K, "mode": mode,
                    "implementation": mode.split("_")[-1] if mode.startswith("w8a8_") else None,
                    "kernel_path": path,
                    "latency": latency,
                    "operator_level_prefill_shaped": mode.startswith("w8a8_operator_"),
                    "model_e2e": None,
                })
            stable = cublaslt_workspace.data_ptrs() == cublaslt_ptrs_before
            for row in results:
                if (row["M"] == m and row["N"] == n and row["mode"].startswith("w8a8_cublaslt")) or (row["M"] == m and row["N"] == n and row["mode"] == "w8a8_operator_cublaslt_autotuned"):
                    row["workspace_data_ptr_stable"] = stable
                    row["selected_metadata"] = cublaslt_workspace.selected_metadata[(m, n, K)]
                    row["baseline_metadata"] = cublaslt_baseline_workspace.selected_metadata[(m, n, K)]
                    row["candidate_count"] = len(cublaslt_workspace.candidates[(m, n, K)])
                    row["layout_results"] = cublaslt_workspace.layout_results[(m, n, K)]
                if row["M"] == m and row["N"] == n and "row_major_baseline" in row["mode"]:
                    row["workspace_data_ptr_stable"] = cublaslt_baseline_workspace.data_ptrs() == cublaslt_baseline_ptrs_before
                    row["selected_metadata"] = cublaslt_baseline_workspace.selected_metadata[(m, n, K)]

    for row in results:
        if not row["mode"].startswith("w8a8_operator_"):
            continue
        same = lambda mode: next(r for r in results if r["M"] == row["M"] and r["N"] == row["N"] and r["mode"] == mode)
        fp16 = same("fp16")["latency"]["p50_ms"]
        w8a16 = same("w8a16_legacy")["latency"]["p50_ms"]
        w8a8_legacy = same("w8a8_operator_legacy")["latency"]["p50_ms"]
        p50 = row["latency"]["p50_ms"]
        row["relative_to_fp16_p50_slowdown"] = p50 / fp16 - 1.0
        row["relative_to_w8a16_p50_speedup"] = w8a16 / p50 - 1.0
        row["relative_to_w8a8_legacy_p50_speedup"] = w8a8_legacy / p50 - 1.0
        row["eligible_for_future_integration"] = (
            row["relative_to_w8a16_p50_speedup"] >= 0.10
            and row["relative_to_fp16_p50_slowdown"] <= 0.10
        )

    break_even = {}
    for n in NS:
        break_even[str(n)] = {}
        for implementation in W8A8_IMPLS:
            mode = "w8a8_operator_cublaslt_autotuned" if implementation == "cublaslt" else f"w8a8_operator_{implementation}"
            eligible = [r["M"] for r in results if r["N"] == n and r["mode"] == mode and r["eligible_for_future_integration"]]
            break_even[str(n)][implementation] = min(eligible) if eligible else "no_break_even_observed"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path("docs/benchmarks")
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"w8a8_break_even_{stamp}.json"
    md_path = outdir / f"w8a8_break_even_{stamp}.md"
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "operator_level_prefill_shaped_linear",
        "model_e2e": None,
        "w8a8_model_integration": False,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "warmup": WARMUP,
            "iterations": ITERS,
            "codegen_evidence": str(codegen) if codegen else None,
            "codegen_status": codegen_data.get("status"),
        },
        "workloads": {"M": list(MS), "N": list(NS), "K": K},
        "typical_prefill_shapes": {
            "128": ["batch=1,q_len=128", "batch=8,q_len=16"],
            "512": ["batch=1,q_len=512", "batch=8,q_len=64"],
            "2048": ["batch=1,q_len=2048", "batch=32,q_len=64"],
        },
        "break_even": break_even,
        "results": results,
        "integration_gate": {
            "eligible_workloads": sum(r.get("eligible_for_future_integration", False) for r in results),
            "operator_results_only": True,
            "note": "W8A8 is not connected to ModelRunner; no model-e2e W8A8 conclusion is made.",
        },
    }
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(json_path)

    lines = [
        "# W8A8 operator-level break-even benchmark", "",
        "This is a prefill-shaped Linear operator benchmark, not full-model prefill: W8A8 is not connected to `ModelRunner`.",
        f"GPU: {data['environment']['gpu']} | CUDA: {data['environment']['cuda']} | PyTorch: {data['environment']['pytorch']} | warmup={WARMUP} | iterations={ITERS}",
        f"Codegen evidence: `{codegen}`", "",
        "## Break-even M", "",
        "| N | auto | legacy | cta16x64 | cta32x64 | cublaslt_autotuned |", "|---:|---|---|---|---|---|",
    ]
    for n in NS:
        b = break_even[str(n)]
        lines.append(f"| {n} | {b['auto']} | {b['legacy']} | {b['cta16x64']} | {b['cta32x64']} | {b['cublaslt']} |")
    lines += ["", "`no_break_even_observed` means no tested M met both the W8A16 +10% speedup and FP16 <=10% slowdown gate.", "", "## Typical prefill-shaped mappings", ""]
    for m, shapes in data["typical_prefill_shapes"].items():
        lines.append(f"- M={m}: " + " or ".join(shapes))
    lines += ["", "## Operator results", "", "| M | N | Mode | Path | Mean ms | P50 ms | P95 ms | vs FP16 | vs W8A16 | vs W8A8 legacy | Eligible |", "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        if not r["mode"].startswith("w8a8_operator_"):
            continue
        s = r["latency"]
        lines.append(f"| {r['M']} | {r['N']} | {r['mode']} | {r['kernel_path']} | {s['mean_ms']:.4f} | {s['p50_ms']:.4f} | {s['p95_ms']:.4f} | {r['relative_to_fp16_p50_slowdown']:.4f} | {r['relative_to_w8a16_p50_speedup']:.4f} | {r['relative_to_w8a8_legacy_p50_speedup']:.4f} | {r['eligible_for_future_integration']} |")
    lines += ["", "Activation quantization, GEMM, and operator end-to-end raw timings are retained in the JSON. No W8A8 model-e2e prefill result is reported."]
    tmp = md_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(md_path)
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
