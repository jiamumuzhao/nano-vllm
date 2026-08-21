"""Decode microbenchmark for legacy WMMA, wide candidate, and FP16 F.linear."""
from __future__ import annotations
import json, platform, statistics, time, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summary(values):
    q = torch.tensor(values)
    return {"mean_ms": statistics.fmean(values), "p50_ms": float(q.quantile(.5)), "p95_ms": float(q.quantile(.95))}


def measure(fn, warmup=30, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(iters):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize(); values.append(start.elapsed_time(end))
    return values


def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    from nanovllm.kernels import _w8a16_tensorcore as ext
    from nanovllm.kernels.w8a16_tensorcore import select_wmma_path
    workloads=[("decode",m,n,1024) for m in (1,4,16) for n in (1024,4096)]
    results=[]
    for family,m,n,k in workloads:
        torch.manual_seed(1000+m+n)
        x=torch.randn(m,k,device="cuda",dtype=torch.float16)
        w=torch.randn(n,k,device="cuda",dtype=torch.float16)
        scale=w.float().abs().amax(1).clamp_min(1e-12).div(127).half()
        wi=torch.round(w.float()/scale.float()[:,None]).clamp(-127,127).to(torch.int8)
        bias=torch.randn(n,device="cuda",dtype=torch.float16)
        out=torch.empty((m,n),device="cuda",dtype=torch.float16)
        modes=[
            ("legacy_single_warp", lambda: ext.linear_cuda_legacy(x,wi,scale,bias,out), "cuda_fp16_tensorcore_wmma_legacy"),
            ("decode_wide_candidate", lambda: ext.linear_cuda_decode_wide_candidate(x,wi,scale,bias,out), "cuda_fp16_tensorcore_wmma_decode_wide_candidate"),
            ("fp16_f_linear", lambda: torch.nn.functional.linear(x,w,bias), "fp16_f_linear"),
        ]
        for mode,fn,path in modes:
            torch.cuda.reset_peak_memory_stats()
            values=measure(fn)
            s=summary(values); s["tflops"]=2*m*n*k/(s["mean_ms"]*1e9)
            results.append({"family":family,"M":m,"N":n,"K":k,"mode":mode,"kernel_path":path,"selected_runtime_path":select_wmma_path(m,n,k),"latency":s,"peak_memory_allocated_bytes":torch.cuda.max_memory_allocated()})
    for r in results:
        legacy=next(x for x in results if x["M"]==r["M"] and x["N"]==r["N"] and x["mode"]=="legacy_single_warp")
        r["relative_to_legacy_p50_speedup"]=(legacy["latency"]["p50_ms"] / r["latency"]["p50_ms"] - 1.0) if r["mode"] != "legacy_single_warp" else 0.0
    codegen=sorted(Path("docs/benchmarks").glob("w8a16_tensorcore_codegen_*.json"))[-1]
    env={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"gpu":torch.cuda.get_device_name(),"compute_capability":list(torch.cuda.get_device_capability()),"pytorch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version(),"warmup":30,"iters":100,"hmma_evidence_path":str(codegen)}
    data={"environment":env,"results":results,"note":"Wide decode is benchmark-only; production selector remains legacy unless all six P50 workloads clear the 5% gate."}
    out=Path("docs/benchmarks"); out.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp=out/f"w8a16_wmma_tiling_{stamp}.json"; mp=out/f"w8a16_wmma_tiling_{stamp}.md"
    tmp=jp.with_suffix(".json.tmp"); tmp.write_text(json.dumps(data,indent=2)+"\n"); tmp.replace(jp)
    lines=["# W8A16 wide-N decode candidate benchmark","",f"GPU: {env['gpu']} | warmup={env['warmup']} | iterations={env['iters']} | HMMA evidence: {env['hmma_evidence_path']}","","| M | N | Mode | Path | Selected runtime path | Mean ms | P50 ms | P95 ms | TFLOPS | P50 speedup vs legacy |","|---:|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        s=r["latency"]; lines.append(f"| {r['M']} | {r['N']} | {r['mode']} | {r['kernel_path']} | {r['selected_runtime_path']} | {s['mean_ms']:.3f} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['tflops']:.3f} | {r['relative_to_legacy_p50_speedup']*100:.2f}% |")
    mt=mp.with_suffix(".md.tmp"); mt.write_text("\n".join(lines)+"\n"); mt.replace(mp)
    print(f"JSON: {jp}\nMarkdown: {mp}")

if __name__ == "__main__": main()
