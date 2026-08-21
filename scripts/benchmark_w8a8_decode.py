"""Benchmark-only W8A8 decode stages against FP16 and W8A16 legacy."""
from __future__ import annotations
import json, platform, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def stats(v):
    q=torch.tensor(v); return {"mean_ms":statistics.fmean(v),"p50_ms":float(q.quantile(.5)),"p95_ms":float(q.quantile(.95)),"raw_ms":v}
def timed(fn, warmup=30, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); v=[]
    for _ in range(iters):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); b.synchronize(); v.append(a.elapsed_time(b))
    return stats(v)

def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    from nanovllm.kernels import _w8a16_tensorcore as w16
    from nanovllm.kernels import _w8a8_tensorcore as w8
    from nanovllm.kernels.w8a8_tensorcore import linear as w8a8_linear
    codegen=sorted(Path("docs/benchmarks").glob("w8a8_tensorcore_codegen_*.json"))[-1]
    cg=json.loads(codegen.read_text())
    results=[]
    for m in (1,4,16,32,64):
      for n in (1024,4096):
        torch.manual_seed(5000+m+n); k=1024
        x=torch.randn(m,k,device="cuda",dtype=torch.float16); wf=torch.randn(n,k,device="cuda",dtype=torch.float16); ws=wf.float().abs().amax(1).clamp_min(1e-12).div(127).half(); wi=torch.round(wf.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8); bias=torch.randn(n,device="cuda",dtype=torch.float16); out=torch.empty((m,n),device="cuda",dtype=torch.float16)
        xq=torch.empty_like(x,dtype=torch.int8); xs=torch.empty((m,),device="cuda",dtype=torch.float16)
        modes=[
          ("fp16_f_linear",lambda:torch.nn.functional.linear(x,wf,bias),None),
          ("w8a16_legacy",lambda:w16.linear_cuda_legacy(x,wi,ws,bias,out),"cuda_fp16_tensorcore_wmma_decode_legacy"),
          ("w8a8_activation_quantization",lambda:w8.quantize_cuda(x,xq,xs),"w8a8_experimental_activation_quant"),
          ("w8a8_int8_gemm_legacy",lambda:w8.gemm_cuda(xq,xs,wi,ws,bias,out),"w8a8_experimental_int8_mma"),
          ("w8a8_int8_gemm_cta16",lambda:w8.gemm_cuda_cta16(xq,xs,wi,ws,bias,out),"w8a8_experimental_int8_mma_cta16x64"),
          ("w8a8_int8_gemm_cta32",lambda:w8.gemm_cuda_cta32(xq,xs,wi,ws,bias,out),"w8a8_experimental_int8_mma_cta32x64"),
          ("w8a8_end_to_end_auto",lambda:w8a8_linear(x,wi,ws,bias,implementation="auto"),"w8a8_experimental_int8_mma"),
          ("w8a8_end_to_end_legacy",lambda:w8a8_linear(x,wi,ws,bias,implementation="legacy"),"w8a8_experimental_int8_mma"),
          ("w8a8_end_to_end_cta16",lambda:w8a8_linear(x,wi,ws,bias,implementation="cta16x64"),"w8a8_experimental_int8_mma_cta16x64"),
          ("w8a8_end_to_end_cta32",lambda:w8a8_linear(x,wi,ws,bias,implementation="cta32x64"),"w8a8_experimental_int8_mma_cta32x64"),
        ]
        for mode,fn,path in modes:
          s=timed(fn); s["tflops"]=2*m*n*k/(s["mean_ms"]*1e9) if mode not in ("w8a8_activation_quantization",) else None
          results.append({"M":m,"N":n,"K":k,"mode":mode,"kernel_path":path,"latency":s})
    for r in results:
      if r["mode"].startswith("w8a8_end_to_end_") and r["mode"] != "w8a8_end_to_end_auto":
        w8a16=next(x for x in results if x["M"]==r["M"] and x["N"]==r["N"] and x["mode"]=="w8a16_legacy"); w8a8=next(x for x in results if x["M"]==r["M"] and x["N"]==r["N"] and x["mode"]=="w8a8_end_to_end_legacy"); fp16=next(x for x in results if x["M"]==r["M"] and x["N"]==r["N"] and x["mode"]=="fp16_f_linear")
        r["relative_to_w8a8_legacy_p50_speedup"]=w8a8["latency"]["p50_ms"]/r["latency"]["p50_ms"]-1; r["relative_to_w8a16_p50_speedup"]=w8a16["latency"]["p50_ms"]/r["latency"]["p50_ms"]-1; r["relative_to_fp16_p50_slowdown"]=r["latency"]["p50_ms"]/fp16["latency"]["p50_ms"]-1; r["eligible_for_future_integration"]=r["relative_to_w8a16_p50_speedup"]>=.10 and r["relative_to_fp16_p50_slowdown"]<=.10
    eligible=[r["eligible_for_future_integration"] for r in results if r["mode"] in ("w8a8_end_to_end_cta16","w8a8_end_to_end_cta32")]
    data={"environment":{"timestamp_utc":datetime.now(timezone.utc).isoformat(),"gpu":torch.cuda.get_device_name(),"compute_capability":list(torch.cuda.get_device_capability()),"pytorch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version(),"warmup":30,"iterations":100,"codegen_evidence":str(codegen),"codegen_status":cg.get("status"),"kernel_path":cg.get("kernel_path")},"results":results,"integration_gate":{"all_shapes_int8_mma":cg.get("status")=="passed","eligible_shapes":sum(eligible),"total_shapes":len(eligible),"eligible_for_future_integration":cg.get("status")=="passed" and any(eligible),"note":"W8A8 remains benchmark-only regardless of this result."}}
    out=Path("docs/benchmarks"); out.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); jp=out/f"w8a8_decode_{stamp}.json"; mp=out/f"w8a8_decode_{stamp}.md"; t=jp.with_suffix(".json.tmp"); t.write_text(json.dumps(data,indent=2)+"\n"); t.replace(jp)
    lines=["# W8A8 experimental decode benchmark","",f"GPU: {data['environment']['gpu']} | warmup=30 | iterations=100 | codegen: {codegen}","","Quantization, each GEMM implementation, and end-to-end paths are measured separately. `auto` is intentionally legacy for every workload.","","| M | N | Mode | Path | Mean ms | P50 ms | P95 ms | TFLOPS | vs W8A8 legacy | vs W8A16 | FP16 slowdown | Eligible |","|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
      s=r["latency"]; lines.append(f"| {r['M']} | {r['N']} | {r['mode']} | {r['kernel_path']} | {s['mean_ms']:.4f} | {s['p50_ms']:.4f} | {s['p95_ms']:.4f} | {s['tflops'] if s['tflops'] is not None else '-'} | {r.get('relative_to_w8a8_legacy_p50_speedup','-')} | {r.get('relative_to_w8a16_p50_speedup','-')} | {r.get('relative_to_fp16_p50_slowdown','-')} | {r.get('eligible_for_future_integration','-')} |")
    lines += ["","## Integration gate",f"Eligible for future model integration discussion: **{data['integration_gate']['eligible_for_future_integration']}**","","W8A8 is benchmark-only and was not connected to ModelRunner, Config, Qwen3, TP=2, or production W8A16 routing."]
    t=mp.with_suffix(".md.tmp"); t.write_text("\n".join(lines)+"\n"); t.replace(mp); print(f"JSON: {jp}\nMarkdown: {mp}")
if __name__=="__main__": main()
