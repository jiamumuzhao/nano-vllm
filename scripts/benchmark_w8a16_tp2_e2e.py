"""Real Qwen3 TP=2 FP16 versus W8A16 eager end-to-end benchmark."""
from __future__ import annotations
import argparse, gc, json, os, platform, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def ints(v): return [int(x) for x in v.split(",") if x]
def summary(v):
    return {"mean_ms": statistics.fmean(v), "p50_ms": float(torch.tensor(v).quantile(.5).item()), "p95_ms": float(torch.tensor(v).quantile(.95).item())}


def codegen_metadata():
    files=sorted(Path(__file__).resolve().parents[1].joinpath("docs/benchmarks").glob("w8a16_tensorcore_codegen_*.json"))
    if not files: return {"implementation":"unverified","evidence":None,"status":"missing"}
    try:
        d=json.loads(files[-1].read_text()); return {"implementation":d.get("kernel_path","unsupported"),"evidence":str(files[-1]),"status":d.get("status","unknown")}
    except Exception: return {"implementation":"unverified","evidence":str(files[-1]),"status":"unreadable"}


def once(engine, prompts, out_len, seed):
    torch.manual_seed(seed)
    params=SamplingParams(temperature=1.0, max_tokens=out_len, ignore_eos=True)
    for p in prompts: engine.add_request(p, params)
    torch.cuda.synchronize(); start=time.perf_counter(); pre_ms=0.; pre_tok=0; dec=[]; first=None
    while not engine.is_finished():
        torch.cuda.synchronize(); t=time.perf_counter()
        _, n, events=engine.step_with_events(); torch.cuda.synchronize()
        ms=(time.perf_counter()-t)*1000
        if n > 0: pre_ms += ms; pre_tok += n
        else: dec.append(ms)
        if first is None and events: first=time.perf_counter()
    total=(time.perf_counter()-start)*1000
    dec_total=statistics.fsum(dec)
    if not pre_tok or not dec or first is None: raise RuntimeError("incomplete prefill/decode events")
    return {"ttft_ms":(first-start)*1000, "prefill_tokens":pre_tok, "prefill_ms":pre_ms,
            "prefill_tok_s":pre_tok*1000/pre_ms, "decode_step_ms":statistics.fmean(dec),
            "decode_tok_s":len(prompts)*out_len*1000/dec_total,
            "end_to_end_tok_s":len(prompts)*out_len*1000/total, "decode_steps":len(dec)}


def env(model):
    cg=codegen_metadata()
    g=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i); g.append({"index":i,"name":p.name,"memory_bytes":p.total_memory,"memory_gib":p.total_memory/2**30})
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit=None
    try: dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except Exception: dirty=None
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"gpu_count":len(g),"gpus":g,
            "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"pytorch_version":torch.__version__,
            "cuda_version":torch.version.cuda,"triton_version":triton.__version__,
            "nccl_version":getattr(torch.cuda,"nccl",None) and torch.cuda.nccl.version(),
            "python_version":platform.python_version(),"model":model,"dtype":"float16","tensor_parallel_size":2,
            "git_commit":commit,"dirty_worktree":dirty,"w8a16_kernel_implementation":cg["implementation"],"w8a16_codegen_evidence":cg["evidence"],"w8a16_codegen_status":cg["status"]}


def run_one(args, mode, batch, length, seed):
    prompts=[[1]*length for _ in range(batch)]
    engine=None; raw=[]; start=time.perf_counter()
    try:
        engine=LLMEngine(args.model,dtype="float16",tensor_parallel_size=2,
            enforce_eager=args.enforce_eager,quantization=mode,max_num_seqs=max(args.batch_sizes),
            max_model_len=max(args.input_lengths)+args.output_len,
            max_num_batched_tokens=max(args.batch_sizes)*(max(args.input_lengths)+args.output_len))
        load_mem=torch.cuda.memory_allocated(); effective_eager=engine.model_runner.enforce_eager
        graph_policy="disabled_w8a16_eager_only" if mode=="w8a16" else ("eager" if effective_eager else "cuda_graph")
        for i in range(args.warmup_runs): once(engine,prompts,args.output_len,seed+i)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        raw=[once(engine,prompts,args.output_len,seed+args.warmup_runs+i) for i in range(args.repeats)]
        torch.cuda.synchronize()
        return {"status":"ok","mode":mode,"batch_size":batch,"input_len":length,"output_len":args.output_len,
          "quantization":mode,"tensor_parallel_size":2,"cuda_graph_policy":graph_policy,
          "load_memory_allocated_bytes":load_mem,"peak_memory_allocated_bytes":torch.cuda.max_memory_allocated(),
          "ttft":summary([x["ttft_ms"] for x in raw]),"prefill_latency":summary([x["prefill_ms"] for x in raw]),
          "prefill_tok_s":summary([x["prefill_tok_s"] for x in raw]),"decode_step_latency":summary([x["decode_step_ms"] for x in raw]),
          "decode_tok_s":summary([x["decode_tok_s"] for x in raw]),"end_to_end_tok_s":summary([x["end_to_end_tok_s"] for x in raw]),"raw_runs":raw,
          "duration_seconds":time.perf_counter()-start}
    except Exception as e:
        return {"status":"failed","mode":mode,"batch_size":batch,"input_len":length,"output_len":args.output_len,
          "quantization":mode,"tensor_parallel_size":2,"error":f"{type(e).__name__}: {e}","raw_runs":[],"duration_seconds":time.perf_counter()-start}
    finally:
        if engine is not None:
            try: engine.exit()
            except Exception: pass
            del engine; gc.collect(); torch.cuda.empty_cache()


def markdown(data):
    lines=["# W8A16 TP=2 End-to-End Inference Benchmark","","This is a real `LLMEngine(tensor_parallel_size=2)` inference comparison, not a Linear correctness record or throughput claim.","",f"Command: `{data['command']}`","", "| Batch | Input | Mode | Status | Graph policy | Load MB | Peak MB | TTFT ms | Prefill ms | Prefill tok/s | Decode step ms | Decode tok/s | E2E tok/s |", "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data['results']:
        if r['status']!='ok': lines.append(f"| {r['batch_size']} | {r['input_len']} | {r['mode']} | **failed** | — | — | — | — | — | — | — | — | — |")
        else: lines.append(f"| {r['batch_size']} | {r['input_len']} | {r['mode']} | ok | {r['cuda_graph_policy']} | {r['load_memory_allocated_bytes']/2**20:.1f} | {r['peak_memory_allocated_bytes']/2**20:.1f} | {r['ttft']['mean_ms']:.2f} | {r['prefill_latency']['mean_ms']:.2f} | {r['prefill_tok_s']['mean_ms']:.1f} | {r['decode_step_latency']['mean_ms']:.2f} | {r['decode_tok_s']['mean_ms']:.1f} | {r['end_to_end_tok_s']['mean_ms']:.1f} |")
    lines += ["","FP16 TP=2 and W8A16 TP=2 use the same prompts, seeds, sampling, limits, warmups, and repeats. With `--enforce-eager`, both modes are eager; W8A16 is always eager-only, while FP16 may use CUDA Graph under the default policy. Therefore default-policy results must not be interpreted as isolating quantization kernel cost.","","## Environment","",json.dumps(data['environment'],indent=2),"","## Coverage","","- Qwen3-0.6B FP16, TP=2 versus W8A16, TP=2","- Prefill/decode/TTFT/end-to-end event accounting","- Load and peak allocated memory","- Raw repeats preserved","", "## Failures", ""]
    for r in data['results']:
        if r['status']!='ok': lines.append(f"- {r['mode']} batch={r['batch_size']} input={r['input_len']}: {r.get('error')}")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--model',default='/root/huggingface/Qwen3-0.6B'); p.add_argument('--batch-sizes',type=ints,default=[1,4]); p.add_argument('--input-lengths',type=ints,default=[128,1024]); p.add_argument('--output-len',type=int,default=32); p.add_argument('--warmup-runs',type=int,default=3); p.add_argument('--repeats',type=int,default=10); p.add_argument('--seed',type=int,default=20260720); p.add_argument('--output-dir',default='docs/benchmarks'); p.add_argument('--enforce-eager',action='store_true'); p.add_argument('--workload-timeout',type=float,default=180.0); p.add_argument('--worker-mode'); p.add_argument('--worker-batch',type=int); p.add_argument('--worker-input',type=int); args=p.parse_args(argv)
    if args.worker_mode:
        result=run_one(args,args.worker_mode,args.worker_batch,args.worker_input,args.seed+args.worker_batch*100000+args.worker_input)
        print('WORKER_RESULT='+json.dumps(result),flush=True)
        return 0 if result['status']=='ok' else 1
    repo=Path(__file__).resolve().parents[1]; out=(repo/args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); jp=out/f'w8a16_tp2_e2e_{stamp}.json'; mp=out/f'w8a16_tp2_e2e_{stamp}.md'; command=' '.join(sys.argv)
    if not torch.cuda.is_available() or torch.cuda.device_count()<2: raise SystemExit('ERROR: TP=2 benchmark requires at least two CUDA GPUs')
    results=[]
    for mode in ('none','w8a16'):
        for b in args.batch_sizes:
            for n in args.input_lengths:
                worker=[sys.executable,str(Path(__file__).resolve()),'--worker-mode',mode,'--worker-batch',str(b),'--worker-input',str(n),'--model',args.model,'--output-len',str(args.output_len),'--warmup-runs',str(args.warmup_runs),'--repeats',str(args.repeats),'--seed',str(args.seed)]
                if args.enforce_eager: worker.append('--enforce-eager')
                try:
                    proc=subprocess.run(worker,env=os.environ.copy(),text=True,capture_output=True,timeout=args.workload_timeout)
                    marker=[line for line in proc.stdout.splitlines() if line.startswith('WORKER_RESULT=')]
                    if marker: r=json.loads(marker[-1].split('=',1)[1])
                    else: r={'status':'failed','mode':mode,'batch_size':b,'input_len':n,'output_len':args.output_len,'quantization':mode,'tensor_parallel_size':2,'error':f'worker exit {proc.returncode} without result','diagnostic_stdout':proc.stdout[-12000:],'diagnostic_stderr':proc.stderr[-12000:],'raw_runs':[]}
                except subprocess.TimeoutExpired as exc:
                    r={'status':'failed','mode':mode,'batch_size':b,'input_len':n,'output_len':args.output_len,'quantization':mode,'tensor_parallel_size':2,'error':f'workload timeout after {args.workload_timeout}s','diagnostic_stdout':(exc.stdout or '')[-12000:] if isinstance(exc.stdout,str) else '', 'diagnostic_stderr':(exc.stderr or '')[-12000:] if isinstance(exc.stderr,str) else '', 'raw_runs':[]}
                results.append(r); print(mode,b,n,r['status'],flush=True)
    data={'environment':env(args.model),'command':command,'parameters':vars(args),'results':results}
    text=json.dumps(data,indent=2)+"\n"; md=markdown(data); jt=jp.with_suffix('.json.tmp'); mt=mp.with_suffix('.md.tmp'); jt.write_text(text); mt.write_text(md); jt.replace(jp); mt.replace(mp); print(f'JSON: {jp}\nMarkdown: {mp}')
    return 0 if all(r['status']=='ok' for r in results) else 1
if __name__=='__main__': raise SystemExit(main())
