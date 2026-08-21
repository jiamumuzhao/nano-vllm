"""Inspect the benchmark-only W8A8 extension's actual INT8 Tensor Core codegen."""
from __future__ import annotations
import json, platform, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def main():
    repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo))
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=repo/"docs/benchmarks"/f"w8a8_tensorcore_codegen_{stamp}.json"; files_dir=out.with_suffix("").with_name(out.stem+"_files"); files_dir.mkdir(parents=True,exist_ok=True)
    data={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"status":"error","kernel_path":"unsupported","extension_build":"unknown","extension_path":None,"ptx_path":None,"sass_path":None,"sass_contains_imma":False,"ptx_contains_int8_mma":False,"path_results":{},"compiler_error":None}
    try:
        import torch, triton
        data.update({"gpu_name":torch.cuda.get_device_name() if torch.cuda.is_available() else None,"compute_capability":list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,"gpu_count":torch.cuda.device_count() if torch.cuda.is_available() else 0,"pytorch_version":torch.__version__,"cuda_version":torch.version.cuda,"triton_version":triton.__version__,"python_version":platform.python_version(),"nvcc_version":subprocess.run(["/usr/local/cuda/bin/nvcc","--version"],capture_output=True,text=True).stdout})
        so=sorted((repo/"nanovllm/kernels").glob("_w8a8_tensorcore*.so"))
        if not so: raise RuntimeError("prebuilt W8A8 extension not found")
        data["extension_build"]="present"; data["extension_path"]=str(so[-1])
        from nanovllm.kernels.w8a8_tensorcore import linear
        for label,m,n,impl,path in (("m1_n1024_legacy",1,1024,"legacy","w8a8_experimental_int8_mma"),("m16_n4096_cta16",16,4096,"cta16x64","w8a8_experimental_int8_mma_cta16x64"),("m32_n4096_cta32",32,4096,"cta32x64","w8a8_experimental_int8_mma_cta32x64")):
            x=torch.randn(m,1024,device="cuda",dtype=torch.float16); wi=torch.randint(-127,128,(n,1024),device="cuda",dtype=torch.int8); ws=torch.ones(n,device="cuda",dtype=torch.float16); linear(x,wi,ws,None,implementation=impl); torch.cuda.synchronize()
            data["path_results"][label]={"shape":{"M":m,"N":n,"K":1024},"runtime_path":path}
        cuobjdump=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        sass=subprocess.run([cuobjdump,"--dump-sass",str(so[-1])],capture_output=True,text=True); sass_text=sass.stdout+sass.stderr; sass_file=files_dir/"w8a8_tensorcore.sass"; sass_file.write_text(sass_text); data["sass_path"]=str(sass_file)
        ptx=subprocess.run([cuobjdump,"--dump-ptx",str(so[-1])],capture_output=True,text=True); ptx_text=ptx.stdout+ptx.stderr; ptx_file=files_dir/"w8a8_tensorcore.ptx"; ptx_file.write_text(ptx_text); data["ptx_path"]=str(ptx_file)
        data["sass_imma_signatures"]=sorted(set(re.findall(r"IMMA\.[A-Z0-9_.]+",sass_text))); data["sass_contains_imma"]=bool(data["sass_imma_signatures"])
        data["ptx_contains_int8_mma"]=bool(re.search(r"mma\.sync\.aligned\.[^;\n]*(?:s8|s32)[^;\n]*(?:s8|s32)",ptx_text,re.I))
        if data["sass_contains_imma"] or data["ptx_contains_int8_mma"]:
            data["status"]="passed"; data["kernel_path"]="int8_mma_verified"
            for v in data["path_results"].values(): v.update({"status":"passed","sass_path":data["sass_path"],"ptx_path":data["ptx_path"],"sass_contains_imma":data["sass_contains_imma"],"ptx_contains_int8_mma":data["ptx_contains_int8_mma"]})
        else:
            data["status"]="failed"; data["kernel_path"]="int8_fma_fallback"; data["blocker"]="no IMMA or INT8 mma.sync in extension codegen"
    except Exception as exc:
        data["compiler_error"]=f"{type(exc).__name__}: {exc}"; data["status"]="error"
    tmp=out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(data,indent=2)+"\n"); tmp.replace(out); print(json.dumps(data,indent=2)); print(f"JSON: {out}"); return 0 if data["status"]=="passed" else 1

if __name__=="__main__": raise SystemExit(main())
