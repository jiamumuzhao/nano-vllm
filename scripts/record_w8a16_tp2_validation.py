"""Run and record the real W8A16 TP=2 correctness validation."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], env: dict[str, str], cwd: Path):
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    return completed.returncode, time.perf_counter() - start, completed.stdout, completed.stderr


def _environment(env: dict[str, str], repo: Path):
    import torch
    import triton
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": props.name,
                "memory_bytes": props.total_memory,
                "memory_gib": props.total_memory / 2**30,
            })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        commit = None
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except Exception:
        dirty = None
    return {
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "nccl_version": getattr(torch.cuda, "nccl", None) and torch.cuda.nccl.version(),
        "git_commit": commit,
        "dirty_worktree": dirty,
    }


def _markdown(data: dict) -> str:
    env = data["environment"]
    lines = [
        "# W8A16 TP=2 Correctness Validation", "",
        "This is a TP=2 correctness acceptance record, not a throughput benchmark.", "",
        "| Field | Value |", "|---|---|",
        f"| Status | **{data['status']}** |",
        f"| Exit code | `{data['exit_code']}` |",
        f"| Duration seconds | `{data['duration_seconds']:.3f}` |",
        f"| CUDA_VISIBLE_DEVICES | `{env.get('cuda_visible_devices')}` |",
        f"| GPU count | `{env.get('gpu_count')}` |",
        f"| PyTorch / CUDA / Triton | `{env.get('pytorch_version')}` / `{env.get('cuda_version')}` / `{env.get('triton_version')}` |",
        f"| NCCL | `{env.get('nccl_version')}` |",
        f"| Git commit | `{env.get('git_commit')}` |",
        f"| Dirty worktree | `{env.get('dirty_worktree')}` |", "",
        "## GPUs", "", "| Index | Name | Memory GiB |", "|---:|---|---:|",
    ]
    for gpu in env.get("gpus", []):
        lines.append(f"| {gpu['index']} | {gpu['name']} | {gpu['memory_gib']:.2f} |")
    lines += ["", "## Command", "", "```bash", data["command"], "```", "",
              "## Coverage", "",
              "- QKVParallelLinear with GQA shape (Q heads=4, KV heads=2, head_dim=8)",
              "- MergedColumnParallelLinear local shard loading then W8A16 quantization",
              "- RowParallelLinear bias rank-0-only addition and NCCL all-reduce",
              "- Local output-channel weight_scale dimensions and FP16 reference comparison",
              "- Real two-rank NCCL process group", "",
              "## Pytest stdout/stderr", "", "<details><summary>Captured output</summary>", "", "```text",
              data.get("stdout", "")[-12000:], "", data.get("stderr", "")[-12000:],
              "```", "</details>", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"w8a16_tp2_validation_{stamp}.json"
    md_path = output_dir / f"w8a16_tp2_validation_{stamp}.md"
    env = os.environ.copy()
    env["NANOVLLM_RUN_TP2"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [sys.executable, "-m", "pytest", "-q", "-s", "tests/test_w8a16_tp.py"]
    command_text = "NANOVLLM_RUN_TP2=1 CUDA_VISIBLE_DEVICES=0,1 " + " ".join(command)
    data = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": command_text,
            "exit_code": None, "status": "error", "duration_seconds": 0.0,
            "environment": {}, "test_scope": ["QKV GQA", "MergedColumnParallel", "RowParallel bias/all-reduce", "local shard scale"],
            "stdout": "", "stderr": ""}
    try:
        data["environment"] = _environment(env, repo)
        if data["environment"]["gpu_count"] < 2:
            data["status"] = "error"
            data["stderr"] = "At least two CUDA GPUs are required for TP=2 validation."
            data["exit_code"] = 2
        else:
            code, duration, stdout, stderr = _run(command, env, repo)
            data.update(exit_code=code, duration_seconds=duration, stdout=stdout, stderr=stderr,
                        status="passed" if code == 0 else "failed")
    except Exception as exc:
        data["stderr"] = f"{type(exc).__name__}: {exc}"
        data["status"] = "error"
        data["exit_code"] = 1
    finally:
        data["duration_seconds"] = float(data.get("duration_seconds", 0.0))
        json_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        md_text = _markdown(data)
        json_tmp = json_path.with_suffix(".json.tmp")
        md_tmp = md_path.with_suffix(".md.tmp")
        json_tmp.write_text(json_text)
        md_tmp.write_text(md_text)
        json_tmp.replace(json_path)
        md_tmp.replace(md_path)
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
    return int(data["exit_code"] if data["exit_code"] is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
