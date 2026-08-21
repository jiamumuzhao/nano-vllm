"""Shared helpers for reproducible offline generation benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False, timeout=20)
    except Exception:
        return None
    text = completed.stdout.strip() or completed.stderr.strip()
    return text or None


def git_revision() -> dict[str, Any]:
    rev = run_text(["git", "rev-parse", "HEAD"])
    status = run_text(["git", "status", "--short"])
    return {
        "commit": rev,
        "is_dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": git_revision(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append({
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
            })
    except Exception as exc:
        info["torch_error"] = repr(exc)
    for package in ("transformers", "triton"):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", None)
        except Exception as exc:
            info[f"{package}_error"] = repr(exc)
    smi = run_text([
        "nvidia-smi",
        "--query-gpu=driver_version,name,compute_cap,memory.total",
        "--format=csv,noheader",
    ])
    if smi:
        info["nvidia_smi"] = smi.splitlines()
    return info


def deterministic_prompt_ids(model: str, num_seqs: int, input_len: int, seed: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    special_ids = {int(x) for x in tokenizer.all_special_ids if x is not None}
    low = 100 if vocab_size > 1000 else 1
    span = max(1, vocab_size - low)
    prompts: list[list[int]] = []
    for seq_idx in range(num_seqs):
        row: list[int] = []
        cursor = (seed + 104729 * (seq_idx + 1)) % span
        while len(row) < input_len:
            token_id = low + ((cursor + 8191 * (len(row) + 1)) % span)
            cursor = (cursor + 65537) % span
            if token_id in special_ids:
                continue
            row.append(int(token_id))
        prompts.append(row)
    payload = json.dumps(prompts, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "prompt_token_ids": prompts,
        "prompt_token_sha256": hashlib.sha256(payload).hexdigest(),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": vocab_size,
        "special_token_ids_excluded": sorted(special_ids),
    }


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def set_seed(seed: int) -> None:
    try:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(item["output_tokens_per_second"]) for item in measurements]
    seconds = [float(item["seconds"]) for item in measurements]
    return {
        "mean_output_tokens_per_second": statistics.fmean(rates) if rates else None,
        "stddev_output_tokens_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "min_output_tokens_per_second": min(rates) if rates else None,
        "max_output_tokens_per_second": max(rates) if rates else None,
        "run_seconds": seconds,
        "mean_seconds": statistics.fmean(seconds) if seconds else None,
    }


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read benchmark JSONL using the repository's one-record-per-line format."""
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def compact_error(stderr: str, stdout: str = "") -> str:
    combined = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
    lines = [line for line in combined.splitlines() if line.strip()]
    if not lines:
        return "unknown error"
    tail = lines[-12:]
    return "\n".join(tail)[-4000:]
