"""Record the offline decision to keep W8A8 out of production routing."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    candidates = sorted(Path("docs/benchmarks").glob("w8a8_break_even_*.json"))
    if not candidates:
        raise SystemExit("no docs/benchmarks/w8a8_break_even_*.json evidence found")
    source = candidates[-1]
    benchmark = json.loads(source.read_text())
    results = benchmark.get("results", [])
    eligible = [r for r in results if r.get("eligible_for_future_integration") is True]
    gate = benchmark.get("integration_gate", {})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path("docs/benchmarks")
    json_path = outdir / f"quantization_routing_decision_{stamp}.json"
    md_path = outdir / f"quantization_routing_decision_{stamp}.md"
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(source),
        "source_timestamp_utc": benchmark.get("timestamp_utc"),
        "environment": {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
        },
        "test_matrix": benchmark.get("workloads"),
        "integration_gate": gate,
        "w8a8_eligible_shape_count": len(eligible),
        "production_routing_decision": {
            "default_performance_route": "fp16",
            "explicit_memory_route": "w8a16",
            "w8a8_production_enabled": False,
        },
        "conditions_to_reconsider_w8a8": [
            "A reproducible benchmark must pass the existing W8A16/FP16 integration gate.",
            "The passing result must be repeated on the target hardware and workload matrix.",
            "A real model end-to-end validation must be completed after the operator result.",
            "Only then may production routing be reconsidered; this record does not change runtime routing.",
        ],
    }
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(json_path)

    lines = [
        "# Quantization production routing decision", "",
        "This is an offline decision record; runtime routing does not read benchmark JSON.", "",
        f"- Source benchmark: `{source}`",
        f"- W8A8 eligible shapes: **{len(eligible)}**",
        f"- GPU: {data['environment']['gpu']}", "",
        "## Production routing", "",
        "| Decision | Value |", "|---|---|",
        "| Default performance route | `fp16` (`quantization=none`) |",
        "| Explicit memory route | `w8a16` |",
        "| W8A8 production enabled | `false` |", "",
        "W8A16 is an explicit weight-memory optimization mode. The RTX 2080 Ti benchmark does not guarantee decode speedup over FP16.",
        "W8A8 WMMA and cuBLASLt remain benchmark-only; no W8A8 model end-to-end result is claimed.", "",
        "## Reconsideration conditions", "",
    ]
    lines.extend(f"- {condition}" for condition in data["conditions_to_reconsider_w8a8"])
    tmp = md_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(md_path)
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
