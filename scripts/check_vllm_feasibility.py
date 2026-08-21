"""Read-only vLLM feasibility check for the current benchmark host."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_schema import environment_info, now_utc


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-4000:],
        "stderr": completed.stderr.strip()[-4000:],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = {
        "created_at": now_utc(),
        "environment": environment_info(),
        "python_version_info": list(sys.version_info[:3]),
        "python_executable": sys.executable,
        "disk_repo": shutil.disk_usage(repo)._asdict(),
        "disk_tmp": shutil.disk_usage(Path("/tmp"))._asdict(),
        "existing_vllm_import": run([sys.executable, "-c", "import vllm; print(vllm.__version__)"]),
        "pip_vllm_index": run([sys.executable, "-m", "pip", "index", "versions", "vllm"]),
        "candidate_env": str(repo / ".venv-vllm-bench"),
        "notes": [],
    }
    if sys.version_info >= (3, 13):
        result["notes"].append("Current benchmark Python is 3.13; vLLM wheels commonly target earlier Python versions, so installation may be blocked without a separate compatible interpreter.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
