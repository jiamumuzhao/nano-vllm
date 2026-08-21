import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_benchmark_regression.py"


def _environment(name="Test GPU", torch="2.5.1"):
    return {
        "gpu_count": 1,
        "gpus": [{"name": name}],
        "cuda_runtime": "12.4",
        "torch": torch,
    }


def _summary(engine="nano_eager", rate=100.0, seconds=1.0, runs=3, status="ok", environment=None, input_len=128, measurements=None):
    return {
        "record_type": "summary",
        "status": status,
        "engine": engine,
        "model": "synthetic-model",
        "dtype": "float16",
        "temperature": 1.0,
        "environment": environment or _environment(),
        "workload": {
            "tensor_parallel_size": 1,
            "num_seqs": 1,
            "input_len": input_len,
            "output_len": 16,
            "max_model_len": 256,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 256,
            "gpu_memory_utilization": 0.5,
        },
        "measurements": measurements if measurements is not None else [
            {"output_tokens_per_second": rate, "seconds": seconds, "run_index": i}
            for i in range(runs)
        ],
    }


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _run(tmp_path, baseline, candidate, *extra):
    base = tmp_path / "baseline.jsonl"
    cand = tmp_path / "candidate.jsonl"
    _write(base, baseline)
    _write(cand, candidate)
    output = tmp_path / "out"
    if output.exists():
        for artifact in output.glob("benchmark_regression_*"):
            artifact.unlink()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(base), str(cand), "--output-dir", str(output), *extra],
        text=True, capture_output=True,
    )
    artifacts = list(output.glob("benchmark_regression_*.json"))
    markdown = list(output.glob("benchmark_regression_*.md"))
    assert artifacts and markdown
    result = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert "| Workload identity |" in markdown[0].read_text(encoding="utf-8")
    return completed, result


def test_gate_passes_and_writes_artifacts(tmp_path):
    completed, result = _run(tmp_path, [_summary()], [_summary(rate=95.0, seconds=1.1)])
    assert completed.returncode == 0
    assert result["status"] == "passed"
    assert result["workloads"][0]["throughput_change_pct"] == -5.0


def test_throughput_and_latency_regressions_fail(tmp_path):
    completed, result = _run(tmp_path, [_summary()], [_summary(rate=80.0, seconds=1.0)])
    assert completed.returncode != 0
    assert any("throughput regression" in error for error in result["errors"])
    completed, result = _run(tmp_path, [_summary()], [_summary(rate=100.0, seconds=1.3)])
    assert completed.returncode != 0
    assert any("latency regression" in error for error in result["errors"])


def test_missing_duplicate_and_insufficient_runs_fail(tmp_path):
    completed, result = _run(tmp_path, [_summary()], [_summary(engine="other")])
    assert completed.returncode != 0
    assert any("missing workload" in error for error in result["errors"])
    completed, result = _run(tmp_path, [_summary(runs=2)], [_summary(runs=2)])
    assert completed.returncode != 0
    assert any("below minimum" in error for error in result["errors"])
    completed, result = _run(tmp_path, [_summary(), _summary()], [_summary()])
    assert completed.returncode != 0
    assert any("duplicate" in error for error in result["errors"])


def test_failed_status_is_not_silently_ignored(tmp_path):
    completed, result = _run(tmp_path, [_summary(status="failed")], [_summary()])
    assert completed.returncode != 0
    assert any("status='failed'" in error for error in result["errors"])


def test_environment_mismatch_requires_explicit_override(tmp_path):
    completed, result = _run(tmp_path, [_summary()], [_summary(environment=_environment(name="Other GPU"))])
    assert completed.returncode != 0
    assert result["strict_comparison"] is False
    completed, result = _run(tmp_path, [_summary()], [_summary(environment=_environment(name="Other GPU"))], "--allow-environment-mismatch")
    assert completed.returncode == 0
    assert result["strict_comparison"] is False
    assert result["allow_environment_mismatch"] is True


def test_invalid_threshold_is_rejected(tmp_path):
    base = tmp_path / "baseline.jsonl"
    cand = tmp_path / "candidate.jsonl"
    _write(base, [_summary()])
    _write(cand, [_summary()])
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(base), str(cand), "--output-dir", str(tmp_path / "out"),
         "--max-throughput-regression-pct", "-1"],
        text=True, capture_output=True,
    )
    assert completed.returncode != 0


@pytest.mark.parametrize("metric, invalid_value, expected_text", [
    ("output_tokens_per_second", float("nan"), "nan"),
    ("output_tokens_per_second", float("inf"), "inf"),
    ("output_tokens_per_second", float("-inf"), "-inf"),
    ("output_tokens_per_second", -1.0, "-1.0"),
    ("output_tokens_per_second", 0.0, "0.0"),
    ("seconds", -1.0, "-1.0"),
    ("seconds", 0.0, "0.0"),
])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_invalid_numeric_measurements_fail_with_diagnostics(tmp_path, metric, invalid_value, expected_text, side):
    measurement = {"output_tokens_per_second": 100.0, "seconds": 1.0, "run_index": 7}
    measurement[metric] = invalid_value
    baseline = [_summary(measurements=[measurement] if side == "baseline" else None)]
    candidate = [_summary(measurements=[measurement] if side == "candidate" else None)]
    completed, result = _run(tmp_path / f"{side}_{metric}_{expected_text}", baseline, candidate)
    assert completed.returncode != 0
    assert result["status"] == "failed"
    assert any("nano_eager" in error and metric in error and "run index 7" in error and expected_text in error for error in result["errors"])
    assert any(metric in error and expected_text in error for error in result["errors"])


@pytest.mark.parametrize("bad_measurement", [
    {"output_tokens_per_second": 100.0},
    {"seconds": 1.0},
    "not-a-measurement",
    17,
    None,
])
def test_missing_or_non_dict_measurements_fail_with_diagnostics(tmp_path, bad_measurement):
    completed, result = _run(
        tmp_path / "invalid_structure",
        [_summary(measurements=[bad_measurement])],
        [_summary()],
    )
    assert completed.returncode != 0
    assert result["status"] == "failed"
    assert any("nano_eager" in error and ("missing" in error or "measurement must be a dict" in error) for error in result["errors"])
    assert any("invalid measurement" in line or "missing" in line or "measurement must be a dict" in line
               for line in (tmp_path / "invalid_structure" / "out").glob("*.md").__next__().read_text(encoding="utf-8").splitlines())
