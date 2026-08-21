import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_serving import (
    build_parser,
    compute_tpot,
    compute_global_timing,
    kv_preflight,
    make_prompts,
    main,
    output_throughput,
    percentile,
    parse_csv_ints,
    timing_summary,
    validate_workload_capacity,
)
from nanovllm.engine.async_engine import AsyncEngine
from nanovllm.engine.scheduler import Scheduler


def test_scheduler_snapshot_tracks_kv_peak_preemption_and_prefix_rates():
    scheduler = object.__new__(Scheduler)
    scheduler.block_manager = SimpleNamespace(
        blocks=[object()] * 10,
        used_block_ids={1, 2, 3},
    )
    scheduler.kv_blocks_peak_used = 7
    scheduler.preemption_count = 2
    scheduler.prefix_cache_requests = 4
    scheduler.prefix_cache_hit_requests = 2
    scheduler.prefix_cache_cached_tokens = 48
    scheduler.prefix_cache_prompt_tokens = 100
    snapshot = scheduler.get_metrics_snapshot()
    assert snapshot["kv_blocks_used"] == 3
    assert snapshot["kv_blocks_peak_used"] == 7
    assert snapshot["kv_usage_peak_ratio"] == 0.7
    assert snapshot["preemption_count"] == 2
    assert snapshot["prefix_cache_hit_rate"] == 0.5
    assert snapshot["prefix_cache_token_hit_rate"] == 0.48


def test_async_engine_snapshot_aggregates_nonterminal_and_terminal_states():
    engine = object.__new__(AsyncEngine)
    engine.max_queue_size = 256
    engine.stream_queue_size = 16
    engine.request_timeout_s = 300.0
    engine._new_requests = asyncio.Queue()
    engine._states = {
        "q": SimpleNamespace(status="queued"),
        "p": SimpleNamespace(status="prefill"),
        "d": SimpleNamespace(status="decode"),
        "f": SimpleNamespace(status="finished"),
        "c": SimpleNamespace(status="cancelled"),
        "e": SimpleNamespace(status="failed"),
    }
    engine.engine = SimpleNamespace(scheduler=SimpleNamespace(get_metrics_snapshot=lambda: {"kv_blocks_used": 0}))
    metrics = engine.get_metrics_snapshot()
    assert metrics["active_requests"] == 3
    assert metrics["queued_requests"] == 1
    assert metrics["prefill_requests"] == 1
    assert metrics["decode_requests"] == 1
    assert metrics["finished_requests"] == 1
    assert metrics["cancelled_requests"] == 1
    assert metrics["failed_requests"] == 1


def test_percentiles_tpot_and_throughput_definitions():
    assert percentile([], 50) is None
    assert percentile([1.0], 99) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert compute_tpot([0.2]) is None
    assert compute_tpot([]) is None
    assert compute_tpot([0.1, 0.3, 0.5]) == pytest.approx(0.2)
    assert timing_summary([None])["p50"] is None
    assert output_throughput(0, 1.0) is None
    assert output_throughput(10, 2.0) == 5.0
    with pytest.raises(ValueError, match="non-decreasing"):
        compute_tpot([0.3, 0.2])


def test_failure_metrics_never_fabricate_throughput():
    assert output_throughput(12, None) is None
    assert output_throughput(12, 0.0) is None
    assert output_throughput(12, -1.0) is None


def test_global_timing_uses_actual_interleaved_submission_and_completion_times():
    records = [
        {"submitted_at_absolute_s": 10.0, "completed_at_absolute_s": 14.0},
        {"submitted_at_absolute_s": 10.5, "completed_at_absolute_s": 18.0},
    ]
    seconds, reason = compute_global_timing(records)
    assert seconds == 8.0
    assert reason is None
    assert compute_global_timing([], False)[0] is None
    assert "no completed" in compute_global_timing([], False)[1]
    assert "failed" in compute_global_timing(records, True)[1]
    reverse = [{"submitted_at_absolute_s": 20.0, "completed_at_absolute_s": 19.0}]
    seconds, reason = compute_global_timing(reverse)
    assert seconds is None and "not later" in reason


def test_prompt_sha_matches_final_prompts_and_changes_with_sharing():
    # Avoid a tokenizer/model dependency while testing the canonical hashing
    # contract by patching the deterministic prompt source.
    import benchmark_serving

    original = benchmark_serving.deterministic_prompt_ids
    benchmark_serving.deterministic_prompt_ids = lambda model, n, length, seed: {
        "prompt_token_ids": [[seed + i * length + j for j in range(length)] for i in range(n)],
        "prompt_token_sha256": "stale",
    }
    try:
        zero = make_prompts("fake", 2, 8, 7, 0.0)
        shared = make_prompts("fake", 2, 8, 7, 0.5)
        repeat = make_prompts("fake", 2, 8, 7, 0.5)
        assert zero["prompt_token_sha256"] == benchmark_serving.prompt_sha256(zero["prompt_token_ids"])
        assert shared["prompt_token_sha256"] == benchmark_serving.prompt_sha256(shared["prompt_token_ids"])
        assert shared["prompt_token_sha256"] == repeat["prompt_token_sha256"]
        assert zero["prompt_token_sha256"] != shared["prompt_token_sha256"]
        assert shared["prompt_token_ids"][0][:4] == shared["prompt_token_ids"][1][:4]
    finally:
        benchmark_serving.deterministic_prompt_ids = original


def test_kv_preflight_capacity_and_diagnostics():
    config = SimpleNamespace(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, intermediate_size=32, vocab_size=100,
                             head_dim=8)
    result = kv_preflight(config, "float16", 1, 0.9, 18000, 20000, 128, 1, 16)
    assert result["ok"]
    assert result["estimated_kv_block_bytes"] > 0
    assert result["theoretical_minimum_utilization_for_one_kv_block"] > 0
    assert "increase gpu_memory_utilization" in result["recommended_action"]
    assert result["verified_smoke_configuration"]["gpu_memory_utilization"] == 0.9
    failed = kv_preflight(config, "float16", 1, 0.01, 9000, 10000, 128, 1, 16)
    assert not failed["ok"]
    for key in ("gpu_memory_utilization", "free_bytes", "total_bytes", "target_bytes",
                "estimated_model_bytes", "estimated_kv_block_bytes", "tensor_parallel_size",
                "theoretical_minimum_utilization_for_one_kv_block", "recommended_action"):
        assert key in failed
    with pytest.raises(ValueError):
        kv_preflight(config, "int8", 1, 0.5, 9000, 10000, 128, 1)
    with pytest.raises(ValueError):
        kv_preflight(config, "float16", 0, 0.5, 9000, 10000, 128, 1)
    with pytest.raises(ValueError):
        kv_preflight(config, "float16", 1, 0.0, 9000, 10000, 128, 1)


def test_preflight_failure_main_skips_engine_and_writes_failed_record(tmp_path, monkeypatch):
    import json
    import benchmark_serving

    constructed = []

    class NeverConstructed:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))
            raise AssertionError("AsyncEngine must not be constructed")

    preflight = {
        "ok": False,
        "reason": "theoretical budget cannot fit one KV block",
        "theoretical_minimum_utilization_for_one_kv_block": 0.72,
        "gpu_memory_utilization": 0.25,
        "free_bytes": 100,
        "total_bytes": 1000,
        "target_bytes": 250,
        "estimated_model_bytes": 900,
        "estimated_kv_block_bytes": 200,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "recommended_action": "increase gpu_memory_utilization or reduce max_model_len/max_num_seqs",
    }
    monkeypatch.setattr(benchmark_serving, "AsyncEngine", NeverConstructed)
    monkeypatch.setattr(benchmark_serving, "run_kv_preflight", lambda args: preflight)
    monkeypatch.setattr(benchmark_serving, "environment_info", lambda: {"cuda_available": False})
    jsonl = tmp_path / "failure.jsonl"
    md = tmp_path / "failure.md"
    rc = main([
        "--model", "fake", "--concurrencies", "1", "--input-lens", "32", "--output-lens", "4",
        "--warmup-runs", "0", "--runs", "1", "--output-jsonl", str(jsonl), "--output-md", str(md),
    ])
    assert rc != 0
    assert constructed == []
    record = json.loads(jsonl.read_text().splitlines()[0])
    assert record["status"] == "failed"
    assert record["ttft_s"] == {"p50": None, "p95": None, "p99": None}
    assert record["tpot_s"] == {"p50": None, "p95": None, "p99": None}
    assert record["e2e_latency_s"] == {"p50": None, "p95": None, "p99": None}
    assert record["global_end_to_end_seconds"] is None
    assert record["global_output_tokens_per_second"] is None
    assert record["cleanup"]["not_started"] is True
    assert "theoretical" in record["failure"] and "increase gpu_memory_utilization" in record["failure"]
    assert "failed" in md.read_text()


def test_preflight_passed_runtime_allocation_failure_has_no_metrics(monkeypatch):
    import asyncio
    import benchmark_serving

    class RuntimeFailEngine:
        def __init__(self, *args, **kwargs):
            raise AssertionError("allocate_kv_cache: num_kvcache_blocks <= 0")

    args = SimpleNamespace(
        model="fake", max_model_len=128, max_num_seqs=1, max_num_batched_tokens=64,
        gpu_memory_utilization=0.9, dtype="float16", tensor_parallel_size=1,
        enforce_eager=True, max_queue_size=256, request_timeout_s=300.0,
        stream_queue_size=16, seed=1, output_lens=[4],
    )
    monkeypatch.setattr(benchmark_serving, "AsyncEngine", RuntimeFailEngine)
    monkeypatch.setattr(benchmark_serving, "_cuda_mem_info", lambda: {"runtime_free_bytes": 10, "runtime_total_bytes": 20})
    monkeypatch.setattr(benchmark_serving, "_cuda_memory", lambda: {"cuda_peak_allocated_bytes": None})
    result = asyncio.run(benchmark_serving.run_once(args, [[1] * 32], 0, False, {
        "ok": True, "gpu_memory_utilization": 0.9,
        "theoretical_minimum_utilization_for_one_kv_block": 0.12,
    }))
    assert result["status"] == "failed"
    assert "preflight passed; runtime allocation failed" in result["failure"]
    assert result["runtime_memory"]["runtime_free_bytes"] == 10
    assert result["global_end_to_end_seconds"] is None
    assert result["global_output_tokens_per_second"] is None


@pytest.mark.parametrize("raw", ["", "1,,2", "1,1", "0,2", "a,2"])
def test_csv_parser_rejects_empty_negative_duplicate_and_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_csv_ints(raw, "concurrencies")


def test_workload_capacity_validation_covers_generated_tokens_and_block_alignment():
    assert validate_workload_capacity(4096, [128, 2048], [64]) == {
        "required_model_len": 2112, "aligned_required_model_len": 2112,
    }
    with pytest.raises(ValueError, match="too small.*use at least 2112"):
        validate_workload_capacity(2048, [2048], [64])
    with pytest.raises(ValueError, match="divisible.*use at least 2112"):
        validate_workload_capacity(2114, [2048], [64])



def test_cli_parser_rejects_invalid_capacity_values():
    parser = build_parser()
    with pytest.raises(SystemExit):
        # argparse itself handles malformed integer syntax; main performs the
        # positive-value validation after parsing.
        parser.parse_args(["--model", "x", "--runs", "-1"])
