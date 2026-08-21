import os
import tempfile
import traceback
from pathlib import Path
from datetime import timedelta

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear


pytestmark = pytest.mark.skipif(
    os.environ.get("NANOVLLM_RUN_TP2") != "1"
    or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="set NANOVLLM_RUN_TP2=1 with at least two CUDA GPUs for real NCCL TP=2 validation",
)


def _check_close(actual, expected):
    # W8A16 int8 weight quantization error; this is intentionally not exact equality.
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=0.25)


def _stage(rank, name):
    print(f"[w8a16-tp2 rank={rank}] {name}", flush=True)


def _worker(rank, init_file):
    initialized = False
    completed = False
    torch.cuda.set_device(rank)
    try:
        _stage(rank, "initializing NCCL")
        dist.init_process_group(
            "nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            device_id=rank,
            timeout=timedelta(seconds=60),
        )
        initialized = True
        _stage(rank, "NCCL initialized")
        torch.manual_seed(20260720)
        dtype = torch.float16

        x = torch.randn(3, 16, device="cuda", dtype=dtype)
        full_q = torch.randn(32, 16, device="cuda", dtype=dtype)
        full_k = torch.randn(16, 16, device="cuda", dtype=dtype)
        full_v = torch.randn(16, 16, device="cuda", dtype=dtype)
        qkv = QKVParallelLinear(16, 8, 4, 2, quantization="w8a16").cuda().half()
        qkv.weight_loader(qkv.weight, full_q, "q")
        qkv.weight_loader(qkv.weight, full_k, "k")
        qkv.weight_loader(qkv.weight, full_v, "v")
        qkv.quantize()
        assert qkv.weight_scale.shape[0] == qkv.weight_int8.shape[0] == 32
        local_qkv = qkv(x)
        gathered = [torch.empty_like(local_qkv) for _ in range(2)]
        dist.all_gather(gathered, local_qkv)
        if rank == 0:
            q_parts = [y[:, :16] for y in gathered]
            k_parts = [y[:, 16:24] for y in gathered]
            v_parts = [y[:, 24:32] for y in gathered]
            actual = torch.cat([
                torch.cat(q_parts, -1),
                torch.cat(k_parts, -1),
                torch.cat(v_parts, -1),
            ], -1)
            _check_close(actual, F.linear(x, torch.cat([full_q, full_k, full_v], 0)))
        _stage(rank, "QKV completed")

        full_gate = torch.randn(24, 16, device="cuda", dtype=dtype)
        full_up = torch.randn(24, 16, device="cuda", dtype=dtype)
        merged = MergedColumnParallelLinear(16, [24, 24], quantization="w8a16").cuda().half()
        merged.weight_loader(merged.weight, full_gate, 0)
        merged.weight_loader(merged.weight, full_up, 1)
        merged.quantize()
        assert merged.weight_scale.shape[0] == merged.weight_int8.shape[0] == 24
        local_merged = merged(x)
        gathered = [torch.empty_like(local_merged) for _ in range(2)]
        dist.all_gather(gathered, local_merged)
        if rank == 0:
            actual = torch.cat([
                torch.cat([y[:, :12] for y in gathered], -1),
                torch.cat([y[:, 12:] for y in gathered], -1),
            ], -1)
            _check_close(actual, F.linear(x, torch.cat([full_gate, full_up], 0)))
        _stage(rank, "Merged completed")

        full_row = torch.randn(20, 32, device="cuda", dtype=dtype)
        bias = torch.randn(20, device="cuda", dtype=dtype)
        row = RowParallelLinear(32, 20, bias=True, quantization="w8a16").cuda().half()
        row.weight_loader(row.weight, full_row)
        row.weight_loader(row.bias, bias)
        row.quantize()
        assert row.weight_scale.shape[0] == row.weight_int8.shape[0] == 20
        full_x = torch.randn(3, 32, device="cuda", dtype=dtype)
        local_x = full_x.chunk(2, -1)[rank]
        actual = row(local_x)
        _check_close(actual, F.linear(full_x, full_row, bias))
        _stage(rank, "RowParallel completed")
        completed = True
        _stage(rank, "normal completion barrier")
    except Exception:
        print(f"[w8a16-tp2 rank={rank}] FAILURE", flush=True)
        traceback.print_exc()
        raise
    finally:
        if initialized and dist.is_initialized():
            if completed:
                dist.barrier()
                _stage(rank, "normal barrier completed")
            # Exception paths intentionally skip barrier so the original error propagates.
            dist.destroy_process_group()
            _stage(rank, "process group destroyed")
            Path(init_file + f".rank{rank}.done").touch()
            # CUDA/NCCL child shutdown can retain a runtime thread after the
            # group is destroyed; successful workers exit directly so mp.spawn
            # cannot wait indefinitely on that unrelated runtime teardown.
            if completed:
                os._exit(0)


def test_w8a16_tp2_nccl_real_shards():
    fd, init_file = tempfile.mkstemp(prefix="nano-vllm-tp2-", suffix=".rendezvous")
    os.close(fd)
    context = None
    try:
        context = mp.spawn(_worker, args=(init_file,), nprocs=2, join=False)
        if not context.join(timeout=60):
            done = [os.path.exists(init_file + f".rank{rank}.done") for rank in range(2)]
            if all(done):
                alive = [(p.pid, p.is_alive(), p.exitcode) for p in context.processes]
                print(f"[w8a16-tp2 parent] all NCCL stages passed; reaping residual runtime processes={alive}", flush=True)
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                context.join(timeout=10)
                return
            alive = [(p.pid, p.is_alive(), p.exitcode) for p in context.processes]
            print(f"[w8a16-tp2 parent] TIMEOUT after 65s; processes={alive}", flush=True)
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            context.join(timeout=10)
            raise TimeoutError(f"TP=2 NCCL test timed out; processes={alive}")
    finally:
        try:
            os.unlink(init_file)
            for rank in range(2):
                os.unlink(init_file + f".rank{rank}.done")
        except FileNotFoundError:
            pass
