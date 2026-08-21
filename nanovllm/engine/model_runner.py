import pickle
import os
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def wait_for_worker_ready_events(ready_events, worker_processes, endpoint, timeout):
    """Wait for all TP workers without an unbounded synchronization primitive."""
    deadline = time.monotonic() + timeout
    while True:
        if all(event.is_set() for event in ready_events):
            return
        diagnostics = [
            {"rank": rank, "pid": process.pid, "is_alive": process.is_alive(),
             "exitcode": process.exitcode, "ready": ready_events[rank - 1].is_set()}
            for rank, process in enumerate(worker_processes, 1)
        ]
        dead = [item["rank"] for item in diagnostics if not item["is_alive"] and not item["ready"]]
        if dead or time.monotonic() >= deadline:
            unready = [item["rank"] for item in diagnostics if not item["ready"]]
            raise RuntimeError(
                f"TP worker readiness timeout: unready_ranks={unready}, workers={diagnostics}, "
                f"endpoint={endpoint}, timeout={timeout}s"
            )
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], ready_event: Event | list[Event] | None = None, worker_processes=None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        # W8A16 is eager-only in the first implementation. Its Triton Linear
        # forward allocates an output tensor, so it must never be captured or replayed.
        self.enforce_eager = self.effective_enforce_eager(config)
        self.quantization_runtime_metadata = self.quantization_metadata(config)
        self.cuda_graph_disabled_reason = (
            "w8a16 eager-only memory-optimization mode: Triton Linear allocates outputs in forward; "
            "none remains the default FP16 performance baseline"
            if config.quantization == "w8a16" else None
        )
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.ready_event = ready_event
        self.worker_processes = worker_processes or []

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_dtype = hf_config.torch_dtype if config.dtype == "auto" else dtype_map[config.dtype]

        torch.cuda.set_device(rank)
        self.model_dtype = model_dtype

        dist.init_process_group(
            "nccl", config.distributed_init_method,
            world_size=self.world_size, rank=rank, device_id=rank,
            timeout=timedelta(seconds=config.tp_startup_timeout),
        )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")
        split_kv_config = dict(
            quantization=config.quantization,
            split_kv_enabled=config.split_kv_enabled,
            split_kv_threshold=config.split_kv_threshold,
            split_kv_partition_size=config.split_kv_partition_size,
            split_kv_max_partitions=config.split_kv_max_partitions,
            paged_prefill_q_tile_mode=config.paged_prefill_q_tile_mode,
            paged_prefill_q_tile_parallel=config.paged_prefill_q_tile_parallel,
            paged_prefill_auto_min_batch_size=config.paged_prefill_auto_min_batch_size,
            paged_prefill_auto_min_q_tiles=config.paged_prefill_auto_min_q_tiles,
            paged_prefill_auto_parallel_enabled=config.paged_prefill_auto_parallel_enabled,
        )
        self.model = Qwen3ForCausalLM(hf_config, split_kv_config)
        # Allocate every layer's Split-KV workspace before warmup and graph capture.
        # Decode only slices these fixed buffers; it never grows or recreates them.
        for module in self.model.modules():
            if hasattr(module, "initialize_split_workspace"):
                module.initialize_split_workspace(min(config.max_num_seqs, 512), torch.device("cuda"))
        load_model(self.model, config.model)
        if config.quantization == "w8a16":
            for module in self.model.modules():
                if hasattr(module, "quantize"):
                    module.quantize()
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.should_capture_cudagraph():
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self._wait_for_workers_ready()
                self.shm = SharedMemory(name=config.shared_memory_name, create=True, size=2**20)
                dist.barrier()
            else:
                self.ready_event.set()
                dist.barrier()
                self.shm = SharedMemory(name=config.shared_memory_name)
                self.loop()

    def _wait_for_workers_ready(self):
        wait_for_worker_ready_events(
            self.ready_event, self.worker_processes,
            self.config.distributed_init_method, self.config.tp_startup_timeout,
        )

    @staticmethod
    def effective_enforce_eager(config: Config) -> bool:
        return config.enforce_eager or config.quantization == "w8a16"

    @staticmethod
    def quantization_metadata(config: Config) -> dict:
        if config.quantization == "none":
            return {
                "quantization": "none",
                "route": "fp16",
                "role": "default FP16 performance baseline",
                "w8a8_production_enabled": False,
            }
        if config.quantization == "w8a16":
            return {
                "quantization": "w8a16",
                "route": "w8a16",
                "role": "explicit weight-memory optimization; not guaranteed faster than FP16",
                "w8a8_production_enabled": False,
            }
        raise ValueError(
            f"unsupported production quantization={config.quantization!r}: W8A8 is benchmark-only; "
            "use none or w8a16"
        )

    def should_capture_cudagraph(self) -> bool:
        return not self.enforce_eager and self.config.quantization == "none"

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        print("Warming up the model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.model_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=self.model_dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]# Tensor must be rectangular, so we pad with -1
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        block_table_rows = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            block_table_rows.append(tuple(seq.block_table))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = block_table_rows if not self.enforce_eager and bs <= 512 else self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def update_graph_block_tables(self, block_table_rows: list[tuple[int, ...]], graph_vars: dict):
        max_rows = len(block_table_rows)
        for row_idx, row in enumerate(block_table_rows):
            if self.graph_block_table_cache[row_idx] == row:
                continue
            old_len = self.graph_block_table_lengths[row_idx]
            row_len = len(row)
            if old_len > row_len:
                graph_vars["block_tables"][row_idx, row_len:old_len].fill_(-1)
            if row_len:
                values = torch.tensor(row, dtype=torch.int32, device="cuda")
                graph_vars["block_tables"][row_idx, :row_len] = values
            self.graph_block_table_cache[row_idx] = row
            self.graph_block_table_lengths[row_idx] = row_len
        for row_idx in range(max_rows, self.graph_block_table_active_rows):
            old_len = self.graph_block_table_lengths[row_idx]
            if old_len:
                graph_vars["block_tables"][row_idx, :old_len].fill_(-1)
                self.graph_block_table_cache[row_idx] = None
                self.graph_block_table_lengths[row_idx] = 0
        self.graph_block_table_active_rows = max_rows

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs].copy_(input_ids, non_blocking=True)
            graph_vars["positions"][:bs].copy_(positions, non_blocking=True)
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs].copy_(context.slot_mapping, non_blocking=True)
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs].copy_(context.context_lens, non_blocking=True)
            if isinstance(context.block_tables, list):
                self.update_graph_block_tables(context.block_tables, graph_vars)
            else:
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)].copy_(context.block_tables, non_blocking=True)
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_logits_for_validation(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        """Return full-vocabulary logits for the TP=1 alignment verifier only.

        The serving path deliberately selects only the last logit of each
        prefill sequence inside ``ParallelLMHead`` and immediately samples it.
        This opt-in validation hook bypasses that sampler and applies the TP=1
        LM head to every valid hidden state.  It does not create a persistent
        logits workspace and is intentionally unavailable for tensor-parallel
        validation, where gathering full logits would change the test setup.
        """
        if self.world_size != 1:
            raise ValueError("run_logits_for_validation supports tensor_parallel_size=1 only")
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        try:
            hidden_states = self.model(input_ids, positions)
            return F.linear(hidden_states, self.model.lm_head.weight)
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        self.graph_block_table_cache = [None] * max_bs
        self.graph_block_table_lengths = [0] * max_bs
        self.graph_block_table_active_rows = 0

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
