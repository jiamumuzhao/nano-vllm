import os
from urllib.parse import urlparse
from dataclasses import dataclass
import math
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 4
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    distributed_init_method: str = "tcp://localhost:2333"
    shared_memory_name: str = "nanovllm"
    tp_startup_timeout: float = 120.0
    tp_shutdown_timeout: float = 30.0
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    dtype: str = "auto"
    quantization: str = "none"
    split_kv_enabled: bool = True
    split_kv_threshold: int = 1024
    split_kv_partition_size: int = 1024
    split_kv_max_partitions: int = 16
    # New selector API. The legacy bool below remains supported when explicitly set.
    paged_prefill_q_tile_mode: str = "auto"
    paged_prefill_q_tile_parallel: bool | None = None
    paged_prefill_auto_min_batch_size: int = 8
    paged_prefill_auto_min_q_tiles: int = 128
    # False because the latest formal e2e matrix showed no clear parallel gain.
    paged_prefill_auto_parallel_enabled: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.dtype in ("auto", "float16", "bfloat16", "float32")
        parsed_init = urlparse(self.distributed_init_method)
        if parsed_init.scheme != "tcp" or not parsed_init.hostname or parsed_init.port is None:
            raise ValueError(
                "distributed_init_method must be a tcp://host:port endpoint, "
                f"got {self.distributed_init_method!r}"
            )
        if not (1 <= parsed_init.port <= 65535):
            raise ValueError("distributed_init_method port must be in 1..65535")
        for name, value in (("tp_startup_timeout", self.tp_startup_timeout), ("tp_shutdown_timeout", self.tp_shutdown_timeout)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number of seconds")
        if self.quantization not in ("none", "w8a16"):
            raise ValueError(
                f"unsupported quantization={self.quantization!r}: W8A8 remains benchmark-only "
                "and no W8A8 workload has passed the integration gate; "
                "available production options are 'none' and 'w8a16'"
            )
        if self.split_kv_threshold < 1:
            raise ValueError("split_kv_threshold must be positive")
        if self.split_kv_partition_size < 1:
            raise ValueError("split_kv_partition_size must be positive")
        if self.split_kv_partition_size % self.kvcache_block_size != 0:
            raise ValueError("split_kv_partition_size must be divisible by kvcache_block_size")
        if self.split_kv_max_partitions not in (1, 2, 4, 8, 16, 32):
            raise ValueError("split_kv_max_partitions must be one of 1, 2, 4, 8, 16, 32")
        if self.paged_prefill_q_tile_mode not in ("serial", "parallel", "auto"):
            raise ValueError("paged_prefill_q_tile_mode must be serial, parallel, or auto")
        if self.paged_prefill_q_tile_parallel is not None and not isinstance(self.paged_prefill_q_tile_parallel, bool):
            raise ValueError("paged_prefill_q_tile_parallel must be bool or None")
        if not isinstance(self.paged_prefill_auto_parallel_enabled, bool):
            raise ValueError("paged_prefill_auto_parallel_enabled must be a bool")
        if self.paged_prefill_auto_min_batch_size < 1 or self.paged_prefill_auto_min_q_tiles < 1:
            raise ValueError("paged prefill auto thresholds must be positive")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
