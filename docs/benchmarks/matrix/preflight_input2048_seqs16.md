# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:56:28+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl`](preflight_input2048_seqs16.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 16 --input-len 2048 --output-len 4 --warmup-runs 0 --runs 1 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl --output-md docs/benchmarks/matrix/preflight_input2048_seqs16.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `16`
- Input tokens per sequence: `2048`
- Output tokens per sequence: `4`
- Warmup runs: `0`
- Measured runs: `1`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `4096`; max sequences: `16`; max batched tokens: `32768`
- Fixed output length: Nano-vLLM uses `ignore_eos=True`; Hugging Face uses `min_new_tokens=max_new_tokens` and `eos_token_id=None`; vLLM uses `ignore_eos=True` plus `min_tokens=max_tokens`.

## Environment

- Python: `/root/workspace/nano-vllm/.venv-vllm-bench/bin/python` / `3.13.9`
- PyTorch: `2.11.0+cu130`
- CUDA runtime: `13.0`
- Triton: `3.6.0`
- Transformers: `5.12.1`
- Git revision: `5fc5453e1245fbd31f3906ca6a618f8276690c1b`; dirty: `True`
- GPU 0: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11337531392` bytes
- GPU 1: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11347623936` bytes

## Results

| Engine | Mode | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Seeds | JSONL |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| nano_eager | eager | ok | 1.59 | 0.00 | 1.59 | 1.59 | 40.1583 | 11234 | `docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl` |
| nano_graph | graph | ok | 1.60 | 0.00 | 1.60 | 1.60 | 39.9609 | 11234 | `docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl` |
| hf | hf | failed | - | - | - | - | - | - | `docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl` |
| vllm | vllm | ok | 11.04 | 0.00 | 11.04 | 11.04 | 5.7973 | 11234 | `docs/benchmarks/matrix/preflight_input2048_seqs16.jsonl` |

## Failures

### hf

- Stage: `worker_process`
```text
...<7 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/opt/anaconda3/lib/python3.13/site-packages/transformers/integrations/sdpa_attention.py", line 92, in sdpa_attention_forward
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
    ...<6 lines>...
        **sdpa_kwargs,
    )
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 10.56 GiB of which 3.49 GiB is free. Including non-PyTorch memory, this process has 7.06 GiB memory in use. Of the allocated memory 6.78 GiB is allocated by PyTorch, and 97.98 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
