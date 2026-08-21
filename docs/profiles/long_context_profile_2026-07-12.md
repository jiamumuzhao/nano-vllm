# Long-Context Profiling - 2026-07-12

This report profiles Nano-vLLM only. It explains the Nano-vLLM side of the strict offline matrix; it does not profile vLLM internals and does not claim any optimization was implemented.

Raw per-workload profiles:

- [nano_profile_eager_input128_seqs8.md](nano_profile_eager_input128_seqs8.md) / [nano_profile_eager_input128_seqs8.json](nano_profile_eager_input128_seqs8.json)
- [nano_profile_graph_input128_seqs8.md](nano_profile_graph_input128_seqs8.md) / [nano_profile_graph_input128_seqs8.json](nano_profile_graph_input128_seqs8.json)
- [nano_profile_eager_input2048_seqs8.md](nano_profile_eager_input2048_seqs8.md) / [nano_profile_eager_input2048_seqs8.json](nano_profile_eager_input2048_seqs8.json)
- [nano_profile_graph_input2048_seqs8.md](nano_profile_graph_input2048_seqs8.md) / [nano_profile_graph_input2048_seqs8.json](nano_profile_graph_input2048_seqs8.json)

## Environment

- Python: `/opt/anaconda3/bin/python`
- Git revision: `5fc5453e1245fbd31f3906ca6a618f8276690c1b`; dirty: `True`
- PyTorch: `2.11.0+cu130`; CUDA runtime: `13.0`; Triton: `3.6.0`; Transformers: `5.12.1`
- GPU 0: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11337531392` bytes
- GPU 1: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11347623936` bytes

## Workloads

- Model: `/root/huggingface/Qwen3-0.6B`
- Modes: `eager`, `graph`
- Workloads: short `num_seqs=8,input_len=128,output_len=64`; long `num_seqs=8,input_len=2048,output_len=64`
- Dtype: `float16`; tensor parallel size: `1`; max model len: `4096`; max batched tokens: `32768`; GPU memory utilization: `0.5`
- Warmup prompt mode: `same`, matching the strict benchmark's prefix-cache behavior.
- Torch profiler captured the first 12 generation steps after warmup.

## Metrics

| Mode | input_len | Prefill tokens | Prefill sec | Prefill tok/s | Decode tokens | Decode sec | Decode tok/s | Decode step mean ms | Decode step p95 ms | Max allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 128 | 1024 | 0.0986 | 10380.35 | 504 | 2.2142 | 227.62 | 35.147 | 35.491 | 3.544 |
| graph | 128 | 1024 | 0.0994 | 10302.04 | 504 | 0.2904 | 1735.30 | 4.610 | 4.702 | 3.552 |
| eager | 2048 | 2048 | 2.2105 | 926.49 | 504 | 2.2099 | 228.07 | 35.077 | 35.379 | 3.571 |
| graph | 2048 | 2048 | 2.2530 | 909.02 | 504 | 0.6353 | 793.32 | 10.084 | 12.131 | 3.579 |

Note: Nano-vLLM emits the first sampled token during the prefill step. The `decode_tokens` column therefore counts the following decode steps: `63 * num_seqs = 504` tokens.

## Key CUDA Events

| Mode | input_len | varlen prefill ms | paged prefill ms | paged decode ms |
| --- | ---: | ---: | ---: | ---: |
| eager | 128 | 72.39 | - | 36.98 |
| graph | 128 | 72.95 | - | 35.73 |
| eager | 2048 | - | 2155.05 | 311.09 |
| graph | 2048 | - | 2191.89 | 360.21 |

## Relation To Matrix

- Strict matrix short context (`input_len=128,num_seqs=8`): Nano Graph `1405.57 tok/s`, vLLM `1393.55 tok/s`.
- Strict matrix long context (`input_len=2048,num_seqs=8`): Nano Graph `174.06 tok/s`, vLLM `310.18 tok/s`.
- The Nano profile shows the long graph workload spends most of the profiled CUDA time in paged-prefix prefill (`_paged_prefill_kernel`) and also raises paged decode latency from about `4.61 ms` to `10.08 ms` per decode step.

## Verified Facts

- The profiled Nano-vLLM runs use the same prompt during warmup and measurement, matching the strict offline benchmark's prefix-cache behavior.
- For input_len=2048,batch=8, the measured Nano-vLLM run schedules 2048 prefill tokens after warmup, consistent with reusing most full prompt blocks from prefix cache and recomputing one 256-token block per sequence.
- In graph mode, input_len=2048 increases synchronized decode step mean latency from about 4.61 ms to about 10.08 ms.
- In graph mode, the long-context profile's top CUDA events include about 2191.89 ms in _paged_prefill_kernel and about 360.21 ms in _decode_paged_kernel for the profiled steps.

## Code-Structure Inferences

- The paged decode Triton grid is (batch, query_head). Each program handles one query head for one sequence and loops over all logical KV blocks, then over BLOCK_KV tiles inside each block.
- Because each decode program scans the full current context for its sequence, paged decode attention work and memory traffic grow roughly linearly with context length for fixed batch and output length.
- CUDA Graph removes launch overhead and Python dispatch from decode, but it does not change the per-token paged attention scan length.

## Unverified Hypotheses

- vLLM's advantage at input_len=2048 likely comes from more parallel long-context attention/paged-prefill kernels and scheduling implementation details, but vLLM kernels were not profiled in this phase.
- Split-KV or partitioned-softmax decode should reduce long-context decode latency by parallelizing the KV scan, but this has not been implemented or measured in Nano-vLLM.

## Optimization Priority

1. Prototype split-KV / partitioned-softmax paged decode and compare against the current one-program-per-(sequence,head) decode kernel.
2. Tune block/tile configuration and decode parallelism for long context on compute capability 7.5.
3. Profile and optimize paged prefill/block-table access for prefix-cache continuation workloads.
4. Only pursue other directions after profiler evidence shows they dominate the long-context path.

No optimization from this list has been implemented in this phase.

## Commands

### eager input_len=128

```bash
/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode eager --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20
```

### graph input_len=128

```bash
/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode graph --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20
```

### eager input_len=2048

```bash
/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode eager --num-seqs 8 --input-len 2048 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20
```

### graph input_len=2048

```bash
/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode graph --num-seqs 8 --input-len 2048 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20
```

