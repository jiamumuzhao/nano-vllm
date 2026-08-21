# Triton 核函数解读

文件：`nanovllm/layers/attention.py`

共包含三个 Triton JIT 核函数，分别处理 **KV Cache 写入**、**变长 Prefill**、**分页 Decode**。

---

## 目录

- [1. `store_kvcache_kernel` — KV Cache 存储](#1-store_kvcache_kernel--kv-cache-存储)
- [2. `_varlen_prefill_kernel` — 变长 Prefill Attention](#2-_varlen_prefill_kernel--变长-prefill-attention)
- [3. `_decode_paged_kernel` — 分页 Decode Attention](#3-_decode_paged_kernel--分页-decode-attention)

---

## 1. `store_kvcache_kernel` — KV Cache 存储

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
```

### 网格配置

```python
store_kvcache_kernel[(N,)](
    key, key.stride(0),
    value, value.stride(0),
    k_cache, v_cache,
    slot_mapping,
    D,
)
```

`N = q.size(0)`，即当前 batch 中所有 token 的总数。

### 程序模型

| 概念 | 对应 |
|------|------|
| `tl.program_id(0)` | token 索引（0..N-1） |
| 每个 program 负责 | 将 1 个 token 的 K 和 V 从临时 buffer 写入连续的 KV Cache 块 |

### 逻辑拆解

```
idx = tl.program_id(0)
slot = tl.load(slot_mapping_ptr + idx)   # 获取物理 slot 编号
```

**slot_mapping**: 从逻辑 token idx 到物理 cache slot 的映射表。值为 `-1` 表示「不缓存」（如 prefix 重用场景下 shared prefix 部分的 token）。

```
if slot == -1: return                     # 跳过无需缓存的 token
```

```
# 临时 buffer 指针
key_offsets   = idx * key_stride   + tl.arange(0, D)
value_offsets = idx * value_stride + tl.arange(0, D)

# 物理 cache 指针
cache_offsets = slot * D + tl.arange(0, D)
```

以 stride 方式计算地址，天然支持 N 维张量打平后的偏移。

### 关键约定

- `key.stride(-1) == 1`：最后一维连续（`head_dim` 维度）
- `key.stride(1) == head_dim`：head 维度 stride
- `k_cache.stride(1) == D = num_heads * head_dim`：cache 中每个 slot 存储所有 head 的 K，扁平成 1D

---

## 2. `_varlen_prefill_kernel` — 变长 Prefill Attention

```python
@triton.jit
def _varlen_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    scale, stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
```

### 网格配置

```python
grid = (batch_size, num_heads)
```

`batch_size = cu_seqlens_q.numel() - 1`，即序列个数。

### 程序模型

| 概念 | 对应 |
|------|------|
| `tl.program_id(0)` | seq_id — 序列索引 |
| `tl.program_id(1)` | head_id — 注意力头索引 |
| 每个 program 负责 | 一个序列的一个注意力头的全部 Prefill 计算 |

### 数据布局：1D 拼接

所有序列的 q/k/v 按 token 维度拼接为 1D：

```
序列长度: [7, 3, 11]
cu_seqlens = [0, 7, 10, 21]

q/k/v shape: (21, num_heads, head_dim)
```

每个 program 通过 `cu_seqlens` 确定自己的序列范围：

```python
seq_q_start = tl.load(cu_seqlens_q_ptr + seq_id)
seq_q_end   = tl.load(cu_seqlens_q_ptr + seq_id + 1)
seq_k_start = tl.load(cu_seqlens_k_ptr + seq_id)
seq_k_end   = tl.load(cu_seqlens_k_ptr + seq_id + 1)

seq_q_len = seq_q_end - seq_q_start
seq_k_len = seq_k_end - seq_k_start
prefix_len = seq_k_len - seq_q_len  # 前缀缓存长度
```

> `prefix_len` 用于前缀缓存场景：KV 侧比 Q 侧多的 token 数（已被缓存的 prefix）。

### GQA (Grouped Query Attention)

```python
if N_HEADS == N_KV_HEADS:
    kv_head_id = head_id          # MHA：一一对应
else:
    kv_head_id = head_id // (N_HEADS // N_KV_HEADS)  # GQA：分组共享
```

### Tiling 双层循环

```
外循环: for q_start in range(seq_q_start, seq_q_end, BLOCK_Q):
内循环: for kv_start in range(seq_k_start, seq_k_end, BLOCK_KV):
```

**参数取值**: `BLOCK_Q=32`, `BLOCK_KV=64`。

Q 侧按 32 个 token tiling，KV 侧按 64 个 token tiling。两层的 tile 大小不同：Q tile 小是因为外层需维护在线 softmax 的中间状态（`m_i, l_i, acc`），越大寄存器压力越大；KV tile 可以大一些以充分利用 Tensor Core 的 `tl.dot` 计算吞吐。

### 在线 Safe Softmax（FlashAttention 风格）

```
m_i = full([-inf])      # row-wise max
l_i = zeros()           # row-wise sum of exp
acc = zeros()           # 加权累积输出

for kv_start in range(seq_k_start, seq_k_end, BLOCK_KV):
    scores = tl.dot(q, k.T) * scale
    # ... 应用 causal mask ...
    m_new = max(m_i, rowmax(scores))
    alpha = exp(m_i - m_new)
    p = exp(scores - m_new)
    l_i = l_i * alpha + rowsum(p)
    acc = acc * alpha[:, None] + p @ v
    m_i = m_new

output = acc / l_i
```

核心思想：**不构造完整的 attention matrix**，逐 tile 累积，每次用 `m_new` 修正之前的累积结果。

推导：

1. 标准 softmax：`softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
2. 分块处理时，第 t 块结束后已知的全局最大值为 `m_new = max(m_i, max(scores_t))`
3. 之前块的 `l_i` 和 `acc` 需乘以修正因子 `alpha = exp(m_i - m_new)`，相当于把旧块的指数值从以 `m_i` 为基改为以 `m_new` 为基

### 因果掩码

```python
q_local = offs_q - seq_q_start                # (BLOCK_Q,)
k_local = offs_kv - seq_k_start               # (BLOCK_KV,)
causal_mask = k_local[None, :] <= q_local[:, None] + prefix_len
valid_mask = q_mask[:, None] & kv_mask[None, :]
scores = tl.where(valid_mask & causal_mask, scores, float("-inf"))
```

`prefix_len` 的作用：如果 kv 侧包含 prefix（已缓存的 token），query token 可以 attend 到 `q_local + prefix_len` 范围内的所有 kv token（包括 prefix）。

```
例：prefix_len=4, seq_q_len=3
q_local=[0,1,2], 可 attend 范围 = q_local + 4 = [4,5,6]
k_local=[0,1,2,3,4,5,6]
causal:  k_local <= q_local + 4
  q=0 → k≤4  → attend [0,1,2,3,4]  (prefix全部+第0个新token)
  q=1 → k≤5  → attend [0,1,2,3,4,5]
  q=2 → k≤6  → attend [0,1,2,3,4,5,6] (全部)
```

### Store

```python
tl.store(
    o_ptr + offs_q[:, None] * stride_o_t + head_id * stride_o_h + offs_d[None, :] * stride_o_d,
    o.to(o_ptr.dtype.element_ty),
    mask=q_mask[:, None],
)
```

输出写入对应的 1D 拼接位置。

---

## 3. `_decode_paged_kernel` — 分页 Decode Attention

```python
@triton.jit
def _decode_paged_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    block_tables_ptr, context_lens_ptr,
    scale, stride_q_b, stride_q_h, stride_q_d,
    stride_kc_blk, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_blk, stride_vc_t, stride_vc_h, stride_vc_d,
    stride_bt_b, stride_o_b, stride_o_h, stride_o_d,
    BLOCK_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
):
```

### 网格配置

```python
grid = (batch_size, num_heads)
```

### 程序模型

| 概念 | 对应 |
|------|------|
| `tl.program_id(0)` | batch_id — 样本索引 |
| `tl.program_id(1)` | head_id — 注意力头索引 |
| 每个 program 负责 | 一个样本的一个注意力头的 1 个 token 的 Decode 计算 |

Decode 阶段每个样本只有 **1 个 query token**，所以没有外层的 Q tile 循环。

### 数据结构

**KV Cache**: `(num_blocks, block_size, num_kv_heads, head_dim)`

**block_tables**: `(batch_size, max_blocks_per_seq)`

```
context_lens = [100, 50, 300, 200]
block_tables = [
    [0, 1, -1, ...],    # seq 0: page 0, page 1
    [2, -1, -1, ...],   # seq 1: page 2
    [3, 4, 5, ...],
    ...
]
```

每个 program 先读取实际长度：

```python
seq_len = tl.load(context_lens_ptr + batch_id)
if seq_len <= 0: return
```

计算需要的 page 数量：

```python
num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
```

### 单 Query 处理

```python
q = tl.load(q_ptr + batch_id * stride_q_b + head_id * stride_q_h + offs_d * stride_q_d)  # (D,)
```

与 Prefill 不同，这里只有一个 `(D,)` 向量，而非 `(BLOCK_Q, D)` 矩阵，因此后续 softmax 的状态是标量而非向量：

```python
m_i = float("-inf")      # 标量
l_i = 0.0                # 标量
acc = zeros([D])         # 向量
```

### 两层内循环：Page → Block

```
for blk_idx in range(0, num_blocks):          # 遍历 page
    blk_id = block_tables[batch_id, blk_idx]   # 物理 page 编号
    blk_start = blk_idx * BLOCK_SIZE
    tokens_in_blk = min(BLOCK_SIZE, seq_len - blk_start)

    for t_start in range(0, tokens_in_blk, BLOCK_KV):  # 遍历 page 内的 tile
        kv_len = min(BLOCK_KV, tokens_in_blk - t_start)
```

第一层遍历 page，第二层在每个 page 内按 `BLOCK_KV=64` tiling（只读有效 token，不读 padding）。

### K/V 读取

```python
k = tl.load(
    k_cache_ptr + blk_id * stride_kc_blk
                + offs_t[:, None] * stride_kc_t
                + kv_head_id * stride_kc_h
                + offs_d[None, :] * stride_kc_d,
    mask=kv_mask[:, None], other=0.0,
)
```

关键 stride 说明：
- `stride_kc_blk`：不同 page 之间的步长（`block_size * num_kv_heads * head_dim`）
- `stride_kc_t`：page 内不同 token 的步长（`num_kv_heads * head_dim`）
- `stride_kc_h`：不同 head 之间的步长（`head_dim`）
- `stride_kc_d`：head 内不同维度的步长（`1`）

### 与 Prefill 的区别

**Score 计算**（单 query 向量 vs KV 矩阵）：

```python
# Prefill（Q 也是矩阵）
scores = tl.dot(q, tl.trans(k))    # (BLOCK_Q, BLOCK_KV)

# Decode（Q 是向量）
scores = tl.sum(q[None, :] * k, axis=1)  # (BLOCK_KV,)
```

Decode 时 Q 是 `(D,)` 向量，不需要 `tl.dot`，直接用 `tl.sum(q * k, axis=1)` 等价于 dot product。

**Softmax 更新**（标量 vs 向量）：

```python
# Prefill: 每个 query 独立维护
m_new = tl.maximum(m_i, tl.max(scores, 1))     # (BLOCK_Q,)
p = tl.exp(scores - m_new[:, None])            # (BLOCK_Q, BLOCK_KV)
l_i = l_i * alpha + tl.sum(p, 1)               # (BLOCK_Q,)

# Decode: 仅一个 query
m_new = tl.maximum(m_i, tl.max(scores))        # 标量
p = tl.exp(scores - m_new)                     # (BLOCK_KV,)
l_i = l_i * alpha + tl.sum(p)                  # 标量
acc = acc * alpha + tl.sum(p[:, None] * v, 0)  # (D,)
```

---

## 在线 Safe Softmax 数学推导

| 步骤 | 公式 | 说明 |
|------|------|------|
| 初始化 | $m=-\infty,\ l=0,\ \text{acc}=0$ | | |
| 对新块 $S$ | $m' = \max(m, \text{rowmax}(S))$ | 更新全局最大值 |
| 修正因子 | $\alpha = e^{m - m'}$ | 旧累积值的 rescale 系数 |
| 局部权重 | $p = e^{S - m'}$ | 当前块的指数权重 |
| 更新分母 | $l = l \cdot \alpha + \text{sum}(p)$ | 保证了与全局 softmax 等价：$\sum e^{x - m'} = (\sum e^{x - m}) \cdot e^{m - m'} + \sum e^{S - m'}$ |
| 更新输出 | $\text{acc} = \text{acc} \cdot \alpha + p \cdot V$ | 同上原理 |

所有中间计算在 fp32 中完成，防止精度溢出。

---

## 三个 Kernel 的对比总结

| | `store_kvcache_kernel` | `_varlen_prefill_kernel` | `_decode_paged_kernel` |
|---|---|---|---|
| **阶段** | 存储 | Prefill | Decode |
| **网格维度** | 1D `(N,)` | 2D `(batch, heads)` | 2D `(batch, heads)` |
| **Query 数** | N/A | 多个 (BLOCK_Q) | 1 个 |
| **数据源** | 临时 K/V buffer | 1D 拼接 K/V | 分页 KV Cache |
| **Tiling 层数** | 0 | 2 (Q, KV) | 2 (Page, KV) |
| **核心技术** | slot_mapping 间接寻址 | 在线 safe softmax + 因果掩码 | block_tables 分页读取 |
| **条件分支** | slot == -1 跳过 | CAUSAL 编译期特化 | seq_len <= 0 跳过 |


---

## 4. Paged Prefill Q-tile 串行与并行路径

\`_prefill_paged()\` 保留两条 eager kernel 路径，并由 selector 选择 grid。CUDA Graph、varlen prefill、paged decode 和 Split-KV decode 不使用这个 selector。

### 两种 grid

串行路径（\`_paged_prefill_kernel_serial\`）：

\`\`\`python
grid = (batch_size, num_heads)
\`\`\`

每个 \`(sequence, head)\` program 在 kernel 内循环处理该 sequence 的全部 Q tile，适合短 Q 或 launch 并行度已经足够的 workload。

并行路径（\`_paged_prefill_kernel\`）：

\`\`\`python
max_q_tiles = (max_seqlen_q + BLOCK_Q - 1) // BLOCK_Q
grid = (batch_size, num_heads, max_q_tiles)
\`\`\`

每个 program 只处理一个 Q tile，并在读取 KV cache/block table 前对短 sequence 的额外 \`q_tile_id\` 直接 return。不同 Q tile 的 query rows 独立，不需要跨 tile reduction。两条路径均保留 paged KV block 映射、prefix cache、GQA、causal mask 和 online safe softmax。

### Selector 三种模式

配置项 \`paged_prefill_q_tile_mode\` 支持：

- \`serial\`：强制旧串行路径。
- \`parallel\`：强制 Q-tile 并行路径。
- \`auto\`：按 \`batch_size\`、\`max_seqlen_q\`、\`BLOCK_Q\` 计算 \`q_tile_count=ceil(max_seqlen_q/BLOCK_Q)\`，再按实测阈值选择。

正式 selector 证据是 `docs/benchmarks/qwen3_eager_q_tile_20260720T032305Z.json`（Qwen3-0.6B、float16、RTX 2080 Ti、eager、batch 1/4/8、input 1024/2048/4096、warmup 3、repeats 10），不是 microbenchmark。最新矩阵中 18 个 workload 的并行 E2E tok/s 均未超过串行，因而 auto 默认对所有 workload 安全回退串行；selector 仍计算 `batch_size`、`max_seqlen_q` 和 `q_tile_count=ceil(max_seqlen_q/BLOCK_Q)`，并保留 `batch>=8`/`q_tile_count>=128` 阈值作为未来重新验证后的配置入口。显式 `serial`/`parallel` 始终覆盖 `auto`。

为兼容旧调用，显式传入 \`paged_prefill_q_tile_parallel=True/False\` 会分别映射为强制 \`parallel\`/\`serial\`；因此旧布尔调用行为不改变。 \`paged_prefill_q_tile_parallel=None\` 时使用新的 mode selector。 \`BLOCK_Q\`、\`BLOCK_KV\` 和 \`num_warps\` 的 benchmark-only 覆盖仍不改变默认线上 tile 参数。

### 当前限制

- selector 阈值来自单张 RTX 2080 Ti 上的一个 Qwen3-0.6B eager 正式矩阵，不代表所有 GPU、模型或服务形态。
- 并行路径可能因 launch、寄存器压力或编译变体变慢；正式 Markdown 保留所有 workload 的原始结果和退化结果。
- 当前不在 CUDA Graph capture 中启用 Split-KV；本节 Q-tile selector 也不改变 CUDA Graph decode 行为。

证据生成脚本：

\`\`\`bash
/opt/anaconda3/bin/python scripts/benchmark_qwen3_eager_q_tile.py
/opt/anaconda3/bin/python scripts/benchmark_paged_prefill_q_tile.py
\`\`\`


## 5. W8A16 Linear

### W8A8 experimental benchmark-only path

`w8a8_experimental` is isolated in `nanovllm/kernels/csrc/w8a8_tensorcore.cu` and `nanovllm/kernels/w8a8_tensorcore.py`; it is not connected to `Config`, `ModelRunner`, Qwen3, TP, or production Linear routing. It uses CUDA per-row symmetric activation quantization followed by an SM75 INT8 WMMA GEMM with FP32 accumulation and FP16 scale/bias epilogue. The codegen record `docs/benchmarks/w8a8_tensorcore_codegen_20260722T080138Z.json` contains SASS `IMMA.8816.S8.S8` for both representative shapes. W8A16 remains the current model path.

The W8A8 dispatcher now retains the legacy single-warp path for decode `M=1/4` and selects the benchmark-only 8-warp CTA candidate (`M_tile=32,N_tile=64`) for `M>=16,N>=1024,K>=1024`. The CTA kernel cooperatively reuses shared activation and transposed/dequantized int8-weight tiles, while each warp keeps an independent int32 WMMA accumulator. It is not connected to production routing. The latest codegen record is `docs/benchmarks/w8a8_tensorcore_codegen_20260722T082225Z.json`; both legacy `M=1,N=1024` and CTA `M=16,N=4096` report SASS `IMMA.8816.S8.S8`.

The latest CTA-inclusive benchmark is `docs/benchmarks/w8a8_decode_20260722T082303Z.json`/`.md`. It preserves activation-quantization, legacy GEMM, CTA GEMM, and end-to-end timings separately. On the RTX 2080 Ti, CTA dispatch was selected for `M=16/32/64` and legacy for `M=1/4`; the future integration gate remained false because no workload met both the W8A16 and FP16 thresholds. W8A8 remains benchmark-only.

The latest split-CTA record is `docs/benchmarks/w8a8_tensorcore_codegen_20260722T092029Z.json` and `docs/benchmarks/w8a8_decode_20260722T092147Z.json`/`.md`. Legacy `M=1` and CTA16 `M=16`/CTA32 `M=32` all contain `IMMA.8816.S8.S8`. The B-tile loaders now read contiguous K elements per output channel from `[N,K]` weights and transpose into WMMA row-major shared memory. `auto` has no benchmark-proven whitelist and therefore always selects legacy; CTA16/CTA32 remain explicit benchmark candidates. The latest measurements show CTA32 wins over W8A8 legacy for some large shapes, but every candidate fails the joint W8A16/FP16 future-integration gate.

The CTA32 cross-grid correctness supplement covers `M=32,N=4096`, `M=64,N=1024`, and `M=64,N=4096`, with both bias variants, FP16-rounded activation scales, and a non-default CUDA stream. The operator-level break-even record is `docs/benchmarks/w8a8_break_even_20260723T022734Z.json`/`.md`; it covers flattened prefill-shaped `M=1..2048` for `N=1024/4096`. It reports `no_break_even_observed` for auto, legacy, CTA16, and CTA32 at both N values. These are Linear operator measurements only, not model prefill results.

### cuBLASLt experimental baseline

The isolated `w8a8_experimental_cublaslt_int8_i32` path uses cuBLASLt `CUBLAS_COMPUTE_32I` for INT8×INT8→INT32, with one-time `[N,K]`→`[K,N]` int8 packing and a separate FP16 scale/bias epilogue. `W8A8Workspace` preallocates activation quantization buffers, INT32 accumulator, FP16 output, packed weight, and cuBLASLt byte workspace; descriptors and heuristic plans are prepared before timing. Repeated calls only slice these buffers and preserve their data pointers. The API evidence is `docs/benchmarks/w8a8_cublaslt_api_20260723T024333Z.json`; it records `cublaslt_api_verified`, not custom-extension SASS evidence.

The cuBLASLt break-even record is `docs/benchmarks/w8a8_break_even_20260723T025525Z.json`/`.md`. The latest API evidence is `docs/benchmarks/w8a8_cublaslt_api_20260723T025452Z.json`; it records bound weight identity fields (`data_ptr`, shape, dtype, device, version), stable normal reuse, and rejection of mismatched/in-place-mutated weights. It remains benchmark-only and does not modify `Config`, `ModelRunner`, Qwen3, TP, or W8A16. The tested shapes had no joint W8A16 +10% / FP16 <=10% eligible shape.

Production quantization routing is intentionally fixed outside the experimental kernels: `none` is the default FP16 performance baseline, while explicit `w8a16` is a weight-memory optimization mode. W8A8 has no production or CUDA Graph route; its WMMA/cuBLASLt modules are offline benchmark-only and Config rejects all W8A8 variants. The routing decision is recorded by `scripts/record_quantization_routing_decision.py` and never loaded at runtime.

quantization="w8a16" applies only to the local TP shards of Qwen3 qkv_proj, o_proj, gate_up_proj, and down_proj. It uses per-output-channel symmetric int8 weights:

scale = max(abs(weight), dim=1) / 127
weight_int8 = clamp(round(weight / scale[:, None]), -127, 127)

The original FP16 Linear weight is released after loading and quantization. Activation and scale are accumulated/applied in FP32, with FP16 output and bias. Embeddings, RMSNorm, KV cache, and LM head are not quantized. The Triton path supports 2D/3D FP16/BF16 activation; the explicit fallback uses FP32 matmul with int8 weights and scales without materializing a full dequantized weight. The first release is globally eager-only for W8A16: ModelRunner forces enforce_eager and never calls capture_cudagraph, because the Triton Linear forward dynamically creates its output buffer; W8A16 must not be silently captured or replayed. The real TP=2 test is tests/test_w8a16_tp.py and runs with NANOVLLM_RUN_TP2=1 CUDA_VISIBLE_DEVICES=0,1; it validates local-shard scales and NCCL collectives.


TP=2 W8A16 correctness records are generated by `scripts/record_w8a16_tp2_validation.py`; they validate real NCCL QKV-GQA, MergedColumnParallel, RowParallel bias/all-reduce, and local output-channel scales. The JSON/Markdown records under `docs/benchmarks/w8a16_tp2_validation_<timestamp>.*` are acceptance evidence, not performance measurements.


### W8A16 tile dequantization and codegen status

The W8A16 CUDA source deliberately removes `out_dtype=tl.float32` from `tl.dot`: this lets Triton see explicit FP16 operands instead of forcing a dot output type. The W8A16 CUDA source keeps `weight_int8` and per-output-channel FP16 `weight_scale` resident. Each GEMM tile loads int8 weights, converts only that tile to FP16, applies the output-channel scale, and passes FP16/BF16 activation and weight operands to `tl.dot`; the accumulator remains FP32. No complete FP16/FP32 weight matrix is materialized after model loading.

The independent fresh-cache codegen record for the RTX 2080 Ti / SM75 is `docs/benchmarks/w8a16_codegen_20260721T064347Z.json` (and its persisted PTX/cubin files). It found no `mma.sync`, no FP16 MMA signature, and no `HMMA`; the generated path contains `fma.rn.f32`. Therefore the actual implementation is `fp32_fma_fallback`, not Tensor Core FP16 MMA. The source-level tile conversion must not be treated as proof of MMA lowering. The codegen inspector exits nonzero for this result and future performance records must not claim FP16 MMA until a fresh PTX or SASS check passes.

The path supports 2D/3D activation, tail M/N/K tiles, bias, and zero channels. CPU/non-CUDA or unsupported activation shapes use an explicit FP32 matmul fallback without persistent dequantized weights. W8A16 remains eager-only because its current forward creates output buffers.

### W8A16 CUDA WMMA extension

The prebuilt CUDA extension in `nanovllm/kernels/csrc/w8a16_tensorcore.cu` is the FP16 Tensor Core route used when the input is CUDA FP16 on SM75+. It loads only an int8 weight tile, dequantizes it to FP16 with the per-output-channel scale, and uses WMMA `16x16x16` with an FP32 accumulator. It supports arbitrary M/N/K through zero-filled K tails and masked stores, 2D/3D activations through the Python wrapper, bias, and zero channels; no complete FP16 weight matrix is persistent.

The instruction-level acceptance record is `docs/benchmarks/w8a16_tensorcore_codegen_20260722T020853Z.json`. It reports SASS `HMMA.1688.F32` on RTX 2080 Ti / SM75, so the actual runtime path is `cuda_fp16_tensorcore_wmma`. If the extension is absent, the device is below SM75, the input is BF16/CPU, or execution fails, the route reports an explicit diagnostic and uses `fp32_fma_fallback` or the CPU matmul fallback. CUDA Graph eager-only policy is unchanged.

The C++ binding obtains `at::cuda::getCurrentCUDAStream()` for every WMMA launch. Thus calls made inside a non-default PyTorch stream execute and complete on that caller stream; the extension does not force work onto the default stream or add hot-path device synchronization.

The CTA-tiled implementation uses a 4-warp `M=16,N=64` decode kernel for flattened `M<=16`, and a 16-warp `M=64,N=64` prefill kernel for `M>16`. Warps are assigned by `threadIdx.x/32` to 16x16 WMMA subtiles; cooperative shared-memory A/B tiles are reused across warp rows/columns. Fresh codegen record `docs/benchmarks/w8a16_tensorcore_codegen_20260722T024958Z.json` executes both representative paths and records `HMMA.1688.F32` for each. The tiling microbenchmark is `docs/benchmarks/w8a16_wmma_tiling_20260722T025201Z.json`; on the RTX 2080 Ti the new prefill path improves the large-N cases, while small decode cases are conservatively retained as measured tradeoffs rather than assumed wins.

The production selector now fixes `M<=16` to `cuda_fp16_tensorcore_wmma_decode_legacy`; the CTA decode kernel remains benchmark-only. For prefill, `M=17..255` or dimensions below the threshold use `cuda_fp16_tensorcore_wmma_prefill_legacy`; CTA prefill is selected only when `M>=256`, `N>=1024`, and `K>=1024`. This excludes the measured-slower `M=64,N=1024,K=1024` case. The three-path evidence is in `docs/benchmarks/w8a16_tensorcore_codegen_20260722T031530Z.json` and the selector-aware benchmark is `docs/benchmarks/w8a16_wmma_tiling_20260722T031653Z.json`. This policy is measured on RTX 2080 Ti and is not a universal GPU autotuner.

The isolated 4-warp decode-wide candidate (`cuda_fp16_tensorcore_wmma_decode_wide_candidate`) also has HMMA evidence in `docs/benchmarks/w8a16_tensorcore_codegen_20260722T072129Z.json`, but it failed the production gate: all six required decode P50 comparisons were slower than legacy by 34.78%–44.91%. It remains benchmark-only and does not change the production selector.

### cuBLASLt autotune evidence

`W8A8Workspace.prepare(..., autotune=True)` enumerates 0/1/4/16/64 MiB
workspace budgets, records every row-major heuristic candidate, times
executable candidates with CUDA Events, and selects the lowest measured P50
plan. `autotune=False` retains an explicit row-major baseline. The selected
plan is reused by `linear()` without search, repacking, or allocation. On the
RTX 2080 Ti, every budget returned the same valid `heuristic_index_0`
(`returned_count=1`), which is recorded as five budget candidates rather than
five distinct algorithms. COL32 is an explicit structured skip because this
extension does not yet validate its weight transform and packed-buffer binding.

Latest API evidence: `docs/benchmarks/w8a8_cublaslt_api_20260723T082738Z.json`.
Latest baseline/autotuned comparison:
`docs/benchmarks/w8a8_break_even_20260723T082812Z.json`/`.md`.

The latest COL32 attempt is `docs/benchmarks/w8a8_cublaslt_api_20260723T092526Z.json`.
For `(M,N,K)=(32,1024,1024)` and `(256,4096,1024)`, cuBLASLt created the
row-major source and `CUBLASLT_ORDER_COL32` destination layouts and completed
the transform with `CUBLAS_STATUS_SUCCESS`. The subsequent COL32 heuristic
stage returned zero valid INT8 candidates (`status_code=0`,
`CUBLAS_STATUS_SUCCESS`); therefore no COL32 matmul was executable, and
row-major `heuristic_index_0` remained selected. This is a runtime heuristic
failure, not a static "not validated" skip. The COL32 packed buffers are still
persistent and included in workspace pointer-stability evidence.
