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
