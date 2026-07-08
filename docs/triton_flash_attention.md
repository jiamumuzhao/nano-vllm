# 手写 Triton FlashAttention 说明

## 背景

原始实现中，prefill 阶段对 batch 内每个 sequence 逐条 `for` 循环调 `F.scaled_dot_product_attention`，decode 阶段将 KV cache 填充到 `max_blocks * block_size` 再调 `F.scaled_dot_product_attention`。两个阶段都无法避免如下问题：

- **Prefill**：batch 内序列长度不同时，无法利用 batch 计算，只能逐条处理
- **Decode**：padding 导致大量无效计算和显存浪费（mask 位置也参与 matmul）

## 核心思路

参考 FlashAttention 的做法：**在 kernel 内部处理变长**，不依赖外部 padding 或对齐。

| 阶段 | 方案 | 关键数据结构 |
|------|------|-------------|
| **Prefill** | Varlen Attention | `cu_seqlens` 标记序列边界 |
| **Decode** | Paged Attention | `block_tables` + `context_lens` |

---

## Prefill：Varlen Attention

### 数据布局

batch 内所有 sequence 的 q/k/v 拼成 1D：

```
序列长度: [7, 3, 11]
cu_seqlens = [0, 7, 10, 21]

q/k/v shape: (21, num_heads, head_dim)   # 拼成 1D
```

### Kernel 结构

**网格**: `(batch_size, num_heads)`

每个 program 处理 `(seq_id, head_id)`。内部逻辑：

1. 从 `cu_seqlens_q/ k` 加载当前 sequence 的 q/kv 范围
2. 外循环：按 `BLOCK_Q（=32）` tiling query 位置
3. 内循环：按 `BLOCK_KV（=64）` tiling kv 位置
4. 在线 safe softmax（FlashAttention 风格）

```python
# 伪代码
seq_q_start = cu_seqlens_q[seq_id]
seq_q_end = cu_seqlens_q[seq_id + 1]
seq_k_start = cu_seqlens_k[seq_id]
seq_k_end = cu_seqlens_k[seq_id + 1]

for q_block in range(seq_q_start, seq_q_end, BLOCK_Q):
    m_i = [-inf], l_i = [0], acc = [0]
    for kv_block in range(seq_k_start, seq_k_end, BLOCK_KV):
        scores = q_block @ k_block^T * scale
        # 因果掩码: k_local <= q_local + prefix_len
        # 不同 sequence 之间的 attention 自然隔离
        m_new = max(m_i, rowmax(scores))
        p = exp(scores - m_new)
        acc = acc * exp(m_i - m_new) + p @ v_block
        l_i = l_i * exp(m_i - m_new) + rowsum(p)
        m_i = m_new
    output = acc / l_i
```

### 因果掩码公式

对于第 i 个 sequence：

```
prefix_len = seq_k_len - seq_q_len   # 缓存的 token 数
q_local = q_flat_idx - seq_q_start
k_local = k_flat_idx - seq_k_start
causal_mask = k_local <= q_local + prefix_len
```

- **普通 prefill（无 prefix）**：`prefix_len = 0`，表示标准的 causal mask
- **前缀缓存 prefill**：`prefix_len > 0`，新的 q 可以 attend 到所有缓存的 prefix token

### GQA 处理

```python
kv_head_id = head_id // (num_heads // num_kv_heads)
```

所有 query head 共享同一个 kv head。

---

## Decode：Paged Attention

### 数据布局

KV cache 以 page 方式存储：`(num_blocks, block_size, num_kv_heads, head_dim)`。

每个 sequence 的 page 分配由 block table 记录：

```
context_lens = [100, 50, 300, 200]
block_tables = [
    [0, 1, -1],   # seq 0: 2 pages
    [2, -1, -1],  # seq 1: 1 page
    ...
]
```

### Kernel 结构

**网格**: `(batch_size, num_heads)`

每个 program 处理 `(batch_id, head_id)`。内部逻辑：

1. 从 `context_lens` 获取当前 sequence 的实际长度
2. 计算需要的 page 数量：`ceil(seq_len / block_size)`
3. 遍历每个 page，只加载有效 block（不加载 padding）
4. 在 page 内按 `BLOCK_KV（=64）` tiling

```python
# 伪代码
num_blocks = ceil(context_lens[batch_id] / block_size)
for blk_idx in range(num_blocks):
    blk_id = block_tables[batch_id, blk_idx]   # page 编号
    tokens_in_blk = min(block_size, seq_len - blk_idx * block_size)
    for t_start in range(0, tokens_in_blk, BLOCK_KV):
        k = k_cache[blk_id, t_start:t_start+BLOCK_KV]
        v = v_cache[blk_id, t_start:t_start+BLOCK_KV]
        # 1 个 query 与多个 kv token 做 dot product
        scores = q @ k^T * scale
        # 在线 softmax 累加
```

### 和之前实现的对比

| | 之前 | 之后 |
|---|---|---|
| 读 KV cache | `k_cache[bt_safe]` 读全部 blocks 并 reshape | 只读 `ceil(seq_len/block_size)` 个 block |
| 计算量 | 读 `max_blocks * block_size` 个 token | 读 `seq_len` 个 token |
| 序列长度不同 | 短序列浪费大量计算在 mask 上 | 短序列自然少量计算 |

---

## 在线 Safe Softmax

两个 kernel 都使用 FlashAttention 的 online safe softmax 算法，避免显式构造完整 attention matrix：

```
m_new = max(m_i, max(scores_rowwise))
alpha = exp(m_i - m_new)
p = exp(scores - m_new)       # p ∈ (0, 1]
l_i = l_i * alpha + sum(p)   # 修正分母
acc = acc * alpha + p @ v    # 修正累积输出
m_i = m_new
```

- 所有中间计算在 fp32 中进行，只有 q/k/v 的 load/store 使用 fp16
- `p = exp(scores - m_new)` 保证 `max(p) = exp(0) = 1`，无溢出风险
- `exp(-inf) = 0`：masked position 对 softmax 贡献为 0

---

## Triton 核心理念

1. **不需要手动管理 shared memory**：`tl.dot` 内部处理 matmul tiling
2. **Program 间独立**：不同 `(seq, head)` 的 program 互不干扰
3. **Runtime 循环**：`for ... in range(runtime_value)` 是 Triton 支持的，编译为 predicated PTX loop
4. **Compile-time 特化**：`tl.constexpr` 参数（如 `D`, `BLOCK_Q`, `N_HEADS`）使 kernel 在编译时针对具体配置优化

---

## 依赖

- **Triton ≥ 2.0**（sm75/Turing 支持）
- 不需要 flash-attn 包（无法在 2080 Ti 上安装）
- 不需要 xformers

## 文件

- `nanovllm/layers/attention.py` — 两个 Triton kernel + Attention 包装类
- `store_kvcache` 保持不变（原有的 Triton kernel）
