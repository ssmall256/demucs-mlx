# Attention Variants: Causal, Sliding Window, and GQA

This guide covers the three most common attention patterns in modern LLMs and how to implement them as custom Metal kernels. All variants build on the online softmax approach described in `references/attention-kernel-guide.md`.

For production use, `mx.fast.scaled_dot_product_attention` supports a mask parameter that handles causal attention. Custom kernels are useful for non-standard patterns (sliding window, GQA) or when fusing attention with other operations.

## Causal Attention

### What It Is

Causal (autoregressive) attention prevents each position from attending to future positions. The attention matrix has a triangular mask where position `i` can only attend to positions `0..i`.

Used by: GPT, Llama, Mistral, Qwen, and most decoder-only models.

### Implementation Strategy

Instead of masking future positions with `-inf` (which still computes the dot products), **bound the K/V loop** to skip them entirely:

```metal
// Standard attention: iterates over ALL K/V positions
for (uint kv = 0; kv < seq_k; kv++) { ... }

// Causal attention: skip future positions entirely
uint kv_end = min(q_row + 1, seq_k);
for (uint kv = 0; kv < kv_end; kv++) { ... }
```

This saves ~50% of compute for full-sequence processing (training/prefill). For single-token generation (`seq_q=1`), there's no savings since all K/V positions are in the past.

### Grid Setup

Same as standard multi-head attention — 3D grid with Z = head index:

```python
grid = (tg, seq_q, num_heads)
threadgroup = (tg, 1, 1)
```

The kernel uses `threadgroup_position_in_grid.y` as the query row and `threadgroup_position_in_grid.z` as the head index.

## Sliding Window Attention

### What It Is

Each query attends only to the `W` nearest past positions instead of the full context. This limits attention to a local window, reducing compute from O(n^2) to O(n * W).

Used by: Mistral, Mixtral, and sliding-window variants.

### Implementation Strategy

Bound both ends of the K/V loop:

```metal
// WINDOW is a compile-time template parameter
uint kv_start = (q_row >= (uint)(WINDOW - 1)) ? (q_row - WINDOW + 1) : 0;
uint kv_end = min(q_row + 1, seq_k);

for (uint kv = kv_start; kv < kv_end; kv++) { ... }
```

Using a template int parameter for the window size allows the compiler to optimize the loop bounds at compile time:

```python
kernel(
    inputs=[q, k, v],
    template=[("T", q.dtype), ("WINDOW", window_size)],
    ...
)
```

### Performance

Speedup is proportional to `seq_len / window_size`. For a 4096-token sequence with a window of 256, the kernel does ~16x less work than full attention.

## Grouped-Query Attention (GQA)

### What It Is

Multiple query heads share a single key/value head. Instead of having equal numbers of Q, K, and V heads, GQA uses fewer K/V heads to reduce KV cache size and memory bandwidth.

Used by: Llama 2 70B (8:1), Llama 3 (8:1), Qwen (varies), Mistral (8:1).

### Memory Layout

```
Q: (num_q_heads, seq, dim)     # e.g., (32, 512, 128)
K: (num_kv_heads, seq, dim)    # e.g., (4, 512, 128)  — 8:1 ratio
V: (num_kv_heads, seq, dim)    # same as K
```

### Implementation Strategy

The grid dispatches one threadgroup per (query_row, q_head) pair. Inside the kernel, divide the query head index to find the corresponding K/V head:

```metal
uint q_head = threadgroup_position_in_grid.z;
uint kv_head = q_head / Q_PER_KV;  // Q_PER_KV is a template int parameter

// Q indexed by q_head, K/V indexed by kv_head
uint q_base = q_head * seq_q * D + q_row * D;
uint k_head_base = kv_head * seq_k * D;
uint v_head_base = kv_head * seq_k * D;
```

The Python wrapper computes the group ratio and passes it as a template parameter:

```python
q_per_kv = num_q_heads // num_kv_heads
assert num_q_heads % num_kv_heads == 0

kernel(
    inputs=[q, k, v],
    template=[("T", q.dtype), ("Q_PER_KV", q_per_kv)],
    grid=(tg, seq_q, num_q_heads),  # Z = num_q_heads, NOT num_kv_heads
    ...
)
```

### Performance Benefit

GQA reduces K/V memory reads proportional to the group ratio. With an 8:1 ratio, K/V bandwidth is reduced by 8x, which significantly improves decode throughput since attention during generation is memory-bound.

## Combining Variants

Real models combine these patterns:

### GQA + Causal (Llama 2/3)

```metal
uint kv_head = q_head / Q_PER_KV;
uint kv_end = min(q_row + 1, seq_k);  // Causal bound

for (uint kv = 0; kv < kv_end; kv++) {
    // Index K/V by kv_head
    dot += (float)q[q_base + d] * (float)k[kv_head * seq_k * D + kv * D + d];
    ...
}
```

### Sliding Window + GQA (Mistral)

```metal
uint kv_head = q_head / Q_PER_KV;
uint kv_start = (q_row >= (uint)(WINDOW - 1)) ? (q_row - WINDOW + 1) : 0;
uint kv_end = min(q_row + 1, seq_k);

for (uint kv = kv_start; kv < kv_end; kv++) {
    dot += (float)q[q_base + d] * (float)k[kv_head * seq_k * D + kv * D + d];
    ...
}
```

Both combined variants use the same online softmax accumulation — only the loop bounds and K/V indexing change.

## Performance Considerations

| Variant | Compute Savings | When It Helps Most |
|---------|-----------------|-------------------|
| Causal | ~2x for full sequences | Prefill / training |
| Sliding window | seq_len / W | Long sequences (4K+) |
| GQA | Reduces K/V bandwidth by group ratio | Decode (memory-bound) |
| Causal + GQA | Both benefits combined | Llama-style decode |

**Note on single-token generation**: During autoregressive decode (`seq_q=1`), causal masking provides no savings (all positions are past). GQA still helps because it reduces K/V cache reads.

## References

- `references/attention-kernel-guide.md` — Online softmax theory, tiling, memory budget
- `scripts/attention_variants_kernel.py` — Working implementations of all three variants
- `scripts/attention_kernel.py` — Basic single-head attention (simpler starting point)
