---
layout: post
title:  "From Math to GPU Metal"
date:   2026-02-11 12:00:00 -0500
category: professional
math: true
---

<!--
=== DRAFT STATUS & CONTEXT FOR FUTURE WORK ===

Series: This is Post 4 of 4 on LLM internals.
  - Post 1 (published): /how-attention-actually-works
  - Post 2 (published): /how-llms-generate-text
  - Post 3 (draft): /how-flash-attention-works (algorithm, numpy)
  - Post 4 (this draft): GPU architecture + Triton implementation

Source material: part4_gpu_metal.ipynb, attention_talk.ipynb

MAJOR GAPS:
1. WRITING STYLE: Uses AI-sounding prose ("paradigm shift", "the main event",
   "workhorse"). Needs rewrite in simple, direct voice. Short sentences.

2. CODE BLOCKS WITH OUTPUT: The Triton code (vector add, tiled matmul, fused
   attention) is here but:
   - The benchmarks are fabricated numbers, not from actual runs.
     Either run them for real or remove them.

3. TABLE OF CONTENTS: Needs a manual TOC at the top.

4. The Triton code requires a GPU to actually run. Add a note about this —
   readers can't just paste into a REPL like Posts 1-3.

5. The tiled matmul ASCII diagram could use a walkthrough with actual
   small matrices and numbers.

6. The "Summary" section is too terse. Match Post 1's "Recap" style.

7. Connect back to Part 3's numpy Flash Attention — show how the same
   algorithm maps to GPU hardware (SRAM/HBM).
===
-->

This is the fourth post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works), [Part 2 covered generation](/how-llms-generate-text), [Part 3 covered the Flash Attention algorithm](/how-flash-attention-works). Now: the GPU hardware that makes it fast.

The code in this post uses [Triton](https://triton-lang.org/) and requires a CUDA GPU to run.

---

## GPU architecture

The marketing line is "a GPU has 10,000 cores." That's misleading — it counts individual ALUs as "cores." By that logic, a CPU with AVX-512 has 16 FP32 "cores" per physical core.

The GPU's actual execution unit is the **SM** (Streaming Multiprocessor, NVIDIA) or **CU** (Compute Unit, AMD). An A100 has 108 SMs. Each SM has its own registers, shared memory (SRAM), and warp schedulers. A **warp** is 32 threads executing the same instruction in lockstep — conceptually similar to CPU SIMD, but you write scalar code per thread and the hardware groups them automatically.

When one warp stalls on a memory load, the SM switches to another. GPUs hide latency by always having more work ready, not with big caches like CPUs.

### The memory hierarchy

This is what matters for Flash Attention:

```
Registers    ~256 KB / SM    ~19 TB/s    (per-thread, fastest)
SRAM         ~164 KB / SM    ~19 TB/s    (shared across threads on one SM)
HBM          40-80 GB        ~2 TB/s     (global GPU memory — "VRAM")
```

SRAM is ~10x faster than HBM. Standard attention writes the full (n, n) score matrix to HBM, reads it back for softmax, writes the weights, reads them back for `@ V`. Flash Attention keeps score tiles in SRAM and never writes the full matrix to HBM.

A GPU "kernel" is a function that runs on SMs. You launch many copies simultaneously, each processing a different chunk of data.

---

## Hello World: Vector Addition in Triton

Before attention, the programming model. CPU vs GPU:

```python
# CPU: process one at a time
def add_cpu(x, y):
    result = np.empty(len(x))
    for i in range(len(x)):
        result[i] = x[i] + y[i]
    return result

# GPU: split across parallel workers
def add_gpu(x, y, num_workers=4):
    result = np.empty(len(x))
    chunk_size = len(x) // num_workers
    for worker_id in range(num_workers):  # these run SIMULTANEOUSLY
        start = worker_id * chunk_size
        end = start + chunk_size
        result[start:end] = x[start:end] + y[start:end]
    return result
```

In Triton, this becomes:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)               # which worker am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                # don't read past the end

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

GPU wins grow with data size — more data means more parallelism to exploit.

---

## Tiled Matrix Multiplication

Matmul is the core operation in attention (`Q @ K.T`, `weights @ V`). The naive approach loads each row and column from HBM for every output element — massive redundant traffic.

**Tiled matmul** loads blocks into SRAM and reuses each element many times:

```
         B (K × N)
        ┌─────────────┐
        │   │ B tile  │
    A   ├───┼─────────┤
  (M×K) │A  │  C tile │ → C (M × N)
        │tile         │
        └───┴─────────┘
```

Each element of A is used BLOCK_N times. Each element of B is used BLOCK_M times. This is the core of why GPUs are fast for matmul — memory reuse through tiling.

---

## Fused Attention with Online Softmax

The full Flash Attention kernel. Same algorithm as [Part 3's numpy simulation](/how-flash-attention-works), but running on a GPU where the SRAM/HBM distinction is real. The Q block lives in registers/SRAM for the entire inner loop. K/V blocks are streamed from HBM one tile at a time. Score tiles are computed and consumed in SRAM — never written to HBM.

```python
# Online softmax accumulation — same as Part 3, now in Triton
for each K,V block:
    scores = q_block @ k_block.T * scale
    m_new = max(m_old, max(scores))           # update running max
    alpha = exp(m_old - m_new)                # rescaling factor
    p = exp(scores - m_new)                   # new contributions
    l = alpha * l + sum(p)                    # update running sum
    acc = alpha * acc + p @ v_block           # accumulate output

output = acc / l  # final normalization
```

When we see a new block with a higher max, we rescale the old accumulator by `exp(old_max - new_max)`. The result is mathematically exact — same as computing the full attention matrix, but using O(n) memory instead of O(n²).

---

## Summary

1. **GPU SMs** are the real execution units — not the marketing "10,000 cores"
2. **SRAM vs HBM** bandwidth gap (~10x) is the entire motivation for Flash Attention
3. **Tiled matmul** avoids redundant memory loads — each element reused many times
4. **Fused attention** keeps score tiles in SRAM, never writes the full matrix to HBM
5. The result: exact attention with O(n) memory, making long contexts practical
