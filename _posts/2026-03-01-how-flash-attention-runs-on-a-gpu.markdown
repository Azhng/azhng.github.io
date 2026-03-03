---
layout: post
title:  "How Flash Attention Runs on a GPU"
date:   2026-03-01 12:00:00 -0500
category: professional
math: true
---

This is the fourth post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works), [Part 2 covered generation](/how-llms-generate-text), [Part 3 covered the Flash Attention algorithm](/how-flash-attention-works). This post takes Part 3's numpy simulation and puts it on a GPU where the SRAM/HBM distinction is real.

Parts 1-3 were all numpy — paste into a REPL and run. This one is different. The code uses [Triton](https://triton-lang.org/) and needs a CUDA GPU. If you want to follow along, a free Colab GPU runtime works (`Runtime -> Change runtime type -> GPU`).

---

**Contents**
- [GPU architecture](#gpu-architecture)
- [The parallel execution model](#the-parallel-execution-model)
- [Vector addition: the GPU hello world](#vector-addition-the-gpu-hello-world)
- [Why tiling matters: matrix multiplication](#why-tiling-matters-matrix-multiplication)
- [Tiled matmul in Triton](#tiled-matmul-in-triton)
- [From 2D indexing to pointer arithmetic](#from-2d-indexing-to-pointer-arithmetic)
- [Fused attention with online softmax](#fused-attention-with-online-softmax)
- [The attention kernel](#the-attention-kernel)
- [Benchmark: standard vs flash attention](#benchmark-standard-vs-flash-attention)
- [Recap](#recap)

---

## GPU architecture {#gpu-architecture}

We need some GPU background before we can write kernels. I'm not going to do a full architecture deep dive here — that deserves its own post. Just enough intuition to understand why the code looks the way it does.

If you took a computer architecture class, you know how a CPU core works: fetch-decode-execute, ALUs, big caches to hide memory latency. A GPU's **SM** (Streaming Multiprocessor — NVIDIA's term; AMD calls it a Compute Unit) takes a different tradeoff. Take the ALUs, duplicate them 32 times, and give them a single program counter. Those 32 threads executing the same instruction in lockstep are called a **warp**. (Yes, I'm glossing over tensor cores and a bunch of other things.)

An A100 has 108 SMs, each running multiple warps simultaneously. A **kernel** is the function you write that runs on the GPU — when you launch one, the GPU scheduler assigns copies of your program to SMs across the chip. You write it in a higher-level language (CUDA, Triton) and the compiler turns it into the machine code that the SM actually executes.

Here's the part I find most interesting. When a warp stalls — say, waiting on a memory read — the SM doesn't sit there. It switches to another warp that's ready to go. If you've taken an OS class, this sounds like context switching. Same idea, but *way* cheaper. On a CPU, a context switch is expensive because the OS has to save all register state to memory and load the next process's state back in. On a GPU, every warp's registers are already resident in the register file at the same time. There's nothing to save or restore — the SM just points at a different warp's registers and keeps issuing instructions. That's why GPUs hide latency with parallelism instead of big caches: there is always more work ready to go.

(The marketing line "a GPU has 10,000 cores" counts individual ALUs as "cores." By that logic, a CPU with AVX-512 has 16 FP32 "cores" per physical core. The actual unit of execution is the SM.)

### The memory hierarchy

This is what matters for Flash Attention:

```
Registers    ~256 KB / SM    ~19 TB/s    (per-thread, fastest)
SRAM         ~164 KB / SM    ~19 TB/s    (shared across threads on one SM)
HBM          40-80 GB        ~2 TB/s     (global GPU memory — "VRAM")
CPU RAM      hundreds of GB  ~50 GB/s    (GPU's view — bottlenecked by PCIe)
```

The gap that matters: on-chip memory (~19 TB/s) vs HBM (~2 TB/s). Roughly 10x.

### Why this matters for attention

Recall the attention formula from [Part 1](/how-attention-actually-works):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

That `Q @ K.T` produces a `(seq_len, seq_len)` score matrix. In [Part 3](/how-flash-attention-works) we showed this is O(n²) memory — at `n = 4096`, that's 16 million elements, about 32 MB in FP16.

Now think about where that matrix lives. A naive implementation **materializes** it — meaning it computes the full score matrix and writes it out to HBM, then reads it back for softmax, writes the weight matrix back to HBM, reads it back again for `@ V`. That's multiple full-matrix round trips through the ~2 TB/s bottleneck. On the order of ~128 MB per head in extra HBM traffic, before you even count Q/K/V/O.

Flash Attention avoids this by keeping score tiles on-chip in SRAM and never writing the full matrix to HBM. Part 3 showed the algorithm with numpy. This post puts it on hardware where the SRAM/HBM gap is a real, physical 10x bandwidth difference.

---

## The parallel execution model {#the-parallel-execution-model}

We are going to use [Triton](https://triton-lang.org/) for the GPU code in this post. Partly because I like reading and writing Python, partly because writing raw C++ CUDA kernels inside a Jupyter notebook is an experience I'd rather not have. (You technically *can* — `nvcc4jupyter` exists — but you end up writing C++ inside Python strings and marshalling data by hand. No thanks.) Triton is also where the momentum is right now in the kernel-writing world, so it's worth getting your hands dirty with.

Triton uses an **SPMD** (Single Program, Multiple Data) model. You write one program that operates on a tile of data, and the runtime launches many copies in parallel.

This is a level above CUDA's **SIMT** model. In CUDA, you write per-thread code and the hardware groups threads into warps. In Triton, you write per-tile code and the compiler handles thread/warp decomposition for you. Same hardware, higher abstraction.

At launch you specify a **grid** — the number of program instances:

```python
grid = (98,)                # launch 98 instances (1D grid)
add_kernel[grid](...)
```

Inside the kernel, each instance calls `tl.program_id(axis=0)` to get its index: 0, 1, 2, ..., 97. This isn't an OS PID — it's just which instance you are within this launch.

Grids can be multi-dimensional. Vector add uses 1D; matmul uses 2D:

```python
grid = (8, 8)                 # 8 tile-rows × 8 tile-cols = 64 instances
pid_row = tl.program_id(0)    # which row of tiles (0-7)
pid_col = tl.program_id(1)    # which col of tiles (0-7)
```

`grid` defines how many instances run. `tl.program_id(axis)` tells each one which it is. Inside each instance, `tl.arange(...)` creates the vector lanes that process individual elements within the tile.

---

## Vector addition: the GPU hello world {#vector-addition-the-gpu-hello-world}

Before we get to anything attention-related, the basic pattern. CPU does elements one at a time. GPU splits them across parallel workers:

```python
# CPU: one element at a time
def add_sequential(x, y):
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = x[i] + y[i]
    return result

# GPU-style: split into chunks, workers run simultaneously
def add_parallel(x, y, num_workers=4):
    result = np.empty_like(x)
    chunk_size = (len(x) + num_workers - 1) // num_workers
    for worker_id in range(num_workers):   # on a GPU, these run SIMULTANEOUSLY
        start = worker_id * chunk_size
        end = min(start + chunk_size, len(x))
        result[start:end] = x[start:end] + y[start:end]
    return result
```

In Triton, this becomes a real GPU kernel:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                        # which worker am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # (BLOCK_SIZE,)
    mask = offsets < n_elements                        # don't read past the end

    x = tl.load(x_ptr + offsets, mask=mask)            # (BLOCK_SIZE,)
    y = tl.load(y_ptr + offsets, mask=mask)            # (BLOCK_SIZE,)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**Why no type annotations on the parameters?** When you call `add_kernel[grid](x, y, ...)` with PyTorch tensors, Triton extracts `.data_ptr()` — a raw memory address. Inside the kernel, `x_ptr` is just a number. There's no shape, no dtype, no tensor metadata. You compute which addresses to read yourself via pointer arithmetic, and `tl.load` gets the dtype from the pointer type that Triton tracks internally. This is why the signature looks untyped — at this level, everything is pointers and integers.

The wrapper launches it:

```python
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    return out
```

Does it work?

```
Results match: True
Launched 98 program instances, each handling up to 1024 elements
```

And here's how CPU vs GPU scales with data size (Colab T4):

```
Vector addition: CPU (NumPy) vs GPU (Triton)
------------------------------------------------------------
  n=     10,000:  CPU    0.004ms  GPU    0.047ms  ->    0.1x
  n=    100,000:  CPU    0.044ms  GPU    0.030ms  ->    1.5x
  n=  1,000,000:  CPU    0.492ms  GPU    0.054ms  ->    9.2x
  n= 10,000,000:  CPU   12.655ms  GPU    0.493ms  ->   25.7x
```

GPU wins grow with data size. More data means more parallelism to fill the SMs. At 10K elements the kernel launch overhead dominates and CPU is faster. At 10M, GPU is ~26x faster.

---

## Why tiling matters: matrix multiplication {#why-tiling-matters-matrix-multiplication}

In [Part 3](/how-flash-attention-works) we tiled the attention computation into small blocks so the full score matrix never had to exist in memory. But we hand-waved over *why* tiling helps beyond memory savings. Now that we have the GPU memory hierarchy in front of us, we can be more concrete.

Attention is dominated by two matrix multiplies — $QK^T$ and $\text{weights} \cdot V$. Matmul is where tiling really pays off, and the reason comes down to how many times you reuse each loaded value.

The naive scalar algorithm loads two values, does one multiply-add, and throws them away. The tiled version loads two small blocks and does a mini matrix multiply — every element in one tile gets multiplied with every corresponding element in the other. Each loaded value is reused many times before eviction.

For `A(rows, inner) @ B(inner, cols)`:

| Approach | Loads per multiply-add | Why |
|---|---|---|
| Scalar (naive) | 2 | Load `A[i,k]` and `B[k,j]`, use once, discard |
| Tiled (block_size=b) | ~2/b | Load `b×b` tiles, each element reused `b` times |

Here are both versions in numpy, counting loads:

```python
def matmul_naive(A, B):
    rows, inner = A.shape
    _, cols = B.shape
    C = np.zeros((rows, cols))
    loads = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                C[i, j] += A[i, k] * B[k, j]
                loads += 2
    return C, loads

def matmul_tiled(A, B, block_size=4):
    rows, inner = A.shape
    _, cols = B.shape
    C = np.zeros((rows, cols))
    loads = 0
    for i0 in range(0, rows, block_size):
        for j0 in range(0, cols, block_size):
            acc = np.zeros((min(block_size, rows-i0), min(block_size, cols-j0)))
            for k0 in range(0, inner, block_size):
                A_tile = A[i0:i0+block_size, k0:k0+block_size]
                B_tile = B[k0:k0+block_size, j0:j0+block_size]
                loads += A_tile.size + B_tile.size
                acc += A_tile @ B_tile
            C[i0:i0+block_size, j0:j0+block_size] = acc
    return C, loads
```

```
Naive:  1,260 memory loads
Tiled:    440 memory loads  (block_size=4)
Ratio:  2.9x fewer loads with tiling
```

Larger tiles = more reuse per load = higher **arithmetic intensity** (FLOPs per byte loaded). GPUs can compute far faster than they can fetch from HBM, so raising arithmetic intensity is what gets you from memory-bound to compute-bound.

---

## Tiled matmul in Triton {#tiled-matmul-in-triton}

Same idea, now on GPU. The grid is 2D — one program instance per output tile:

- `axis=0` → which row of tiles
- `axis=1` → which col of tiles
- Inner loop walks the shared dimension, loading A and B tiles and accumulating into `acc`

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    rows, cols, inner,
    stride_a_row, stride_a_inner,
    stride_b_inner, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # (BLOCK_ROWS,)
    offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)  # (BLOCK_COLS,)

    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

    for k_start in range(0, inner, BLOCK_INNER):
        offs_inner = k_start + tl.arange(0, BLOCK_INNER)        # (BLOCK_INNER,)

        # Build 2D address grids via broadcasting (explained below)
        a_ptrs = (A_ptr
                  + offs_row[:, None] * stride_a_row
                  + offs_inner[None, :] * stride_a_inner)
        b_ptrs = (B_ptr
                  + offs_inner[:, None] * stride_b_inner
                  + offs_col[None, :] * stride_b_col)

        a_mask = (offs_row[:, None] < rows) & (offs_inner[None, :] < inner)
        b_mask = (offs_inner[:, None] < inner) & (offs_col[None, :] < cols)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)   # (BLOCK_ROWS, BLOCK_INNER)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)   # (BLOCK_INNER, BLOCK_COLS)
        acc += tl.dot(a, b)                            # (BLOCK_ROWS, BLOCK_COLS)

    c_ptrs = (C_ptr
              + offs_row[:, None] * stride_c_row
              + offs_col[None, :] * stride_c_col)
    c_mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
    tl.store(c_ptrs, acc, mask=c_mask)
```

```
Results match: True
Grid: 8 x 8 = 64 program instances
```

A couple things to unpack here:

`tl.dot` is the tile-level matrix multiply — the mini-GEMM from the numpy tiled version. `(BLOCK_ROWS, BLOCK_INNER) @ (BLOCK_INNER, BLOCK_COLS)` produces `(BLOCK_ROWS, BLOCK_COLS)`. This is where all the reuse pays off: each loaded element participates in `BLOCK_ROWS` (or `BLOCK_COLS`) multiply-adds instead of just one. When shapes and dtypes line up, Triton can lower `tl.dot` to Tensor Core instructions.

The `other=0.0` in `tl.load` is the fill value for out-of-bounds positions. When the mask is `False` for some elements (tail tiles that go past the end of the matrix), `tl.load` returns `0.0` for those instead of reading garbage memory. Since `0.0` is the identity for addition, the accumulation stays correct.

**Why strides instead of hardcoding layout?** `A.T` in PyTorch doesn't copy data — it swaps the strides. A transposed `(8, 8)` matrix has `stride_row = 1, stride_inner = 8`. By taking strides as kernel parameters, the same kernel handles any memory layout without copies.

---

## From 2D indexing to pointer arithmetic {#from-2d-indexing-to-pointer-arithmetic}

This is the part I had to stare at for a while before it clicked. Inside a kernel there are no tensors — just a base address and flat memory. An `(8, 8)` matrix stored row-major is 64 contiguous floats:

```
Address:  A_ptr+0   A_ptr+1   A_ptr+2  ...  A_ptr+7   A_ptr+8   A_ptr+9  ...
Element:  A[0,0]    A[0,1]    A[0,2]  ...  A[0,7]    A[1,0]    A[1,1]  ...
```

To read `A[i, k]`: `A_ptr + i * stride_row + k * stride_inner`. For row-major `(8, 8)`, `stride_row = 8` and `stride_inner = 1`. So `A[5, 3]` is at `A_ptr + 5*8 + 3*1 = A_ptr + 43`.

**The broadcasting trick.** To load an entire tile, we broadcast row offsets against column offsets to build a 2D grid of addresses. Say `pid_row=1` owns rows 4-7 and we're loading inner columns 0-3:

```
offs_row   = [4, 5, 6, 7]    # shape (4,)
offs_inner = [0, 1, 2, 3]    # shape (4,)

# offs_row[:, None] is (4, 1), offs_inner[None, :] is (1, 4)
# Broadcasting gives a (4, 4) grid of addresses:

a_ptrs = A_ptr + offs_row[:, None] * 8  +  offs_inner[None, :] * 1

       = A_ptr + [[32, 33, 34, 35],      ← addresses for A[4, 0..3]
                  [40, 41, 42, 43],      ← addresses for A[5, 0..3]
                  [48, 49, 50, 51],      ← addresses for A[6, 0..3]
                  [56, 57, 58, 59]]      ← addresses for A[7, 0..3]
```

That's a `(4, 4)` block of pointers — one per element. `tl.load(a_ptrs)` loads all 16 values. Within each row, addresses are contiguous (`32, 33, 34, 35`), so the hardware can **coalesce** them into wide cache-line loads. If `stride_inner` were larger (column-major layout), each element in a row would land on a different cache line — same result, much slower.

**Walking the inner dimension.** One program instance owns a fixed output tile (e.g., `C[4:8, 0:4]`). To compute it, it needs `A[4:8, :] @ B[:, 0:4]` — but it can't load the whole inner dimension at once. So it walks in chunks of `BLOCK_INNER`:

```
Iteration 1 (k_start=0):               Iteration 2 (k_start=4):
offs_inner = [0, 1, 2, 3]              offs_inner = [4, 5, 6, 7]

A_ptr + [[32, 33, 34, 35],             A_ptr + [[36, 37, 38, 39],
         [40, 41, 42, 43],                      [44, 45, 46, 47],
         [48, 49, 50, 51],                      [52, 53, 54, 55],
         [56, 57, 58, 59]]                      [60, 61, 62, 63]]

         A[4:8, 0:4]                             A[4:8, 4:8]
```

And for B — same `offs_inner`, but now it indexes *rows* (the contracted dimension), while `offs_col` indexes columns:

```
Iteration 1 (k_start=0):               Iteration 2 (k_start=4):
offs_inner = [0, 1, 2, 3]              offs_inner = [4, 5, 6, 7]

B_ptr + [[ 0,  1,  2,  3],             B_ptr + [[32, 33, 34, 35],
         [ 8,  9, 10, 11],                      [40, 41, 42, 43],
         [16, 17, 18, 19],                      [48, 49, 50, 51],
         [24, 25, 26, 27]]                      [56, 57, 58, 59]]

         B[0:4, 0:4]                             B[4:8, 0:4]
```

A slides horizontally (same rows, next chunk of the inner dimension). B slides vertically (next chunk of the inner dimension, same columns). Each iteration loads a `(BLOCK_ROWS, BLOCK_INNER)` slice of A and a `(BLOCK_INNER, BLOCK_COLS)` slice of B, does a `tl.dot`, and accumulates into `acc`.

**A note on production kernels.** Our kernel recomputes the address grid from scratch each iteration. This is intentional — you can read any single iteration and see exactly which addresses it loads. Production kernels (including the Triton tutorials) typically slide the pointer grid instead:

```python
# Before the loop (compute once):
a_ptrs = A_ptr + offs_row[:, None] * stride_a_row + offs_inner[None, :] * stride_a_inner

for k_start in range(0, inner, BLOCK_INNER):
    a = tl.load(a_ptrs, ...)
    a_ptrs += BLOCK_INNER * stride_a_inner   # slide the grid
```

Same result, one fewer multiply per iteration. Tiny savings compared to `tl.dot`, but it's a pattern you'll see everywhere in Triton/CUDA code.

---

## Fused attention with online softmax {#fused-attention-with-online-softmax}

Now we put it all together. This is Part 3's causal flash attention algorithm, translated into a Triton kernel. Same math, same running states — but this time the tiles are real SRAM, not numpy arrays pretending to be SRAM.

For each query tile, we keep three running states (same names as Part 3):

- `running_max` — largest score seen so far, per query row
- `running_sum` — denominator accumulator, per query row
- `acc` — numerator accumulator (weighted sum of V tiles)

Per K/V tile update:

$$new\_max = \max(running\_max, \max(\text{scores}))$$

$$rescale = e^{running\_max - new\_max}$$

$$weights = e^{\text{scores} - new\_max}$$

$$running\_sum \leftarrow rescale \cdot running\_sum + \sum weights$$

$$acc \leftarrow rescale \cdot acc + weights \cdot V$$

Final output is `acc / running_sum`. That's the entire point of Part 3's derivation: by breaking softmax apart, the numerator and denominator can be accumulated independently, tile by tile. The full `(seq_len, seq_len)` score matrix never needs to exist.

### Where things live on-chip

The Q tile gets loaded once and stays resident in SRAM for the entire inner loop. K and V tiles are streamed from HBM one at a time — load, use, discard. Score tiles are computed on-chip and never touch HBM.

One thing that confused me coming from a systems background: there's no SRAM annotation in the code. No `__shared__` keyword like CUDA. In Triton, local variables live on-chip by default — registers or shared memory, decided by the compiler. `tl.load` reads from HBM into on-chip, `tl.store` writes back out. Everything in between — `acc`, `running_max`, `running_sum`, `scores`, `weights` — stays on-chip. (With a caveat: if the compiler runs out of registers, it can spill to global memory. In practice, reasonable tile sizes avoid this.)

The tile sizes need to fit in SRAM. At `BLOCK_Q=64`, `BLOCK_KV=64`, `d_k=64`, FP32:

```
q tile:          64 × 64 × 4B = 16 KB  (loaded once, stays resident)
k tile:          64 × 64 × 4B = 16 KB  (streamed per inner loop iter)
v tile:          64 × 64 × 4B = 16 KB  (streamed per inner loop iter)
scores:          64 × 64 × 4B = 16 KB  (computed, used, discarded)
weights:         64 × 64 × 4B = 16 KB  (computed, used, discarded)
acc:             64 × 64 × 4B = 16 KB  (accumulator, stays resident)
running_max/sum: 64 × 4B × 2  =  0.5 KB
                          Total ≈ 96 KB  ← fits in ~164 KB SRAM per SM
```

Bigger tiles means more reuse per HBM load (good), but they exceed SRAM capacity (bad). Smaller tiles means less reuse and underutilized compute. 64 is a common sweet spot.

### About `d_k`

I got confused by this at first. `d_k` is the head dimension — it's NOT a block/tile size. Real models split the full embedding across many heads, so each head only sees a small slice:

| Model | d_model | Heads | d_k = d_model / heads |
|---|---|---|---|
| GPT-2 | 768 | 12 | 64 |
| Llama 3 8B | 4096 | 32 | 128 |
| Llama 3 70B | 8192 | 64 | 128 |

64-128 elements per head. That's small enough to load whole rows without tiling over this dimension. The kernel loads `(BLOCK_Q, d_k)` tiles for Q and `(BLOCK_KV, d_k)` tiles for K/V — no inner loop over the head dimension.

If `d_k` were larger (like the full embedding), we'd need a `BLOCK_HEAD` and another inner loop. But for typical head sizes, the full row fits in SRAM.

---

## The attention kernel {#the-attention-kernel}

{% raw %}
```python
@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_q_seq, stride_q_head,
    stride_kv_seq, stride_kv_head,    # K and V share the same layout
    stride_o_seq, stride_o_head,
    seq_len, scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    d_k: tl.constexpr,
):
    pid_q = tl.program_id(0)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)       # (BLOCK_Q,)
    offs_kv = tl.arange(0, BLOCK_KV)                        # (BLOCK_KV,)
    offs_head = tl.arange(0, d_k)                            # (d_k,)

    # Q tile stays resident for the entire inner loop
    q_ptrs = Q_ptr + offs_q[:, None] * stride_q_seq + offs_head[None, :] * stride_q_head
    q_mask = offs_q[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)              # (BLOCK_Q, d_k)

    # Online softmax running state (same names as Part 3)
    acc = tl.zeros((BLOCK_Q, d_k), dtype=tl.float32)
    running_max = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Causal early-stop: only iterate over KV positions ≤ the last query in this tile.
    # Same as Part 3's range(0, q_end, BLOCK_SIZE).
    q_end = (pid_q + 1) * BLOCK_Q
    for kv_start in range(0, q_end, BLOCK_KV):
        cur_offs_kv = kv_start + offs_kv                     # (BLOCK_KV,)

        # K tile
        k_ptrs = (K_ptr
                  + cur_offs_kv[:, None] * stride_kv_seq
                  + offs_head[None, :] * stride_kv_head)
        k_mask = cur_offs_kv[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)          # (BLOCK_KV, d_k)

        # scores = Q @ K^T, scaled by 1/sqrt(d_k)
        scores = tl.dot(q, tl.trans(k)) * scale               # (BLOCK_Q, BLOCK_KV)

        # Causal mask for tiles straddling the diagonal
        causal_mask = offs_q[:, None] >= cur_offs_kv[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax update
        tile_max = tl.max(scores, axis=1)                     # (BLOCK_Q,)
        new_max = tl.maximum(running_max, tile_max)
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max[:, None])           # (BLOCK_Q, BLOCK_KV)
        running_sum = rescale * running_sum + tl.sum(weights, axis=1)
        running_max = new_max

        # V tile and accumulation
        v_ptrs = (V_ptr
                  + cur_offs_kv[:, None] * stride_kv_seq
                  + offs_head[None, :] * stride_kv_head)
        v_mask = cur_offs_kv[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)           # (BLOCK_KV, d_k)
        acc = rescale[:, None] * acc + tl.dot(weights, v)     # (BLOCK_Q, d_k)

    # Final normalization: numerator / denominator
    acc = acc / running_sum[:, None]                           # (BLOCK_Q, d_k)

    out_ptrs = Out_ptr + offs_q[:, None] * stride_o_seq + offs_head[None, :] * stride_o_head
    out_mask = offs_q[:, None] < seq_len
    tl.store(out_ptrs, acc, mask=out_mask)
```
{% endraw %}

Here's an aha moment I had while working through this. I'd been mentally treating Q as a 1D sequence — each position is "a vector" (the embedding), and you just iterate over positions. But looking at the pointer arithmetic forced me to confront that Q is explicitly a 2D matrix `(seq_len, d_k)`. The kernel indexes into BOTH dimensions: `offs_q` picks which rows (sequence positions) and `offs_head` picks which columns (embedding elements). The scores are a real matrix multiply (`tl.dot(q, tl.trans(k))`), not some element-wise thing. Once I stopped thinking of Q as "a list of opaque vectors" and started seeing it as a 2D grid of numbers, the pointer arithmetic and the shapes all made sense.

A few more pieces that tripped me up:

**`running_sum[:, None]` in the final division.** `running_sum` is `(BLOCK_Q,)` — one sum per query row. `acc` is `(BLOCK_Q, d_k)` — a full 2D tile. The `[:, None]` reshapes to `(BLOCK_Q, 1)` so the division broadcasts across all `d_k` columns. Each query row gets divided by its own softmax denominator.

**Causal early-stop vs causal mask.** Two separate mechanisms working together. The early-stop (`range(0, q_end, BLOCK_KV)`) skips entire K/V tiles that are completely in the future — no point loading them at all. The causal mask handles tiles that _straddle_ the diagonal, where some K positions are past some Q positions within the same tile. Both do the same thing as Part 3's numpy version.

**Why Q-outer, not KV-outer?** The original Flash Attention paper (v1) uses KV as the outer loop — each K/V tile is loaded from HBM once, and the Q tiles stream through. For the forward pass, Q-outer turns out to be better: the running state (`acc`, `running_max`, `running_sum`) stays in SRAM for the entire inner loop. With KV-outer, you'd need to read and write that running state from HBM on every inner iteration — extra round trips. Flash Attention v2 switched to Q-outer for the forward pass, in part for this reason.

**`tl.where(causal_mask, scores, float('-inf'))`** sets future positions to $-\infty$. After `exp()`, those become 0 — zero weight in the softmax, just like Part 3.

**Load/compute ordering.** The kernel interleaves `tl.load` and compute in whatever order reads clearest. In CUDA you'd think carefully about prefetching, double buffering, async copies. In Triton, the compiler handles instruction scheduling — it sees all the loads and compute, reorders them to issue loads early, and overlaps memory latency with arithmetic. Write whatever ordering is clearest; the compiler will reorder aggressively regardless. (That said, extreme differences in staging or register pressure can still affect the generated code — Triton is smart, not magic.)

The wrapper and verification:

```python
def triton_attention(Q, K, V):
    seq_len, d_k = Q.shape
    Out = torch.empty_like(Q)
    BLOCK_Q, BLOCK_KV = 64, 64
    grid = (triton.cdiv(seq_len, BLOCK_Q),)
    scale = 1.0 / (d_k ** 0.5)
    attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        Out.stride(0), Out.stride(1),
        seq_len, scale,
        BLOCK_Q, BLOCK_KV, d_k,
    )
    return Out
```

```
Results match: True
Max difference: 0.000031

Kernel computed causal attention on 256 tokens
without creating a 256x256 score matrix in HBM.
```

Matches the PyTorch reference to ~1e-5. The full `(seq_len, seq_len)` matrix was never allocated.

---

## Benchmark: standard vs flash attention {#benchmark-standard-vs-flash-attention}

I wanted to see how our kernel actually performs. Not just "GPU faster than CPU" (obviously), but against PyTorch's standard attention (unfused, materializes the full n×n score matrix in HBM) and against `F.scaled_dot_product_attention` (SDPA — production flash attention using FlashAttention-2 under the hood).

**Standard attention** is what you get when you write `softmax(Q @ K.T) @ V` in PyTorch. In eager mode, each operation — matmul, softmax, matmul — is a separate CUDA kernel dispatch. The n×n score matrix lives in HBM between steps. No fusion, no tiling.

**SDPA** is PyTorch's built-in that uses FlashAttention-2 (or other optimized backends). Fully fused, warp-specialized, double-buffered — all the things our teaching kernel doesn't do.

Results on a Colab T4:

```
Causal Attention Benchmark
d_k=64, 10 iterations after 3 warmup
     n    scores      standard    our triton          SDPA     cpu numpy    std/SDPA
-------------------------------------------------------------------------------------
   128     0.1MB       0.546ms       0.324ms       0.074ms       1.091ms       7.4x
   256     0.2MB       0.796ms       0.390ms       0.186ms       3.107ms       4.3x
   512     1.0MB       0.368ms       0.355ms       0.114ms      16.915ms       3.2x
  1024     4.0MB       0.311ms       0.539ms       0.211ms      34.609ms       1.5x
  2048    16.0MB       1.178ms       1.440ms       0.502ms             —       2.3x
  4096    64.0MB       4.654ms       5.567ms       1.275ms             —       3.7x
  8192   256.0MB      19.086ms      20.390ms       3.945ms             —       4.8x
 16384  1024.0MB      77.059ms      81.196ms      15.295ms             —       5.0x
 32768  4096.0MB           OOM     338.959ms      92.026ms             —           —
```

This was humbling. A few things jumped out:

**Our kernel doesn't beat standard PyTorch.** At every size, our Triton flash attention is roughly tied with — or slower than — naive PyTorch doing separate matmul and softmax calls. Writing a correct fused kernel is not enough. PyTorch dispatches to cuBLAS (matmul) and cuDNN (softmax), which are hand-tuned by NVIDIA engineers with warp specialization, double-buffering, architecture-specific tuning. Our 64×64 tiles with no pipelining can't compete.

**SDPA (production flash attention) crushes everything.** 5x faster than standard at n=16384. This is what a properly optimized flash attention implementation looks like — FlashAttention-2 with all the tricks: warp specialization (some warps load data while others compute), double-buffering (prefetch the next tile while computing the current one), and tile sizes tuned for specific GPU architectures.

**The OOM at n=32768 is the capacity win.** Standard attention tries to allocate a 4 GB score matrix and dies. Our kernel and SDPA keep running because they never materialize it. Even when our kernel is slower in raw time, it runs on inputs that standard attention physically can't handle.

I verified that standard attention is actually materializing the score matrix by tracking peak GPU memory:

```
n=4096, d_k=64
  Score matrix size:        64.0 MB
  Standard peak alloc:     196.0 MB  ← includes score matrix
  SDPA peak alloc:           2.0 MB  ← no score matrix
```

The gap is real — standard attention allocates the full n×n matrix. SDPA doesn't.

---

## Recap {#recap}

| Part 3 concept | Triton kernel mapping | Why it matters |
|---|---|---|
| Tiling over scores | `program_id`, `BLOCK_Q/BLOCK_KV`, loop over `kv_start` | Work is partitioned into on-chip tiles |
| Online softmax running max/sum | `running_max`, `running_sum`, `rescale`, `weights` | Exact softmax without storing full rows |
| Running numerator accumulation | `acc = rescale * acc + weights @ v` | Fuses softmax and `@ V` in one pass |
| O(n²) intermediate avoidance | No global `scores` or `weights` materialization | Cuts HBM traffic and memory footprint |

The benchmark taught me something I didn't expect going in: understanding the algorithm is the first step, not the last. Our kernel is _correct_ — it computes exact attention without materializing the score matrix, and it handles inputs that standard attention OOMs on. But it's not _fast_. The gap between "I implemented the algorithm" and "I beat cuBLAS" is filled with warp specialization, double-buffering, architecture-specific tile sizes, and years of engineering effort.

The real payoff of flash attention shows in `std/SDPA`: production implementations (FlashAttention-2) that combine the algorithm with GPU-specific optimization get 5x+ over standard attention at long sequences and handle inputs that would otherwise OOM. Our kernel demonstrates the algorithm on real hardware. SDPA demonstrates the engineering.

Part 3 explained _why_ Flash Attention works. This post showed _where_ each piece lives on GPU hardware — and that putting the algorithm on a chip is only the beginning.
