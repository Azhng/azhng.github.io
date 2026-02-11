---
layout: post
title:  "How LLMs Generate Text (And Why It's Expensive)"
date:   2026-02-11 12:00:00 -0400
category: professional
math: true
---

<!--
=== DRAFT STATUS & CONTEXT FOR FUTURE WORK ===

Series: This is Post 2 of 3 on LLM internals.
  - Post 1 (published): /how-attention-actually-works
  - Post 2 (this draft): generation, KV caching, VRAM costs
  - Post 3 (draft): /flash-attention-from-math-to-gpu-metal

Source material: ../attention_talk.ipynb (Jupyter notebook with the original talk)

MAJOR GAPS:
1. WRITING STYLE: This draft was generated with AI-sounding prose. Post 1 went
   through a full rewrite to use simple, direct, human language — short sentences,
   no sophisticated phrases, no "critical insight" / "emerges from" / "paradigm"
   type wording. This draft needs the same treatment. Read Post 1 for the target
   voice.

2. CODE BLOCKS WITH OUTPUT: Post 1 includes runnable Python code with copy-paste
   outputs (notebook style). This draft has almost no runnable code. Need to add:
   - A simple autoregressive generation loop (numpy, ~10 lines)
   - KV cache demo showing the speedup
   - VRAM calculation code so readers can plug in their own model sizes

3. MATH RENDERING: Added `math: true` to frontmatter but the LaTeX in this draft
   hasn't been tested. The $$...$$ display math gets converted to \[...\] by
   kramdown — MathJax config in _includes/head.html handles both. Inline math
   uses $...$. Variable references in prose should use backticks: `d_k`, `Q`, etc.

4. TABLE OF CONTENTS: Post 1 has a manual TOC at the top linking to each section.
   This draft needs one too.

5. COLLAPSIBLE SECTIONS: Any heavy math proofs should go in <details> blocks.
   Use this pattern (the <div markdown="1"> is required for kramdown to process
   the markdown inside the HTML block):
     <details>
     <summary>Plain text summary here</summary>
     <div markdown="1">
     markdown and $$LaTeX$$ content here
     </div>
     </details>

6. The "Claude Code spiraling" example (lines 33-37) is good insight but may
   need rephrasing to not sound like marketing copy.

7. The VRAM table numbers should be double-checked against actual model specs.
===
-->

This is the second post in a series on LLM internals. [Part 1 covered attention](/how-attention-actually-works) — how tokens look at each other and share information. Now: how that machinery produces text, one token at a time, and why long contexts cost so much.

---

## The Language Modeling Objective

The model is trained to predict:

$$P(x_1, x_2, ..., x_n) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdot ... \cdot P(x_n|x_1,...,x_{n-1})$$

At each position: predict the next token given all previous tokens. That's it. Everything else — reasoning, coding, conversation — emerges from this.

---

## Autoregressive Generation

Generation works as a loop:
1. Feed all tokens so far through the model
2. Take the probability distribution over next tokens
3. Sample one token
4. Append it to the sequence
5. Repeat

The critical insight: **once a token is generated, it's fixed.** The model conditions on its own outputs. It cannot revise earlier tokens.

This is why Claude Code can "spiral":
1. Early tokens commit to an approach ("Let's use recursion...")
2. All subsequent tokens are conditioned on that commitment
3. If the approach is wrong, the model keeps building on it
4. It's not "stubborn" — it literally cannot revise earlier tokens

---

## KV Caching

Look at the causal attention matrix. When we add a new token, **all previous rows stay the same** — the causal mask guarantees no previous token ever attends to a future one.

So to generate the next token, we only compute **one new row**. Everything above? Cached.

```
Step 0: generate "The"     → compute 1 row
Step 1: generate "cat"     → compute 1 row, reuse "The"'s K,V
Step 2: generate "sat"     → compute 1 row, reuse "The" and "cat"
Step 3: generate "on"      → compute 1 row, reuse everything above
```

### Complexity

| | Without Cache | With Cache |
|---|---|---|
| Per new token | O(n²) recompute all | O(n) attend to cache |
| Memory | O(1) | O(n) for cache |
| Generating n tokens | O(n³) total | O(n²) total |

With caching, generating token t costs O(t) — it attends to t cached keys. Total:

```
1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²)
```

This is the minimum possible. Without caching, we'd redundantly recompute all previous attention at every step: O(n³).

---

## Where Your VRAM Actually Goes

The KV cache formula: **2 (K+V) × num_layers × d_model × dtype_bytes** per token.

Real-world sizes:

| Model | KV per token | At 4K context | At 128K context |
|-------|-------------|---------------|-----------------|
| Llama 3 8B | 0.5 MB | 2 GB | 64 GB |
| Llama 3 70B | 2.6 MB | 10.7 GB | >300 GB |
| GPT-4 scale | 5.9 MB | 24 GB | >700 GB |

This is **on top of model weights** (8B model ≈ 16 GB in fp16, 70B ≈ 140 GB). This is why:
- Long contexts are expensive (KV cache grows linearly with sequence length)
- Serving many users needs massive VRAM (each user has their own KV cache)
- Context window limits exist — it's a VRAM constraint, not a model limitation
- Quantization helps — halving dtype_bytes halves KV cache too

---

## Prefill vs Decode: Why You See "Pause Then Stream"

When you use Claude, there's a pause before tokens start flowing:

1. **Prefill** (the pause): Process your entire prompt at once. Populates the KV cache for all layers. Cost: O(S²).
2. **Decode** (the stream): Generate one token at a time, adding one new row to the KV cache per step. Cost: O(S+t) per token.

The pause = **Time to First Token (TTFT)** = mostly prefill time. The longer your prompt, the longer you wait.

---

## O(n²) Scaling

The quadratic cost is unavoidable in standard attention — every query must look at every key:

```
Attention cost scaling (relative to 8K context):
   8K:     1.0x  █
  16K:     4.0x  ████
  32K:    16.0x  ████████████████
  64K:    64.0x  (off the chart)
 128K:   256.0x
 200K:   625.0x
```

---

## "Lost in the Middle"

Empirically, models retrieve information better from the **beginning** and **end** of context. Middle content gets relatively less effective attention — the "lost in the middle" effect (Liu et al., 2023).

**Practical implication:** Position your most important context at the start or end. Don't bury critical instructions at token 50,000.

---

## Connecting to Observable Behaviors

| Behavior | Mechanism | Mitigation |
|----------|-----------|------------|
| Keeps doing the same wrong thing | Autoregressive: early mistake → all subsequent tokens build on it | Shorter contexts, fresh starts, explicit "reconsider" prompts |
| Seems to forget instructions | Long context → attention dilution. Instructions at position 0 compete with 50K tokens of code | Repeat key instructions, structure context deliberately |
| Hallucinates file contents | Model predicts what would plausibly come next. If file isn't in context, it generates plausible-looking content | Ensure files are actually in context, verify with reads |
| Slower on long contexts | O(n²) attention. 100K context is 156x more expensive than 8K | Use minimum necessary context |

---

## The Mindset Shift

**Stop thinking:** "I'm telling the model things"

**Start thinking:** "I'm shaping the probability distribution over next tokens by controlling what the model attends to"

Next up: [Flash Attention](/flash-attention-from-math-to-gpu-metal) — how we make all of this fast enough to be practical.
