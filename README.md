# 🚀 Qwen MoE Benchmark: Running a 35B Model on 12GB VRAM

> **Verified: 75 tok/s avg, peak 87.5 tok/s** — APEX-MTP + n-cpu-moe on RTX 3060 12GB + 32GB DDR4  
> 30 May – 4 Jul 2026

This repo records experiments verifying `--n-cpu-moe` and MTP (Multi-Token Prediction) on a $300 RTX 3060 12GB card — can a 35B-parameter model really run at usable speeds?

---

## 🧠 TL;DR

- **The trick is real**: Qwen3.6-35B-A3B is MoE — 35B total, only **~3B active per token**
- **--n-cpu-moe 16**: Offloads expert weights to CPU RAM, GPU keeps active path → **62 tok/s**
- **+ MTP**: Self-speculative decoding → **75 tok/s avg, peak 87.5 tok/s**
- **Optimal config**: `-ngl 99 --n-cpu-moe 16 --spec-type draft-mtp --spec-draft-n-max 3`
- **Needs ≥48GB DDR5** for the 130 tok/s claim (Alan Dao)

---

## 📊 Results

### APEX-MTP-Compact (17GB Q4_K) — RTX 3060 + 5800X + 32GB DDR4

| Config | Avg (tok/s) | Peak code | Wall (9 prompts) | Note |
|--------|:----------:|:---------:|:----------------:|------|
| `-ngl 99`, no n-cpu-moe, no MTP | **OOM** | — | — | 17GB model + KV cache > 12GB VRAM |
| `-ngl 0` (full CPU), no n-cpu-moe | **~1-2** | — | — | 35B params on CPU only → unusable |
| `-ngl 99`, n-cpu-moe 16, no MTP | **62.4** | — | 25.3s | Clean baseline |
| `-ngl 99`, **n-cpu-moe 16 + MTP n=3** 🏆 | **75.1** | **86.1** | **21.9s** | **Recommended** |
| `-ngl 99`, n-cpu-moe 16 + MTP n=4 | 75.5 | 87.5 | 21.4s | ~same as n=3 (hw limit) |
| `-ngl 99`, n-cpu-moe 32 + MTP n=3 | 74.8 | 87.2 | 21.4s | |
| `-ngl 99`, -fitt 2560 + MTP n=3 | 73.5 | 85.8 | 21.7s | Alternative offload method |
| `-ngl 99`, -fitt 1536 + MTP n=2 | ~66 | ~73 | ~24.6s | From janvitos benchmark |

> **4.7x speedup? Wrong.** That compares MTP against `-fitt` baseline (15 tok/s).  
> Against the same config (`n-cpu-moe 16`), MTP gives **~1.2x** (62 → 75 tok/s).

**Key takeaway:** Without `--n-cpu-moe`, the 17GB model doesn't fit on 12GB VRAM at all. With it, you get 62 tok/s base and 75 tok/s with MTP.

### Cross-Reference Comparison

| Setup | Gen (tok/s) | ×Colab | Source |
|-------|:-----------:|:------:|-------|
| **RTX 3060 + MTP n=3** ⚡ | **75 / 87 peak** | **~10×** | **This repo** |
| RTX 3060 + n-cpu-moe (Qwen3.6) ⚡ | **54-62** | ~7-8× | This repo |
| RTX 3060 + n-cpu-moe (Qwen3.5) ✅ | **49.9** | **6.6×** | This repo |
| RTX 3060 + MTP (SpecPicks #2) | 80 (Q4_K) | ~11× | [SpecPicks](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-mtp-deep-dive-2026) |
| RTX 4070S + MTP (janvitos) | 76-82 | ~10× | [xiaoxinsoft](https://www.xiaoxinsoft.com/en/qwen35b-mtp-12gb-vram-guide) |
| RTX 3060 + MTP (SpecPicks #1) | 31 (Q3_K) | ~4× | [SpecPicks](https://specpicks.com/reviews/qwen-36-35b-a3b-rtx-3060-12gb-llama-cpp-2026) |
| RTX 3060 + 3700X + 32GB (knightli) | 33-36 | ~4.7× | [knightli](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/) |
| Colab T4 (free) | 7.5 | 1× | This repo |

---

## 📦 Model Used

All benchmarks use the **same base model** — different quant variant sizes:

| Variant | Quant | Size | MTP | Note |
|---------|:-----:|:----:|:---:|------|
| **APEX-MTP-Compact** (this bench) | Q4_K | **17 GB** | ✅ | Q4_K quant, retains MTP heads |
| APEX-MTP-Balanced | Q4_K_M | ~19 GB | ✅ | Higher quality, more VRAM |
| APEX-MTP-Quality | — | ~22 GB | ✅ | Lightest quant, highest quality |
| APEX-MTP-Nano | Q3_K | ~14 GB | ✅ | Smallest with MTP |
| APEX-MTP-non-MTP (mudler) | Q4_K | ~16 GB | ❌ | Without MTP head |

> **17 GB is quantized (Q4_K)** — FP16 original is ~70 GB. MTP head adds ~1 GB.  
> All variants are **Qwen3.6-35B-A3B** (35B MoE, ~3B active/token).

---

## 🧠 Flag Guide (Feynman Style)

### `--n-cpu-moe 16` — MoE Expert Offloading

> An MoE model is like a company with 64 specialists. Each answer only needs the 2 best ones.  
> `n-cpu-moe 16` = "Keep 16 frequently-used specialists at your desk (GPU). The other 48 wait in the warehouse (RAM). When needed, page them in."
>
> GPU = small desk (12GB), RAM = warehouse (32GB). The number 16 = "specialists within arm's reach" — fast enough, doesn't overflow the desk.

**Why needed:** Without this flag, *all* 64 expert weight copies (35B total) must be on GPU. With Q4_K quantization that's 17GB — exceeding 12GB VRAM. By keeping only the active computation path (~3B/token) on GPU and offloading the idle expert weights to CPU RAM, the model fits.

**Tuning:** Values 8/16/32 all perform similarly (~75 tok/s with MTP). 16 is the safe default.

### `--spec-type draft-mtp` — Multi-Token Prediction

> MTP = guess 3 words at once instead of one at a time. "To…day…the…" → "Today the weather."  
> Verify once; if correct, use all 3. Like touch-typing instead of hunt-and-peck.

**Why it works:** Qwen3.6-35B-A3B was trained with MTP heads — extra output layers that predict tokens 2, 3, and 4 ahead in parallel. With `--draft-mtp`, llama.cpp uses these heads as a lightweight "draft model" in a speculative decoding loop: draft the next 3 tokens cheaply, then verify them all with one forward pass. Good guesses → big speedup.

### `--spec-draft-n-max 3` — Draft Length

> "Guess at most 3 more tokens each time." 2 is safer (easier to guess right), 3 is greedier (faster if correct).  
> On RTX 3060, n=3 is the sweet spot.

**Tradeoff:** Higher `n` = more tokens per speculative step = bigger speedup if guesses are correct. But wrong guesses get discarded and waste compute. n=4 gives no improvement over n=3 on this hardware — the model's MTP head accuracy drops beyond 3.

### `-ngl 99` — GPU Layers

> "Put everything possible on the GPU." 99 is a sentinel meaning "all layers."

**Why 99 not 40:** The model has ~40 transformer layers + embeddings + LM head. `-ngl 99` guarantees all of them land on GPU. Combined with `--n-cpu-moe`, only the *expert weights* go to CPU — the attention computation, shared experts, and output projection stay on GPU where they're fast.

### `--flash-attn on` — Flash Attention

> Instead of computing the full attention matrix (expensive and memory-heavy), Flash Attention divides and conquers — computes attention in tiles, never materializing the full matrix.

**Why needed:** Without Flash Attention, the KV cache for 64K context alone would eat ~4-5GB of VRAM. With it, attention uses ~2-3× less memory *and* runs faster. Essential for fitting long-context inference in 12GB.

### `--jinja` — Jinja2 Prompt Templates

> Enables Jinja2 template processing for chat-format models.

Qwen's tokenizer uses Jinja2 templates for chat formatting (e.g., `<|im_start|>system\n...<|im_end|>`). Without `--jinja`, the server can't parse chat-style prompts correctly.

### `-c 65536` — Context Length (64K tokens)

> "Remember the last 64K tokens of conversation."

On 12GB VRAM with Flash Attention + Q4_K KV cache, 64K is the maximum feasible context. The KV cache alone takes ~3-4GB at this length. Going higher would spill VRAM.

### `-t 8` — CPU Threads

> "Use 8 CPU threads for the offloaded expert computation."

The MoE experts offloaded to CPU (`--n-cpu-moe`) still need compute when paged in. 8 threads on the 5800X (8C/16T) balances CPU expert eval vs. GPU forward pass. Too few threads → CPU becomes bottleneck. Too many → contention with GPU.

### `-b 512 -ub 128` — Batch / Unbatch Size

> `-b 512` = "Process up to 512 tokens per batch." `-ub 128` = "Split into 128-token chunks internally."

Controls memory-vs-speed for prompt processing. Lower `-b` uses less VRAM but is slower. 512/128 is tuned for 12GB with 64K context — fits without spilling while staying fast.

### `-fitt 1536` / `-fitt 2560` — Flash Inference Trick (alternative offload)

> Another way to split the work: "Keep 1.5GB on GPU for MTP. Everything else goes to CPU."

**Why it's worse:** `-fitt` uses a blanket "this many MB on GPU" approach. `--n-cpu-moe` is smarter — it knows which weights are *expert* vs. *attention* vs. *shared* and keeps the right things on GPU. Result: n-cpu-moe baseline = 62 tok/s, fitt baseline = 15 tok/s (4× slower).

---

## 🎯 Lessons Learned

1. **--n-cpu-moe + MTP are compatible** — need llama.cpp built from source with CUDA 12.4, pre-built binaries don't support this combination
2. **n-cpu-moe >> -fitt** — n-cpu-moe baseline 62 tok/s vs. fitt baseline 15 tok/s
3. **MTP gives ~20% speedup** — 62 → 75 tok/s, worth it for daily coding agent use
4. **n-cpu-moe value barely matters** — 8/16/32 all give ~75 tok/s with MTP
5. **Maxing out the GPU** — n-cpu-moe 16 + MTP n=3 uses 11.3/12 GB VRAM

---

## 📁 Repo Structure

```
qwen-moe-benchmark/
├── scripts/
│   ├── colab-bench.sh        # Colab benchmark wrapper
│   ├── colab_bench.py        # Benchmark runner
│   └── sweep-mtp.sh          # MTP sweep: n-max, moe, fitt
├── docs/
│   ├── setup.md              # Colab CLI setup
│   ├── conclusion.md         # Analysis & conclusions
│   └── references.md         # Reference list
├── results/
│   ├── bench-rtx3060-ckey-20260704.md   # Baseline (49.9 tok/s)
│   ├── bench-colab-t4-20260704-1055.json # Colab (7.5 tok/s)
│   └── bench-mtp-20260704.md            # MTP benchmark (75 tok/s) ⭐
└── README.md
```

---

## 🔗 References

- **[knightli](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/)** — RTX 3060 + n-cpu-moe guide (33-36 tok/s)
- **[SpecPicks #1](https://specpicks.com/reviews/qwen-36-35b-a3b-rtx-3060-12gb-llama-cpp-2026)** — 3060 12GB benchmarks (19-31 tok/s)
- **[SpecPicks #2](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-mtp-deep-dive-2026)** — MTP deep dive (80 tok/s claim)
- **[janvitos Reddit](https://www.xiaoxinsoft.com/en/qwen35b-mtp-12gb-vram-guide)** — 4070S + MTP @ 76-82 tok/s
- **[InsiderLLM](https://insiderllm.com/guides/wicked-fast-qwen-3-6-27b-mtp-rtx-3090/)** — 27B MTP on RTX 3090
- **[PR #22673](https://github.com/ggerganov/llama.cpp)** — MTP support in llama.cpp
