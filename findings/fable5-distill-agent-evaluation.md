# Fable 5 Distill Models — Hermes Agent Use Case Evaluation

> **Goal:** Evaluate Fable 5 distilled models for roles in Hermes Agent ecosystem:
> - **Pi** — Hermes Agent core (tool calling, multi-turn, instruction following)
> - **OMX / OMC** — Planning & coding architect (code gen, debug, architecture)
>
> **Constraint:** Local hardware = GTX 1050 Ti 4GB VRAM.
> Rental (ckey.vn RTX 3060 12GB / Colab T4 16GB) available for development/benchmark.
>
> **Date:** 2026-07-05

---

## 1. Model Landscape (10 Models)

| # | Model | Architecture | Params | Q4 Size | 4GB Local? | 12GB RTX? |
|---|-------|:-----------:|:------:|:-------:|:----------:|:---------:|
| 1 | armand0e/Qwen3.6-35B-A3B-Fable-5-Distill | MoE 35B→3B act | 35.1B | ~17 GB | ❌ | ✅ n-cpu-moe |
| 2 | deadbydawn101/RavenX-OpenFable-Qwopus-Coder-Holo3-35B-MTP | MoE 35B→3B act | 35B | ~17 GB | ❌ | ✅ n-cpu-moe+MTP |
| 3 | Kyle Hessling Qwopus3.6-35B-A3B-v1 | MoE 35B→3B act | 35B | ~18 GB | ❌ | ✅ n-cpu-moe |
| 4 | TeichAI/Gemma-4-31B-Fable-5-Agent-Distill | dense | 31B | ~17 GB | ❌ | ❌ |
| 5 | armand0e/Qwen3.6-27B-Fable-5-Undercooked | dense | 27B | safetensors | ❌ | ❌ |
| 6 | yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1 | dense | 12B | ~7 GB | ❌ | ✅ |
| 7 | **armand0e/Qwen3.5-9B-Fable-5-v1** | dense | 9B | **~5.2 GB** | ⚠️ Q3 | ✅ |
| 8 | **empero-ai/Qwythos-9B-Claude-Mythos-5-1M** | dense | 9B | **~5.5 GB** | ⚠️ Q3 | ✅ |
| 9 | developerjeremylive/gpt-oss-120b-Fable-5-Distilled | MoE | 120B | safetensors | ❌ | ❌ |
| 10 | ermiaazarkhalili/Qwen3.5-2B-Fable5-Glint | dense | 2B | ~1.5 GB | ✅ | ✅ |

**Models in bold = primary candidates** (fit 12GB, reasonable quality-to-size).

---

## 2. Published Benchmark Scores

Researched via AGY subagent — 15 web searches across HF model cards, community blogs, benchmark aggregators. **Only 2/10 models have published scores.**

### Qwopus3.6-35B-A3B-v1 (Kyle Hessling / Jackrong)

| Benchmark | Score | Comparable to |
|-----------|:-----:|---------------|
| **SWE-bench** (300 cases, thinking off) | **62.4%** | Claude Opus 4 ~55%, GPT-5 ~60%, GPT-4 ~42% |
| Japanese I-Mini coding | 80/100 | — |
| Tool-call reliability (asiai.dev) | **100%** clean, 0 bugs | — |
| Deep context stress | **100%** clean (vs dense 27B: 88.9%) | — |

⚡ **SWE-bench 62.4% is exceptionally high** — suggests real coding architecture capability.

### Qwythos-9B-Claude-Mythos-5-1M (Empero AI)

| Benchmark | Score | vs Qwen3.5-9B Base | Notes |
|-----------|:-----:|:------------------:|-------|
| MMLU | **57.5%** | +34.3 pts🏆 | Absolute still low (base ~71%) — likely compared vs weaker base |
| GSM8K strict | **81%** | +30 pts | Solid |
| GSM8K flex | **86%** | +19 pts | ✅ |
| ARC-challenge | **49%** | +2 pts | Mediocre |
| GPQA-diamond | **58%** | **-5 pts** (drop) | ⚠️ |
| Tool-augmented QA | **7/7 perfect** | — | Critical for Pi role ✅ |
| HumanEval | — | — | **No data** |

### Models with ZERO published scores

- armand0e's 3 models (#1, #5, #7) — main distiller but no eval at all
- RavenX MTP (#2) — no data
- TeichAI Gemma 4 31B (#4) — no data
- Gemma 4 12B coder (#6) — no data
- GPT OSS 120B (#9) — no data
- 2B Glint (#10) — no data

**HumanEval is absent from ALL model cards.** Any HumanEval run would be the first published data point for these models.

### Reference: Base Model Scores (Community Benchmarks)

| Model | HumanEval | MMLU |
|-------|:---------:|:----:|
| Qwen3.6-27B (base) | 78% | ~76% |
| Qwen3.6-35B-A3B (base) | 72% | ~66% |
| Qwen3.5-9B (base) | — | ~71% |

---

## 3. Hardware Reality Check

### Local Machine (daily driver)
- **GPU:** NVIDIA GTX 1050 Ti — **4 GB VRAM**
- **CPU:** Intel HD Graphics 630 (iGPU, ko dùng được)
- **Implication:** Cannot run any Q4 model above ~4.2 GB

### Options on 4GB

| Strategy | Model | Quality Impact | Estimated Tok/s |
|----------|-------|:--------------:|:---------------:|
| Qwythos-9B **Q3_K_M** (~3.8 GB) | Qwythos | ~10-15% quality loss vs Q4 | ~15-20 |
| Qwen3.5-9B **Q3_K_M** (~3.6 GB) | Qwen3.5-9B-Fable5 | ~10-15% loss | ~15-20 |
| Qwythos-9B Q4_K_M **--ngl 24** (partial GPU) | Qwythos | Full quality, slower | ~8-12 |
| Qwen3.5-2B-Fable5-Glint **Q4** (~1.5 GB) | 2B Glint | Minimal quality (2B) | ~40-50 |

**Recommendation for 4GB:** Qwythos-9B Q3_K_M (tight fit) or partial offload `--ngl 24` (stable).

### Development/Rental Hardware
- **ckey.vn:** RTX 3060 12GB — ~1,680 VND/h (budget ~14h remaining)
- **Colab:** T4 16GB — free tier quota exhausted, Pro available
- **Local (Pi host):** RTX 3060 12GB for inference server

---

## 4. Role-Fit Analysis

### Pi Core (Hermes Agent)

**Required capabilities:**
- Tool calling reliability (BFCL / tool-augmented QA)
- Multi-turn conversation
- Fast response (low latency, high tok/s)
- Long context (agent sessions)
- Always-on inference

**Ranking:**

| Model | Tool Score | Speed | Context | Verdict |
|-------|:----------:|:-----:|:-------:|:-------:|
| **Qwythos-9B-MTP** 🏆 | **7/7** perfect | **52 tok/s** | **1M ctx** | ✅ Fit perfect |
| Qwen3.5-9B-Fable5 | — | 52 tok/s | ? | ⚠️ No tool data |
| Gemma4-12B-Fable5 | — | 35 tok/s | ? | ⚠️ Slower |
| Qwopus 35B MoE | 100% tool-call | 66 tok/s | ? | ✅ But heavy |

**→ Winner: Qwythos-9B-MTP** — tool-QA perfect, fastest, lightest, long context. Proven agent capability.

### OMX / OMC (Planning & Coding Architect)

**Required capabilities:**
- Code generation quality
- Debugging / reasoning depth
- Architecture planning
- Can tolerate higher latency

**Ranking:**

| Model | SWE-bench | HumanEval (est) | Verdict |
|-------|:---------:|:---------------:|:-------:|
| **Qwopus 35B MoE** 🏆 | **62.4%** | ~75% (base MoE) | ✅ Best coder |
| Qwythos-9B-MTP | — | — | ⚠️ No code data |
| Qwen3.5-9B-Fable5 | — | — | ⚠️ No code data |

**→ Winner: Qwopus 35B MoE** (needs n-cpu-moe trick) or **Qwythos-9B-MTP as fallback** (balanced).

*Note: SWE-bench 62.4% on 35B MoE suggests real architectural coding capability — likely the best open-weight coding model available for this size.*

---

## 5. Recommended Stack

### Tier 1 (Best Quality — needs rental GPU)

```
Role     Model                    Stack                    When
Pi       Qwythos-9B-MTP           llama.cpp GGUF, Q4_K_M   Always-on inference server
OMX      Qwopus 35B MoE           llama.cpp + n-cpu-moe 32  Spawn per-task on ckey/colab
OMC      Qwopus 35B MoE           Same as OMX              Spawn per-task
```

### Tier 2 (All-in-1 on 12GB — daily driver)

```
All roles → Qwythos-9B-MTP Q4_K_M (~5.5 GB)
            llama.cpp, --ctx-size 32768, --cont-batching
            ~52 tok/s, 1M context capability
```

### Tier 3 (4GB Local — degraded)

```
All roles → Qwythos-9B-MTP Q3_K_M (~3.8 GB)
            hoặc partial offload --ngl 24
            ~8-20 tok/s, quality reduced
```

---

## 6. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pi core model | **Qwythos-9B-MTP** | Tool-QA perfect, fast, light, long ctx |
| Coding model | **Qwopus 35B MoE** | SWE-bench 62.4% = best available |
| Local inference | **Qwythos-9B Q3_K_M** | Only fit on 4GB with acceptable quality |
| Benchmark priority | **BFCL + HumanEval** | Cover both Pi & OMX/OMC use cases |
| Stack simplicity | **1 model for all roles** | Qwythos-9B-MTP is balanced enough |

---

## 7. What's Missing (Bench Later)

| Data Point | Why Needed | Effort |
|------------|------------|--------|
| **BFCL v3** scores for Qwythos-9B | Tool calling !== tool-QA. Need formal BFCL. | ~2h/3 models |
| **HumanEval** for all models | Code quality metric, missing from all model cards | ~2h/3 models |
| Qwypus 35B MoE on 12GB local | Can n-cpu-moe sustain OMX workload? | ~1h setup |
| Qwythos vs Qwen3.5-Fable5 comparison | Direct A/B for Pi role | ~1h eval |

---

## References

- Fable 5 distill model list: [colab-gpu-bench skill references/fable5-distill-models.md]
- Speed benchmark results: [bench-fable5-distill-20260705.json](https://github.com/tanhaien/qwen-moe-benchmark/results/)
- AGY research report: 2026-07-05 subagent task (15 web searches)
- Qwopus SWE-bench: [asiai.dev](https://asiai.dev)
- Qwythos official scores: [empero.org](https://empero.org)
