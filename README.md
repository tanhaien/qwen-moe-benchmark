# 🚀 Qwen MoE Benchmark: Chạy 35B Model trên 12GB VRAM

> **Verified: 75 tok/s avg, peak 87.5 tok/s** — APEX-MTP + n-cpu-moe trên RTX 3060 12GB + 32GB DDR4  
> 30/05 → 04/07/2026

Repo này ghi lại thí nghiệm verify trick `--n-cpu-moe` và MTP (Multi-Token Prediction) trên RTX 3060 12GB — liệu chạy được model 35B tham số ngang Llama 3.3 70B trên 1 con card $300?

---

## 🧠 TL;DR

- **Trick có thật**: Qwen3.6-35B-A3B là MoE — 35B tổng, chỉ ~3B active/token
- **--n-cpu-moe 16**: offload expert weights lên CPU RAM, GPU giữ active path → **62 tok/s**
- **+ MTP**: thêm speculative decoding → **75 tok/s avg, peak 87.5 tok/s**
- **Config tối ưu**: `-ngl 99 --n-cpu-moe 16 --spec-type draft-mtp --spec-draft-n-max 3`
- **Cần ≥48GB DDR5** cho 130 tok/s (Alan Dao claim)

---

## 📊 Kết Quả Chính

### APEX-MTP-Compact (17GB Q4_K) — RTX 3060 + 5800X + 32GB DDR4

| Config | Avg (tok/s) | Peak code | Wall 9pr | Ghi chú |
|--------|:----------:|:---------:|:--------:|---------|
| n-cpu-moe 16 (no MTP) | **62.4** | — | 25.3s | Baseline sạch |
| **n-cpu-moe 16 + MTP n=3** 🏆 | **75.1** | **86.1** | **21.9s** | **Best** |
| n-cpu-moe 16 + MTP n=4 | 75.5 | 87.5 | 21.4s | ~n=3 (model limit) |
| n-cpu-moe 32 + MTP n=3 | 74.8 | 87.2 | 21.4s | |
| -fitt 2560 + MTP n=3 | 73.5 | 85.8 | 21.7s | |
| -fitt 1536 + MTP n=2 | ~66 | ~73 | ~24.6s | Bài janvitos |

> **4.7x speedup? Sai!** Đó là so sánh MTP với `-fitt` baseline (15 tok/s).  
> So với cùng config (`n-cpu-moe 16`), MTP cho **~1.2x** (62 → 75 tok/s).

### So sánh tổng hợp

| Setup | Gen (tok/s) | Gấp Colab | Nguồn |
|-------|:-----------:|:---------:|-------|
| **RTX 3060 + MTP n=3** ⚡ | **75 / 87 peak** | **~10x** | **Bài này** |
| RTX 3060 + n-cpu-moe (Qwen3.6) ⚡ | **54-62** | ~7-8x | Bài này |
| RTX 3060 + n-cpu-moe (Qwen3.5) ✅ | **49.9** | **6.6x** | Bài này |
| RTX 3060 + MTP (SpecPicks #2) | 80 (Q4_K) | ~11x | [SpecPicks](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-mtp-deep-dive-2026) |
| RTX 4070S + MTP (janvitos Reddit) | 76-82 | ~10x | [xiaoxinsoft](https://www.xiaoxinsoft.com/en/qwen35b-mtp-12gb-vram-guide) |
| RTX 3060 + MTP (SpecPicks #1) | 31 (Q3_K) | ~4x | [SpecPicks](https://specpicks.com/reviews/qwen-36-35b-a3b-rtx-3060-12gb-llama-cpp-2026) |
| RTX 3060 + 3700X + 32GB (knightli) | 33-36 | ~4.7x | [knightli](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/) |
| Colab T4 (free) | 7.5 | 1x | Bài này |

---

## 📦 Model sử dụng

Benchmark dùng **1 model duy nhất** — các dòng ở dưới là variant/cỡ khác nhau của cùng Qwen3.6-35B-A3B:

| Variant | Quant | Size | MTP | Note |
|---------|:-----:|:----:|:---:|------|
| **APEX-MTP-Compact** (dùng cho bench này) | Q4_K | **17 GB** | ✅ | Bản Q4_K quant, giữ MTP heads |
| APEX-MTP-Balanced | Q4_K_M | ~19 GB | ✅ | Chất lượng cao hơn, tốn thêm VRAM |
| APEX-MTP-Quality | — | ~22 GB | ✅ | Quant nhẹ nhất, quality cao nhất |
| APEX-MTP-Nano | Q3_K | ~14 GB | ✅ | Nhỏ nhất có MTP |
| APEX-MTP-non-MTP (mudler APEX repo) | Q4_K | ~16 GB | ❌ | Ko có MTP head |

> **17 GB là đã quantize (Q4_K)** — model FP16 gốc ~70 GB. MTP head thêm ~1 GB.  
> Tất cả variants trên đều là **Qwen3.6-35B-A3B** (35B MoE, ~3B active/token).

---

## 🧠 Giải Thích (Feynman Style)

### `--n-cpu-moe 16`

> Model AI như công ty có 64 chuyên gia. Mỗi lần trả lời chỉ cần gọi 2 chuyên gia giỏi nhất.  
> `n-cpu-moe 16` = "GPU giữ 16 chuyên gia thường dùng. 48 người còn lại ở kho (RAM). Khi cần, chạy lên gọi."
>
> GPU = bàn làm việc nhỏ (12GB), RAM = kho chứa (32GB). Số 16 là "số chuyên gia để sẵn ở bàn" — đủ nhanh, ko tràn.

### `--spec-type draft-mtp`

> MTP = đoán 3 chữ cùng lúc thay vì từng chữ một. "Hôm... nay... trời..." → "Hôm nay trời".  
> Kiểm tra 1 lần, nếu đúng thì dùng. Như đánh máy 10 ngón thay vì 1 ngón.

### `--spec-draft-n-max 3`

> "Hãy đoán tối đa 3 chữ tiếp theo mỗi lần". 2 an toàn hơn (dễ đoán đúng), 3 tham hơn (nếu đúng thì nhanh hơn nhiều).  
> Trên RTX 3060, n=3 là số đẹp.

### `-fitt 1536`

> Cách khác để chia việc: "Hãy giữ 1.5GB trên GPU cho MTP. Phần còn lại xuống CPU."  
> Thua `--n-cpu-moe` vì nó là dao kéo cứng nhắc, n-cpu-moe là bàn tay thông minh.

---

## 🎯 Lessons Learned

1. **--n-cpu-moe và MTP compatible** — build CUDA 12.4 từ source, ko dùng pre-built
2. **n-cpu-moe > -fitt** — n-cpu-moe cho baseline 62 tok/s, fitt chỉ 15 tok/s
3. **MTP ~20% speedup** — 62 → 75 tok/s, đáng nếu chạy coding agent hằng ngày
4. **n-cpu-moe value ít ảnh hưởng** — 8/16/32 đều ~75 tok/s
5. **Tận dụng GPU tối đa** — n-cpu-moe 16 + MTP n=3 dùng 11.3/12 GB VRAM

---

## 📁 Repo Structure

```
qwen-moe-benchmark/
├── scripts/
│   ├── colab-bench.sh        # Colab benchmark wrapper
│   ├── colab_bench.py        # Benchmark runner
│   └── sweep-mtp.sh          # MTP sweep: n-max, moe, fitt
├── docs/
│   ├── setup.md              # Setup Colab CLI
│   ├── conclusion.md         # Phân tích + kết luận
│   └── references.md         # Danh sách tham khảo
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
