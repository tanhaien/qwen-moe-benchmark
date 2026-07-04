# Qwen MoE Benchmark: Chạy 35B Model trên 12GB VRAM?

Repo này ghi lại thí nghiệm verify cái trick `--n-cpu-moe` trên Colab T4: 
liệu có thật là chạy được model 35B tham số trên GPU chỉ có 12-16GB VRAM?

## TL;DR

- **Trick có thật**: Qwen3.5-35B-A3B là MoE (Mixture of Experts) — 35B tổng, 
  nhưng mỗi lần xử lý chỉ dùng ~3B. `--n-cpu-moe 32` offload expert weights 
  lên CPU RAM, GPU chỉ giữ phần active.
- **Speed depends on CPU RAM**: Desktop DDR5 đạt 35-130 tok/s. 
  Colab T4 (DDR4 chậm) chỉ ~7.5 tok/s.
- **Cần GPU + CPU RAM bandwidth cao** để đạt 100+ tok/s.

## Câu Chuyện

Tháng 5/2026, [knightli.com](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/) 
post 1 bài: RTX 3060 12GB chạy Qwen3.5-35B-A3B được **33-36 tok/s** 
(chất lượng ngang Llama 3.3 70B). Trước đó model 35B chỉ chạy được trên GPU 24GB+.

Sau đó Alan Dao (Facebook) claim **130 tok/s** trên 12GB với MTP.

→ Tụi mình thử trên Colab T4 free để verify.

## The Trick (1 dòng)

```bash
--n-cpu-moe 32
```

**Tại sao nó hoạt động?** Qwen3.5-35B-A3B có kiến trúc MoE:
- 35B tham số tổng cộng — nhưng ko phải tất cả đều dùng cùng lúc
- Mỗi token, model chọn 8/256 "chuyên gia" nhỏ → chỉ ~3B tham số active
- `--n-cpu-moe 32` bảo GPU tính ~3B active, còn các "chuyên gia" còn lại để trên CPU RAM

Full config từ knightli:
```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 32 \
  --flash-attn on \
  -c 65536 \
  -t 8 \
  -b 512 -ub 128 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

## Kết Quả Benchmark (Colab T4)

| n-cpu-moe | Gen (tok/s) | PP (tok/s) | Ghi chú |
|-----------|-------------|------------|---------|
| 0 (all-GPU) | — | — | OOM (ko fit VRAM) |
| **16** | **7.53** | 5.30 | Nhanh nhất trên T4 |
| 32 | 3.92 | 2.27 | Alan Dao config |
| 64 | 2.78 | 2.52 | Offload nhiều hơn |
| 128 | 2.60 | 1.97 | Offload tối đa |

**Model**: Qwen3.5-35B-A3B Q3_K_M (15.2 GB, 34.6B params)  
**GPU**: Tesla T4 (15360 MiB)  
**CPU**: Intel Xeon @ 2.00GHz  
**llama.cpp**: build b9860, CUDA 12.8, FlashAttn on

### So sánh với community

| Setup | Speed | Nguồn |
|-------|-------|-------|
| Colab T4 (bài này) | 7.5 tok/s | Benchmark thực tế |
| RTX 3060 + DDR4 + ko MTP | 33-36 tok/s | [knightli](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/) |
| RTX 3060 + MTP | 60-80 tok/s | [SpecPicks](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-12gb-mtp-2026) |
| RTX 3060 + MTP + Q3_K_M | 80-130 tok/s | Alan Dao (Facebook) |
| RTX 3090 (all-GPU, 24GB) | 133-142 tok/s | [aminrj](https://aminrj.com/posts/llamacpp-qwen36-35b/) |

## Tại sao Colab T4 chậm?

| Yếu tố | Desktop RTX 3060 | Colab T4 | Ảnh hưởng |
|--------|-----------------|----------|-----------|
| GPU memory bandwidth | 360 GB/s | 320 GB/s | ~1x |
| **CPU RAM bandwidth** | DDR4 51 GB/s | ~25 GB/s | **~2x** |
| **CPU speed** | Ryzen 3700X 4.4 GHz | Xeon 2.0 GHz | **~2x** |
| PCIe bus | Gen3 x16 | Gen3 x4-x8 | ~1.5x |
| **Tổng** | | | **~5-6x** |

Colab T4 dùng CPU chậm + DDR4 bandwidth thấp → bottleneck khi swap expert weights giữa CPU ↔ GPU.

**Bài học**: MoE + `--n-cpu-moe` phụ thuộc nhiều vào CPU RAM bandwidth, ko chỉ GPU.

## Cần gì để đạt 100+ tok/s?

1. GPU ≤ 12GB VRAM (đủ active weights ~3B)
2. CPU RAM ≥ 48GB DDR5 (chứa cold expert pool ~13GB)
3. CPU RAM bandwidth > 50 GB/s
4. `--mtp 6` (MTP: Multi-Token Prediction, predict nhiều token cùng lúc)
5. llama.cpp build có MTP support (PR #22673)

## Repo Structure

```
qwen-moe-benchmark/
├── scripts/
│   ├── colab-bench.sh        # Wrapper: provision Colab VM + chạy benchmark
│   └── colab_bench.py        # Benchmark runner (chạy trên Colab)
├── docs/
│   ├── setup.md              # Hướng dẫn setup Colab CLI
│   ├── conclusion.md         # Phân tích kết quả + kết luận chi tiết
│   └── references.md         # Danh sách tham khảo
├── results/
│   ├── README.md             # Bảng kết quả
│   └── bench-colab-t4-*.json # Raw output từ llama-bench
└── README.md
```

## References

- **[knightli](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/)** — RTX 3060 + --n-cpu-moe guide (33-36 tok/s)
- **[SpecPicks](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-12gb-mtp-2026)** — MTP benchmark (60-80 tok/s)
- **[aminrj](https://aminrj.com/posts/llamacpp-qwen36-35b/)** — RTX 3090 all-GPU benchmark
- **[ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda)** — Pre-built CUDA binaries
- **[PR #22673](https://github.com/ggerganov/llama.cpp)** — MTP support in llama.cpp
