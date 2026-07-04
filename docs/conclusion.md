# Kết Luận: Qwen MoE Benchmark

## Vấn đề

Community claim: 130 tok/s trên RTX 3060 12GB với `--n-cpu-moe 32` (Alan Dao).
Benchmark thực tế trên Colab T4: **chỉ 7.5 tok/s**.

## Kết quả verified

### Colab T4 (Tesla T4 15GB, Xeon 2.0GHz, DDR4)
- n-cpu-moe=16 → 7.5 tok/s gen
- Bottleneck: CPU RAM bandwidth thấp ~25 GB/s + Xeon 2.0GHz chậm

### RTX 3060 ckey.vn (RTX 3060 12GB, Ryzen 5800X, 32GB DDR4)
- Qwen3.5 Q3_K_M, n-cpu-moe=16 → **49.9 tok/s** gen (6.6x Colab T4)
- Qwen3.6 UD-Q3_K_M, n-cpu-moe=16 → **54.4 tok/s** gen (9% nhanh hơn Qwen3.5)
- n-cpu-moe=16 cho tốc độ tốt nhất (ko phải 32 như knightli khuyên)

### MTP Status — ✅ Verified
- Qwen3.5: **Không** hỗ trợ MTP
- Qwen3.6 Unsloth GGUF: Có MTP architecture nhưng **bị strip heads** khi quant
- **Qwen3.6 APEX-MTP GGUF (Compact)**: Preserve MTP heads — **✅ Verified 75 tok/s avg, peak 87.5 tok/s**
- Config: `-ngl 99 --n-cpu-moe 16 --spec-type draft-mtp --spec-draft-n-max 3`
- Key: dùng `--n-cpu-moe` (không dùng `-fitt`) + MTP. Cả 2 tương thích với llama.cpp build từ source CUDA 12.4

## Tại sao Colab T4 chậm?

MoE model: 35B total, ~3B active/token. `--n-cpu-moe` offload expert weights → CPU RAM.
Mỗi token, router chọn experts mới → swap từ CPU RAM qua PCIe.

Bottleneck chính: **CPU RAM bandwidth**, ko phải GPU compute.

| Factor | Desktop (5800X + DDR4) | Colab T4 | Chênh lệch |
|--------|----------------------|----------|------------|
| GPU mem BW | 360 GB/s | 320 GB/s | 1.1x |
| **CPU RAM BW** | DDR4-3200 dual **51 GB/s** | DDR4-? **~25 GB/s** | **~2x** |
| **CPU speed** | Ryzen 5800X **4.7 GHz** | Xeon **2.0 GHz** | **~2x** |
| PCIe | Gen3 x16 (16 GB/s) | Gen3 x4-x8 (~4-8 GB/s) | ~2-4x |
| **Tổng** | | | **~5-6x** |

SpecPicks confirm: *"CPU bandwidth matters more than core count"*
— cold experts (~13 GB) swap liên tục qua CPU RAM.

So với knightli (RTX 3060 + 3700X + 32GB):
- Cùng GPU RTX 3060 12GB
- CPU 5800X nhanh hơn 3700X ~19% IPC
- Q3_K_M nhẹ hơn Q4_K_M → expert swap nhanh hơn
- Kết quả: 49.9 tok/s (bài này) vs 33-36 tok/s (knightli)

## So sánh chi tiết

| Setup | Gen (tok/s) | PP (tok/s) | Nguồn |
|-------|:-----------:|:-----------:|-------|
| Colab T4 — n-cpu-moe=16 | **7.5** | 5.3 | Bài này |
| RTX 3060 + 5800X + 32GB (Qwen3.5) | **49.9** | 228.4 | Bài này |
| RTX 3060 + 5800X + 32GB (Qwen3.6) | **54.4** | 222.4 | Bài này |
| RTX 3060 + 3700X + 32GB (Q4_K_M) | 33-36 | — | knightli |
| RTX 3060 + 64GB (Q3_K_M) | 12-18 | — | InsiderLLM |
| RTX 3060 + MTP (dự kiến) | 60-80+ | — | Đang test |

## Cần gì để đạt 100+ tok/s?

1. GPU ≥ 12GB VRAM (đủ active weights ~3B)
2. **CPU RAM ≥ 48GB DDR5** (cold expert pool ~13GB)
3. **CPU RAM bandwidth > 50 GB/s**
4. **APEX-MTP GGUF** (Unsloth strip MTP heads)
5. `--spec-type draft-mtp --spec-draft-n-max 6`
6. llama.cpp build có MTP support (PR #22673)

## Build Notes

- **ai-dock pre-built binary** (CUDA 12.8): Không chạy được với CUDA 12.4 driver
  - Lỗi: "PTX compiled with unsupported toolchain"
- **Cần build từ source** với CUDA 12.4 nvcc
  - `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"`
  - Thời gian build: ~5-10 phút trên 5800X
- **Ubuntu `nvidia-cuda-toolkit`** (CUDA 11.5) — quá cũ, compile lỗi C++17
- **Cần cài `cuda-nvcc-12-4` từ NVIDIA repo** để build đúng

## Kết luận

- **--n-cpu-moe trick verified** — chạy 35B model trên 12GB GPU là thật
- Colab T4 free: 7.5 tok/s → quá chậm cho interactive use
- Desktop RTX 3060 + DDR4: 62-75 tok/s → usable (với MTP)
- **MTP verified: 75 tok/s avg, peak 87.5 tok/s** trên APEX-MTP-Compact + n-cpu-moe 16
- 130 tok/s claim của Alan Dao cần DDR5 để đạt (SpecPicks confirm)
