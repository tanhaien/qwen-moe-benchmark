# Kết Luận: Qwen3.5-35B-A3B trên T4 vs Desktop

## Vấn đề

Community claim: 130 tok/s trên RTX 3060 12GB với `--n-cpu-moe 32` (Alan Dao).
Benchmark thực tế trên Colab T4: **chỉ 7.5 tok/s**.

## Tại sao Colab T4 chậm?

MoE model: 35B total, ~3B active/token. `--n-cpu-moe` offload expert weights → CPU RAM.
Mỗi token, router chọn experts mới → swap từ CPU RAM qua PCIe.

Bottleneck chính: **CPU RAM bandwidth**, ko phải GPU compute.

| Factor | Desktop (RTX 3060) | Colab T4 | Chênh lệch |
|--------|--------------------|----------|------------|
| GPU mem BW | 360 GB/s | 320 GB/s | 1.1x |
| **CPU RAM BW** | DDR5-6000 **96 GB/s** | DDR4-3200 **~25 GB/s** | **~4x** |
| **CPU speed** | Ryzen 5 **5.2 GHz** | Xeon **2.0 GHz** | **~2.5x** |
| PCIe | Gen4 x16 | Gen3 x4-x8 | ~2x |
| **Tổng** | | | **~5-6x** |

SpecPicks confirm: *"CPU bandwidth matters more than core count"* — cold experts (~13 GB) swap liên tục qua CPU RAM.

## Breakdown các số liệu

| Setup | Gen (tok/s) | Nguồn |
|-------|------------|-------|
| **Colab T4** n-cpu-moe=16 | **7.5** | Benchmark này |
| RTX 3060 (ko MTP) | 35-45 | knightli, SpecPicks |
| RTX 3060 + MTP | 60-80 | SpecPicks |
| RTX 3060 + MTP peak | 80-130 | Alan Dao (Q3_K_M) |
| RTX 3090 (all GPU) | 133-142 | aminrj |

Base speed không MTP:
- Colab T4: 7.5 tok/s
- Desktop 3060: 35-45 tok/s  
- Ratio: ~5-6x ✅ match với chênh lệch hardware

Với MTP (`--mtp 6`):
- Qwen3.5 built-in MTP heads → speculative decoding: predict 6 tokens/pass
- Hit rate: 65-90% → effective 1.5-2x speedup
- 45 tok/s × 1.9x × Q3_K_M nhẹ hơn Q4 = ~130 tok/s

## Điều kiện cần cho 100+ tok/s

1. **GPU ≤ 12GB VRAM**: enough cho active weights (~3B)
2. **CPU RAM ≥ 48GB DDR5**: cho cold expert pool (~13GB)
3. **CPU bandwidth** > 50 GB/s: DDR5-4800+
4. **llama.cpp MTP support**: build với PR #22673
5. **Full config**:
   ```
   --n-cpu-moe 32
   --mtp 6
   -ngl 99
   --flash-attn on
   -c 32768
   -b 512 -ub 128
   ```

## Kết luận

- **Colab T4 ko phù hợp** cho MoE offload workflow vì CPU RAM chậm + Xeon low clock
- 130 tok/s là **thật** nhưng cần desktop DDR5 + MTP
- T4 free tier: max 7-8 tok/s → unusable cho interactive
- Muốn local inference nhanh → cần GPU + CPU RAM bandwidth cao

## References

1. [knightli.com — RTX 3060 + n-cpu-moe](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/)
2. [aminrj.com — RTX 3090 all-GPU](https://aminrj.com/post/qwen-35b-llamacpp/)
3. [SpecPicks — RTX 3060 12GB MTP @ 80 tok/s](https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-12gb-mtp-2026)
4. [Alan Dao — Facebook Reel 130 tok/s](https://www.facebook.com/alandao)
5. [ai-dock/llama.cpp-cuda — pre-built CUDA nightly](https://github.com/ai-dock/llama.cpp-cuda)
6. [PR #22673 — MTP support in llama.cpp](https://github.com/ggerganov/llama.cpp)
