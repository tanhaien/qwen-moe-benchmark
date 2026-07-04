# References

## Bài gốc (nguồn cảm hứng)

### [knightli.com] RTX 3060 12GB Local 35B Guide
- **Link**: https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/
- **Tóm tắt**: RTX 3060 12GB + Ryzen 3700X + 32GB DDR4 chạy Qwen3.6-35B-A3B Q4_K_M
- **Speed**: ~33-36 tok/s (ko MTP)
- **Key**: `--n-cpu-moe 32` cho phép chạy 35B model trên 12GB VRAM
- **Full config**: `-ngl 99 --n-cpu-moe 32 --flash-attn on -c 65536 -t 8 -b 512 -ub 128 --cache-type-k q4_0 --cache-type-v q4_0`

## Benchmark verified

### [SpecPicks] RTX 3060 12GB MTP @ 80 tok/s
- **Link**: https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-12gb-mtp-2026
- **Tóm tắt**: RTX 3060 + Ryzen 5 7600 + DDR5 → Qwen3.6-35B-A3B Q4_K_M
- **Speed**: 35-45 tok/s (ko MTP), 60-80 tok/s (với `--spec-type draft-mtp`)
- **Key**: MTP cho 1.5-2x speedup, cold expert traffic chỉ ~320 MB/s
- **Lưu ý**: Cần APEX-MTP GGUF (Unsloth strip MTP heads)

### [aminrj] RTX 3090 24GB all-GPU
- **Link**: https://aminrj.com/posts/llamacpp-qwen36-35b/
- **Speed**: 133-142 tok/s (all-GPU, ko cần offload)

### [InsiderLLM] Qwen3.6 35B Hardware Guide
- **Link**: https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/
- **Speed table**: RTX 3060 + 64GB RAM → 12-18 tok/s (Q3_K_M)
- **Key**: MTP spec decode ko hiệu quả trên MoE (memory thrash)

## Thí nghiệm này

### Colab T4 Benchmark
- **Platform**: Google Colab free tier, GPU Tesla T4 15GB
- **CPU**: Intel Xeon @ 2.00GHz, DDR4 ~25 GB/s
- **llama.cpp**: ai-dock pre-built (b9860, CUDA 12.8) — gặp lỗi CUDA 12.8 vs 12.4 driver
- **Result**: n-cpu-moe=16 → 7.5 tok/s gen
- **Bottleneck**: CPU RAM bandwidth quá thấp cho expert offload

### ckey.vn RTX 3060 Benchmark
- **Platform**: ckey.vn (community GPU rental, Belarus)
- **GPU**: NVIDIA GeForce RTX 3060 12GB (cc 8.6)
- **CPU**: AMD Ryzen 7 5800X 8-Core Processor
- **RAM**: 32 GB DDR4
- **CUDA**: 12.4 (nvidia repo cuda-nvcc-12-4)
- **llama.cpp**: Build từ source (2d97363)
- **Results**:
  - Qwen3.5 Q3_K_M, n-cpu-moe=16 → **49.9 tok/s**
  - Qwen3.6 UD-Q3_K_M, n-cpu-moe=16 → **54.4 tok/s**

## Models

### Qwen3.5-35B-A3B (Single-token prediction)
- **GGUF (Unsloth)**: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
- **Không** hỗ trợ MTP

### Qwen3.6-35B-A3B (Multi-token prediction)
- **GGUF (Unsloth)**: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
  - Có MTP architecture nhưng bị strip heads khi quant
  - UD-Q3_K_M: 16.6 GB
- **APEX-MTP GGUF (mudler)**: https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF
  - Preserve MTP draft heads
  - Balanced / Compact variants

### Alan Dao (Facebook)
- **Link**: https://www.facebook.com/alandao
- **Claim**: 130 tok/s peak trên 12GB với APEX-MTP + Q3_K_M
- **Verification**: Chưa verified (cần build APEX-MTP GGUF + test MTP)

## Build notes

### ai-dock pre-built (không dùng được)
- **Link**: https://github.com/ai-dock/llama.cpp-cuda
- Build với CUDA 12.8 → PTX ko compatible với CUDA 12.4 driver
- Lỗi: "the provided PTX was compiled with an unsupported toolchain"

### Build từ source (recommended)
- Cần cài `cuda-nvcc-12-4` từ NVIDIA repo (ko dùng Ubuntu nvidia-cuda-toolkit vì là CUDA 11.5)
- `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DLLAMA_CUDA_F16=ON`
- Time: ~5-10 phút trên 5800X 8C/16T

## Tools

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **google-colab-cli** (v0.6.0): https://github.com/googlecolab/google-colab-cli
- **huggingface_hub**: pip install huggingface_hub (resumable downloads)

## Community

- Reddit r/LocalLLaMA: "80 tok/sec on 12GB VRAM with Qwen3.6 35B A3B"
- ComfyUI-Qwen3.5: https://github.com/DanielBartolic/ComfyUI-Qwen3.5
