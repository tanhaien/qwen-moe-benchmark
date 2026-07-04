# References

## Bài gốc (nguồn cảm hứng)

### [knightli.com] RTX 3060 12GB Local 35B Guide
- **Link**: https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/
- **Tóm tắt**: RTX 3060 12GB + Ryzen 3700X + DDR4 chạy Qwen3.6-35B-A3B Q4_K_M
- **Speed**: ~33-36 tok/s (ko MTP)
- **Key insight**: `--n-cpu-moe 32` cho phép chạy 35B model trên 12GB VRAM
- **Full command template**:
  ```
  -ngl 99 --n-cpu-moe 32 --flash-attn on -c 65536 -t 8 -b 512 -ub 128
  --cache-type-k q4_0 --cache-type-v q4_0 -np 1 --cache-ram 0
  ```

## Benchmark khác

### [SpecPicks] RTX 3060 12GB MTP @ 80 tok/s
- **Link**: https://specpicks.com/reviews/qwen36-35b-a3b-rtx-3060-12gb-mtp-2026
- **Tóm tắt**: RTX 3060 + Ryzen 5 7600 + DDR5 → Qwen3.6-35B-A3B Q4_K_M
- **Speed**: 35-45 tok/s (ko MTP), 60-80 tok/s (với `--mtp 6`)
- **Key insight**: Multi-Token Prediction (PR #22673) cho 1.5-2x speedup
- **PCIe note**: Cold expert traffic chỉ ~320 MB/s, PCIe 3.0 x4 vẫn OK

### [aminrj] RTX 3090 24GB all-GPU
- **Link**: https://aminrj.com/posts/llamacpp-qwen36-35b/
- **Tóm tắt**: Qwen3.5-35B-A3B trên RTX 3090 (all-GPU, ko cần offload)
- **Speed**: 133-142 tok/s
- **Chứng minh**: MoE active ~3B compute có thể đạt 130+ tok/s trên GPU mạnh

### Alan Dao (Facebook)
- **Link**: https://www.facebook.com/alandao
- **Claim**: 130 tok/s peak trên 12GB với Q3_K_M + MTP
- **Verification**: Chưa verified độc lập

## Pre-built llama.cpp CUDA

- **ai-dock/llama.cpp-cuda**: https://github.com/ai-dock/llama.cpp-cuda
- Nightly CUDA 12.8 builds, support T4 (cc 7.5)
- Includes: `llama-server`, `llama-bench`, `llama-cli`

## Tools & Model

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **google-colab-cli** (v0.6.0): https://github.com/googlecolab/google-colab-cli
- **Model GGUF**: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
- **Official model**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## Community

- Reddit r/LocalLLaMA: "80 tok/sec on 12GB VRAM with Qwen3.6 35B A3B"
- ComfyUI-Qwen3.5: https://github.com/DanielBartolic/ComfyUI-Qwen3.5
