# References

## Core Sources

### Qwen3.5-35B-A3B MoE Trick (130 tok/s on 12GB)
- **knightli.com** (2026-05-26): "RTX 3060 12GB Local 35B Guide"
  https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/
  - Exact command template: `--n-cpu-moe 32 -ngl 99 --flash-attn on`
  - Reports 33-36 tok/s at Q4_K_M on RTX 3060

- **aminrj.com** (2026-04-18): "Qwen3.6 on 24GB VRAM: Benchmark"
  https://aminrj.com/posts/llamacpp-qwen36-35b/
  - Qwen3.5-35B-A3B on RTX 3090: 133-142 tok/s (all-GPU, 24GB)
  - Confirms MoE: 35B total, ~3.6B active/token

- **Alan Dao** (2026-07): 130 tok/s peak on 12GB with Q3_K_M + MTP
  Facebook Reel (not independently verified in this research)

### Pre-built Linux CUDA Binaries
- **ai-dock/llama.cpp-cuda** (nightly CUDA builds)
  https://github.com/ai-dock/llama.cpp-cuda
  - Tracks upstream releases
  - CUDA 12.8, supports T4 (cc 7.5)
  - `llama-server`, `llama-bench`, `llama-cli` included

### Google Colab CLI
- **google-colab-cli** (v0.6.0, June 2026)
  https://github.com/googlecolab/google-colab-cli
  - Official CLI for Colab provisioning
  - GPU: T4 (free), L4, A100, H100 (paid)
  - `colab run --gpu T4 script.py` — one-shot

### Colab MCP Server (alternative)
- **googlecolab/colab-mcp** (705★)
  https://github.com/googlecolab/colab-mcp
  - MCP server running inside Colab notebook
  - Requires browser session

## Model

- **HuggingFace**: `unsloth/Qwen3.5-35B-A3B-GGUF`
  https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
  - Quants available: Q2-Q8, UD-IQ2 to UD-Q8

- **Official**: `Qwen/Qwen3.5-35B-A3B`
  https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## Tools

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **llama.cpp releases**: https://github.com/ggml-org/llama.cpp/releases
- **llama.cpp Docker (CUDA)**: `ghcr.io/ggml-org/llama.cpp:full-cuda`

## Related Community Benchmarks

- Reddit r/LocalLLaMA: "80 tok/sec and 128K context on 12GB VRAM with Qwen3.6 35B A3B"
- ComfyUI-Qwen3.5: GGUF pipeline @ 152 tok/s
  https://github.com/DanielBartolic/ComfyUI-Qwen3.5
- Speculative decoding benchmarks (RTX 3090):
  https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090

## Key Parameters

```bash
# Production config (knightli, 12GB)
llama-server \
  -m Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 32 \
  --flash-attn on \
  --jinja \
  -c 65536 \
  -t 8 \
  -b 512 -ub 128 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

```bash
# Benchmark mode
llama-bench \
  -m Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe N \
  --flash-attn 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -p 64 -n 256 \
  -t 8 -b 512 -ub 128
```
