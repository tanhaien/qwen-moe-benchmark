# Qwen MoE Benchmark

Benchmark Qwen3.5-35B-A3B (Mixture of Experts) trên Google Colab T4 16GB.

Mục tiêu: đo tốc độ inference (tok/s) với các giá trị `--n-cpu-moe` khác nhau.

## Background

Qwen3.5-35B-A3B là MoE model: 35B total params, chỉ ~3B active/token.
Với `--n-cpu-moe 32`, expert weights offload lên CPU RAM, GPU chỉ giữ active computation path.
=> Cho phép chạy 35B model trên 12-16GB VRAM.

Ref: [knightli.com guide](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/), [Alan Dao trick](https://www.facebook.com/alandao)

## Quick Start

### Prerequisites
- Google account (Colab free tier)
- `uv tool install google-colab-cli` (Colab CLI v0.6.0+)

### One-time Auth
```bash
colab --auth=oauth2 new  # browser login, paste code
# Or if you have gcloud ADC:
colab --auth=adc whoami  # verify
```

### Run Benchmark
```bash
# One-shot (wrapper script)
./scripts/colab-bench.sh

# Or step by step
colab new -s bench --gpu T4
colab exec -f scripts/colab_bench.py
colab download -s bench results.json ./results/
colab stop -s bench
```

## Benchmark Matrix

| n-cpu-moe | Description | Expected VRAM |
|-----------|-------------|---------------|
| 0 | All GPU (wont fit 16GB) | >16GB |
| 16 | Mostly GPU, minimal offload | ~13-14GB |
| 32 | Balanced (Alan Dao spec) | ~7-9GB |
| 64 | More CPU offload | ~5-6GB |
| 128 | Heavy CPU offload | ~3-4GB |

Model: `unsloth/Qwen3.5-35B-A3B-Q3_K_M.gguf` (~16GB)

## Results

| n-cpu-moe | Gen (tok/s) | PP (tok/s) | Status |
|-----------|-------------|------------|--------|
| 0         | —           | —          | OOM    |
| 16        | **7.53**    | 5.30       | OK     |
| 32        | 3.92        | 2.27       | OK     |
| 64        | 2.78        | 2.52       | OK     |
| 128       | 2.60        | 1.97       | OK     |

Tesla T4 15 GB, Xeon 2.0 GHz, DDR4, 256 gen, 64 pp, ctx 2k.
Full results & analysis: [results/](results/) & [docs/conclusion.md](docs/conclusion.md)

## Repo Structure

```
qwen-moe-benchmark/
├── scripts/
│   ├── colab-bench.sh      # Wrapper: orchestrate Colab VM
│   └── colab_bench.py      # Benchmark runner (runs on Colab VM)
├── docs/
│   ├── setup.md            # Setup instructions
│   └── references.md       # Research references
├── results/                # Benchmark outputs
└── README.md
```

## How It Works

1. **Pre-built llama.cpp binary**: `ai-dock/llama.cpp-cuda` (Linux CUDA, nightly builds)
2. **Model**: HuggingFace `unsloth/Qwen3.5-35B-A3B-GGUF`
3. **Colab CLI**: `google/colab-cli` v0.6.0 — provision GPU VM, exec, teardown

No need to compile from source — saves 30+ minutes.

## Tech Stack

- [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) — terminal-based Colab provisioning
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference (CUDA)
- [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda) — pre-built CUDA binaries
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — MoE model
