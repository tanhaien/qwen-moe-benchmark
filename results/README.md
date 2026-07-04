# Benchmark Results

**Model**: Qwen3.5-35B-A3B Q3_K_M (15.2 GB / 34.66B params)
**GPU**: Tesla T4 (15360 MiB, cc 7.5)
**CPU**: Intel Xeon @ 2.00GHz (8 threads)
**llama.cpp**: build fdb1db8, CUDA backend
**Batch**: 512 / ubatch 128
**KV cache**: q4_0, FlashAttn on
**Date**: 2026-07-04

## Results

| n-cpu-moe | VRAM used | PP (tok/s) | Gen (tok/s) | Status |
|-----------|-----------|------------|-------------|--------|
| 0         | >15.2 GB  | —          | —           | OOM    |
| 16        | ~13-14 GB | 5.30       | **7.53**    | OK     |
| 32        | ~7-9 GB   | 2.27       | 3.92        | OK     |
| 64        | ~5-6 GB   | 2.52       | 2.78        | OK     |
| 128       | ~3-4 GB   | 1.97       | 2.60        | OK     |

## Notable

- **n-cpu-moe=0** (all-GPU) fails: model 15.2 GB barely exceeds T4 15 GB → OOM
- **n-cpu-moe=16** fastest at 7.5 tok/s gen, but VRAM nearly saturated
- Increasing CPU offload (32→128) degrades speed monotonically
- Colab Xeon 2.0GHz + DDR4 is the main bottleneck for expert offload path

## Raw Data

See `bench-colab-t4-20260704-1055.json` for full llama-bench output per config.
