# Benchmark Results

## Colab T4 (Tesla T4 15GB, Xeon 2.0GHz, DDR4)

**Model**: Qwen3.5-35B-A3B Q3_K_M (15.2 GB / 34.66B params)
**llama.cpp**: build fdb1db8 (ai-dock), CUDA 12.8

| n-cpu-moe | PP (tok/s) | Gen (tok/s) | Status |
|-----------|:----------:|:-----------:|--------|
| 0         | —          | —           | OOM |
| **16**    | 5.30       | **7.53**    | OK |
| 32        | 2.27       | 3.92        | OK |
| 64        | 2.52       | 2.78        | OK |
| 128       | 1.97       | 2.60        | OK |

Full JSON: `bench-colab-t4-20260704-1055.json`

## RTX 3060 — ckey.vn (12GB, Ryzen 5800X, 32GB DDR4)

**llama.cpp**: build 2d97363 (tự build source), CUDA 12.4

| Model | n-cpu-moe | PP (tok/s) | Gen (tok/s) | Ghi chú |
|-------|:---------:|:----------:|:-----------:|---------|
| Qwen3.5 Q3_K_M | 16 | 228.40 | **49.94** | Tối ưu |
| Qwen3.5 Q3_K_M | 32 | 151.04 | 43.44 | |
| Qwen3.5 Q3_K_M | 64 | 124.57 | 39.21 | |
| Qwen3.5 Q3_K_M | 128 | 121.67 | 39.48 | |
| Qwen3.6 UD-Q3_K_M | 16 | 222.35 | **54.39** | 9% nhanh hơn |
| Qwen3.6 APEX-MTP | 16 | _pending_ | _pending_ | MTP test |

## Comparison

| Setup | Gen (tok/s) | vs Colab |
|-------|:-----------:|:--------:|
| Colab T4 (Qwen3.5) | 7.5 | 1x |
| RTX 3060 + 5800X (Qwen3.5) | 49.9 | 6.6x |
| RTX 3060 + 5800X (Qwen3.6) | 54.4 | 7.2x |
| RTX 3060 + 5800X + MTP | _pending_ | _pending_ |

## Raw Data

- `bench-colab-t4-20260704-1055.json` — Colab T4 raw llama-bench output
- `bench-rtx3060-ckey-20260704.md` — RTX 3060 ckey.vn results
