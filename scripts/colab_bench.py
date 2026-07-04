#!/usr/bin/env python3
"""
colab_bench.py — Qwen MoE benchmark runner for Colab VM.

Downloads pre-built llama.cpp CUDA binary + GGUF model,
then benchmarks --n-cpu-moe values and saves results.

Usage (on Colab VM):
    python3 /content/colab_bench.py

Environment:
    MODEL_QUANT: Q3_K_M (default) or Q4_K_M, IQ4_XS, etc.
    N_CPU_MOE: comma-separated values (default: 0,16,32,64,128)
    BENCH_N: tokens to generate (default: 256)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

# Config
MODEL_QUANT = os.getenv("MODEL_QUANT", "Q3_K_M")
MODEL_FILE = f"Qwen3.5-35B-A3B-{MODEL_QUANT}.gguf"
MODEL_URL = f"https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/{MODEL_FILE}"

LLAMA_VERSION = os.getenv("LLAMA_VERSION", "b9860")
LLAMA_URL = f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/{LLAMA_VERSION}/llama.cpp-{LLAMA_VERSION}-cuda-12.8-amd64.tar.gz"

N_CPU_MOE_VALUES = [int(x) for x in os.getenv("N_CPU_MOE", "0,16,32,64,128").split(",")]
BENCH_N = int(os.getenv("BENCH_N", "256"))

BIN_DIR = "/content/llama-cuda/cuda-12.8"
MODEL_PATH = f"/content/{MODEL_FILE}"
RESULTS_PATH = "/content/results.json"


def log(msg):
    print(f"[BENCH] {msg}", flush=True)


def run(cmd, timeout=300):
    """Run subprocess and return CompletedProcess."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def download_prebuilt_llamacpp():
    """Download pre-built llama.cpp CUDA binary from ai-dock."""
    log("=== Downloading pre-built llama.cpp CUDA binary ===")
    
    if os.path.exists(f"{BIN_DIR}/llama-bench"):
        log("Binary already exists")
        return
    
    # Download
    tar_path = "/tmp/llama.tar.gz"
    log(f"Downloading {LLAMA_URL}...")
    urllib.request.urlretrieve(LLAMA_URL, tar_path)
    log("Extracting...")
    
    os.makedirs("/content/llama-cuda", exist_ok=True)
    subprocess.run(["tar", "-xzf", tar_path, "-C", "/content/llama-cuda"], check=True)
    os.remove(tar_path)
    
    # Verify
    for b in ["llama-bench", "llama-server", "llama-cli"]:
        p = f"{BIN_DIR}/{b}"
        if os.path.exists(p):
            os.chmod(p, 0o755)
            ver = subprocess.run([p, "--version"], capture_output=True, text=True).stdout[:80]
            log(f"  ✓ {b}: {ver}")
        else:
            log(f"  ✗ {b} not found!")
    
    if os.path.exists(f"{BIN_DIR}/VERSION.txt"):
        v = open(f"{BIN_DIR}/VERSION.txt").read().strip()
        log(f"  Version: {v}")


def download_model():
    """Download GGUF model from HuggingFace."""
    log("=== Downloading model ===")
    
    if os.path.exists(MODEL_PATH):
        sz = os.path.getsize(MODEL_PATH) / (1024**3)
        log(f"Model exists: {sz:.1f} GB")
        return
    
    log(f"Downloading {MODEL_FILE} (this may take 3-5 min)...")
    log(f"URL: {MODEL_URL}")
    
    start = time.time()
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    elapsed = time.time() - start
    sz = os.path.getsize(MODEL_PATH) / (1024**3)
    log(f"Downloaded: {sz:.1f} GB in {elapsed/60:.1f} min")


def gpu_info():
    """Get GPU info string."""
    r = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"])
    return r.stdout.strip()


def bench_n_cpu_moe(n_cpu_moe):
    """Benchmark with a specific --n-cpu-moe value using llama-bench."""
    log(f"--- Benchmark: --n-cpu-moe {n_cpu_moe} ---")
    
    entry = {"n_cpu_moe": n_cpu_moe}
    
    cmd = [
        f"{BIN_DIR}/llama-bench",
        "-m", MODEL_PATH,
        "-ngl", "99",
        "--n-cpu-moe", str(n_cpu_moe),
        "--flash-attn", "1",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "-p", "64",
        "-n", str(BENCH_N),
        "-t", "8",
        "-b", "512",
        "-ub", "128",
    ]
    
    try:
        r = run(cmd, timeout=300)
        entry["output"] = r.stdout
        
        # Parse llama-bench output
        for line in r.stdout.split("\n"):
            line = line.strip()
            if not line or not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) >= 9 and cols[0].isdigit():
                entry.update({
                    "model_size": cols[1],
                    "backend": cols[2],
                    "n_batch": cols[3],
                    "n_ubatch": cols[4],
                    "flash_attn": cols[5],
                    "n_gpu_layers": cols[6],
                    "n_cpu_moe_actual": cols[7],
                    "result": cols[8],
                    "test": cols[0],
                })
                log(f"  → {cols[8]}")
                break
            # Fallback parse
            if "tok/s" in line.lower() and "|" not in line:
                for part in line.split():
                    if "/s" in part and not part.startswith("|"):
                        entry["result"] = part
                        log(f"  → {part}")
                        break
    except subprocess.TimeoutExpired:
        entry["error"] = "TIMEOUT"
        log("  TIMEOUT")
    except Exception as e:
        entry["error"] = str(e)
        log(f"  ERROR: {e}")
    
    return entry


def main():
    log("=" * 60)
    log("Qwen3.5-35B-A3B MoE Benchmark on Colab T4")
    log("=" * 60)
    
    # GPU info
    gpu = gpu_info()
    log(f"GPU: {gpu}")
    
    # Download pre-built binaries
    download_prebuilt_llamacpp()
    
    # Download model
    download_model()
    
    # Run benchmarks
    results = {
        "gpu": {"info": gpu},
        "config": {
            "model": MODEL_FILE,
            "llama_version": LLAMA_VERSION,
            "n_predict": BENCH_N,
            "n_cpu_moe_values": N_CPU_MOE_VALUES,
            "model_size_gb": round(os.path.getsize(MODEL_PATH) / (1024**3), 1) if os.path.exists(MODEL_PATH) else None,
        },
        "benchmarks": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    for n in N_CPU_MOE_VALUES:
        entry = bench_n_cpu_moe(n)
        results["benchmarks"].append(entry)
        time.sleep(1)
    
    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for b in results["benchmarks"]:
        r = b.get("result", "ERROR")
        log(f"  n-cpu-moe={b['n_cpu_moe']:3d}  |  {str(r):>20s}")
    
    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n✓ Results: {RESULTS_PATH}")
    
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
