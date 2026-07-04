#!/usr/bin/env bash
set -euo pipefail

# colab-bench.sh — Orchestrate Qwen MoE benchmark on Google Colab T4
# Usage: colab-bench.sh [--gpu T4|L4|A100] [--quant Q3_K_M|Q4_K_M|IQ4_XS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="qwen-bench-$(date +%Y%m%d-%H%M)"
GPU="${GPU:-T4}"
QUANT="${QUANT:-Q3_K_M}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: colab-bench.sh [--gpu T4|L4|A100] [--quant Q3_K_M|Q4_K_M|IQ4_XS]"
            echo ""
            echo "Runs Qwen3.5-35B-A3B MoE benchmark on Colab T4."
            echo "Downloads pre-built llama.cpp CUDA binary + GGUF model."
            echo ""
            echo "Environment:"
            echo "  N_CPU_MOE=0,16,32,64,128  (default)"
            echo "  BENCH_N=256                (tokens to generate)"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }

echo ""
echo "=========================================="
echo "  Qwen MoE Benchmark Orchestrator"
echo "  GPU: $GPU | Model: Q3_K_M"
echo "=========================================="
echo ""

command -v colab >/dev/null 2>&1 || fail "colab CLI not found. Run: uv tool install google-colab-cli"

echo ">>> Step 1: Checking auth..."
colab sessions 2>&1 | grep -q "Error" && warn "Auth needed: colab --auth=adc whoami"

echo ">>> Step 2: Provisioning $GPU VM..."
colab new -s "$SESSION_NAME" --gpu "$GPU" 2>&1 || fail "Failed to provision VM"
ok "VM provisioned"

echo ">>> Step 3: Uploading benchmark..."
colab upload -s "$SESSION_NAME" "$REPO_DIR/scripts/colab_bench.py" /content/colab_bench.py 2>&1
ok "Script uploaded"

echo ">>> Step 4: Running benchmark (10-15 min)..."
echo "  - Downloads pre-built llama.cpp CUDA binary"
echo "  - Downloads Qwen3.5-35B-A3B-Q3_K_M.gguf (~16GB)"
echo "  - Benchmarks n-cpu-moe = 0, 16, 32, 64, 128"
echo ""

# Note: Using 'console' instead of 'exec' to avoid Jupyter kernel timeout
{ echo "python3 /content/colab_bench.py 2>&1"; cat; } | colab --auth=adc console -s "$SESSION_NAME" 2>&1

echo ""
echo ">>> Step 5: Downloading results..."
RESULTS_LOCAL="$REPO_DIR/results/bench-$(date +%Y%m%d-%H%M).json"
mkdir -p "$REPO_DIR/results"
colab download -s "$SESSION_NAME" /content/results.json "$RESULTS_LOCAL" 2>&1 || warn "Download failed"
ok "Results saved to: $RESULTS_LOCAL"

echo ">>> Step 6: Teardown..."
colab stop -s "$SESSION_NAME" 2>&1 || warn "Stop had issues"
ok "VM released"

echo ""
echo "=========================================="
echo "  BENCHMARK COMPLETE"
echo "  Results: $RESULTS_LOCAL"
echo "=========================================="
python3 -c "
import json
with open('$RESULTS_LOCAL') as f:
    d = json.load(f)
print(f'  GPU: {d.get(\"gpu\",{}).get(\"info\",\"?\")}')
print(f'  Model: {d.get(\"config\",{}).get(\"model_size_gb\",\"?\")} GB')
print()
for b in d.get('benchmarks', []):
    print(f'  n-cpu-moe={b[\"n_cpu_moe\"]:3d}  |  {str(b.get(\"result\",\"ERROR\")):>20s}')
" 2>&1
