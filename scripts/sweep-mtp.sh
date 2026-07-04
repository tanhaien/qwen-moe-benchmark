#!/bin/bash
# MTP Sweep Benchmark — run all configs and compare
# Usage: bash sweep-mtp.sh
# Requires: llama-server built with CUDA, APEX-MTP-Compact.gguf

MODEL=# set path to APEX-MTP-Compact.gguf
PORT=8080

bench() {
    local label="$1"
    shift

    pkill -f "llama-server.*$PORT" 2>/dev/null
    sleep 2

    llama-server \
        -m "$MODEL" "$@" \
        -c 8192 -fa on -np 1 --no-mmap \
        -ctk q8_0 -ctv q8_0 \
        --host 127.0.0.1 --port $PORT &
    local pid=$!

    for i in $(seq 1 20); do
        sleep 3
        curl -s -m 3 "http://127.0.0.1:$PORT/completion" \
            -d '{"prompt":"x","n_predict":1}' | grep -q content && break
    done

    echo "=== $label ==="
    python3 mtp-bench.py
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    echo ""
}

# Sweep spec-draft-n-max
for n in 2 3 4; do
    bench "n-cpu-moe=16 + MTP n=$n" \
        -ngl 99 --n-cpu-moe 16 \
        --spec-type draft-mtp --spec-draft-n-max $n
done

# Sweep n-cpu-moe
for moe in 8 32; do
    bench "n-cpu-moe=$moe + MTP n=3" \
        -ngl 99 --n-cpu-moe $moe \
        --spec-type draft-mtp --spec-draft-n-max 3
done

# Sweep -fitt
for fitt in 2048 2560; do
    bench "fitt=$fitt + MTP n=3" \
        -fitt $fitt \
        --spec-type draft-mtp --spec-draft-n-max 3
done
