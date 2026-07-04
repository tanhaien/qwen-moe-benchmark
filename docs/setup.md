# Setup Guide

## 1. Install Colab CLI

```bash
uv tool install google-colab-cli
colab --help
```

Requires Python 3.13+ — `uv tool install` handles this automatically.

## 2. Authenticate

### Option A: OAuth2 (browser)
```bash
colab --auth=oauth2 new
# Opens URL → login Google → copy code → paste
```

### Option B: ADC (gcloud, agent-friendly)
```bash
gcloud auth application-default login --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory

colab --auth=adc whoami
```

## 3. Run Benchmark

### One-shot
```bash
chmod +x scripts/colab-bench.sh
./scripts/colab-bench.sh [--gpu T4] [--quant Q3_K_M]
```

### Manual steps
```bash
# Provision
colab new -s bench --gpu T4

# Run benchmark script
colab exec -f scripts/colab_bench.py

# Download results
colab download -s bench results.json ./results/bench-$(date +%Y%m%d-%H%M).json

# Cleanup
colab stop -s bench
```

### Interactive (tmux console)
```bash
# Upload script first
colab upload -s bench /path/to/colab_bench.sh /content/run.sh

# Run via raw TTY (no Jupyter timeout)
echo "bash /content/run.sh" | colab console -s bench
```

## Colab CLI Quick Reference

| Command | Description |
|---------|-------------|
| `colab new -s NAME --gpu T4` | Provision T4 GPU VM |
| `colab exec -f FILE` | Run local Python/notebook file |
| `colab console -s NAME` | Raw TTY shell (tmux) |
| `colab install PKG` | Install packages on VM |
| `colab upload LOCAL REMOTE` | Upload file to VM |
| `colab download REMOTE LOCAL` | Download from VM |
| `colab stop -s NAME` | Teardown VM |
| `colab sessions` | List active sessions |
| `colab log -s NAME -n 20` | View session log |

## GPU Types

| Flag | GPU | VRAM | Access |
|------|-----|------|--------|
| (none) | CPU | - | Free |
| `--gpu T4` | T4 | 16GB | Free |
| `--gpu L4` | L4 | 24GB | Pro |
| `--gpu A100` | A100 | 40GB | Pro+ |
| `--gpu H100` | H100 | 80GB | Pro+ |

## Troubleshooting

- **"Connection was lost"**: Jupyter kernel timeout. Use `colab console` instead
- **"redirect_uri_mismatch"**: Use `--auth=adc` instead of default OAuth2
- **"No GPU quota"**: Try smaller GPU type or upgrade Colab plan
- **Build too slow**: Use ai-dock pre-built binaries (save 30+ min)
- **Colab VM disk**: ~66GB free, enough for model + binaries
