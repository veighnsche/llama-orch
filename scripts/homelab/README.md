# Homelab Remote Testing CLI

**Created by:** TEAM-018, TEAM-022  
**Version:** 0.2.0

## Overview

**Architecture:** Local-only management scripts that can be called remotely via SSH.

**Local scripts** (in `scripts/`):
- **`llorch-models`** - Model download, verification, and management
- **`llorch-git`** - Git operations with submodule support

**Remote wrapper** (in `scripts/homelab/`):
- **`llorch-remote`** - Calls local scripts on remote hosts via SSH

## Installation

```bash
# Local scripts (from repo root)
cd /path/to/llama-orch
ln -s $(pwd)/scripts/llorch-models ~/.local/bin/llorch-models
ln -s $(pwd)/scripts/llorch-git ~/.local/bin/llorch-git

# Remote wrapper
cd scripts/homelab
ln -s $(pwd)/llorch-remote ~/.local/bin/llorch-remote

# Or use directly
./scripts/llorch-models --help
./scripts/llorch-git --help
./scripts/homelab/llorch-remote --help
```

## Quick Start

### Local Usage

```bash
# Model management (local)
llorch-models list
llorch-models download tinyllama
llorch-models info tinyllama

# Git management (local)
llorch-git status
llorch-git pull
llorch-git submodules
```

### Remote Usage

```bash
# Git operations on remote
llorch-remote mac.home.arpa status
llorch-remote mac.home.arpa pull

# Model management on remote
llorch-remote mac.home.arpa models-list
llorch-remote mac.home.arpa models-download tinyllama

# Build and test on remote
llorch-remote mac.home.arpa build metal
llorch-remote mac.home.arpa test metal

# Full workflow
llorch-remote mac.home.arpa all metal
```

## Usage

### llorch-models (local)

```
llorch-models <ACTION> [MODEL] [OPTIONS]
```

**Arguments:**
- **ACTION**: list, catalog, download, info, verify, delete, disk-usage
- **MODEL**: Model identifier (required for download/info/verify/delete)

**Available models:** tinyllama, qwen, qwen-fp16, phi3, llama3, llama32, llama2, mistral, gpt2, granite

### llorch-git (local)

```
llorch-git <ACTION> [OPTIONS]
```

**Arguments:**
- **ACTION**: status, pull, sync, submodules, submodule-branch, submodule-update, submodule-reset, clean, branches, log

**Submodules:** reference/candle, reference/candle-vllm, reference/llama.cpp, reference/mistral.rs

### llorch-remote (remote wrapper)

```
llorch-remote <HOST> <ACTION> [BACKEND] [OPTIONS]
```

**Arguments:**
- **HOST**: Remote host (e.g., `mac.home.arpa`, `workstation.home.arpa`)
- **ACTION**: Action to perform (see below)
- **BACKEND**: Backend type (`cpu`, `cuda`, `metal`) - required for build/test/inference actions

### llorch-remote Actions

| Action | Backend Required? | Description |
|--------|------------------|-------------|
| `status` | No | Show git status and system info (calls llorch-git) |
| `pull` | No | Pull latest changes and update submodules (calls llorch-git) |
| `sync` | No | Sync to origin/main hard reset (calls llorch-git) |
| `models-list` | No | List all models on remote (calls llorch-models) |
| `models-download` | No | Download model on remote (calls llorch-models) |
| `models-info` | No | Show model info on remote (calls llorch-models) |
| `build` | Yes | Build backend binary (release mode) |
| `test` | Yes | Run all tests for backend |
| `smoke` | Yes | Run smoke tests only |
| `unit` | Yes | Run unit tests only |
| `integration` | Yes | Run integration tests only |
| `inference` | Yes | Generate a test story with actual model |
| `debug-inference` | Yes | Run inference with detailed logging |
| `logs` | Yes | Show worker logs from last run |
| `info` | Yes | Show backend and hardware info |
| `clean` | No | Clean build artifacts |
| `all` | Yes | Run: pull → build → test → inference |

### llorch-remote Options

- `--model PATH` - Model path for inference (default: `$LLORCH_TEST_MODEL_PATH`)
- `--port PORT` - Port for worker (default: 8080)
- `--device ID` - Device ID for GPU backends (default: 0)
- `--help, -h` - Show help message
- `--version, -v` - Show version

### llorch-models Options

- `--force` - Force re-download even if model exists
- `--help, -h` - Show help message
- `--version, -v` - Show version

### llorch-git Options

- `--submodule NAME` - Submodule name (for submodule-* actions)
- `--branch NAME` - Branch name (for submodule-branch)
- `--all` - Apply to all submodules
- `--force` - Force operation (skip confirmations)
- `--help, -h` - Show help message
- `--version, -v` - Show version

## Examples

### Local Model Management

```bash
# View catalog with verified HuggingFace repos
llorch-models catalog

# Download TinyLlama (669 MB)
llorch-models download tinyllama

# Download Llama-3 8B (4.9 GB)
llorch-models download llama3

# List downloaded models
llorch-models list

# Show model info
llorch-models info tinyllama

# Verify model integrity
llorch-models verify tinyllama

# Delete model
llorch-models delete phi3

# Show disk usage
llorch-models disk-usage
```

### Local Git Management

```bash
# Show status with submodules
llorch-git status

# Pull latest changes
llorch-git pull

# Hard reset to origin/main
llorch-git sync

# List all submodules
llorch-git submodules

# Show candle submodule branch
llorch-git submodule-branch --submodule reference/candle

# Switch candle to metal-fixes branch
llorch-git submodule-branch --submodule reference/candle --branch metal-fixes

# Update candle to latest on current branch
llorch-git submodule-update --submodule reference/candle

# Reset candle to committed version
llorch-git submodule-reset --submodule reference/candle

# Update all submodules
llorch-git submodule-update --all
```

### Remote Operations

```bash
# Git operations on remote
llorch-remote mac.home.arpa status
llorch-remote mac.home.arpa pull
llorch-remote mac.home.arpa sync

# Model management on remote
llorch-remote mac.home.arpa models-list
llorch-remote mac.home.arpa models-download tinyllama
llorch-remote mac.home.arpa models-info tinyllama

# Build and test
llorch-remote workstation.home.arpa build cuda
llorch-remote mac.home.arpa test metal
```

### Testing

```bash
# Run all tests
./llorch-remote mac.home.arpa test metal

# Run smoke tests only
./llorch-remote mac.home.arpa smoke metal

# Run unit tests only
./llorch-remote workstation.home.arpa unit cuda

# Run integration tests
./llorch-remote mac.home.arpa integration metal
```

### Model Management

```bash
# List all models
./llorch-models mac.home.arpa list

# Download TinyLlama
./llorch-models mac.home.arpa download tinyllama

# Download Mistral with Q5 quantization
./llorch-models workstation.home.arpa download mistral --quant Q5_K_M

# Show model info
./llorch-models mac.home.arpa info tinyllama

# Verify model integrity
./llorch-models mac.home.arpa verify tinyllama

# Delete a model
./llorch-models mac.home.arpa delete phi2

# Show disk usage
./llorch-models mac.home.arpa disk-usage
```

### Inference

```bash
# Generate test story
./llorch-remote mac.home.arpa inference metal

# Debug inference with detailed logs
./llorch-remote mac.home.arpa debug-inference metal

# View logs from last run
./llorch-remote mac.home.arpa logs metal
```

### Full Workflow

```bash
# Pull, build, test, and run inference
./llorch-remote mac.home.arpa all metal
```

## Environment Variables

| Variable | Tool | Description | Default |
|----------|------|-------------|---------|
| `LLORCH_REPO_URL` | llorch-remote | Repository URL | `https://github.com/veighnsche/llama-orch.git` |
| `LLORCH_REMOTE_PATH` | Both | Remote path | `~/Projects/llama-orch` |
| `LLORCH_TEST_MODEL_PATH` | llorch-remote | Model path for inference tests | - |
| `LLORCH_MODEL_BASE_DIR` | llorch-models | Model base directory | `.test-models` |

## Backend-Specific Binaries

| Backend | Binary Name |
|---------|-------------|
| `cpu` | `llorch-cpu-candled` |
| `cuda` | `llorch-cuda-candled` |
| `metal` | `llorch-metal-candled` |

## SSH Configuration

The tool uses fail-fast SSH with:
- `BatchMode=yes` - No interactive prompts
- `ConnectTimeout=10` - 10-second timeout

Ensure SSH keys are configured for passwordless access:

```bash
# Generate SSH key (if needed)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy to remote host
ssh-copy-id user@mac.home.arpa
```

## Homelab Setup

### Example Hosts

| Hostname | Role | Backend | Hardware |
|----------|------|---------|----------|
| `blep.home.arpa` | Dev Box | CPU | x86_64 CPU |
| `workstation.home.arpa` | GPU Box | CUDA | NVIDIA RTX 3060/3090 |
| `mac.home.arpa` | Apple Silicon | Metal | M1/M2/M3/M4 |

### SSH Config

Add to `~/.ssh/config`:

```
Host mac.home.arpa
    HostName mac.home.arpa
    User vinceliem
    IdentityFile ~/.ssh/id_ed25519

Host workstation.home.arpa
    HostName workstation.home.arpa
    User vince
    IdentityFile ~/.ssh/id_ed25519
```

## Observability Hooks

The tool includes `# TEAM-019:` comment markers for future observability integration:
- Structured logging
- Metrics collection (Prometheus)
- Build/test telemetry
- Real-time streaming (SSE)

See `.specs/TEAM_019_HANDOFF.md` for details.

## Troubleshooting

### SSH Connection Failed

```bash
# Test SSH connection
ssh -o BatchMode=yes -o ConnectTimeout=10 mac.home.arpa echo "OK"

# Check SSH keys
ssh-add -l
```

{{ ... }}

```bash
# Run smoke tests only
./llorch-remote mac.home.arpa smoke metal

# View test logs
./llorch-remote mac.home.arpa logs metal
```

### Inference Failed

```bash
# Run debug inference with detailed logging
./llorch-remote mac.home.arpa debug-inference metal

# View worker logs
./llorch-remote mac.home.arpa logs metal

# Check if model exists
./llorch-models mac.home.arpa list
```

### Common Issues

**Model not found:**
```bash
./llorch-models mac.home.arpa download tinyllama
```

**Worker crashes on startup:**
```bash
# Check detailed logs
./llorch-remote mac.home.arpa debug-inference metal
./llorch-remote mac.home.arpa logs metal
```

**Broadcasting errors (Metal/CUDA):**
- Fixed in TEAM-019 (cache recreation workaround)
- See `.specs/METAL_CUDA_INFERENCE_BUG_REPORT.md` for details

## Migration from Old Scripts

Old hardcoded scripts:
```bash
{{ ... }}
./scripts/homelab/mac-test.sh
```

New CLI equivalent:
```bash
./llorch-remote mac.home.arpa build metal
./llorch-remote mac.home.arpa test metal
```

## Available Models

| Model | Parameters | Size (Q4_K_M) | Description |
|-------|-----------|---------------|-------------|
| `tinyllama` | 1.1B | 669 MB | Standard Llama architecture, simplest for testing |
| `phi2` | 2.7B | 1.6 GB | Microsoft's efficient model, good quality |
| `mistral` | 7B | 4.1 GB | High quality instruction following |

## Future Enhancements

- [x] Model download/staging (llorch-models)
- [ ] Benchmark mode
- [ ] Parallel execution (multiple hosts)
- [ ] JSON output mode
- [ ] CI/CD integration
- [ ] Metrics export (Prometheus)
- [ ] Real-time log streaming
- [ ] Interactive mode (TUI)
- [ ] Model conversion utilities
- [ ] Automatic quantization selection

## See Also

- `.specs/TEAM_018_HANDOFF.md` - Metal migration details
- `.specs/TEAM_019_HANDOFF.md` - Observability roadmap
- `bin/rbees-workerd/docs/metal.md` - Metal backend guide
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Multi-backend architecture

---

**Created:** 2025-10-09  
**Team:** TEAM-018  
**Status:** ✅ Production-ready
