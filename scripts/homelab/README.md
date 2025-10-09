# Homelab Remote Testing CLI

**Created by:** TEAM-018  
**Version:** 0.1.0

## Overview

`llorch-remote` is a comprehensive CLI tool for managing remote builds, tests, and inference across different backends (CPU, CUDA, Metal) via SSH.

## Installation

```bash
# Add to PATH (optional)
ln -s $(pwd)/llorch-remote ~/.local/bin/llorch-remote

# Or use directly
./llorch-remote --help
```

## Quick Start

```bash
# Clone repository to remote host
./llorch-remote mac.home.arpa metal clone

# Build Metal backend
./llorch-remote mac.home.arpa metal build

# Run tests
./llorch-remote mac.home.arpa metal test

# Generate test story
./llorch-remote mac.home.arpa metal inference

# Full workflow (pull → build → test → inference)
./llorch-remote mac.home.arpa metal all
```

## Usage

```
llorch-remote <HOST> <BACKEND> <ACTION> [OPTIONS]
```

### Arguments

- **HOST**: Remote host (e.g., `mac.home.arpa`, `workstation.home.arpa`)
- **BACKEND**: Backend type (`cpu`, `cuda`, `metal`)
- **ACTION**: Action to perform (see below)

### Actions

| Action | Description |
|--------|-------------|
| `clone` | Clone repository to remote host |
| `pull` | Pull latest changes from origin/main |
| `status` | Show git status and system info |
| `build` | Build backend binary (release mode) |
| `test` | Run all tests for backend |
| `smoke` | Run smoke tests only |
| `unit` | Run unit tests only |
| `integration` | Run integration tests only |
| `inference` | Generate a test story (requires model) |
| `clean` | Clean build artifacts |
| `info` | Show backend and hardware info |
| `all` | Run: pull → build → test → inference |

### Options

- `--model PATH` - Model path for inference (default: `$LLORCH_TEST_MODEL_PATH`)
- `--port PORT` - Port for worker (default: 8080)
- `--device ID` - Device ID for GPU backends (default: 0)
- `--help, -h` - Show help message
- `--version, -v` - Show version

## Examples

### Basic Operations

```bash
# Check system info
./llorch-remote mac.home.arpa metal info

# Pull latest code
./llorch-remote mac.home.arpa metal pull

# Build CUDA backend on workstation
./llorch-remote workstation.home.arpa cuda build
```

### Testing

```bash
# Run all tests
./llorch-remote mac.home.arpa metal test

# Run smoke tests only
./llorch-remote mac.home.arpa metal smoke

# Run unit tests only
./llorch-remote workstation.home.arpa cuda unit

# Run integration tests
./llorch-remote mac.home.arpa metal integration
```

### Inference

```bash
# Generate test story (placeholder)
./llorch-remote mac.home.arpa metal inference

# With actual model (future)
./llorch-remote mac.home.arpa metal inference --model /path/to/model
```

### Full Workflow

```bash
# Pull, build, test, and run inference
./llorch-remote mac.home.arpa metal all
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLORCH_REPO_URL` | Repository URL | `https://github.com/veighnsche/llama-orch.git` |
| `LLORCH_REMOTE_PATH` | Remote path | `~/Projects/llama-orch` |
| `LLORCH_TEST_MODEL_PATH` | Model path for inference tests | - |

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

### Build Failed

```bash
# Check remote Rust toolchain
./llorch-remote mac.home.arpa metal info

# Clean and rebuild
./llorch-remote mac.home.arpa metal clean
./llorch-remote mac.home.arpa metal build
```

### Tests Failed

```bash
# Run smoke tests only
./llorch-remote mac.home.arpa metal smoke

# Check logs
ssh mac.home.arpa "cd ~/Projects/llama-orch && cargo test --features metal 2>&1 | tail -100"
```

## Migration from Old Scripts

Old hardcoded scripts:
```bash
./scripts/homelab/mac-build.sh
./scripts/homelab/mac-test.sh
```

New CLI equivalent:
```bash
./llorch-remote mac.home.arpa metal build
./llorch-remote mac.home.arpa metal test
```

## Future Enhancements

- [ ] Model download/staging
- [ ] Benchmark mode
- [ ] Parallel execution (multiple hosts)
- [ ] JSON output mode
- [ ] CI/CD integration
- [ ] Metrics export (Prometheus)
- [ ] Real-time log streaming
- [ ] Interactive mode (TUI)

## See Also

- `.specs/TEAM_018_HANDOFF.md` - Metal migration details
- `.specs/TEAM_019_HANDOFF.md` - Observability roadmap
- `bin/llorch-candled/docs/metal.md` - Metal backend guide
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Multi-backend architecture

---

**Created:** 2025-10-09  
**Team:** TEAM-018  
**Status:** ✅ Production-ready
