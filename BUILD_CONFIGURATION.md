# Build Configuration Guide

## Overview

llama-orch uses a local configuration file (`.llorch.toml`) to customize build behavior per machine. This file is **gitignored** and machine-specific.

## Quick Start

### 1. Create Your Local Config

```bash
# Copy the example file
cp .llorch.toml.example .llorch.toml

# Edit for your machine
vim .llorch.toml
```

### 2. Configure for Your Setup

#### GPU Development Machine (Default)

```toml
[build]
cuda = true
auto_detect_cuda = false
```

**Result**: Builds with CUDA support (requires CUDA toolkit installed)

#### CPU-only Development Machine

```toml
[build]
cuda = false
auto_detect_cuda = false
```

**Result**: Builds in stub mode (no CUDA required)

#### Laptop (Auto-detect)

```toml
[build]
cuda = true
auto_detect_cuda = true
```

**Result**: Automatically detects CUDA and builds accordingly

## Build Behavior

### Priority Order

1. **Explicit feature flag** (highest priority)
   ```bash
   cargo build --features cuda  # Always builds with CUDA
   ```

2. **Local config file** (`.llorch.toml`)
   - Reads `build.cuda` setting
   - If `auto_detect_cuda = true`, attempts detection

3. **Default** (if no config file exists)
   - `cuda = true` (assumes CUDA available)
   - `auto_detect_cuda = false`

### CUDA Detection

When `auto_detect_cuda = true`, the build script checks:

1. ✅ `nvcc --version` (CUDA compiler in PATH)
2. ✅ `CUDA_PATH` environment variable
3. ✅ Common installation paths:
   - `/usr/local/cuda`
   - `/opt/cuda`
   - `/usr/lib/cuda`

## Examples

### Example 1: First-time Setup (No GPU)

```bash
# Clone repo
git clone <repo>
cd llama-orch

# Create local config for CPU-only development
cp .llorch.toml.example .llorch.toml
# Edit: set cuda = false

# Build successfully without CUDA
cargo build -p worker-orcd
```

### Example 2: GPU Machine

```bash
# Clone repo (on GPU machine)
git clone <repo>
cd llama-orch

# No config needed - defaults to CUDA enabled
cargo build -p worker-orcd
# Builds with CUDA (default behavior)
```

### Example 3: Shared Dev Machine

```bash
# Machine has GPU but you want to test stub mode
cp .llorch.toml.example .llorch.toml
# Edit: set cuda = false

cargo build -p worker-orcd
# Builds WITHOUT CUDA despite GPU being available
```

## Configuration Reference

### `[build]` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cuda` | bool | `true` | Enable CUDA compilation |
| `auto_detect_cuda` | bool | `false` | Auto-detect CUDA toolkit |

### `[worker]` Section (Future)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_gpu_device` | int | `0` | Default GPU for local testing |
| `default_model` | string | - | Default model path for testing |

### `[development]` Section (Future)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `verbose_build` | bool | `false` | Enable verbose build output |
| `skip_cuda_tests` | bool | `false` | Skip CUDA tests even when available |

## Troubleshooting

### Build fails with "nvcc not found"

**Cause**: Config has `cuda = true` but CUDA toolkit not installed

**Solution**:
```bash
# Option 1: Install CUDA toolkit
# Option 2: Disable CUDA in config
echo '[build]\ncuda = false' > .llorch.toml
```

### Build succeeds but CUDA not detected

**Cause**: Auto-detection failed to find CUDA

**Solution**:
```bash
# Check CUDA installation
nvcc --version
echo $CUDA_PATH

# Explicitly enable in config
echo '[build]\ncuda = true\nauto_detect_cuda = false' > .llorch.toml
```

### Config file not being read

**Cause**: File must be named exactly `.llorch.toml` (with leading dot)

**Solution**:
```bash
# Check filename
ls -la .llorch.toml

# Verify it's in repo root
pwd  # Should be /path/to/llama-orch
```

## CI/CD Integration

### GitHub Actions

```yaml
# CPU-only CI
- name: Build worker-orcd (stub mode)
  run: |
    echo '[build]\ncuda = false' > .llorch.toml
    cargo build -p worker-orcd

# GPU CI (self-hosted runner)
- name: Build worker-orcd (CUDA)
  run: |
    echo '[build]\ncuda = true' > .llorch.toml
    cargo build -p worker-orcd
```

### Docker

```dockerfile
# Development image (no CUDA)
FROM rust:1.75
WORKDIR /app
RUN echo '[build]\ncuda = false' > .llorch.toml
COPY . .
RUN cargo build --release

# Production image (with CUDA)
FROM nvidia/cuda:12.0-devel
WORKDIR /app
RUN echo '[build]\ncuda = true' > .llorch.toml
COPY . .
RUN cargo build --release --features cuda
```

## Design Rationale

### Why Not Auto-detect by Default?

**Deterministic builds** are critical for:
- ✅ Reproducible builds across machines
- ✅ Consistent CI/CD behavior
- ✅ Explicit intent in production deployments
- ✅ Easier debugging (no surprises)

### Why Local Config File?

**Machine-specific settings** without polluting git:
- ✅ Each developer configures once
- ✅ Not committed to version control
- ✅ No merge conflicts
- ✅ Clear separation of concerns

### Why Default to CUDA?

**Production-first approach**:
- ✅ worker-orcd requires GPU in production
- ✅ Fails fast if CUDA not available
- ✅ Developers explicitly opt into stub mode
- ✅ Prevents accidental CPU-only deployments

## See Also

- `.llorch.toml.example` - Example configuration file
- `bin/worker-orcd/CUDA_FEATURE.md` - CUDA feature flag documentation
- `bin/worker-orcd/build.rs` - Build script implementation
