# sd-worker-rbee

**Stable Diffusion inference worker for rbee**

---

## What Is This?

`sd-worker-rbee` is a Candle-based Stable Diffusion worker that generates images from text prompts. It's spawned by `rbee-hive` and provides HTTP endpoints for image generation.

### Features

- ✅ **Text-to-image** generation
- ✅ **Image-to-image** transformation  
- ✅ **Inpainting** support
- ✅ **Multiple backends** (CPU, CUDA, Metal)
- ✅ **Streaming progress** via SSE
- ✅ **Multiple SD models** (1.5, 2.1, XL, Turbo, SD3)

---

## Binaries

Three feature-gated binaries for different hardware:

| Binary | Features | Platforms |
|--------|----------|-----------|
| `sd-worker-rbee-cpu` | `--features cpu` | Linux, macOS, Windows |
| `sd-worker-rbee-cuda` | `--features cuda` | Linux, Windows (NVIDIA GPU) |
| `sd-worker-rbee-metal` | `--features metal` | macOS (Apple Silicon) |

---

## Building

```bash
# CPU variant
cargo build --release --no-default-features --features cpu

# CUDA variant (requires CUDA toolkit)
cargo build --release --no-default-features --features cuda

# Metal variant (macOS only)
cargo build --release --no-default-features --features metal
```

Binaries output to: `../../target/release/sd-worker-rbee`

---

## Installation via PKGBUILD

Available in the Worker Catalog:

```bash
# Download PKGBUILD
curl http://localhost:8787/workers/sd-worker-rbee-cpu/PKGBUILD > PKGBUILD

# Build and install
makepkg -si
```

Variants: `sd-worker-rbee-cpu`, `sd-worker-rbee-cuda`, `sd-worker-rbee-metal`

---

## Usage

### Spawned by rbee-hive

The worker is spawned by `rbee-hive` with:
- Port assignment
- Model path
- Device configuration

### HTTP API

```bash
# Health check
GET /health

# Generate image from text
POST /execute
{
  "prompt": "A beautiful sunset over mountains",
  "steps": 30,
  "guidance_scale": 7.5
}

# Image-to-image
POST /img2img
{
  "prompt": "Same scene but at night",
  "image": "base64_encoded_image",
  "strength": 0.8
}

# Inpainting
POST /inpaint
{
  "prompt": "Add a lake in the foreground",
  "image": "base64_encoded_image",
  "mask": "base64_encoded_mask"
}
```

### Streaming Progress

All generation endpoints support SSE for real-time progress:

```
Accept: text/event-stream
```

---

## Architecture

```
┌─────────────────────────────────────┐
│       sd-worker-rbee Worker          │
├─────────────────────────────────────┤
│  HTTP Server (Axum)                 │
│  ├─ /health                         │
│  ├─ /execute (text-to-image)        │
│  ├─ /img2img                        │
│  └─ /inpaint                        │
├─────────────────────────────────────┤
│  CandleSDBackend                    │
│  ├─ CLIP Text Encoder               │
│  ├─ UNet / MMDiT                    │
│  ├─ VAE (Encoder/Decoder)           │
│  └─ Scheduler (DDIM/Euler)          │
├─────────────────────────────────────┤
│  Candle (ML Framework)              │
│  └─ Device (CPU/CUDA/Metal)         │
└─────────────────────────────────────┘
```

---

## Models

Supports multiple Stable Diffusion models:

- **SD 1.5** - Classic Stable Diffusion
- **SD 2.1** - Improved quality
- **SDXL** - High resolution (1024x1024)
- **SD Turbo** - Fast generation (1-4 steps)
- **SD 3 / 3.5** - Latest models with MMDiT

Models are loaded from `~/.cache/rbee/models/` by rbee-hive.

---

## Configuration

Configured via command-line arguments (set by rbee-hive):

```bash
sd-worker-rbee \
  --port 8080 \
  --model-path ~/.cache/rbee/models/stable-diffusion-v1-5 \
  --device cuda \
  --worker-id worker-123
```

---

## Development

### Project Structure

```
bin/31_sd_worker_rbee/
├── src/
│   ├── main.rs           # Entry point
│   ├── backend/          # Candle SD backend
│   ├── http/             # HTTP server (Axum)
│   ├── generation/       # Image generation logic
│   └── models/           # Model loading
├── ui/                   # Web UI (optional)
├── Cargo.toml            # Dependencies
└── README.md             # This file
```

### Running Locally

```bash
# Build
cargo build --release --features cpu

# Run
../../target/release/sd-worker-rbee \
  --port 8080 \
  --model-path /path/to/sd-model \
  --device cpu
```

### Testing

```bash
# Unit tests
cargo test --features cpu

# Integration tests
cargo test --features cpu --test '*'
```

---

## Dependencies

- **Candle** - ML framework
- **candle-transformers** - Stable Diffusion implementation
- **Axum** - HTTP server
- **Tokio** - Async runtime
- **Serde** - JSON serialization

---

## License

GPL-3.0-or-later

---

## Implementation Status

### ✅ Backend (Production Ready)
- **Text-to-image generation** - Fully implemented
- **Image-to-image transformation** - Fully implemented (TEAM-487)
- **Inpainting with masks** - Fully implemented (TEAM-487)
- **LoRA support** - Fully implemented (TEAM-488)
- **Multiple SD models** - 1.5, 2.1, XL, Turbo, inpainting variants
- **Streaming progress** - SSE implemented
- **HTTP API** - `/v1/jobs` endpoint operational
- **Job queue system** - Full job management

### ⚠️ UI (Stub Implementation)
- **WASM SDK** - Stub implementation (TEAM-391, awaiting TEAM-399+)
- **React hooks** - Stub implementation
- **Web UI** - Basic structure only

### ❌ Not Yet Implemented
- ControlNet support
- ROCm (AMD GPU) variant
- SD 3/3.5 models
- FLUX integration (partial)

**Bottom Line:** The worker backend is fully functional via HTTP API. Only the UI SDK needs completion.

---

## Documentation

- **Archived docs**: `.archive/` - Historical implementation docs
- **Worker Catalog**: `bin/80-hono-worker-catalog/` - PKGBUILD files
- **Hive Integration**: `bin/20_rbee_hive/` - Worker spawning logic

---

**Status:** ✅ Backend Production Ready, UI SDK Stubbed

**Version:** 0.1.0  
**Created by:** TEAM-390+  
**Last Updated:** 2025-11-04
