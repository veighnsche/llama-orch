# Apple Metal Backend (Pre-Release)

**Status:** 🚧 Pre-release  
**Platform:** macOS with Apple Silicon (M1/M2/M3/M4)  
**Created:** 2025-10-09 by TEAM-018

---

## Overview

The Metal backend enables GPU-accelerated inference on Apple Silicon Macs using Apple's Metal API. This is Apple's equivalent to NVIDIA CUDA and provides significant performance improvements over CPU-based inference.

### Metal vs Accelerate

| Feature | Accelerate | Metal |
|---------|-----------|-------|
| **Hardware** | CPU | GPU |
| **Framework** | Apple Accelerate | Apple Metal |
| **Performance** | ~10-20 tok/s | ~50-100 tok/s (estimated) |
| **Use Case** | CPU-optimized | GPU inference |
| **Binary** | ❌ Removed | `llorch-metal-candled` |

**Key Point:** Accelerate has been **removed** from llorch-candled as of TEAM-018. It was CPU-bound and too slow for production inference. Metal is the recommended backend for Apple Silicon.

---

## Requirements

### Hardware
- **Apple Silicon Mac** (M1, M2, M3, M4 series)
- Minimum 8GB unified memory (16GB+ recommended for larger models)

### Software
- **macOS 10.15+** (Catalina or later)
- **Xcode Command Line Tools** (for Metal framework)

### Not Supported
- ❌ Intel-based Macs (use CPU backend instead)
- ❌ Older Apple Silicon (A-series chips)

---

## Building

### Build Metal Binary

```bash
cd bin/llorch-candled

# Build release binary with Metal support
cargo build --release --features metal --bin llorch-metal-candled

# Binary location
./target/release/llorch-metal-candled
```

### Verify Build

```bash
# Check binary exists and shows help
./target/release/llorch-metal-candled --help
```

---

## Running

### Basic Usage

```bash
./target/release/llorch-metal-candled \
  --worker-id test-worker-001 \
  --model /path/to/llama-2-7b/ \
  --port 8080 \
  --callback-url http://localhost:9999 \
  --metal-device 0
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--worker-id` | ✅ | - | Worker UUID (assigned by pool-managerd) |
| `--model` | ✅ | - | Path to model directory (GGUF or SafeTensors) |
| `--port` | ✅ | - | HTTP server port |
| `--callback-url` | ✅ | - | Pool manager callback URL |
| `--metal-device` | ❌ | 0 | Metal device ID (usually 0) |

### Test Mode

To run without pool-managerd callback:

```bash
./target/release/llorch-metal-candled \
  --worker-id test-worker \
  --model /path/to/model/ \
  --port 8080 \
  --callback-url http://localhost:9999
```

The worker will skip the callback if URL contains `localhost:9999`.

---

## Supported Models

Metal backend supports all model architectures:

- ✅ **Llama** (Llama-2, Llama-3)
- ✅ **Mistral** (Mistral-7B, Mixtral)
- ✅ **Phi** (Phi-2, Phi-3)
- ✅ **Qwen** (Qwen, Qwen-2)

Model architecture is auto-detected from `config.json`.

---

## Performance

### Expected Performance (Estimated)

| Model | Size | Metal (M1 Pro) | Metal (M2 Max) |
|-------|------|----------------|----------------|
| Llama-2-7B | 7B | ~40-60 tok/s | ~60-80 tok/s |
| Mistral-7B | 7B | ~45-65 tok/s | ~65-85 tok/s |
| Llama-2-13B | 13B | ~20-30 tok/s | ~30-45 tok/s |

**Note:** These are estimates. Actual performance depends on:
- Model size and quantization
- Unified memory bandwidth
- Thermal throttling
- Background processes

### Benchmarking

To benchmark your specific hardware:

```bash
# Run inference and measure tokens/second
# (Metrics exposed via HTTP /metrics endpoint)
curl http://localhost:8080/metrics
```

---

## Troubleshooting

### "Metal device initialization failed"

**Cause:** Metal framework not available or incompatible hardware.

**Solution:**
1. Verify you're on Apple Silicon (not Intel)
2. Update macOS to latest version
3. Install Xcode Command Line Tools: `xcode-select --install`

### "Out of memory" errors

**Cause:** Model too large for available unified memory.

**Solution:**
1. Use a smaller model (7B instead of 13B)
2. Close other applications to free memory
3. Use quantized models (GGUF format)

### Performance slower than expected

**Cause:** Thermal throttling or background processes.

**Solution:**
1. Ensure good ventilation (laptop not on soft surface)
2. Close unnecessary applications
3. Monitor Activity Monitor for CPU/GPU usage

---

## Deployment

### Pool-Manager Integration

Pool-managerd will automatically:
1. Detect Apple Silicon GPU
2. Launch `llorch-metal-candled` binary
3. Pass `--metal-device 0` argument
4. Monitor worker health

### Capability Reporting

Metal workers report capability:
```json
{
  "capabilities": ["metal"],
  "device": "apple_silicon_gpu",
  "backend": "metal"
}
```

---

## Development

### Running Tests

```bash
# Run unit tests with Metal feature
cargo test --features metal

# Run device verification test
cargo test --features metal test_metal_device_init
```

### Debug Logging

Enable debug logging:

```bash
RUST_LOG=debug ./target/release/llorch-metal-candled \
  --worker-id test \
  --model /path/to/model \
  --port 8080 \
  --callback-url http://localhost:9999
```

---

## Known Limitations

### Pre-Release Status

⚠️ Metal backend is **pre-release** and has not been extensively tested on real Apple Silicon hardware.

**What this means:**
- ✅ Code compiles and follows CUDA pattern
- ⚠️ Runtime behavior not validated
- ⚠️ Performance not benchmarked
- ⚠️ Edge cases not tested

**Before production use:**
1. Test on target Apple Silicon hardware
2. Benchmark performance vs expectations
3. Validate model loading for all architectures
4. Run stress tests (long-running inference)

### Candle Metal Support

Metal support depends on Candle's Metal implementation:
- Candle provides Metal kernels for common operations
- Some operations may fall back to CPU
- Performance may vary by operation type

---

## Roadmap

### Phase 1: Foundation ✅
- [x] Add Metal feature flag
- [x] Create `init_metal_device()` function
- [x] Create `llorch-metal-candled` binary
- [x] Follow CUDA pattern

### Phase 2: Validation (Pending)
- [ ] Test on Apple Silicon Mac
- [ ] Benchmark vs CPU backend
- [ ] Verify model loading (all architectures)
- [ ] Test warmup and inference

### Phase 3: Production (Future)
- [ ] Performance optimization
- [ ] Memory management tuning
- [ ] CI/CD integration (macOS runners)
- [ ] Production deployment guide

---

## References

### Internal Documentation
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Multi-backend pattern
- `.specs/TEAM_018_HANDOFF.md` - Accelerate → Metal transition
- `README.md` - Project overview

### Candle Metal Support
- `reference/candle/candle-core/` - Metal device implementation
- `reference/candle/candle-metal-kernels/` - Metal GPU kernels

### Apple Documentation
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Apple Silicon Performance](https://developer.apple.com/documentation/apple-silicon)

---

**Last Updated:** 2025-10-09  
**Status:** 🚧 Pre-release (not production-ready)  
**Contact:** TEAM-018
