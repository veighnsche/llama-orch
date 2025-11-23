# SD Worker Implementation Plans

**Last Updated:** 2025-11-12  
**Status:** Plans complete, ready for implementation

---

## Overview

This directory contains detailed implementation plans for bringing the SD worker to MVP status.

**Current State:** Basic text-to-image only  
**Target State:** Full-featured image generation worker with FLUX support

---

## Implementation Plans

### 🔴 MUST HAVE (2-3 weeks)

These are **critical** for MVP. The worker is not production-ready without them.

| # | Plan | Effort | Priority | Status |
|---|------|--------|----------|--------|
| 01 | [Image-to-Image](./01_IMAGE_TO_IMAGE.md) | 2-3 days | 🔴 CRITICAL | ✅ **COMPLETE (TEAM-487)** |
| 02 | [Inpainting](./02_INPAINTING.md) | 3-4 days | 🔴 CRITICAL | ✅ **COMPLETE (TEAM-487)** |
| 03 | [Model Loading Verification](./03_MODEL_LOADING_VERIFICATION.md) | 1-2 days | 🔴 CRITICAL | ✅ **COMPLETE (TEAM-487)** |

**Total:** 6-9 days

---

### 🟠 SHOULD HAVE (2-3 weeks)

These **dramatically improve** marketplace compatibility and unlock thousands of models.

| # | Plan | Effort | Priority | Status |
|---|------|--------|----------|--------|
| 04 | [LoRA Support](./04_LORA_SUPPORT.md) | 5-7 days | 🟠 HIGH | ❌ Not started |
| 05 | [ControlNet Support](./05_CONTROLNET_SUPPORT.md) | 7-10 days | 🟠 HIGH | ❌ Not started |
| 06 | [FLUX.1 Support](./06_FLUX_SUPPORT.md) | 4-6 days | 🟠 HIGH | ❌ Not started |

**Total:** 16-23 days

---

## Quick Reference

### What Works Now ✅
- Text-to-image generation
- Multiple backends (CPU, CUDA, Metal, ROCm)
- Streaming progress via SSE
- Basic model variants (V1.5, V2.1, XL, Turbo)

### What's Broken ❌
- Image-to-image (stub only)
- Inpainting (stub only)
- LoRA loading (no code)
- ControlNet (no code)
- FLUX (not integrated)

### Marketplace Impact

**Current (Accurate):**
```typescript
civitai: {
  modelTypes: ['Checkpoint'],  // ONLY checkpoints
  baseModels: [
    'SD 1.4', 'SD 1.5',
    'SD 2.0', 'SD 2.1',
    'SDXL 1.0', 'SDXL Turbo',
    'SD 3', 'SD 3.5',
  ],
}
```

**After LoRA + ControlNet:**
```typescript
civitai: {
  modelTypes: [
    'Checkpoint',
    'LORA',        // +100K models
    'Controlnet',  // +1K models
  ],
  // ... same base models ...
}
```

**After FLUX:**
```typescript
civitai: {
  modelTypes: ['Checkpoint', 'LORA', 'Controlnet'],
  baseModels: [
    // ... existing ...
    'Flux.1 D',   // State-of-the-art quality
    'Flux.1 S',   // Fast generation
  ],
}
```

---

## Implementation Order

### Phase 1: Core Features (Week 1-2)
1. **Image-to-Image** (2-3 days)
   - Most requested feature
   - Enables variations and refinement
   - Foundation for inpainting

2. **Inpainting** (3-4 days)
   - Essential for editing workflows
   - Requires img2img foundation
   - Uses special inpainting models

3. **Model Verification** (1-2 days)
   - Verify all 7 model variants work
   - Create test suite
   - Document verified models

### Phase 2: Marketplace Expansion (Week 3-4)
4. **LoRA Support** (5-7 days)
   - Unlocks 100K+ CivitAI models
   - Most impactful feature
   - Requires candle-transformers fork

5. **ControlNet Support** (7-10 days)
   - Professional workflows
   - Precise control over generation
   - Complex implementation

### Phase 3: Next-Gen Models (Week 5)
6. **FLUX.1 Support** (4-6 days)
   - State-of-the-art quality
   - Candle already has it!
   - Just needs integration
   - Competitive advantage

---

## Key Insights from Candle Reference

### What Candle Provides

**Stable Diffusion:**
- ✅ Full UNet implementation
- ✅ VAE encoder/decoder
- ✅ CLIP text encoder
- ✅ Multiple schedulers (DDIM, Euler, etc.)
- ✅ Inpainting model support

**FLUX (Discovered!):**
- ✅ Full FLUX transformer implementation
- ✅ T5-XXL + CLIP text encoders
- ✅ FLUX VAE
- ✅ Quantized GGUF support
- ✅ Working example code

**What Candle Doesn't Provide:**
- ❌ LoRA loading/application
- ❌ ControlNet architecture
- ❌ Preprocessors (Canny, Depth, OpenPose)

---

## Technical Challenges

### Easy (1-3 days each)
- ✅ Image-to-image (VAE encoder + noise addition)
- ✅ Model verification (just testing)
- ✅ FLUX integration (Candle has it)

### Medium (3-7 days each)
- ⚠️ Inpainting (9-channel UNet, mask blending)
- ⚠️ LoRA support (weight loading + application)

### Hard (7-10 days each)
- ❌ ControlNet (full architecture + preprocessors)

---

## Success Metrics

### MVP Complete When:
- [ ] All 3 MUST HAVE items complete
- [ ] At least 2 SHOULD HAVE items complete
- [ ] All model variants verified
- [ ] Test suite passing
- [ ] Documentation updated
- [ ] Marketplace compatibility accurate

### Stretch Goals:
- [ ] All 6 implementation plans complete
- [ ] FLUX generating high-quality images
- [ ] LoRA stacking works
- [ ] Multiple ControlNets work
- [ ] Performance optimized

---

## Candle Reference Files

### For Image-to-Image (Plan 01)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-examples/examples/stable-diffusion/main.rs`
- Lines 531-826: Full generation loop with VAE encode/decode
- Function: `run()` - Shows img2img pattern

**VAE Encoder:**
- `candle-transformers/src/models/stable_diffusion/vae.rs`
- Struct: `AutoEncoderKL`
- Method: `encode()` - Converts image to latents

**Key Pattern:**
```rust
// Encode image to latents
let latents = vae.encode(&image_tensor)?;
// Add noise based on strength
let noisy_latents = add_noise(latents, strength)?;
// Denoise from start_step to end
for step in start_step..num_steps { /* ... */ }
```

---

### For Inpainting (Plan 02)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-transformers/src/models/stable_diffusion/`
- `unet_2d.rs` - UNet architecture (9-channel input for inpainting)
- `vae.rs` - VAE encode/decode

**Mask Processing:**
- Look at how Candle handles multi-channel inputs
- `candle-core/src/` - Tensor operations for mask blending

**Key Pattern:**
```rust
// Concatenate: [latents (4ch) + mask (1ch) + masked_image (4ch)] = 9ch
let unet_input = Tensor::cat(&[latents, mask, masked_image], 1)?;
let noise_pred = unet.forward(&unet_input, timestep, text_emb)?;
```

---

### For Model Loading Verification (Plan 03)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-examples/examples/stable-diffusion/main.rs`
- Lines 1-200: Model loading setup
- Lines 400-500: Device initialization

**Model Loading Pattern:**
```rust
// Load model components
let vb = VarBuilder::from_mmaped_safetensors(&files, dtype, device)?;
let unet = UNet2DConditionModel::new(&config, vb.pp("unet"))?;
let vae = AutoEncoderKL::new(&config, vb.pp("vae"))?;
```

**Test Structure:**
- `candle-examples/examples/stable-diffusion/` - Full working example
- Use as integration test template

---

### For LoRA Support (Plan 04)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-core/src/safetensors.rs`
- Function: `load()` - Loading SafeTensors files
- Shows how to read weight tensors

**Weight Manipulation:**
- `candle-nn/src/var_builder.rs` - VarBuilder pattern
- `candle-core/src/tensor.rs` - Tensor operations (matmul, add)

**Key Pattern:**
```rust
// Load LoRA weights
let lora_tensors = candle_core::safetensors::load(path, device)?;
// Apply LoRA: W' = W + α * (A × B)
let delta = lora_up.matmul(&lora_down)?;
let new_weight = (original_weight + (delta * alpha)?)?;
```

**Challenge:** Candle's UNet doesn't expose weights for modification
- May need to fork: `candle-transformers/src/models/stable_diffusion/unet_2d.rs`
- Add method: `apply_delta(&mut self, layer_name: &str, delta: &Tensor)`

---

### For ControlNet Support (Plan 05)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-transformers/src/models/stable_diffusion/unet_2d.rs`
- Lines 1-500: Full UNet architecture
- Struct: `UNet2DConditionModel`
- Method: `forward()` - Shows layer structure

**Architecture to Mirror:**
- ControlNet mirrors UNet structure
- Need to implement similar blocks with zero convolutions
- Inject control signals at each down/mid/up block

**Key Files:**
- `unet_2d.rs` - Base architecture to mirror
- `candle-nn/src/conv.rs` - Convolution layers
- `candle-nn/src/group_norm.rs` - Normalization

**No Direct Reference:** ControlNet not in Candle, must implement from scratch
- Reference: https://github.com/lllyasviel/ControlNet (PyTorch)
- Port architecture to Candle patterns

---

### For FLUX Support (Plan 06)
**Primary Reference:** `/home/vince/Projects/rbee/reference/candle/candle-examples/examples/flux/main.rs`
- **COMPLETE WORKING EXAMPLE** ✅
- Lines 63-267: Full FLUX generation pipeline

**FLUX Implementation:**
- `candle-transformers/src/models/flux/model.rs` - Main transformer
  - Struct: `Flux` (line 526)
  - Method: `forward()` (line 580)
  - Configs: `Config::dev()`, `Config::schnell()`

- `candle-transformers/src/models/flux/autoencoder.rs` - FLUX VAE
  - Struct: `AutoEncoder` (line 100+)
  - Methods: `encode()`, `decode()`

- `candle-transformers/src/models/flux/sampling.rs` - Scheduling
  - Function: `get_noise()` (line 20+)
  - Function: `get_schedule()` (line 50+)
  - Struct: `State` - Sampling state

- `candle-transformers/src/models/flux/quantized_model.rs` - GGUF support
  - Quantized FLUX for lower memory

**Text Encoders:**
- `candle-transformers/src/models/t5/` - T5-XXL encoder
- `candle-transformers/src/models/clip/` - CLIP encoder

**Key Pattern (from main.rs lines 164-200):**
```rust
// 1. Load models
let t5_model = t5::T5EncoderModel::load(vb, &config)?;
let clip_model = clip::ClipTextTransformer::new(vb, &config)?;
let flux_model = flux::model::Flux::new(&cfg, vb)?;
let vae = flux::autoencoder::AutoEncoder::new(&cfg, vb)?;

// 2. Encode text
let t5_emb = t5_model.forward(&t5_tokens)?;
let clip_emb = clip_model.forward(&clip_tokens)?;

// 3. Initialize and denoise
let img = flux::sampling::get_noise(1, height, width, device)?;
let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
let timesteps = flux::sampling::get_schedule(50, None);

// 4. Denoising loop
for timestep in timesteps {
    let pred = flux_model.forward(&img, &state, &timestep)?;
    img = (img + pred)?;
}

// 5. Decode
let output = vae.decode(&img)?;
```

---

## External References

### Automatic1111 (SD Reference Implementation)
- **LoRA:** `extensions-builtin/Lora/`
- **ControlNet:** `extensions-builtin/ControlNet/`
- **Inpainting:** `modules/processing.py`

### ComfyUI (Node-based SD)
- **LoRA Loading:** `comfy/sd.py`
- **ControlNet:** `comfy/controlnet.py`

### HuggingFace Diffusers (Official)
- **Pipelines:** `diffusers/pipelines/stable_diffusion/`
- **LoRA:** `diffusers/loaders/lora.py`
- **ControlNet:** `diffusers/models/controlnet.py`

### Black Forest Labs (FLUX Official)
- **GitHub:** https://github.com/black-forest-labs/flux
- **Models:** https://huggingface.co/black-forest-labs/

---

## Notes

### Why These Plans?

1. **Based on actual source code** - Not documentation or assumptions
2. **Candle-first** - Uses what Candle provides, doesn't reinvent
3. **Incremental** - Each plan builds on previous work
4. **Realistic timelines** - Based on complexity analysis
5. **Marketplace-driven** - Focused on unlocking CivitAI models

### What Makes FLUX Special?

- **Candle already has it!** - Just needs integration
- **Better quality** - State-of-the-art results
- **Future-proof** - Transformer architecture (not UNet)
- **Competitive edge** - Most workers don't support it yet
- **CivitAI ready** - Flux.1 D and Flux.1 S already in types

---

**Ready to implement? Start with [01_IMAGE_TO_IMAGE.md](./01_IMAGE_TO_IMAGE.md)**
