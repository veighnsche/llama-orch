# FLUX.1 Support Implementation

**Priority:** 🟠 HIGH - SHOULD HAVE  
**Estimated Effort:** 4-6 days  
**Status:** ❌ NOT IMPLEMENTED  
**Assignee:** TBD

---

## Overview

**FLUX.1 = Next-Generation Image Model:**
- 12B parameter rectified flow transformer
- State-of-the-art quality (better than SDXL)
- Two variants: Dev (guidance) and Schnell (fast, 4-step)
- **Candle already has full FLUX support!** ✅

**Models:**
- `FLUX.1-dev` - Guidance-distilled model (50 steps, best quality)
- `FLUX.1-schnell` - Fast model (4 steps, good quality)

**Candle Reference:**
- `/home/vince/Projects/rbee/reference/candle/candle-transformers/src/models/flux/`
- `/home/vince/Projects/rbee/reference/candle/candle-examples/examples/flux/main.rs`

---

## Why FLUX?

### Advantages Over Stable Diffusion

**Quality:**
- Better prompt adherence
- More photorealistic outputs
- Better text rendering in images
- Fewer artifacts

**Architecture:**
- Rectified flow (simpler than diffusion)
- Transformer-based (not UNet)
- More efficient training

**Speed:**
- Schnell: 4 steps (vs SD Turbo's 4 steps)
- Dev: 50 steps (vs SD's 20-30 steps)
- Comparable or better quality per step

### Marketplace Impact

**CivitAI Support:**
- Flux.1 D and Flux.1 S base models already in types
- Growing library of FLUX fine-tunes
- Future-proof architecture

---

## Current Candle Implementation

### Available Components

**From `/reference/candle/candle-transformers/src/models/flux/`:**

1. **`model.rs`** - Main FLUX transformer
   - `Flux` struct with forward pass
   - `Config::dev()` and `Config::schnell()`
   - Full transformer implementation

2. **`autoencoder.rs`** - VAE for FLUX
   - `AutoEncoder` struct
   - Encode/decode to latent space

3. **`sampling.rs`** - Sampling/scheduling
   - `get_noise()` - Initialize noise
   - `get_schedule()` - Timestep schedule
   - `State` - Sampling state management

4. **`quantized_model.rs`** - Quantized FLUX (GGUF)
   - Memory-efficient version
   - Faster inference

### Text Encoders

FLUX uses **two text encoders:**
1. **T5-XXL** - Large language model (4.7B params)
2. **CLIP** - Vision-language model

Both are already in Candle:
- `candle_transformers::models::t5`
- `candle_transformers::models::clip`

---

## Implementation Plan

### Step 1: Add FLUX to SDVersion Enum

**File:** `src/backend/models/mod.rs`

```rust
pub enum SDVersion {
    // Existing SD models
    V1_5,
    V1_5Inpaint,
    V2_1,
    V2Inpaint,
    XL,
    XLInpaint,
    Turbo,
    
    // NEW: FLUX models
    FluxDev,      // FLUX.1-dev (50 steps, guidance)
    FluxSchnell,  // FLUX.1-schnell (4 steps, fast)
}

impl SDVersion {
    pub fn repo(&self) -> &'static str {
        match self {
            // ... existing ...
            Self::FluxDev => "black-forest-labs/FLUX.1-dev",
            Self::FluxSchnell => "black-forest-labs/FLUX.1-schnell",
        }
    }
    
    pub fn default_size(&self) -> (usize, usize) {
        match self {
            // ... existing ...
            Self::FluxDev | Self::FluxSchnell => (1024, 1024), // FLUX default
        }
    }
    
    pub fn default_steps(&self) -> usize {
        match self {
            // ... existing ...
            Self::FluxDev => 50,      // Dev needs more steps
            Self::FluxSchnell => 4,   // Schnell is fast
        }
    }
    
    pub fn default_guidance_scale(&self) -> f64 {
        match self {
            // ... existing ...
            Self::FluxDev => 3.5,     // FLUX uses lower guidance
            Self::FluxSchnell => 0.0, // Schnell doesn't use guidance
        }
    }
    
    pub fn is_flux(&self) -> bool {
        matches!(self, Self::FluxDev | Self::FluxSchnell)
    }
}
```

---

### Step 2: FLUX Model Loading

**File:** `src/backend/models/flux_loader.rs` (NEW FILE)

```rust
use anyhow::Result;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, t5};
use std::path::Path;

/// FLUX model components
pub struct FluxComponents {
    pub version: super::SDVersion,
    pub device: Device,
    pub dtype: DType,
    
    // Text encoders
    pub t5_tokenizer: tokenizers::Tokenizer,
    pub t5_model: t5::T5EncoderModel,
    pub clip_tokenizer: tokenizers::Tokenizer,
    pub clip_model: clip::ClipTextTransformer,
    
    // FLUX transformer
    pub flux_model: Box<dyn flux::WithForward>,
    
    // VAE
    pub vae: flux::autoencoder::AutoEncoder,
}

impl FluxComponents {
    /// Load FLUX model from HuggingFace cache
    ///
    /// # Arguments
    /// * `model_path` - Path to model directory
    /// * `version` - FLUX variant (Dev or Schnell)
    /// * `device` - Device to load on
    /// * `use_f16` - Use F16 precision (recommended for GPU)
    /// * `quantized` - Use quantized model (GGUF)
    ///
    /// # Returns
    /// Loaded FLUX components
    pub fn load(
        model_path: &str,
        version: super::SDVersion,
        device: &Device,
        use_f16: bool,
        quantized: bool,
    ) -> Result<Self> {
        let dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let model_path = Path::new(model_path);
        
        tracing::info!("Loading FLUX {:?} from {:?}", version, model_path);
        
        // 1. Load T5 text encoder
        tracing::info!("Loading T5 tokenizer and model...");
        let t5_tokenizer = {
            let tokenizer_path = model_path.join("tokenizer_2/tokenizer.json");
            tokenizers::Tokenizer::from_file(tokenizer_path)?
        };
        
        let t5_model = {
            let config = t5::Config::v1_1_xxl();
            let weights_path = model_path.join("text_encoder_2/model.safetensors");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
            };
            t5::T5EncoderModel::load(vb, &config)?
        };
        
        // 2. Load CLIP text encoder
        tracing::info!("Loading CLIP tokenizer and model...");
        let clip_tokenizer = {
            let tokenizer_path = model_path.join("tokenizer/tokenizer.json");
            tokenizers::Tokenizer::from_file(tokenizer_path)?
        };
        
        let clip_model = {
            let config = clip::Config::v2_1();
            let weights_path = model_path.join("text_encoder/model.safetensors");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
            };
            clip::ClipTextTransformer::new(vb, &config)?
        };
        
        // 3. Load FLUX transformer
        tracing::info!("Loading FLUX transformer...");
        let flux_model: Box<dyn flux::WithForward> = if quantized {
            // Load quantized GGUF model
            let gguf_path = model_path.join("flux1-schnell.gguf");
            let mut file = std::fs::File::open(gguf_path)?;
            let model = flux::quantized_model::Flux::from_gguf(&mut file, device)?;
            Box::new(model)
        } else {
            // Load full precision model
            let config = match version {
                super::SDVersion::FluxDev => flux::model::Config::dev(),
                super::SDVersion::FluxSchnell => flux::model::Config::schnell(),
                _ => unreachable!(),
            };
            let weights_path = model_path.join("transformer/diffusion_pytorch_model.safetensors");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
            };
            let model = flux::model::Flux::new(&config, vb)?;
            Box::new(model)
        };
        
        // 4. Load VAE
        tracing::info!("Loading VAE...");
        let vae = {
            let config = match version {
                super::SDVersion::FluxDev => flux::autoencoder::Config::dev(),
                super::SDVersion::FluxSchnell => flux::autoencoder::Config::schnell(),
                _ => unreachable!(),
            };
            let weights_path = model_path.join("vae/diffusion_pytorch_model.safetensors");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
            };
            flux::autoencoder::AutoEncoder::new(&config, vb)?
        };
        
        tracing::info!("FLUX model loaded successfully");
        
        Ok(Self {
            version,
            device: device.clone(),
            dtype,
            t5_tokenizer,
            t5_model,
            clip_tokenizer,
            clip_model,
            flux_model,
            vae,
        })
    }
}
```

---

### Step 3: FLUX Generation Function

**File:** `src/backend/flux_generation.rs` (NEW FILE)

```rust
use anyhow::Result;
use candle_core::{DType, IndexOp, Module, Tensor};
use candle_transformers::models::flux;
use image::DynamicImage;

use super::models::flux_loader::FluxComponents;
use super::sampling::SamplingConfig;

/// Generate image with FLUX
///
/// Based on: reference/candle/candle-examples/examples/flux/main.rs
///
/// # Arguments
/// * `config` - Sampling configuration
/// * `models` - FLUX model components
/// * `progress_callback` - Progress updates
///
/// # Returns
/// Generated image
pub fn generate_flux<F>(
    config: &SamplingConfig,
    models: &FluxComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    config.validate()?;
    
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    // 1. Encode text with T5
    tracing::info!("Encoding text with T5...");
    let t5_tokens = models.t5_tokenizer
        .encode(&config.prompt, true)
        .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {}", e))?;
    let t5_token_ids = Tensor::new(t5_tokens.get_ids(), &models.device)?
        .unsqueeze(0)?;
    let t5_emb = models.t5_model.forward(&t5_token_ids)?;
    
    // 2. Encode text with CLIP
    tracing::info!("Encoding text with CLIP...");
    let clip_tokens = models.clip_tokenizer
        .encode(&config.prompt, true)
        .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {}", e))?;
    let clip_token_ids = Tensor::new(clip_tokens.get_ids(), &models.device)?
        .unsqueeze(0)?;
    let clip_emb = models.clip_model.forward(&clip_token_ids)?;
    
    // 3. Initialize noise
    let img = flux::sampling::get_noise(
        1,
        config.height,
        config.width,
        &models.device,
    )?.to_dtype(models.dtype)?;
    
    // 4. Create sampling state
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
    
    // 5. Get timestep schedule
    let timesteps = match models.version {
        super::models::SDVersion::FluxDev => {
            // Dev: 50 steps with shift
            flux::sampling::get_schedule(
                config.steps,
                Some((state.img.dim(1)?, 0.5, 1.15)),
            )
        }
        super::models::SDVersion::FluxSchnell => {
            // Schnell: 4 steps, no shift
            flux::sampling::get_schedule(config.steps, None)
        }
        _ => unreachable!(),
    };
    
    tracing::info!("Starting FLUX generation with {} steps", timesteps.len());
    
    // 6. Denoising loop
    let mut img = state.img.clone();
    let guidance = if config.guidance_scale > 1.0 {
        Some(Tensor::new(&[config.guidance_scale as f32], &models.device)?)
    } else {
        None
    };
    
    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        let timestep_tensor = Tensor::new(&[timestep], &models.device)?;
        
        // FLUX forward pass
        let pred = models.flux_model.forward(
            &img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &timestep_tensor,
            &state.vec,
            guidance.as_ref(),
        )?;
        
        // Update image
        img = (img + pred)?;
        
        progress_callback(step_idx + 1, timesteps.len());
    }
    
    // 7. Decode with VAE
    tracing::info!("Decoding latents with VAE...");
    let img = models.vae.decode(&img)?;
    
    // 8. Convert to image
    let img = ((img / 2.)? + 0.5)?
        .to_device(&candle_core::Device::Cpu)?
        .clamp(0f32, 1f32)?;
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        anyhow::bail!("Expected 3 channels, got {}", channel);
    }
    let img = (img * 255.)?
        .to_dtype(DType::U8)?
        .i(0)?;
    
    crate::backend::image_utils::tensor_to_image(&img)
}
```

---

### Step 4: Integration with Model Loader

**File:** `src/backend/model_loader.rs`

```rust
use super::models::{SDVersion, ModelComponents, flux_loader::FluxComponents};

pub enum LoadedModel {
    StableDiffusion(ModelComponents),
    Flux(FluxComponents),
}

pub fn load_model(
    model_path: &str,
    version: SDVersion,
    device: &Device,
    use_f16: bool,
) -> Result<LoadedModel> {
    if version.is_flux() {
        // Load FLUX model
        let components = FluxComponents::load(
            model_path,
            version,
            device,
            use_f16,
            false, // quantized
        )?;
        Ok(LoadedModel::Flux(components))
    } else {
        // Load Stable Diffusion model
        let components = load_sd_components(model_path, version, device, use_f16)?;
        Ok(LoadedModel::StableDiffusion(components))
    }
}
```

---

### Step 5: Update Generation Engine

**File:** `src/backend/generation_engine.rs`

```rust
async fn process_request(
    request: GenerationRequest,
    models: Arc<LoadedModel>,
) {
    let result = match models.as_ref() {
        LoadedModel::Flux(flux_models) => {
            // FLUX generation
            crate::backend::flux_generation::generate_flux(
                &request.config,
                flux_models,
                |step, total| {
                    tracing::debug!("FLUX progress: {}/{}", step, total);
                },
            )
        }
        LoadedModel::StableDiffusion(sd_models) => {
            // SD generation (existing code)
            match (request.input_image, request.mask) {
                (Some(image), Some(mask)) => {
                    crate::backend::generation::inpaint(/* ... */)
                }
                (Some(image), None) => {
                    crate::backend::generation::image_to_image(/* ... */)
                }
                (None, _) => {
                    crate::backend::generation::generate_image(/* ... */)
                }
            }
        }
    };
    
    // Send response...
}
```

---

## Testing Plan

### Unit Tests

**File:** `tests/flux_loading.rs`

```rust
#[test]
fn test_flux_dev_loading() {
    // Test loading FLUX.1-dev model
}

#[test]
fn test_flux_schnell_loading() {
    // Test loading FLUX.1-schnell model
}
```

### Integration Tests

**File:** `tests/flux_generation.rs`

```rust
#[tokio::test]
#[ignore]
async fn test_flux_schnell_generation() {
    // Load FLUX Schnell
    // Generate with 4 steps
    // Verify output quality
}

#[tokio::test]
#[ignore]
async fn test_flux_dev_generation() {
    // Load FLUX Dev
    // Generate with 50 steps
    // Verify output quality
}
```

---

## Acceptance Criteria

- [ ] FLUX models can be loaded
- [ ] FLUX Schnell generates in ~4 steps
- [ ] FLUX Dev generates in ~50 steps
- [ ] Text encoding (T5 + CLIP) works
- [ ] VAE decode produces valid images
- [ ] Image quality matches reference
- [ ] Memory usage acceptable
- [ ] Performance acceptable
- [ ] Unit tests pass
- [ ] Integration tests pass

---

## Marketplace Compatibility Update

**File:** `bin/80-global-worker-catalog/src/data.ts`

```typescript
civitai: {
  modelTypes: ['Checkpoint'],
  baseModels: [
    // ... existing SD models ...
    // FLUX models
    'Flux.1 D',    // ✅ FLUX.1-dev
    'Flux.1 S',    // ✅ FLUX.1-schnell
  ],
}
```

---

## Advantages of FLUX Implementation

### 1. **Candle Already Has It**
- Full implementation in `candle-transformers`
- Working example in `candle-examples`
- Just need to integrate, not implement from scratch

### 2. **Better Quality**
- State-of-the-art image generation
- Better prompt adherence
- Fewer artifacts

### 3. **Future-Proof**
- Modern architecture (transformer, not UNet)
- Growing ecosystem
- CivitAI already supports it

### 4. **Competitive Advantage**
- Most SD workers don't support FLUX yet
- Early mover advantage
- Attracts professional users

---

## Estimated Timeline

- **Day 1:** Add FLUX to SDVersion enum, create flux_loader.rs
- **Day 2:** Implement FLUX model loading (T5, CLIP, transformer, VAE)
- **Day 3:** Implement flux_generation.rs
- **Day 4:** Integration with model loader and generation engine
- **Day 5:** Testing and bug fixes
- **Day 6:** Documentation and marketplace compatibility update

**Total:** 4-6 days

---

## References

- **Candle FLUX Implementation:** `/reference/candle/candle-transformers/src/models/flux/`
- **Candle FLUX Example:** `/reference/candle/candle-examples/examples/flux/main.rs`
- **FLUX GitHub:** https://github.com/black-forest-labs/flux
- **FLUX HuggingFace:** https://huggingface.co/black-forest-labs/FLUX.1-dev
- **FLUX Paper:** "FLUX: Fast Latent Unified X-formers"
