# Image-to-Image (img2img) Implementation

**Priority:** 🔴 CRITICAL - MUST HAVE  
**Estimated Effort:** 2-3 days  
**Status:** ❌ NOT IMPLEMENTED  
**Assignee:** TBD

---

## Problem

**Current State:**
```rust
// job_router.rs line 97-104
async fn execute_image_transform(...) -> Result<JobResponse> {
    Err(anyhow!("ImageTransform not yet implemented - requires img2img pipeline"))
}
```

Users cannot:
- Transform existing images with new prompts
- Create variations of images
- Apply style transfer
- Refine generated images

**This is a core SD feature** - without it, the worker is severely limited.

---

## What Is Image-to-Image?

Image-to-image takes an **existing image** and transforms it based on a text prompt.

**Key Difference from Text-to-Image:**
- Text-to-image: Start from pure noise → denoise to image
- Image-to-image: Start from existing image → add noise → partially denoise

**Strength Parameter (0.0 - 1.0):**
- `0.0` = No change (output = input)
- `0.3` = Subtle changes (keep composition, change details)
- `0.7` = Major changes (keep rough structure, change everything else)
- `1.0` = Full regeneration (equivalent to text-to-image)

---

## Implementation Plan

### Step 1: Add VAE Encoder Function

**File:** `src/backend/generation.rs`

```rust
/// Encode image to latent space
///
/// Takes an RGB image and encodes it to latent space using VAE encoder.
/// This is the reverse of VAE decoding (latents → image).
///
/// # Arguments
/// * `image` - Input image (RGB, any size)
/// * `vae` - VAE model (contains encoder)
/// * `device` - Device to run on
/// * `dtype` - Data type (F32 or F16)
///
/// # Returns
/// Latent tensor (shape: [1, 4, height/8, width/8])
pub fn encode_image_to_latents(
    image: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // 1. Resize image to match model requirements
    let (target_width, target_height) = (512, 512); // TODO: Get from config
    let resized = image.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );
    
    // 2. Convert to tensor (normalize to [-1, 1])
    let image_tensor = crate::backend::image_utils::image_to_tensor(&resized, device, dtype)?;
    
    // 3. Encode to latent space
    let latents = vae.encode(&image_tensor)?;
    
    // 4. Apply VAE scaling factor
    let latents = (latents * 0.18215)?; // SD VAE scaling factor
    
    Ok(latents)
}
```

---

### Step 2: Add Noise to Latents

**File:** `src/backend/generation.rs`

```rust
/// Add noise to latents based on strength parameter
///
/// For img2img, we don't start from pure noise. Instead:
/// 1. Encode input image to latents
/// 2. Add noise proportional to strength
/// 3. Denoise from that point
///
/// # Arguments
/// * `latents` - Clean latents from input image
/// * `strength` - How much to change (0.0 = no noise, 1.0 = full noise)
/// * `scheduler` - Scheduler to determine noise schedule
/// * `num_steps` - Total denoising steps
///
/// # Returns
/// Noisy latents + starting timestep
pub fn add_noise_for_img2img(
    latents: &Tensor,
    strength: f64,
    scheduler: &dyn crate::backend::scheduler::Scheduler,
    num_steps: usize,
) -> Result<(Tensor, usize)> {
    // Calculate starting step based on strength
    // strength=0.0 → start at step 0 (no denoising)
    // strength=1.0 → start at step num_steps (full denoising)
    let start_step = ((1.0 - strength) * num_steps as f64) as usize;
    
    // Get noise schedule
    let timesteps = scheduler.timesteps(num_steps);
    let t = timesteps[start_step];
    
    // Generate noise
    let noise = latents.randn_like(0.0, 1.0)?;
    
    // Add noise according to scheduler
    let noisy_latents = scheduler.add_noise(latents, &noise, t)?;
    
    Ok((noisy_latents, start_step))
}
```

---

### Step 3: Create Image-to-Image Generation Function

**File:** `src/backend/generation.rs`

```rust
/// Generate image from existing image (img2img)
///
/// Based on: reference/candle/candle-examples/examples/stable-diffusion/main.rs
/// Similar to generate_image() but starts from encoded input image
///
/// # Arguments
/// * `config` - Sampling configuration (prompt, steps, guidance, etc.)
/// * `models` - Loaded model components (VAE, UNet, CLIP, etc.)
/// * `input_image` - Starting image to transform
/// * `strength` - Transformation strength (0.0-1.0)
/// * `progress_callback` - Called after each denoising step
///
/// # Returns
/// Transformed image
pub fn image_to_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    input_image: &DynamicImage,
    strength: f64,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    // Validate inputs
    config.validate()?;
    if !(0.0..=1.0).contains(&strength) {
        return Err(Error::InvalidInput(format!(
            "Strength must be between 0.0 and 1.0, got {}",
            strength
        )));
    }
    
    // Set seed if provided
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    let use_guide_scale = config.guidance_scale > 1.0;
    
    // 1. Generate text embeddings (same as text-to-image)
    let text_embeddings = text_embeddings(
        &config.prompt,
        config.negative_prompt.as_deref().unwrap_or(""),
        &models.tokenizer,
        &models.clip_config,
        &models.clip_weights,
        &models.device,
        models.dtype,
        use_guide_scale,
    )?;
    
    // 2. Encode input image to latents
    let init_latents = encode_image_to_latents(
        input_image,
        &models.vae,
        &models.device,
        models.dtype,
    )?;
    
    // 3. Add noise based on strength
    let (noisy_latents, start_step) = add_noise_for_img2img(
        &init_latents,
        strength,
        models.scheduler.as_ref(),
        config.steps,
    )?;
    
    // 4. Denoise from start_step to end (partial denoising)
    let mut latents = noisy_latents;
    let timesteps = models.scheduler.timesteps(config.steps);
    
    for (step_idx, &t) in timesteps.iter().enumerate().skip(start_step) {
        // Expand latents for classifier-free guidance
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        // Predict noise
        let noise_pred = models.unet.forward(&latent_model_input, t as f64, &text_embeddings)?;
        
        // Apply classifier-free guidance
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * config.guidance_scale)?)?
        } else {
            noise_pred
        };
        
        // Scheduler step
        latents = models.scheduler.step(&noise_pred, t, &latents)?;
        
        // Progress callback
        progress_callback(step_idx + 1, config.steps);
    }
    
    // 5. Decode latents to image (same as text-to-image)
    let image = models.vae.decode(&(&latents / models.vae_scale)?)?;
    let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let image = (image.clamp(0f32, 1f32)? * 255.)?
        .to_dtype(DType::U8)?
        .i(0)?;
    
    crate::backend::image_utils::tensor_to_image(&image)
}
```

---

### Step 4: Wire Up to Job Router

**File:** `src/job_router.rs`

```rust
/// Execute image transform operation (img2img)
async fn execute_image_transform(
    state: JobState,
    req: operations_contract::ImageTransformRequest,
) -> Result<JobResponse> {
    // 1. Decode base64 input image
    let input_image_bytes = base64::decode(&req.image)
        .context("Failed to decode base64 image")?;
    let input_image = image::load_from_memory(&input_image_bytes)
        .context("Failed to load input image")?;
    
    // 2. Create sampling config
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps.unwrap_or(20),
        guidance_scale: req.guidance_scale.unwrap_or(7.5),
        seed: req.seed,
        width: req.width.unwrap_or(512),
        height: req.height.unwrap_or(512),
    };
    
    // 3. Submit to generation queue
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    state.queue.send(GenerationRequest {
        config,
        input_image: Some(input_image),
        strength: req.strength.unwrap_or(0.8), // Default 80% transformation
        response_tx: tx,
    }).await?;
    
    // 4. Wait for result
    let response = rx.await?;
    
    // 5. Encode output image to base64
    let mut output_bytes = Vec::new();
    response.image.write_to(&mut Cursor::new(&mut output_bytes), image::ImageFormat::Png)?;
    let output_base64 = base64::encode(&output_bytes);
    
    Ok(JobResponse::ImageGeneration(ImageGenerationResponse {
        image: output_base64,
        seed: response.seed,
        steps: response.steps,
    }))
}
```

---

### Step 5: Update Request Queue

**File:** `src/backend/request_queue.rs`

```rust
pub struct GenerationRequest {
    pub config: SamplingConfig,
    pub input_image: Option<DynamicImage>, // NEW: For img2img
    pub strength: f64,                      // NEW: For img2img
    pub response_tx: oneshot::Sender<GenerationResponse>,
}
```

**File:** `src/backend/generation_engine.rs`

```rust
async fn process_request(
    request: GenerationRequest,
    models: Arc<ModelComponents>,
) {
    let result = if let Some(input_image) = request.input_image {
        // Image-to-image
        crate::backend::generation::image_to_image(
            &request.config,
            &models,
            &input_image,
            request.strength,
            |step, total| {
                tracing::debug!("img2img progress: {}/{}", step, total);
            },
        )
    } else {
        // Text-to-image
        crate::backend::generation::generate_image(
            &request.config,
            &models,
            |step, total| {
                tracing::debug!("txt2img progress: {}/{}", step, total);
            },
        )
    };
    
    // Send response...
}
```

---

## Testing Plan

### Unit Tests

**File:** `src/backend/generation.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_image_to_latents() {
        // Test that encoding produces correct latent shape
        // Expected: [1, 4, height/8, width/8]
    }
    
    #[test]
    fn test_add_noise_for_img2img() {
        // Test noise addition at different strength levels
        // strength=0.0 → start_step=0 (no denoising)
        // strength=1.0 → start_step=num_steps (full denoising)
    }
    
    #[test]
    fn test_strength_validation() {
        // Test that invalid strength values are rejected
        // Should error on strength < 0.0 or > 1.0
    }
}
```

### Integration Tests

**File:** `tests/img2img_integration.rs`

```rust
#[tokio::test]
#[ignore] // Requires model files
async fn test_img2img_low_strength() {
    // Load model
    // Generate base image
    // Transform with strength=0.3
    // Verify output is similar to input
}

#[tokio::test]
#[ignore]
async fn test_img2img_high_strength() {
    // Load model
    // Generate base image
    // Transform with strength=0.9
    // Verify output is different from input
}

#[tokio::test]
#[ignore]
async fn test_img2img_style_transfer() {
    // Load model
    // Load photo
    // Transform with prompt "oil painting"
    // Verify style changed
}
```

---

## Acceptance Criteria

- [ ] `encode_image_to_latents()` function implemented
- [ ] `add_noise_for_img2img()` function implemented
- [ ] `image_to_image()` function implemented
- [ ] Job router wired up to handle `ImageTransform` operations
- [ ] Request queue supports optional input image
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Strength parameter works correctly (0.0-1.0)
- [ ] Output quality matches reference implementations
- [ ] Memory usage stays within limits

---

## References

- **Candle Example:** `reference/candle/candle-examples/examples/stable-diffusion/main.rs`
- **HuggingFace Diffusers:** `diffusers.StableDiffusionImg2ImgPipeline`
- **Automatic1111:** `modules/processing.py` (img2img implementation)

---

## Estimated Timeline

- **Day 1:** Implement VAE encoder + noise addition functions
- **Day 2:** Implement main img2img function + wire up job router
- **Day 3:** Testing + bug fixes + documentation

**Total:** 2-3 days
