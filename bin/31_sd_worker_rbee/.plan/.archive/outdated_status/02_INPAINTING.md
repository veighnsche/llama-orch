# Inpainting Implementation

**Priority:** 🔴 CRITICAL - MUST HAVE  
**Estimated Effort:** 3-4 days  
**Status:** ❌ NOT IMPLEMENTED  
**Assignee:** TBD

---

## Problem

**Current State:**
```rust
// job_router.rs line 106-113
async fn execute_inpaint(...) -> Result<JobResponse> {
    Err(anyhow!("ImageInpaint not yet implemented - requires inpainting pipeline"))
}
```

Users cannot:
- Edit specific regions of images
- Remove objects from images
- Add objects to images
- Fix/repair parts of images

**This is essential for editing workflows** - without it, users must regenerate entire images.

---

## What Is Inpainting?

Inpainting fills in **masked regions** of an image based on a text prompt.

**How It Works:**
1. User provides: original image + mask + prompt
2. Mask defines region to regenerate (white = inpaint, black = keep)
3. Model regenerates only the masked region
4. Result blends seamlessly with original

**Use Cases:**
- Remove unwanted objects ("remove the person")
- Add new objects ("add a cat on the couch")
- Change backgrounds ("replace sky with sunset")
- Fix artifacts or errors

---

## Inpainting Models

**Special Models Required:**

The worker already has inpainting model variants defined:

```rust
// src/backend/models/mod.rs
pub enum SDVersion {
    V1_5Inpaint,  // stable-diffusion-v1-5/stable-diffusion-inpainting
    V2Inpaint,    // stabilityai/stable-diffusion-2-inpainting
    XLInpaint,    // diffusers/stable-diffusion-xl-1.0-inpainting-0.1
}
```

**Key Difference:**
- **Regular models:** 4-channel input (RGB latents)
- **Inpainting models:** 9-channel input (RGB latents + mask + masked image)

---

## Implementation Plan

### Step 1: Process Mask Image

**File:** `src/backend/image_utils.rs` (already has stub)

```rust
/// Process mask image for inpainting
///
/// Converts mask to proper format:
/// - White (255) = inpaint this region
/// - Black (0) = keep this region
/// - Resize to match latent dimensions
///
/// # Arguments
/// * `mask` - Input mask (any format, will be converted to grayscale)
/// * `target_width` - Target width (must match image width)
/// * `target_height` - Target height (must match image height)
///
/// # Returns
/// Processed mask (grayscale, resized, normalized)
pub fn process_mask(
    mask: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Result<DynamicImage> {
    // 1. Convert to grayscale
    let gray = mask.to_luma8();
    
    // 2. Resize to target dimensions
    let resized = image::DynamicImage::ImageLuma8(gray).resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );
    
    // 3. Threshold to binary (0 or 255)
    let mut binary = resized.to_luma8();
    for pixel in binary.pixels_mut() {
        pixel[0] = if pixel[0] > 127 { 255 } else { 0 };
    }
    
    Ok(image::DynamicImage::ImageLuma8(binary))
}

/// Convert mask to latent space
///
/// Inpainting models need mask in latent space (1/8 resolution)
///
/// # Arguments
/// * `mask` - Processed mask (binary, full resolution)
/// * `device` - Device to create tensor on
/// * `dtype` - Data type
///
/// # Returns
/// Mask tensor in latent space (shape: [1, 1, height/8, width/8])
pub fn mask_to_latent_tensor(
    mask: &DynamicImage,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // 1. Resize mask to latent dimensions (1/8 of original)
    let (width, height) = (mask.width(), mask.height());
    let latent_mask = mask.resize_exact(
        width / 8,
        height / 8,
        image::imageops::FilterType::Nearest, // Use nearest for binary mask
    );
    
    // 2. Convert to tensor [0.0, 1.0]
    let mask_tensor = image_to_tensor(&latent_mask, device, dtype)?;
    
    // 3. Normalize: 0.0 = keep, 1.0 = inpaint
    let mask_tensor = (mask_tensor / 255.0)?;
    
    Ok(mask_tensor)
}
```

---

### Step 2: Prepare Masked Image Latents

**File:** `src/backend/generation.rs`

```rust
/// Prepare masked image latents for inpainting
///
/// Inpainting models need:
/// 1. Original image latents (what to keep)
/// 2. Mask latents (where to inpaint)
/// 3. Masked image latents (original * (1 - mask))
///
/// # Arguments
/// * `image` - Original image
/// * `mask` - Binary mask (white = inpaint, black = keep)
/// * `vae` - VAE for encoding
/// * `device` - Device
/// * `dtype` - Data type
///
/// # Returns
/// (image_latents, mask_latents, masked_image_latents)
pub fn prepare_inpainting_latents(
    image: &DynamicImage,
    mask: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    // 1. Encode original image to latents
    let image_latents = encode_image_to_latents(image, vae, device, dtype)?;
    
    // 2. Convert mask to latent space
    let mask_latents = crate::backend::image_utils::mask_to_latent_tensor(mask, device, dtype)?;
    
    // 3. Create masked image (original * (1 - mask))
    // This shows the model what to keep
    let inverted_mask = (Tensor::ones_like(&mask_latents)? - &mask_latents)?;
    let masked_image_latents = (image_latents.clone() * inverted_mask)?;
    
    Ok((image_latents, mask_latents, masked_image_latents))
}
```

---

### Step 3: Inpainting Generation Function

**File:** `src/backend/generation.rs`

```rust
/// Generate inpainted image
///
/// Fills in masked regions based on text prompt.
/// Uses special inpainting models (9-channel input).
///
/// # Arguments
/// * `config` - Sampling configuration
/// * `models` - Model components (MUST be inpainting variant)
/// * `input_image` - Original image
/// * `mask` - Binary mask (white = inpaint, black = keep)
/// * `progress_callback` - Progress updates
///
/// # Returns
/// Inpainted image
pub fn inpaint<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    input_image: &DynamicImage,
    mask: &DynamicImage,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    // Validate that this is an inpainting model
    if !models.version.is_inpainting() {
        return Err(Error::InvalidInput(format!(
            "Model {:?} is not an inpainting model. Use V1_5Inpaint, V2Inpaint, or XLInpaint.",
            models.version
        )));
    }
    
    config.validate()?;
    
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    let use_guide_scale = config.guidance_scale > 1.0;
    
    // 1. Generate text embeddings
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
    
    // 2. Process mask
    let processed_mask = crate::backend::image_utils::process_mask(
        mask,
        config.width,
        config.height,
    )?;
    
    // 3. Prepare inpainting latents
    let (image_latents, mask_latents, masked_image_latents) = prepare_inpainting_latents(
        input_image,
        &processed_mask,
        &models.vae,
        &models.device,
        models.dtype,
    )?;
    
    // 4. Initialize noise latents
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    let mut latents = Tensor::randn(
        0f32,
        1f32,
        (1, 4, latent_height, latent_width),
        &models.device,
    )?.to_dtype(models.dtype)?;
    
    // 5. Scale initial noise
    latents = (latents * models.scheduler.init_noise_sigma())?;
    
    // 6. Denoising loop
    let timesteps = models.scheduler.timesteps(config.steps);
    
    for (step_idx, &t) in timesteps.iter().enumerate() {
        // Concatenate latents with mask and masked image
        // Inpainting UNet expects 9 channels:
        // - 4 channels: noisy latents
        // - 1 channel: mask
        // - 4 channels: masked image latents
        let latent_model_input = Tensor::cat(
            &[&latents, &mask_latents, &masked_image_latents],
            1, // Concatenate along channel dimension
        )?;
        
        // Expand for classifier-free guidance
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latent_model_input, &latent_model_input], 0)?
        } else {
            latent_model_input
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
        
        // Blend with original image in non-masked regions
        // This ensures masked regions are regenerated, non-masked regions stay the same
        let inverted_mask = (Tensor::ones_like(&mask_latents)? - &mask_latents)?;
        latents = ((&latents * &mask_latents)? + (&image_latents * &inverted_mask)?)?;
        
        progress_callback(step_idx + 1, config.steps);
    }
    
    // 7. Decode latents to image
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
/// Execute inpaint operation
async fn execute_inpaint(
    state: JobState,
    req: operations_contract::ImageInpaintRequest,
) -> Result<JobResponse> {
    // 1. Decode base64 input image and mask
    let input_image_bytes = base64::decode(&req.image)
        .context("Failed to decode base64 image")?;
    let input_image = image::load_from_memory(&input_image_bytes)
        .context("Failed to load input image")?;
    
    let mask_bytes = base64::decode(&req.mask)
        .context("Failed to decode base64 mask")?;
    let mask = image::load_from_memory(&mask_bytes)
        .context("Failed to load mask")?;
    
    // 2. Validate that loaded model is an inpainting model
    // TODO: Check models.version.is_inpainting() before submitting
    
    // 3. Create sampling config
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps.unwrap_or(20),
        guidance_scale: req.guidance_scale.unwrap_or(7.5),
        seed: req.seed,
        width: req.width.unwrap_or(512),
        height: req.height.unwrap_or(512),
    };
    
    // 4. Submit to generation queue
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    state.queue.send(GenerationRequest {
        config,
        input_image: Some(input_image),
        mask: Some(mask), // NEW
        strength: 1.0,    // Not used for inpainting
        response_tx: tx,
    }).await?;
    
    // 5. Wait for result
    let response = rx.await?;
    
    // 6. Encode output image to base64
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
    pub input_image: Option<DynamicImage>,
    pub mask: Option<DynamicImage>,  // NEW: For inpainting
    pub strength: f64,
    pub response_tx: oneshot::Sender<GenerationResponse>,
}
```

**File:** `src/backend/generation_engine.rs`

```rust
async fn process_request(
    request: GenerationRequest,
    models: Arc<ModelComponents>,
) {
    let result = match (request.input_image, request.mask) {
        (Some(image), Some(mask)) => {
            // Inpainting
            crate::backend::generation::inpaint(
                &request.config,
                &models,
                &image,
                &mask,
                |step, total| {
                    tracing::debug!("inpaint progress: {}/{}", step, total);
                },
            )
        }
        (Some(image), None) => {
            // Image-to-image
            crate::backend::generation::image_to_image(
                &request.config,
                &models,
                &image,
                request.strength,
                |step, total| {
                    tracing::debug!("img2img progress: {}/{}", step, total);
                },
            )
        }
        (None, _) => {
            // Text-to-image
            crate::backend::generation::generate_image(
                &request.config,
                &models,
                |step, total| {
                    tracing::debug!("txt2img progress: {}/{}", step, total);
                },
            )
        }
    };
    
    // Send response...
}
```

---

## Testing Plan

### Unit Tests

**File:** `src/backend/image_utils.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_mask_binary() {
        // Test that mask is properly thresholded to binary
        // Input: grayscale with values 0-255
        // Output: only 0 or 255
    }
    
    #[test]
    fn test_mask_to_latent_tensor() {
        // Test mask downsampling to latent space
        // Input: 512x512 mask
        // Output: 64x64 latent mask (1/8 resolution)
    }
}
```

### Integration Tests

**File:** `tests/inpainting_integration.rs`

```rust
#[tokio::test]
#[ignore] // Requires inpainting model
async fn test_inpaint_remove_object() {
    // Load inpainting model
    // Load image with object
    // Create mask around object
    // Inpaint with prompt "empty background"
    // Verify object is removed
}

#[tokio::test]
#[ignore]
async fn test_inpaint_add_object() {
    // Load inpainting model
    // Load image
    // Create mask in empty region
    // Inpaint with prompt "add a cat"
    // Verify cat appears in masked region
}

#[tokio::test]
#[ignore]
async fn test_inpaint_non_inpainting_model_error() {
    // Load regular (non-inpainting) model
    // Try to inpaint
    // Verify error: "Model is not an inpainting model"
}
```

---

## Acceptance Criteria

- [ ] `process_mask()` function implemented
- [ ] `mask_to_latent_tensor()` function implemented
- [ ] `prepare_inpainting_latents()` function implemented
- [ ] `inpaint()` function implemented
- [ ] Job router wired up to handle `ImageInpaint` operations
- [ ] Request queue supports optional mask
- [ ] Model validation (must be inpainting variant)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Mask blending works correctly
- [ ] Output quality matches reference implementations
- [ ] Non-masked regions stay unchanged

---

## Common Pitfalls

### 1. **Using Regular Model for Inpainting**
❌ **Wrong:** Load SD 1.5 model, try to inpaint  
✅ **Right:** Load SD 1.5 Inpaint model specifically

### 2. **Incorrect Mask Format**
❌ **Wrong:** RGB mask with colors  
✅ **Right:** Grayscale mask (0 = keep, 255 = inpaint)

### 3. **Forgetting Latent Blending**
❌ **Wrong:** Only denoise, don't blend with original  
✅ **Right:** Blend denoised latents with original in non-masked regions

### 4. **Wrong Channel Count**
❌ **Wrong:** Pass 4-channel input to inpainting UNet  
✅ **Right:** Pass 9-channel input (4 latents + 1 mask + 4 masked image)

---

## References

- **Candle Example:** `reference/candle/candle-examples/examples/stable-diffusion-inpainting/`
- **HuggingFace Diffusers:** `diffusers.StableDiffusionInpaintPipeline`
- **Automatic1111:** `modules/processing.py` (inpainting implementation)
- **Inpainting Paper:** "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"

---

## Estimated Timeline

- **Day 1:** Implement mask processing functions
- **Day 2:** Implement inpainting latent preparation
- **Day 3:** Implement main inpaint function + wire up job router
- **Day 4:** Testing + bug fixes + documentation

**Total:** 3-4 days
