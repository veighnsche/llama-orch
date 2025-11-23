# ControlNet Support Implementation

**Priority:** 🟠 HIGH - SHOULD HAVE  
**Estimated Effort:** 7-10 days  
**Status:** ❌ NOT IMPLEMENTED  
**Assignee:** TBD

---

## Problem

**Current State:** No ControlNet support exists

**Impact:**
- Cannot control pose, depth, edges, etc.
- Professional workflows require ControlNet
- CivitAI has thousands of ControlNet models
- Limits use cases to basic generation

**ControlNet = Conditional Control:**
- Guides generation with additional inputs (pose, depth, edges, etc.)
- Enables precise control over composition
- Essential for professional/commercial work
- Can stack multiple ControlNets (pose + depth + style)

---

## What Is ControlNet?

**Technical Overview:**
- Separate neural network that conditions the UNet
- Takes conditioning image (pose skeleton, depth map, edges, etc.)
- Injects control signals at multiple UNet layers
- Original UNet remains unchanged (frozen)

**Architecture:**
```
Input Image → Preprocessor → Conditioning Image
                                    ↓
Text Prompt → CLIP → Text Embeddings
                                    ↓
Noise Latents → UNet ← ControlNet ← Conditioning Image
                  ↓
              VAE Decode → Output Image
```

**Common ControlNet Types:**
- **Canny:** Edge detection (precise lines)
- **Depth:** Depth map (3D structure)
- **OpenPose:** Human pose skeleton
- **Scribble:** Hand-drawn sketches
- **Normal:** Surface normals (lighting)
- **Seg:** Semantic segmentation
- **Lineart:** Line art extraction
- **MLSD:** Straight line detection

---

## Implementation Plan

### Step 1: ControlNet Model Structure

**File:** `src/backend/controlnet.rs` (NEW FILE)

```rust
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{Module, VarBuilder};
use std::path::Path;

/// ControlNet model
///
/// Mirrors UNet structure but with additional input channels
/// for conditioning image
#[derive(Debug)]
pub struct ControlNetModel {
    /// Model name/type (e.g., "canny", "depth", "openpose")
    pub name: String,
    
    /// ControlNet neural network
    /// NOTE: This will need to be implemented based on Candle's UNet structure
    pub model: ControlNet,
    
    /// Conditioning scale (0.0-2.0, typically 1.0)
    pub conditioning_scale: f64,
    
    /// Device
    pub device: Device,
}

impl ControlNetModel {
    /// Load ControlNet from SafeTensors file
    ///
    /// # Arguments
    /// * `path` - Path to ControlNet model directory
    /// * `name` - ControlNet type (e.g., "canny", "depth")
    /// * `device` - Device to load on
    /// * `dtype` - Data type
    ///
    /// # Returns
    /// Loaded ControlNet model
    pub fn load(
        path: impl AsRef<Path>,
        name: String,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading ControlNet '{}' from {:?}", name, path);
        
        // Load ControlNet weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.join("diffusion_pytorch_model.safetensors")],
                dtype,
                device,
            )?
        };
        
        // Build ControlNet model
        // NOTE: This requires implementing ControlNet architecture in Candle
        let model = ControlNet::new(&get_controlnet_config(), vb)?;
        
        Ok(Self {
            name,
            model,
            conditioning_scale: 1.0,
            device: device.clone(),
        })
    }
    
    /// Set conditioning scale
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.conditioning_scale = scale;
        self
    }
}

/// ControlNet neural network
///
/// Based on: https://github.com/lllyasviel/ControlNet
/// Architecture mirrors UNet but with:
/// - Additional input channels for conditioning
/// - Zero convolutions for gradual control injection
/// - Outputs added to UNet at multiple layers
pub struct ControlNet {
    // TODO: Implement ControlNet architecture
    // This is a simplified placeholder
    encoder: ControlNetEncoder,
    middle: ControlNetMiddle,
}

impl ControlNet {
    pub fn new(config: &ControlNetConfig, vb: VarBuilder) -> Result<Self> {
        // TODO: Implement ControlNet construction
        unimplemented!("ControlNet architecture not yet implemented")
    }
    
    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Noisy latents
    /// * `timestep` - Current timestep
    /// * `encoder_hidden_states` - Text embeddings
    /// * `controlnet_cond` - Conditioning image (preprocessed)
    ///
    /// # Returns
    /// Control signals for each UNet layer
    pub fn forward(
        &self,
        x: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        controlnet_cond: &Tensor,
    ) -> Result<Vec<Tensor>> {
        // TODO: Implement ControlNet forward pass
        unimplemented!("ControlNet forward pass not yet implemented")
    }
}

// Placeholder structs
struct ControlNetEncoder;
struct ControlNetMiddle;
struct ControlNetConfig;

fn get_controlnet_config() -> ControlNetConfig {
    ControlNetConfig
}
```

---

### Step 2: Conditioning Image Preprocessing

**File:** `src/backend/controlnet_preprocessors.rs` (NEW FILE)

```rust
use anyhow::Result;
use image::DynamicImage;

/// Preprocess image for ControlNet conditioning
///
/// Different ControlNet types require different preprocessing:
/// - Canny: Edge detection
/// - Depth: Depth estimation
/// - OpenPose: Pose detection
/// - etc.
pub trait ControlNetPreprocessor {
    /// Preprocess image to conditioning format
    fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage>;
}

/// Canny edge detection preprocessor
pub struct CannyPreprocessor {
    pub low_threshold: f32,
    pub high_threshold: f32,
}

impl ControlNetPreprocessor for CannyPreprocessor {
    fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage> {
        // Convert to grayscale
        let gray = image.to_luma8();
        
        // Apply Canny edge detection
        // NOTE: This requires an edge detection library
        // Options: imageproc, opencv-rust, or custom implementation
        let edges = crate::backend::image_utils::canny_edge_detection(
            &gray,
            self.low_threshold,
            self.high_threshold,
        )?;
        
        Ok(DynamicImage::ImageLuma8(edges))
    }
}

/// Depth estimation preprocessor
pub struct DepthPreprocessor {
    // Depth estimation model (e.g., MiDaS)
    // This would require loading a separate depth estimation model
}

impl ControlNetPreprocessor for DepthPreprocessor {
    fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage> {
        // Run depth estimation model
        // This is complex and may require a separate implementation
        unimplemented!("Depth estimation not yet implemented")
    }
}

/// OpenPose skeleton detection preprocessor
pub struct OpenPosePreprocessor {
    // OpenPose model for human pose detection
}

impl ControlNetPreprocessor for OpenPosePreprocessor {
    fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage> {
        // Run OpenPose model
        unimplemented!("OpenPose not yet implemented")
    }
}

/// Passthrough preprocessor (user provides pre-processed image)
pub struct PassthroughPreprocessor;

impl ControlNetPreprocessor for PassthroughPreprocessor {
    fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage> {
        // No preprocessing, use image as-is
        Ok(image.clone())
    }
}
```

---

### Step 3: Integration with Generation

**File:** `src/backend/generation.rs`

```rust
use crate::backend::controlnet::{ControlNetModel, ControlNetPreprocessor};

/// Generate image with ControlNet conditioning
///
/// # Arguments
/// * `config` - Sampling configuration
/// * `models` - Model components
/// * `controlnets` - List of ControlNet models to apply
/// * `conditioning_images` - Conditioning images for each ControlNet
/// * `progress_callback` - Progress updates
///
/// # Returns
/// Generated image with ControlNet guidance
pub fn generate_with_controlnet<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    controlnets: &[ControlNetModel],
    conditioning_images: &[DynamicImage],
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    if controlnets.len() != conditioning_images.len() {
        return Err(Error::InvalidInput(format!(
            "ControlNet count ({}) doesn't match conditioning image count ({})",
            controlnets.len(),
            conditioning_images.len()
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
    
    // 2. Preprocess conditioning images to latent space
    let conditioning_latents: Vec<Tensor> = conditioning_images
        .iter()
        .map(|img| {
            // Resize to match generation size
            let resized = img.resize_exact(
                config.width,
                config.height,
                image::imageops::FilterType::Lanczos3,
            );
            
            // Convert to tensor
            crate::backend::image_utils::image_to_tensor(
                &resized,
                &models.device,
                models.dtype,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    
    // 3. Initialize noise latents
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    let mut latents = Tensor::randn(
        0f32,
        1f32,
        (1, 4, latent_height, latent_width),
        &models.device,
    )?.to_dtype(models.dtype)?;
    
    latents = (latents * models.scheduler.init_noise_sigma())?;
    
    // 4. Denoising loop with ControlNet
    let timesteps = models.scheduler.timesteps(config.steps);
    
    for (step_idx, &t) in timesteps.iter().enumerate() {
        // Expand latents for classifier-free guidance
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        // Get ControlNet conditioning signals
        let mut control_signals = Vec::new();
        for (controlnet, cond_latent) in controlnets.iter().zip(conditioning_latents.iter()) {
            let signals = controlnet.model.forward(
                &latent_model_input,
                t as f64,
                &text_embeddings,
                cond_latent,
            )?;
            
            // Scale by conditioning_scale
            let scaled_signals: Vec<Tensor> = signals
                .into_iter()
                .map(|s| (s * controlnet.conditioning_scale).unwrap())
                .collect();
            
            control_signals.push(scaled_signals);
        }
        
        // Predict noise with ControlNet conditioning
        let noise_pred = models.unet.forward_with_controlnet(
            &latent_model_input,
            t as f64,
            &text_embeddings,
            &control_signals,
        )?;
        
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
        
        progress_callback(step_idx + 1, config.steps);
    }
    
    // 5. Decode latents to image
    let image = models.vae.decode(&(&latents / models.vae_scale)?)?;
    let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let image = (image.clamp(0f32, 1f32)? * 255.)?
        .to_dtype(DType::U8)?
        .i(0)?;
    
    crate::backend::image_utils::tensor_to_image(&image)
}
```

---

### Step 4: UNet Modification for ControlNet

**File:** Fork of `candle-transformers/src/models/stable_diffusion/unet_2d.rs`

```rust
impl UNet2DConditionModel {
    /// Forward pass with ControlNet conditioning
    ///
    /// # Arguments
    /// * `x` - Noisy latents
    /// * `timestep` - Current timestep
    /// * `encoder_hidden_states` - Text embeddings
    /// * `controlnet_signals` - Control signals from ControlNet(s)
    ///
    /// # Returns
    /// Predicted noise
    pub fn forward_with_controlnet(
        &self,
        x: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        controlnet_signals: &[Vec<Tensor>],
    ) -> Result<Tensor> {
        // Regular UNet forward pass
        let mut h = self.conv_in.forward(x)?;
        
        // Down blocks with ControlNet injection
        let mut down_block_res_samples = Vec::new();
        for (i, down_block) in self.down_blocks.iter().enumerate() {
            h = down_block.forward(&h, encoder_hidden_states)?;
            
            // Add ControlNet signals if available
            if !controlnet_signals.is_empty() {
                for signals in controlnet_signals {
                    if i < signals.len() {
                        h = (h + &signals[i])?;
                    }
                }
            }
            
            down_block_res_samples.push(h.clone());
        }
        
        // Middle block with ControlNet injection
        h = self.mid_block.forward(&h, encoder_hidden_states)?;
        if !controlnet_signals.is_empty() {
            for signals in controlnet_signals {
                if let Some(mid_signal) = signals.last() {
                    h = (h + mid_signal)?;
                }
            }
        }
        
        // Up blocks (no ControlNet injection here)
        for up_block in &self.up_blocks {
            let res_sample = down_block_res_samples.pop().unwrap();
            h = up_block.forward(&h, &res_sample, encoder_hidden_states)?;
        }
        
        // Output
        self.conv_out.forward(&h)
    }
}
```

---

## Major Challenges

### 1. **ControlNet Architecture Implementation**
- Need to implement full ControlNet architecture in Candle
- Mirrors UNet structure (complex)
- Requires zero convolutions
- Estimated: 3-4 days

### 2. **Preprocessing Models**
- Canny: Relatively easy (edge detection)
- Depth: Requires MiDaS or similar (separate model)
- OpenPose: Requires pose detection model (separate model)
- Estimated: 2-3 days per preprocessor

### 3. **UNet Modification**
- Need to fork candle-transformers
- Add ControlNet injection points
- Maintain compatibility with existing code
- Estimated: 1-2 days

---

## Phased Implementation

### Phase 1: Basic ControlNet (3-4 days)
- [ ] Implement ControlNet architecture
- [ ] Add Canny preprocessor (easiest)
- [ ] Modify UNet for control injection
- [ ] Basic generation with single ControlNet

### Phase 2: Multiple ControlNets (1-2 days)
- [ ] Support stacking multiple ControlNets
- [ ] Test combinations (Canny + Depth)
- [ ] Optimize performance

### Phase 3: Additional Preprocessors (2-3 days each)
- [ ] Depth estimation (MiDaS)
- [ ] OpenPose detection
- [ ] Scribble/Lineart
- [ ] Normal maps

---

## Testing Plan

### Unit Tests

**File:** `src/backend/controlnet.rs`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_controlnet_loading() {
        // Test loading ControlNet model
    }
    
    #[test]
    fn test_conditioning_scale() {
        // Test scale parameter (0.0-2.0)
    }
}
```

### Integration Tests

**File:** `tests/controlnet_integration.rs`

```rust
#[tokio::test]
#[ignore]
async fn test_canny_controlnet() {
    // Load base model + Canny ControlNet
    // Generate with edge conditioning
    // Verify output follows edges
}

#[tokio::test]
#[ignore]
async fn test_multiple_controlnets() {
    // Load base model + Canny + Depth
    // Generate with both conditionings
    // Verify combined effect
}
```

---

## Acceptance Criteria

- [ ] ControlNet architecture implemented in Candle
- [ ] At least Canny preprocessor working
- [ ] Single ControlNet generation works
- [ ] Multiple ControlNet stacking works
- [ ] Conditioning scale parameter works
- [ ] UNet modification complete
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Performance acceptable (< 30% slowdown)

---

## Marketplace Impact

**Before ControlNet:**
- ✅ Checkpoint models
- ❌ NO ControlNet (thousands of models unavailable)

**After ControlNet:**
- ✅ Checkpoint models
- ✅ ControlNet models (thousands now available!)
- 🎯 Professional workflows enabled

---

## References

- **ControlNet Paper:** "Adding Conditional Control to Text-to-Image Diffusion Models"
- **ControlNet GitHub:** https://github.com/lllyasviel/ControlNet
- **Diffusers:** `diffusers.ControlNetModel`
- **Automatic1111:** `extensions-builtin/ControlNet/`

---

## Estimated Timeline

- **Day 1-4:** Implement ControlNet architecture
- **Day 5-6:** Implement Canny preprocessor + integration
- **Day 7-8:** Modify UNet, add injection points
- **Day 9:** Multiple ControlNet support
- **Day 10:** Testing + bug fixes + documentation

**Total:** 7-10 days (Phase 1 only)

**Additional preprocessors:** +2-3 days each
