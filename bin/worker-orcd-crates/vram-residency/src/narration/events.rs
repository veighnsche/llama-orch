//! Narration events for VRAM operations
//!
//! Provides human-readable narration with structured fields for observability.

use observability_narration_core::{narrate, NarrationFields};

/// Narrate VRAM manager initialization
///
/// # Parameters
///
/// * `gpu_count` - Number of GPUs detected
/// * `total_vram_gb` - Total VRAM capacity in GB
/// * `worker_id` - Worker identifier
pub fn narrate_vram_manager_init(gpu_count: usize, total_vram_gb: f64, worker_id: Option<&str>) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "init",
        target: format!("gpu-count-{}", gpu_count),
        human: format!(
            "Initialized VRAM manager with {} GPU(s), {:.1} GB total VRAM",
            gpu_count, total_vram_gb
        ),
        cute: Some(format!(
            "Woke up and found {} friendly GPU{}! They have {:.1} GB of cozy VRAM space! 🎉✨",
            gpu_count, if gpu_count == 1 { "" } else { "s" }, total_vram_gb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("{} GPUs", gpu_count)),
        ..Default::default()
    });
}

/// Narrate model seal operation
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `gpu_device` - GPU device index
/// * `vram_mb` - VRAM allocated in MB
/// * `duration_ms` - Operation duration in milliseconds
/// * `worker_id` - Worker identifier
/// * `correlation_id` - Request correlation ID
pub fn narrate_model_sealed(
    shard_id: &str,
    gpu_device: u32,
    vram_mb: usize,
    duration_ms: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "seal",
        target: shard_id.to_string(),
        human: format!(
            "Sealed model shard '{}' in {} MB VRAM on GPU {} ({} ms)",
            shard_id, vram_mb, gpu_device, duration_ms
        ),
        cute: Some(format!(
            "Tucked '{}' safely into GPU{}'s warm {} MB nest! Sweet dreams! 🛏️✨",
            shard_id, gpu_device, vram_mb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate seal verification success
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `gpu_device` - GPU device index
/// * `duration_ms` - Verification duration in milliseconds
/// * `worker_id` - Worker identifier
/// * `correlation_id` - Request correlation ID
pub fn narrate_seal_verified(
    shard_id: &str,
    gpu_device: u32,
    duration_ms: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "verify",
        target: shard_id.to_string(),
        human: format!(
            "Verified seal for shard '{}' on GPU {} ({} ms)",
            shard_id, gpu_device, duration_ms
        ),
        cute: Some(format!(
            "Checked on '{}' — still sleeping soundly on GPU{}! All is well! 🔍💕",
            shard_id, gpu_device
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate seal verification failure (CRITICAL)
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `gpu_device` - GPU device index
/// * `reason` - Why verification failed
/// * `worker_id` - Worker identifier
/// * `correlation_id` - Request correlation ID
pub fn narrate_seal_verification_failed(
    shard_id: &str,
    gpu_device: u32,
    reason: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "verify_failed",
        target: shard_id.to_string(),
        human: format!(
            "CRITICAL: Seal verification failed for shard '{}' on GPU {}: {}",
            shard_id, gpu_device, reason
        ),
        cute: Some(format!(
            "Uh oh! '{}' on GPU{} doesn't look right — {}! Time to investigate! 😟🔍",
            shard_id, gpu_device, reason
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        error_kind: Some(reason.to_string()),
        ..Default::default()
    });
}

/// Narrate VRAM allocation
///
/// # Parameters
///
/// * `requested_mb` - Requested allocation in MB
/// * `allocated_mb` - Actual allocated in MB
/// * `available_mb` - Available VRAM in MB
/// * `gpu_device` - GPU device index
/// * `duration_ms` - Allocation duration in milliseconds
/// * `worker_id` - Worker identifier
pub fn narrate_vram_allocated(
    requested_mb: usize,
    allocated_mb: usize,
    available_mb: usize,
    gpu_device: u32,
    duration_ms: u64,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "allocate",
        target: format!("GPU{}", gpu_device),
        human: format!(
            "Allocated {} MB VRAM on GPU {} (requested {} MB, {} MB available, {} ms)",
            allocated_mb, gpu_device, requested_mb, available_mb, duration_ms
        ),
        cute: Some(format!(
            "Found a perfect {} MB spot on GPU{}! {} MB still available for friends! 🏠✨",
            allocated_mb, gpu_device, available_mb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate VRAM allocation failure
///
/// # Parameters
///
/// * `requested_mb` - Requested allocation in MB
/// * `available_mb` - Available VRAM in MB
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub fn narrate_vram_allocation_failed(
    requested_mb: usize,
    available_mb: usize,
    gpu_device: u32,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "allocate_failed",
        target: format!("GPU{}", gpu_device),
        human: format!(
            "VRAM allocation failed on GPU {}: requested {} MB, only {} MB available",
            gpu_device, requested_mb, available_mb
        ),
        cute: Some(format!(
            "Oh dear! GPU{} doesn't have enough room (need {} MB, only {} MB free). Let's try elsewhere! 😟",
            gpu_device, requested_mb, available_mb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        error_kind: Some("insufficient_vram".to_string()),
        ..Default::default()
    });
}

/// Narrate VRAM deallocation
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `freed_mb` - Freed VRAM in MB
/// * `remaining_mb` - Remaining used VRAM in MB
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub fn narrate_vram_deallocated(
    shard_id: &str,
    freed_mb: usize,
    remaining_mb: usize,
    gpu_device: u32,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "deallocate",
        target: shard_id.to_string(),
        human: format!(
            "Deallocated {} MB VRAM for shard '{}' on GPU {} ({} MB still in use)",
            freed_mb, shard_id, gpu_device, remaining_mb
        ),
        cute: Some(format!(
            "Said goodbye to '{}' and tidied up {} MB on GPU{}! Room for new friends! 👋🧹",
            shard_id, freed_mb, gpu_device
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        ..Default::default()
    });
}

/// Narrate policy violation (CRITICAL)
///
/// # Parameters
///
/// * `violation` - Description of the violation
/// * `gpu_device` - GPU device index
/// * `action_taken` - Action taken in response
/// * `worker_id` - Worker identifier
pub fn narrate_policy_violation(
    violation: &str,
    gpu_device: u32,
    action_taken: &str,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "policy_violation",
        target: format!("GPU{}", gpu_device),
        human: format!(
            "CRITICAL: VRAM-only policy violated on GPU {}: {}. Action: {}",
            gpu_device, violation, action_taken
        ),
        cute: Some(format!(
            "Oops! GPU{} has a problem: {}. We need to {}! 🛑",
            gpu_device, violation, action_taken.to_lowercase()
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        error_kind: Some("policy_violation".to_string()),
        ..Default::default()
    });
}

/// Narrate digest computation
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `data_mb` - Data size in MB
/// * `duration_ms` - Computation duration in milliseconds
/// * `worker_id` - Worker identifier
pub fn narrate_digest_computed(
    shard_id: &str,
    data_mb: usize,
    duration_ms: u64,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "digest",
        target: shard_id.to_string(),
        human: format!(
            "Computed SHA-256 digest for {} MB data ({} ms)",
            data_mb, duration_ms
        ),
        cute: Some(format!(
            "Created a unique fingerprint for '{}' ({} MB of data)! 🔐✨",
            shard_id, data_mb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate signature generation
///
/// # Parameters
///
/// * `shard_id` - Shard identifier
/// * `duration_ms` - Generation duration in milliseconds
/// * `worker_id` - Worker identifier
pub fn narrate_signature_generated(
    shard_id: &str,
    duration_ms: u64,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "sign",
        target: shard_id.to_string(),
        human: format!(
            "Generated HMAC-SHA256 signature for shard '{}' ({} ms)",
            shard_id, duration_ms
        ),
        cute: Some(format!(
            "Put a special safety seal on '{}'! Now it's protected! 🔏✨",
            shard_id
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate GPU context initialization
///
/// # Parameters
///
/// * `gpu_device` - GPU device index
/// * `gpu_name` - GPU name
/// * `vram_gb` - Total VRAM in GB
/// * `worker_id` - Worker identifier
pub fn narrate_cuda_context_init(
    gpu_device: u32,
    gpu_name: &str,
    vram_gb: f64,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "cuda_init",
        target: format!("GPU{}", gpu_device),
        human: format!(
            "Initialized CUDA context on GPU {} ({}, {:.1} GB VRAM)",
            gpu_device, gpu_name, vram_gb
        ),
        cute: Some(format!(
            "Said hello to GPU{} ({})! It has {:.1} GB of VRAM ready to help! 👋💚",
            gpu_device, gpu_name, vram_gb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}: {}", gpu_device, gpu_name)),
        ..Default::default()
    });
}

/// Narrate input validation failure
///
/// # Parameters
///
/// * `input_type` - Type of input (e.g., "shard_id", "model_size")
/// * `reason` - Why validation failed
/// * `worker_id` - Worker identifier
pub fn narrate_validation_failed(
    input_type: &str,
    reason: &str,
    worker_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "validate_failed",
        target: input_type.to_string(),
        human: format!(
            "Input validation failed for {}: {}",
            input_type, reason
        ),
        cute: Some(format!(
            "Hmm, the {} doesn't look right: {}. Let's fix that! 🤔",
            input_type, reason
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        error_kind: Some("validation_error".to_string()),
        ..Default::default()
    });
}

/// Narrate capacity query
///
/// * `available_mb` - Available VRAM in MB
/// * `total_mb` - Total VRAM in MB
/// * `used_mb` - Used VRAM in MB
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub fn narrate_capacity_query(
    available_mb: usize,
    total_mb: usize,
    used_mb: usize,
    gpu_device: u32,
    worker_id: Option<&str>,
) {
    let usage_percent = (used_mb as f64 / total_mb as f64 * 100.0) as usize;
    
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "capacity_query",
        target: format!("GPU{}", gpu_device),
        human: format!(
            "GPU {} capacity: {} MB available, {} MB used ({}% utilization)",
            gpu_device, available_mb, used_mb, usage_percent
        ),
        cute: Some(format!(
            "GPU{} status check: {} MB free, {} MB busy ({}% full)! {}",
            gpu_device, available_mb, used_mb, usage_percent,
            if usage_percent > 90 { "Getting cozy! 🏠" } else { "Plenty of room! ✨" }
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        device: Some(format!("GPU{}", gpu_device)),
        ..Default::default()
    });
}
