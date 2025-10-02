//! Narration events for model loading operations
//!
//! Provides human-readable and cute children's book style narration
//! for all model-loader operations.

use observability_narration_core::{narrate, NarrationFields};

/// Narrate model load start
pub fn narrate_load_start(
    model_path: &str,
    max_size_gb: f64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "load_start",
        target: model_path.to_string(),
        human: format!(
            "Loading model from {} (max size: {:.1} GB)",
            model_path, max_size_gb
        ),
        cute: Some(format!(
            "Looking for model at {}! Let's load it up! 📦✨",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

/// Narrate path validation success
pub fn narrate_path_validated(
    canonical_path: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "path_validated",
        target: canonical_path.to_string(),
        human: format!(
            "Validated path: {} (within allowed root)",
            canonical_path
        ),
        cute: Some(format!(
            "Found the model at {}! Path looks safe! ✅🗂️",
            canonical_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

/// Narrate path validation failure
pub fn narrate_path_validation_failed(
    attempted_path: &str,
    error_kind: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "path_validation_failed",
        target: attempted_path.to_string(),
        human: format!(
            "Path validation failed: {} in '{}'",
            error_kind, attempted_path
        ),
        cute: Some(format!(
            "Whoa! That path looks suspicious ({})! Nice try, but no! 🛑🔍",
            attempted_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        error_kind: Some(error_kind.to_string()),
        ..Default::default()
    });
}

/// Narrate file size check
pub fn narrate_size_checked(
    model_path: &str,
    file_size_gb: f64,
    max_size_gb: f64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "size_checked",
        target: model_path.to_string(),
        human: format!(
            "File size: {:.2} GB (within {:.1} GB limit)",
            file_size_gb, max_size_gb
        ),
        cute: Some(format!(
            "Model is {:.2} GB — fits perfectly within our {:.1} GB limit! 📏✨",
            file_size_gb, max_size_gb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

/// Narrate file too large
pub fn narrate_size_check_failed(
    model_path: &str,
    file_size_gb: f64,
    max_size_gb: f64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "size_check_failed",
        target: model_path.to_string(),
        human: format!(
            "File too large: {:.2} GB (max: {:.1} GB)",
            file_size_gb, max_size_gb
        ),
        cute: Some(format!(
            "Oh no! Model is {:.2} GB but we can only handle {:.1} GB! Too big! 😟📦",
            file_size_gb, max_size_gb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        error_kind: Some("file_too_large".to_string()),
        ..Default::default()
    });
}

/// Narrate hash verification started
pub fn narrate_hash_verify_start(
    model_path: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "hash_verify_start",
        target: model_path.to_string(),
        human: format!("Verifying SHA-256 hash for {}", model_path),
        cute: Some(format!(
            "Checking {}'s fingerprint to make sure it's authentic! 🔍🔐",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

/// Narrate hash verification success
pub fn narrate_hash_verified(
    model_path: &str,
    hash_prefix: &str,
    duration_ms: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "hash_verified",
        target: model_path.to_string(),
        human: format!("Hash verified: {}... (SHA-256 match)", hash_prefix),
        cute: Some(format!(
            "Perfect! {}'s fingerprint matches! All authentic! ✅✨",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate hash verification failed
pub fn narrate_hash_verification_failed(
    model_path: &str,
    expected_prefix: &str,
    actual_prefix: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "hash_verification_failed",
        target: model_path.to_string(),
        human: format!(
            "Hash mismatch: expected {}..., got {}... (integrity violation)",
            expected_prefix, actual_prefix
        ),
        cute: Some(format!(
            "Uh oh! {}'s fingerprint doesn't match! Expected one thing, got another! 😟❌",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        error_kind: Some("hash_mismatch".to_string()),
        ..Default::default()
    });
}

/// Narrate GGUF validation started
pub fn narrate_gguf_validate_start(
    model_path: &str,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "gguf_validate_start",
        target: model_path.to_string(),
        human: format!("Validating GGUF format for {}", model_path),
        cute: Some(format!(
            "Checking if {} is a valid GGUF file! 📋✨",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

/// Narrate GGUF validation success
pub fn narrate_gguf_validated(
    model_path: &str,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    duration_ms: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "gguf_validated",
        target: model_path.to_string(),
        human: format!(
            "GGUF format validated: version {}, {} tensors, {} metadata KV pairs",
            version, tensor_count, metadata_count
        ),
        cute: Some(format!(
            "{} is a perfect GGUF file (v{}, {} tensors)! Ready to load! 🎉📦",
            model_path, version, tensor_count
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate GGUF validation failed (invalid magic)
pub fn narrate_gguf_validation_failed_magic(
    model_path: &str,
    expected: u32,
    actual: u32,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "gguf_validation_failed",
        target: model_path.to_string(),
        human: format!(
            "Invalid GGUF magic number: expected 0x{:x}, got 0x{:x}",
            expected, actual
        ),
        cute: Some(format!(
            "Hmm, {} doesn't look like a GGUF file! Wrong magic number! 🤔❌",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        error_kind: Some("invalid_magic".to_string()),
        ..Default::default()
    });
}

/// Narrate GGUF validation failed (bounds check)
pub fn narrate_gguf_validation_failed_bounds(
    model_path: &str,
    limit_type: &str,
    actual: u64,
    max: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "gguf_validation_failed",
        target: model_path.to_string(),
        human: format!(
            "GGUF bounds check failed: {} {} exceeds maximum {}",
            limit_type, actual, max
        ),
        cute: Some(format!(
            "Whoa! {} claims to have {} {}! That's way too many! 😟📊",
            model_path, actual, limit_type
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        error_kind: Some("bounds_check_failed".to_string()),
        ..Default::default()
    });
}

/// Narrate model load complete
pub fn narrate_load_complete(
    model_path: &str,
    file_size_gb: f64,
    duration_ms: u64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "load_complete",
        target: model_path.to_string(),
        human: format!(
            "Model loaded: {} ({:.2} GB, validated, ready for VRAM)",
            model_path, file_size_gb
        ),
        cute: Some(format!(
            "Hooray! {} is loaded and validated! Ready to go to VRAM! 🎉🚀",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        duration_ms: Some(duration_ms),
        ..Default::default()
    });
}

/// Narrate bytes validation (memory mode)
pub fn narrate_bytes_validated(
    size_gb: f64,
    hash_verified: bool,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "bytes_validated",
        target: "memory".to_string(),
        human: format!(
            "Validated model bytes: {:.2} GB, hash {}, GGUF format valid",
            size_gb,
            if hash_verified {
                "verified"
            } else {
                "not checked"
            }
        ),
        cute: Some(format!(
            "Checked the model bytes — {:.2} GB of perfect GGUF data! All good! ✅✨",
            size_gb
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}
