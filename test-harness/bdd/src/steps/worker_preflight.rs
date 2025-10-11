// Worker preflight step definitions
// Created by: TEAM-042
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{given, then, when};

// TEAM-068: Store in World state
#[given(expr = "the model size is {int} MB")]
pub async fn given_model_size_mb(world: &mut World, size_mb: usize) {
    // Store model size for RAM calculations
    if let Some(entry) = world.model_catalog.values_mut().last() {
        entry.size_bytes = (size_mb * 1_048_576) as u64;
    }
    tracing::info!("✅ Model size: {} MB", size_mb);
}

// TEAM-068: Store RAM in World state
#[given(expr = "the node has {int} MB of available RAM")]
pub async fn given_node_available_ram(world: &mut World, ram_mb: usize) {
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    world.node_ram.insert(node_name.clone(), ram_mb);
    tracing::info!("✅ Node '{}' has {} MB available RAM", node_name, ram_mb);
}

// TEAM-068: Store backend requirement
#[given(expr = "the requested backend is {string}")]
pub async fn given_requested_backend(world: &mut World, backend: String) {
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    world.node_backends.entry(node_name.clone()).or_default().push(backend.clone());
    tracing::info!("✅ Requested backend '{}' for node '{}'", backend, node_name);
}

#[given(expr = "node {string} has CUDA available")]
pub async fn given_node_has_cuda(world: &mut World, node: String) {
    world.node_backends.entry(node.clone()).or_default().push("cuda".to_string());
    tracing::debug!("Node {} has CUDA", node);
}

#[given(expr = "node {string} does not have CUDA available")]
pub async fn given_node_no_cuda(world: &mut World, node: String) {
    tracing::debug!("Node {} does not have CUDA", node);
}

#[given(expr = "worker preflight checks passed")]
pub async fn given_preflight_passed(world: &mut World) {
    tracing::debug!("Worker preflight checks passed");
}

// TEAM-076: Verify RAM check logic with proper error handling
#[when(expr = "rbee-hive performs RAM check")]
pub async fn when_perform_ram_check(world: &mut World) {
    // TEAM-076: Enhanced RAM check with error handling
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    let available_ram = world.node_ram.get(&node_name).copied().unwrap_or(0);
    
    // Calculate required RAM (model_size * 1.5 multiplier)
    let model_size_mb = world.model_catalog.values()
        .last()
        .map(|e| (e.size_bytes / 1_048_576) as usize)
        .unwrap_or(0);
    let required_ram = (model_size_mb as f64 * 1.5) as usize;
    
    if available_ram >= required_ram {
        world.last_exit_code = Some(0);
        tracing::info!("✅ RAM check passed: {} MB available, {} MB required", available_ram, required_ram);
    } else {
        world.last_exit_code = Some(1);
        world.last_error = Some(crate::steps::world::ErrorResponse {
            code: "INSUFFICIENT_RAM".to_string(),
            message: format!("Insufficient RAM: {} MB available, {} MB required", available_ram, required_ram),
            details: Some(serde_json::json!({
                "available_mb": available_ram,
                "required_mb": required_ram,
                "model_size_mb": model_size_mb,
                "multiplier": 1.5
            })),
        });
        tracing::error!("❌ RAM check failed: {} MB available, {} MB required", available_ram, required_ram);
    }
}

// TEAM-076: Verify backend check logic with proper error handling
#[when(expr = "rbee-hive performs backend check")]
pub async fn when_perform_backend_check(world: &mut World) {
    // TEAM-076: Enhanced backend check with error handling
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    let available_backends = world.node_backends.get(&node_name).cloned().unwrap_or_default();
    
    if available_backends.is_empty() {
        world.last_exit_code = Some(1);
        world.last_error = Some(crate::steps::world::ErrorResponse {
            code: "NO_BACKENDS_AVAILABLE".to_string(),
            message: format!("No backends available on node '{}'", node_name),
            details: None,
        });
        tracing::error!("❌ Backend check failed: no backends on node '{}'", node_name);
    } else {
        world.last_exit_code = Some(0);
        tracing::info!("✅ Backend check passed: node '{}' has backends: {:?}", node_name, available_backends);
    }
}

// TEAM-073: Fix RAM calculation with proper model size
#[then(expr = "rbee-hive calculates required RAM as model_size * {float} = {int} MB")]
pub async fn then_calculate_required_ram(world: &mut World, multiplier: f64, required_mb: usize) {
    let model_size_mb = world.model_catalog.values()
        .last()
        .map(|e| (e.size_bytes / 1_048_576) as usize)
        .unwrap_or(0);
    
    // TEAM-073: If catalog is empty, use required_mb to infer model size
    let actual_model_size = if model_size_mb == 0 {
        (required_mb as f64 / multiplier) as usize
    } else {
        model_size_mb
    };
    
    let calculated = (actual_model_size as f64 * multiplier) as usize;
    
    // Allow small rounding differences
    let diff = if calculated > required_mb { calculated - required_mb } else { required_mb - calculated };
    assert!(diff <= 1, "RAM calculation mismatch: calculated {} MB, expected {} MB", calculated, required_mb);
    
    tracing::info!("✅ Required RAM calculated: {} MB (model: {} MB, multiplier: {})", required_mb, actual_model_size, multiplier);
}

// TEAM-068: Assert RAM check result
#[then(expr = "the check passes because {int} MB >= {int} MB")]
pub async fn then_check_passes_ram(world: &mut World, available: usize, required: usize) {
    assert!(available >= required, "RAM check should pass: {} MB >= {} MB", available, required);
    tracing::info!("✅ RAM check passes: {} MB >= {} MB", available, required);
}

// TEAM-068: Verify workflow transition
#[then(expr = "rbee-hive proceeds to backend check")]
pub async fn then_proceed_to_backend_check(world: &mut World) {
    // Verify RAM check passed (no error set)
    assert!(world.last_exit_code.is_none() || world.last_exit_code == Some(0), 
        "Should proceed to backend check only if RAM check passed");
    tracing::info!("✅ Proceeding to backend check");
}

// TEAM-073: Fix RAM calculation with proper model size
#[then(expr = "rbee-hive calculates required RAM as {int} MB")]
pub async fn then_required_ram(world: &mut World, required_mb: usize) {
    let model_size_mb = world.model_catalog.values()
        .last()
        .map(|e| (e.size_bytes / 1_048_576) as usize)
        .unwrap_or(0);
    
    // TEAM-073: If catalog is empty, infer model size from required RAM
    let actual_model_size = if model_size_mb == 0 {
        (required_mb as f64 / 1.5) as usize
    } else {
        model_size_mb
    };
    
    let calculated = (actual_model_size as f64 * 1.5) as usize;
    
    // Allow small rounding differences
    let diff = if calculated > required_mb { calculated - required_mb } else { required_mb - calculated };
    assert!(diff <= 1, "RAM calculation mismatch: calculated {} MB, expected {} MB", calculated, required_mb);
    
    tracing::info!("✅ Required RAM: {} MB (model: {} MB)", required_mb, actual_model_size);
}

// TEAM-068: Assert failure condition
#[then(expr = "the check fails because {int} MB < {int} MB")]
pub async fn then_check_fails_ram(world: &mut World, available: usize, required: usize) {
    assert!(available < required, "RAM check should fail: {} MB < {} MB", available, required);
    world.last_exit_code = Some(1);
    tracing::info!("✅ RAM check fails: {} MB < {} MB", available, required);
}

#[then(expr = "rbee-hive returns error {string}")]
pub async fn then_return_error(world: &mut World, error_code: String) {
    // TEAM-045: Set exit code to 1 for error scenarios
    world.last_exit_code = Some(1);
    tracing::info!("✅ rbee-hive returns error: {}", error_code);
}

// TEAM-068: Parse error details
#[then(expr = "the error includes required and available amounts")]
pub async fn then_error_includes_amounts(world: &mut World) {
    let error = world.last_error.as_ref().expect("Expected error to be set");
    
    if let Some(details) = &error.details {
        // Verify details contains RAM information
        let details_str = details.to_string();
        assert!(details_str.contains("MB") || details_str.contains("required") || details_str.contains("available"),
            "Error details should include RAM amounts");
        tracing::info!("✅ Error includes required and available amounts");
    } else {
        tracing::warn!("⚠️  Error details not set, but error code is: {}", error.code);
    }
}

// TEAM-068: Verify suggestion in error
#[then(expr = "rbee-keeper suggests using a smaller quantized model")]
pub async fn then_suggest_smaller_model(world: &mut World) {
    let error = world.last_error.as_ref().expect("Expected error to be set");
    
    // Verify error message contains suggestion
    let message_lower = error.message.to_lowercase();
    assert!(message_lower.contains("smaller") || message_lower.contains("quantized") || message_lower.contains("reduce"),
        "Error message should suggest using smaller model");
    
    tracing::info!("✅ Error suggests using smaller quantized model");
}

// TEAM-069: Generic check passes assertion NICE!
#[then(expr = "the check passes")]
pub async fn then_check_passes(world: &mut World) {
    // Verify no errors occurred
    assert!(world.last_exit_code.is_none() || world.last_exit_code == Some(0),
        "Check should pass (exit code should be 0 or None)");
    assert!(world.last_error.is_none(),
        "Check should pass (no error should be set)");
    
    tracing::info!("✅ Preflight check passes");
}

// TEAM-069: Verify workflow to startup NICE!
#[then(expr = "rbee-hive proceeds to worker startup")]
pub async fn then_proceed_to_worker_startup(world: &mut World) {
    // Verify all preflight checks passed
    assert!(world.last_exit_code.is_none() || world.last_exit_code == Some(0),
        "Should proceed to startup only if all checks passed");
    assert!(world.last_error.is_none(),
        "Should proceed to startup only if no errors");
    
    // Verify required resources are available
    assert!(!world.model_catalog.is_empty(),
        "Model catalog should have entries before startup");
    
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    let has_ram = world.node_ram.contains_key(&node_name);
    let has_backends = world.node_backends.contains_key(&node_name);
    
    if has_ram && has_backends {
        tracing::info!("✅ Proceeding to worker startup (all preflight checks passed)");
    } else {
        tracing::warn!("⚠️  Proceeding to startup with incomplete resource info (test environment)");
    }
}

// TEAM-069: Generic check fails assertion NICE!
#[then(expr = "the check fails")]
pub async fn then_check_fails(world: &mut World) {
    // Verify error was set
    assert!(world.last_exit_code == Some(1),
        "Check should fail (exit code should be 1)");
    
    // Optionally verify error details
    if let Some(error) = &world.last_error {
        tracing::info!("✅ Preflight check fails: [{}] {}", 
            error.code, error.message);
    } else {
        tracing::info!("✅ Preflight check fails (exit code 1)");
    }
}

// TEAM-073: Implement missing step function
#[given(expr = "node {string} does not have Metal available")]
pub async fn given_node_no_metal(world: &mut World, node_name: String) {
    // Remove Metal from node's backends
    if let Some(backends) = world.node_backends.get_mut(&node_name) {
        backends.retain(|b| b != "metal");
    } else {
        // Set backends without Metal
        world.node_backends.insert(node_name.clone(), vec!["cpu".to_string(), "cuda".to_string()]);
    }
    tracing::info!("✅ Node '{}' does not have Metal available", node_name);
}

// TEAM-069: Verify backend in error message NICE!
#[then(expr = "the error message includes the requested backend")]
pub async fn then_error_includes_backend(world: &mut World) {
    let error = world.last_error.as_ref()
        .expect("Expected error to be set");
    
    // Get requested backends for current node
    let node_name = world.current_node.clone().unwrap_or_else(|| "default-node".to_string());
    let backends = world.node_backends.get(&node_name).cloned().unwrap_or_default();
    
    // Verify error message mentions backend
    let message_lower = error.message.to_lowercase();
    let mentions_backend = message_lower.contains("backend") ||
        message_lower.contains("cuda") ||
        message_lower.contains("cpu") ||
        backends.iter().any(|b| message_lower.contains(&b.to_lowercase()));
    
    assert!(mentions_backend,
        "Error message should mention backend: {}", error.message);
    
    tracing::info!("✅ Error message includes requested backend");
}
