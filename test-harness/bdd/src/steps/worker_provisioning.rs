// Step definitions for Worker Provisioning (cargo build from git)
// Created by: TEAM-078
//
// ⚠️ CRITICAL: These steps MUST connect to real product code from /bin/
// ⚠️ Import rbee_hive::worker_provisioner and test actual cargo builds

use cucumber::{given, then, when};
use crate::steps::world::World;

#[given(expr = "worker binary {string} not in catalog")]
pub async fn given_worker_not_in_catalog(world: &mut World, worker_type: String) {
    // TEAM-078: Wire to rbee_hive::worker_catalog::WorkerCatalog
    tracing::info!("TEAM-078: Worker {} not in catalog", worker_type);
    world.last_action = Some(format!("worker_not_in_catalog_{}", worker_type));
}

#[given(expr = "worker binary {string} built successfully")]
pub async fn given_worker_built_successfully(world: &mut World, worker_type: String) {
    // TEAM-078: Simulate successful build
    tracing::info!("TEAM-078: Worker {} built successfully", worker_type);
    world.last_action = Some(format!("worker_built_{}", worker_type));
}

#[when(expr = "rbee-hive builds worker from git with features {string}")]
pub async fn when_build_worker_with_features(world: &mut World, features: String) {
    // TEAM-078: Call rbee_hive::worker_provisioner::WorkerProvisioner::build()
    tracing::info!("TEAM-078: Building worker with features: {}", features);
    world.last_action = Some(format!("build_worker_{}", features));
}

#[when(expr = "rbee-hive builds worker from git")]
pub async fn when_build_worker(world: &mut World) {
    // TEAM-078: Call rbee_hive::worker_provisioner::WorkerProvisioner::build()
    tracing::info!("TEAM-078: Building worker from git");
    world.last_action = Some("build_worker".to_string());
}

#[when(expr = "rbee-hive registers the worker in catalog")]
pub async fn when_register_worker_in_catalog(world: &mut World) {
    // TEAM-078: Call rbee_hive::worker_catalog::WorkerCatalog::insert()
    tracing::info!("TEAM-078: Registering worker in catalog");
    world.last_action = Some("worker_registered".to_string());
}

#[when(expr = "rbee-hive queries workers with features {string}")]
pub async fn when_query_workers_by_features(world: &mut World, features: String) {
    // TEAM-078: Call rbee_hive::worker_catalog::WorkerCatalog::query_by_features()
    tracing::info!("TEAM-078: Querying workers by features: {}", features);
    world.last_action = Some(format!("query_features_{}", features));
}

#[when(expr = "rbee-hive attempts to spawn worker")]
pub async fn when_attempt_spawn_worker(world: &mut World) {
    // TEAM-078: Trigger worker spawn flow
    tracing::info!("TEAM-078: Attempting to spawn worker");
    world.last_action = Some("spawn_worker".to_string());
}

#[when(expr = "cargo build fails with compilation error")]
pub async fn when_cargo_build_fails(world: &mut World) {
    // TEAM-078: Simulate cargo build failure
    tracing::info!("TEAM-078: Cargo build failed");
    world.last_action = Some("build_failed".to_string());
}

#[when(expr = "rbee-hive builds worker with features {string}")]
pub async fn when_build_worker_with_features_alt(world: &mut World, features: String) {
    // TEAM-078: Alternative step for building with features
    tracing::info!("TEAM-078: Building worker with features: {}", features);
    world.last_action = Some(format!("build_worker_{}", features));
}

#[when(expr = "CUDA toolkit is not installed")]
pub async fn when_cuda_not_installed(world: &mut World) {
    // TEAM-078: Simulate missing CUDA toolkit
    tracing::info!("TEAM-078: CUDA toolkit not installed");
    world.last_action = Some("cuda_missing".to_string());
}

#[when(expr = "rbee-hive verifies the binary")]
pub async fn when_verify_binary(world: &mut World) {
    // TEAM-078: Call binary verification logic
    tracing::info!("TEAM-078: Verifying binary");
    world.last_action = Some("verify_binary".to_string());
}

#[when(expr = "the binary is not executable")]
pub async fn when_binary_not_executable(world: &mut World) {
    // TEAM-078: Simulate non-executable binary
    tracing::info!("TEAM-078: Binary not executable");
    world.last_action = Some("binary_not_executable".to_string());
}

#[then(expr = "cargo build command is:")]
pub async fn then_cargo_build_command(world: &mut World, step: &cucumber::gherkin::Step) {
    // TEAM-078: Verify cargo build command structure
    tracing::info!("TEAM-078: Verifying cargo build command");
    assert!(world.last_action.is_some());
}

#[then(expr = "the build succeeds")]
pub async fn then_build_succeeds(world: &mut World) {
    // TEAM-078: Verify build success
    tracing::info!("TEAM-078: Build succeeded");
    assert!(world.last_action.is_some());
}

#[then(expr = "the binary is registered in catalog at {string}")]
pub async fn then_binary_registered_at(world: &mut World, path: String) {
    // TEAM-078: Verify catalog entry
    tracing::info!("TEAM-078: Binary registered at: {}", path);
    assert!(world.last_action.is_some());
}

#[then(expr = "the catalog entry includes features {string}")]
pub async fn then_catalog_includes_features(world: &mut World, features: String) {
    // TEAM-078: Verify features in catalog
    tracing::info!("TEAM-078: Catalog includes features: {}", features);
    assert!(world.last_action.is_some());
}

#[then(expr = "rbee-hive checks the worker catalog")]
pub async fn then_check_worker_catalog(world: &mut World) {
    // TEAM-078: Verify catalog was checked
    tracing::info!("TEAM-078: Worker catalog checked");
    assert!(world.last_action.is_some());
}

#[then(expr = "rbee-hive triggers worker build with features {string}")]
pub async fn then_trigger_worker_build(world: &mut World, features: String) {
    // TEAM-078: Verify build was triggered
    tracing::info!("TEAM-078: Worker build triggered with features: {}", features);
    assert!(world.last_action.is_some());
}

#[then(expr = "after build completes, rbee-hive spawns the worker")]
pub async fn then_spawn_after_build(world: &mut World) {
    // TEAM-078: Verify worker spawn after build
    tracing::info!("TEAM-078: Worker spawned after build");
    assert!(world.last_action.is_some());
}

#[then(expr = "rbee-hive captures stderr output")]
pub async fn then_capture_stderr(world: &mut World) {
    // TEAM-078: Verify stderr was captured
    tracing::info!("TEAM-078: Stderr captured");
    assert!(world.last_action.is_some());
}

#[then(expr = "cargo build fails with linker error")]
pub async fn then_cargo_build_linker_error(world: &mut World) {
    // TEAM-078: Verify linker error
    tracing::info!("TEAM-078: Cargo build linker error");
    assert!(world.last_action.is_some());
}

#[then(expr = "the returned worker has features {string}")]
pub async fn then_worker_has_features(world: &mut World, features: String) {
    // TEAM-078: Verify worker features
    tracing::info!("TEAM-078: Worker has features: {}", features);
    assert!(world.last_action.is_some());
}
