//! Hash verification step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};

#[given("a GGUF model file with hash {string}")]
async fn given_gguf_with_hash(world: &mut BddWorld, _hash: String) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test-model.gguf");

    // Create valid GGUF file
    let gguf = BddWorld::create_valid_gguf();
    std::fs::write(&model_path, &gguf).unwrap();

    // Compute actual hash
    let actual_hash = BddWorld::compute_hash(&gguf);

    world.temp_dir = Some(temp_dir);
    world.model_path = Some(model_path);
    world.expected_hash = Some(actual_hash);
}

#[when("I load the model with hash verification")]
async fn when_load_with_hash(world: &mut BddWorld) {
    let model_path = world.model_path.as_ref().expect("No model path set");
    let expected_hash = world.expected_hash.as_ref().expect("No hash set");

    let request = LoadRequest::new(model_path)
        .with_hash(expected_hash)
        .with_max_size(100_000_000_000);

    world.load_result = Some(world.loader.load_and_validate(request));
}

#[when("I load the model with wrong hash")]
async fn when_load_with_wrong_hash(world: &mut BddWorld) {
    let model_path = world.model_path.as_ref().expect("No model path set");

    let wrong_hash = "0".repeat(64); // Wrong hash

    let request = LoadRequest::new(model_path)
        .with_hash(&wrong_hash)
        .with_max_size(100_000_000_000);

    world.load_result = Some(world.loader.load_and_validate(request));
}

#[then("the load fails with hash mismatch")]
async fn then_fails_hash_mismatch(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::HashMismatch { .. }) => {
            // Expected
        }
        other => panic!("Expected HashMismatch, got: {:?}", other),
    }
}
