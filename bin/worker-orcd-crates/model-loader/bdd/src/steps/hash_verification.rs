//! Hash verification step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};

#[given(regex = r#"a GGUF model file with hash "(.*)""#)]
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
    
    // Set up loader with allowed root
    if world.temp_dir.is_some() {
        let temp_path = world.temp_dir.as_ref().unwrap().path().to_path_buf();
        world.loader = model_loader::ModelLoader::with_allowed_root(temp_path);
    }

    let request = LoadRequest::new(model_path)
        .with_hash(expected_hash)
        .with_max_size(100_000_000_000);

    world.load_result = Some(world.loader.load_and_validate(request));
}

#[when("I load the model with wrong hash")]
async fn when_load_with_wrong_hash(world: &mut BddWorld) {
    let model_path = world.model_path.as_ref().expect("No model path set");

    let wrong_hash = "0".repeat(64); // Wrong hash
    
    // Set up loader with allowed root
    if world.temp_dir.is_some() {
        let temp_path = world.temp_dir.as_ref().unwrap().path().to_path_buf();
        world.loader = model_loader::ModelLoader::with_allowed_root(temp_path);
    }

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

#[when(regex = r#"I validate with hash "(.*)""#)]
async fn when_validate_with_hash(world: &mut BddWorld, hash: String) {
    let bytes = world.model_bytes.as_ref().expect("No bytes set");
    world.validation_result = Some(world.loader.validate_bytes(bytes, Some(&hash)));
}

#[when("I validate with computed hash")]
async fn when_validate_with_computed_hash(world: &mut BddWorld) {
    let bytes = world.model_bytes.as_ref().expect("No bytes set");
    let hash = BddWorld::compute_hash(bytes);
    world.validation_result = Some(world.loader.validate_bytes(bytes, Some(&hash)));
}

#[then("the validation fails with invalid format")]
async fn then_validation_fails_invalid_format(world: &mut BddWorld) {
    let result = world.validation_result.as_ref().expect("No validation result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::InvalidFormat(_)) => {
            // Expected
        }
        other => panic!("Expected InvalidFormat, got: {:?}", other),
    }
}

#[then("the validation succeeds")]
async fn then_validation_succeeds(world: &mut BddWorld) {
    let result = world.validation_result.as_ref().expect("No validation result");
    assert!(result.is_ok(), "Expected success, got error: {:?}", result);
}
