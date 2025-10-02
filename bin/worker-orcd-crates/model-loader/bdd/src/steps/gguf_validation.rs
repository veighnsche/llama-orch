//! GGUF format validation step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};

#[given("a valid GGUF model file")]
async fn given_valid_gguf_file(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    
    // Create valid GGUF file
    let gguf = BddWorld::create_valid_gguf();
    let file_path = temp_dir.path().join("test-model.gguf");
    std::fs::write(&file_path, &gguf).unwrap();
    
    // Set up loader with allowed root FIRST
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Store the full path (it will be validated against allowed_root)
    world.model_path = Some(file_path);
    world.temp_dir = Some(temp_dir);
}

#[given("a model file with invalid magic number")]
async fn given_invalid_magic(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    
    // Invalid magic number
    let invalid = BddWorld::create_invalid_gguf();
    let file_path = temp_dir.path().join("invalid-model.gguf");
    std::fs::write(&file_path, &invalid).unwrap();
    
    // Set up loader with allowed root FIRST
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Store the full path
    world.model_path = Some(file_path);
    world.temp_dir = Some(temp_dir);
}

#[given("valid GGUF bytes in memory")]
async fn given_valid_bytes_in_memory(world: &mut BddWorld) {
    world.model_bytes = Some(BddWorld::create_valid_gguf());
}

#[given("invalid GGUF bytes in memory")]
async fn given_invalid_bytes_in_memory(world: &mut BddWorld) {
    world.model_bytes = Some(BddWorld::create_invalid_gguf());
}

#[when("I load and validate the model")]
async fn when_load_and_validate(world: &mut BddWorld) {
    let model_path = world.model_path.as_ref().expect("No model path set");
    
    // Loader should already be set up with allowed_root in the Given step
    let request = LoadRequest::new(model_path).with_max_size(100_000_000_000);

    world.load_result = Some(world.loader.load_and_validate(request));
}

#[when("I validate the bytes in memory")]
async fn when_validate_bytes(world: &mut BddWorld) {
    let bytes = world.model_bytes.as_ref().expect("No bytes set");
    world.validation_result = Some(world.loader.validate_bytes(bytes, None));
}

#[then("the model loads successfully")]
async fn then_loads_successfully(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    assert!(result.is_ok(), "Expected success, got error: {:?}", result);
}

#[then("the load fails with invalid format")]
async fn then_fails_invalid_format(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::InvalidFormat(_)) => {
            // Expected
        }
        other => panic!("Expected InvalidFormat, got: {:?}", other),
    }
}

// Note: Removed duplicate step definitions - using ones from hash_verification.rs

#[then("the loaded bytes match the file contents")]
async fn then_bytes_match_file(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    let loaded_bytes = result.as_ref().expect("Load failed");

    let model_path = world.model_path.as_ref().expect("No model path");
    let file_bytes = std::fs::read(model_path).expect("Failed to read file");

    assert_eq!(loaded_bytes, &file_bytes, "Loaded bytes don't match file");
}
