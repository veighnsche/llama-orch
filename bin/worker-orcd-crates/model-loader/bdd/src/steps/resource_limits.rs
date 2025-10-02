//! Resource limit step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};

#[given("a model file that is too large")]
async fn given_oversized_file(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large-model.gguf");

    // Create file larger than typical max_size for tests
    let large_data = vec![0u8; 1_000_000]; // 1MB
    std::fs::write(&file_path, &large_data).unwrap();
    
    // Set up loader with allowed root
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());

    world.temp_dir = Some(temp_dir);
    // Store the full path
    world.model_path = Some(file_path);
}

#[when(regex = r"I load the model with max size (\d+) bytes")]
async fn when_load_with_max_size(world: &mut BddWorld, max_size: usize) {
    let model_path = world.model_path.as_ref().expect("No model path set");

    let request = LoadRequest::new(model_path).with_max_size(max_size);

    world.load_result = Some(world.loader.load_and_validate(request));
}

#[then("the load fails with file too large")]
async fn then_fails_too_large(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::TooLarge { .. }) => {
            // Expected
        }
        other => panic!("Expected TooLarge, got: {:?}", other),
    }
}

#[given(regex = r"a GGUF file with (\d+) tensors")]
async fn given_gguf_with_tensor_count(world: &mut BddWorld, count: u64) {
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    bytes.extend_from_slice(&count.to_le_bytes()); // Tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    
    world.model_bytes = Some(bytes);
}

#[given("a GGUF file with oversized string")]
async fn given_gguf_with_oversized_string(world: &mut BddWorld) {
    // Create GGUF with string length > MAX_STRING_LEN
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    
    // Try to add a 20MB string (exceeds 10MB limit)
    let huge_len = 20_000_000u64;
    bytes.extend_from_slice(&huge_len.to_le_bytes());
    
    world.model_bytes = Some(bytes);
}

#[given(regex = r"a GGUF file with (\d+) metadata pairs")]
async fn given_gguf_with_metadata_pairs(world: &mut BddWorld, count: u64) {
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
    bytes.extend_from_slice(&count.to_le_bytes()); // Metadata count
    
    world.model_bytes = Some(bytes);
}

#[then("the validation fails with tensor count exceeded")]
async fn then_fails_tensor_count(world: &mut BddWorld) {
    let result = world.validation_result.as_ref().expect("No validation result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::TensorCountExceeded { .. }) => {
            // Expected
        }
        other => panic!("Expected TensorCountExceeded, got: {:?}", other),
    }
}

#[then("the validation fails with string too long")]
async fn then_fails_string_too_long(world: &mut BddWorld) {
    let result = world.validation_result.as_ref().expect("No validation result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::StringTooLong { .. }) => {
            // Expected
        }
        other => panic!("Expected StringTooLong, got: {:?}", other),
    }
}
