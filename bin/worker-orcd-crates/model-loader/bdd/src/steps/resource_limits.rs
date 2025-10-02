//! Resource limit step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};

#[given("a model file that is too large")]
async fn given_oversized_file(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let model_path = temp_dir.path().join("large-model.gguf");

    // Create file larger than typical max_size for tests
    let large_data = vec![0u8; 1_000_000]; // 1MB
    std::fs::write(&model_path, &large_data).unwrap();

    world.temp_dir = Some(temp_dir);
    world.model_path = Some(model_path);
}

#[when("I load the model with max size {int} bytes")]
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
