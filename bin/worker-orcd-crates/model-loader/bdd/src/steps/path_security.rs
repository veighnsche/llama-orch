//! Path security step definitions

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use model_loader::{LoadError, LoadRequest};
use std::os::unix::fs as unix_fs;

#[given("a model file with path traversal sequence")]
async fn given_path_traversal(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Set malicious path with traversal
    world.model_path = Some(temp_dir.path().join("../../../etc/passwd"));
    world.temp_dir = Some(temp_dir);
}

#[given("a symlink pointing outside allowed directory")]
async fn given_symlink_escape(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let outside_dir = tempfile::TempDir::new().unwrap();
    
    // Create file outside allowed root
    let outside_file = outside_dir.path().join("secret.gguf");
    std::fs::write(&outside_file, b"secret").unwrap();
    
    // Create symlink inside allowed root pointing outside
    let symlink_path = temp_dir.path().join("escape.gguf");
    unix_fs::symlink(&outside_file, &symlink_path).unwrap();
    
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    world.model_path = Some(symlink_path);
    world.temp_dir = Some(temp_dir);
    world.metadata.insert("outside_dir".to_string(), format!("{:?}", outside_dir.path()));
}

#[given("a model path with null byte")]
async fn given_null_byte_path(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    
    // Path with null byte
    let path_str = format!("{}/model\0.gguf", temp_dir.path().display());
    world.model_path = Some(std::path::PathBuf::from(path_str));
    world.temp_dir = Some(temp_dir);
}

#[given("a valid model file in allowed directory")]
async fn given_valid_file_in_allowed_dir(world: &mut BddWorld) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.gguf");
    
    // Create valid GGUF file
    let gguf = BddWorld::create_valid_gguf();
    std::fs::write(&model_path, &gguf).unwrap();
    
    world.loader = model_loader::ModelLoader::with_allowed_root(temp_dir.path().to_path_buf());
    world.model_path = Some(model_path);
    world.temp_dir = Some(temp_dir);
}

#[when("I attempt to load the model")]
async fn when_attempt_load(world: &mut BddWorld) {
    let model_path = world.model_path.as_ref().expect("No model path set");
    let request = LoadRequest::new(model_path).with_max_size(100_000_000_000);
    world.load_result = Some(world.loader.load_and_validate(request));
}

#[then("the load fails with path validation error")]
async fn then_fails_path_validation(world: &mut BddWorld) {
    let result = world.load_result.as_ref().expect("No load result");
    assert!(result.is_err(), "Expected error, got success");

    match result {
        Err(LoadError::PathValidationFailed(_)) => {
            // Expected
        }
        other => panic!("Expected PathValidationFailed, got: {:?}", other),
    }
}
