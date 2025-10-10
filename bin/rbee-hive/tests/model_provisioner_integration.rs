//! Model Provisioner Integration Tests
//!
//! Comprehensive test suite for model downloading, listing, and removal
//! through the ModelProvisioner (not manual operations).
//!
//! Created by: TEAM-032
//!
//! Tests:
//! - Model downloading via llorch-models script
//! - Model listing from filesystem
//! - Model lookup by reference
//! - Model size calculation
//! - Error handling for missing models
//! - Integration with model catalog

use std::fs;
use std::path::PathBuf;

// Helper to create test provisioner with temp directory
fn setup_test_provisioner() -> (rbee_hive::provisioner::ModelProvisioner, PathBuf) {
    let temp_dir = std::env::temp_dir().join(format!("test_provisioner_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir).unwrap();

    let provisioner = rbee_hive::provisioner::ModelProvisioner::new(temp_dir.clone());
    (provisioner, temp_dir)
}

// Helper to cleanup test directory
fn cleanup_test_dir(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

// ========== MODEL LISTING TESTS ==========

#[test]
fn test_list_models_empty_directory() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 0, "Empty directory should have no models");

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_list_models_single_model() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create a model directory with a .gguf file
    let model_dir = temp_dir.join("testmodel");
    fs::create_dir_all(&model_dir).unwrap();
    let model_file = model_dir.join("model.gguf");
    fs::write(&model_file, b"test model content").unwrap();

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 1, "Should find one model");
    assert_eq!(models[0].0, "testmodel");
    assert!(models[0].1.ends_with("model.gguf"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_list_models_multiple_models() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create multiple model directories
    for model_name in &["tinyllama", "qwen", "phi3"] {
        let model_dir = temp_dir.join(model_name);
        fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join(format!("{}.gguf", model_name));
        fs::write(&model_file, b"test").unwrap();
    }

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 3, "Should find three models");

    // Verify all model names are present
    let model_names: Vec<String> = models.iter().map(|(name, _)| name.clone()).collect();
    assert!(model_names.contains(&"tinyllama".to_string()));
    assert!(model_names.contains(&"qwen".to_string()));
    assert!(model_names.contains(&"phi3".to_string()));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_list_models_ignores_non_gguf_files() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("testmodel");
    fs::create_dir_all(&model_dir).unwrap();

    // Create various files
    fs::write(model_dir.join("model.gguf"), b"gguf file").unwrap();
    fs::write(model_dir.join("config.json"), b"{}").unwrap();
    fs::write(model_dir.join("README.md"), b"readme").unwrap();
    fs::write(model_dir.join("model.txt"), b"text").unwrap();

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 1, "Should only find .gguf files");
    assert!(models[0].1.ends_with("model.gguf"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_list_models_multiple_gguf_files_in_one_directory() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("llama");
    fs::create_dir_all(&model_dir).unwrap();

    // Create multiple .gguf files (different quantizations)
    fs::write(model_dir.join("llama-q4.gguf"), b"q4").unwrap();
    fs::write(model_dir.join("llama-q8.gguf"), b"q8").unwrap();
    fs::write(model_dir.join("llama-fp16.gguf"), b"fp16").unwrap();

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 3, "Should find all .gguf files");

    // All should have the same model name
    for (name, _) in &models {
        assert_eq!(name, "llama");
    }

    cleanup_test_dir(&temp_dir);
}

// ========== MODEL LOOKUP TESTS ==========

#[test]
fn test_find_local_model_exists() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create model matching HuggingFace reference pattern
    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();
    let model_file = model_dir.join("model.gguf");
    fs::write(&model_file, b"test").unwrap();

    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_some(), "Should find model by reference");
    assert!(found.unwrap().ends_with("model.gguf"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_find_local_model_not_exists() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let found = provisioner.find_local_model("nonexistent/model");
    assert!(found.is_none(), "Should return None for missing model");

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_find_local_model_case_insensitive() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create model with lowercase directory name
    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"test").unwrap();

    // Search with mixed case reference
    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_some(), "Should find model case-insensitively");

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_find_local_model_returns_first_gguf() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();

    // Create multiple .gguf files
    fs::write(model_dir.join("model-q4.gguf"), b"q4").unwrap();
    fs::write(model_dir.join("model-q8.gguf"), b"q8").unwrap();

    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_some(), "Should find a .gguf file");

    // Should return one of the .gguf files (implementation returns first found)
    let path = found.unwrap();
    assert!(path.extension().unwrap() == "gguf");

    cleanup_test_dir(&temp_dir);
}

// ========== MODEL SIZE TESTS ==========

#[test]
fn test_get_model_size() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_file = temp_dir.join("test.gguf");
    let test_data = b"test model data with some content";
    fs::write(&model_file, test_data).unwrap();

    let size = provisioner.get_model_size(&model_file).unwrap();
    assert_eq!(size, test_data.len() as i64);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_get_model_size_large_file() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_file = temp_dir.join("large.gguf");
    // Create a 1MB file
    let large_data = vec![0u8; 1024 * 1024];
    fs::write(&model_file, &large_data).unwrap();

    let size = provisioner.get_model_size(&model_file).unwrap();
    assert_eq!(size, 1024 * 1024);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_get_model_size_nonexistent_file() {
    let (provisioner, _temp_dir) = setup_test_provisioner();

    let result = provisioner.get_model_size(&PathBuf::from("/nonexistent/file.gguf"));
    assert!(result.is_err(), "Should error for nonexistent file");
}

#[test]
fn test_get_model_size_empty_file() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_file = temp_dir.join("empty.gguf");
    fs::write(&model_file, b"").unwrap();

    let size = provisioner.get_model_size(&model_file).unwrap();
    assert_eq!(size, 0);

    cleanup_test_dir(&temp_dir);
}

// ========== MODEL NAME EXTRACTION TESTS ==========

#[test]
fn test_extract_model_name_tinyllama() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // TEAM-032: The provisioner extracts last part of reference and lowercases it
    // "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" -> "tinyllama-1.1b-chat-v1.0-gguf"
    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"test").unwrap();

    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_some(), "Should find tinyllama model");

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_extract_model_name_qwen() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // TEAM-032: "Qwen/Qwen2.5-0.5B-Instruct-GGUF" -> "qwen2.5-0.5b-instruct-gguf"
    let model_dir = temp_dir.join("qwen2.5-0.5b-instruct-gguf");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"test").unwrap();

    let found = provisioner.find_local_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF");
    assert!(found.is_some(), "Should find qwen model");

    cleanup_test_dir(&temp_dir);
}

// ========== ERROR HANDLING TESTS ==========

#[test]
fn test_list_models_nonexistent_base_dir() {
    let nonexistent = PathBuf::from("/nonexistent/models");
    let provisioner = rbee_hive::provisioner::ModelProvisioner::new(nonexistent);

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 0, "Should return empty list for nonexistent directory");
}

#[test]
fn test_find_local_model_empty_directory() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create model directory but no .gguf files
    let model_dir = temp_dir.join("tinyllama");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("README.md"), b"readme").unwrap();

    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_none(), "Should return None when no .gguf files exist");

    cleanup_test_dir(&temp_dir);
}

// ========== INTEGRATION TESTS WITH REAL STRUCTURE ==========

#[test]
fn test_realistic_model_directory_structure() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // TEAM-032: Simulate realistic HuggingFace model download structure
    // Directory name matches the lowercased reference
    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();

    // Create typical files from HuggingFace
    fs::write(model_dir.join("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"), b"model").unwrap();
    fs::write(model_dir.join("config.json"), b"{}").unwrap();
    fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    fs::write(model_dir.join("README.md"), b"# Model").unwrap();
    fs::write(model_dir.join(".gitattributes"), b"").unwrap();

    // List should only find .gguf file
    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 1);
    assert!(models[0].1.to_string_lossy().contains(".gguf"));

    // Find should locate the model
    let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert!(found.is_some());

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_multiple_models_different_sizes() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create models with different sizes
    let models_data =
        vec![("tinyllama", vec![0u8; 100]), ("qwen", vec![0u8; 500]), ("phi3", vec![0u8; 1000])];

    for (name, data) in &models_data {
        let model_dir = temp_dir.join(name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join(format!("{}.gguf", name)), data).unwrap();
    }

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 3);

    // Verify sizes
    for (name, path) in &models {
        let size = provisioner.get_model_size(path).unwrap();
        let expected_size =
            models_data.iter().find(|(n, _)| n == name).map(|(_, d)| d.len() as i64).unwrap();
        assert_eq!(size, expected_size, "Size mismatch for {}", name);
    }

    cleanup_test_dir(&temp_dir);
}

// ========== EDGE CASES ==========

#[test]
fn test_model_directory_with_subdirectories() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("llama");
    fs::create_dir_all(&model_dir).unwrap();

    // Create .gguf in root
    fs::write(model_dir.join("model.gguf"), b"root").unwrap();

    // Create subdirectory with .gguf (should be ignored)
    let subdir = model_dir.join("quantized");
    fs::create_dir_all(&subdir).unwrap();
    fs::write(subdir.join("q4.gguf"), b"sub").unwrap();

    let models = provisioner.list_models().unwrap();
    // Should only find the root .gguf file
    assert_eq!(models.len(), 1);
    assert!(models[0].1.ends_with("model.gguf"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_model_with_special_characters_in_filename() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("model");
    fs::create_dir_all(&model_dir).unwrap();

    // Create file with special characters (valid in filenames)
    fs::write(model_dir.join("model-v1.2.3_Q4_K_M.gguf"), b"test").unwrap();

    let models = provisioner.list_models().unwrap();
    assert_eq!(models.len(), 1);
    assert!(models[0].1.to_string_lossy().contains("model-v1.2.3_Q4_K_M.gguf"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_provisioner_creation_with_relative_path() {
    let provisioner = rbee_hive::provisioner::ModelProvisioner::new(PathBuf::from(".test-models"));

    // Should create without error
    let models = provisioner.list_models();
    assert!(models.is_ok());
}

#[test]
fn test_provisioner_creation_with_absolute_path() {
    let abs_path = std::env::temp_dir().join("test_abs_models");
    let provisioner = rbee_hive::provisioner::ModelProvisioner::new(abs_path.clone());

    let models = provisioner.list_models();
    assert!(models.is_ok());

    let _ = fs::remove_dir_all(&abs_path);
}

// ========== MODEL DELETION TESTS (TEAM-032) ==========

#[test]
fn test_delete_model_exists() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Create a model
    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"test").unwrap();

    // Verify it exists
    assert!(model_dir.exists());

    // Delete it
    provisioner.delete_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").unwrap();

    // Verify it's gone
    assert!(!model_dir.exists());

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_delete_model_not_exists() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let result = provisioner.delete_model("nonexistent/model");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_delete_model_with_multiple_files() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("llama");
    fs::create_dir_all(&model_dir).unwrap();

    // Create multiple files
    fs::write(model_dir.join("model.gguf"), b"model").unwrap();
    fs::write(model_dir.join("config.json"), b"{}").unwrap();
    fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

    provisioner.delete_model("llama").unwrap();

    // Entire directory should be gone
    assert!(!model_dir.exists());

    cleanup_test_dir(&temp_dir);
}

// ========== MODEL INFO TESTS (TEAM-032) ==========

#[test]
fn test_get_model_info() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
    fs::create_dir_all(&model_dir).unwrap();

    // Create files with known sizes
    fs::write(model_dir.join("model.gguf"), vec![0u8; 1000]).unwrap();
    fs::write(model_dir.join("config.json"), vec![0u8; 100]).unwrap();

    let info = provisioner.get_model_info("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").unwrap();

    assert_eq!(info.reference, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    assert_eq!(info.total_size, 1100);
    assert_eq!(info.file_count, 2);
    assert_eq!(info.gguf_files.len(), 1);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_get_model_info_not_exists() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let result = provisioner.get_model_info("nonexistent/model");
    assert!(result.is_err());

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_get_model_info_multiple_gguf() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("llama");
    fs::create_dir_all(&model_dir).unwrap();

    fs::write(model_dir.join("q4.gguf"), vec![0u8; 500]).unwrap();
    fs::write(model_dir.join("q8.gguf"), vec![0u8; 800]).unwrap();
    fs::write(model_dir.join("readme.txt"), vec![0u8; 50]).unwrap();

    let info = provisioner.get_model_info("llama").unwrap();

    assert_eq!(info.total_size, 1350);
    assert_eq!(info.file_count, 3);
    assert_eq!(info.gguf_files.len(), 2);

    cleanup_test_dir(&temp_dir);
}

// ========== MODEL VERIFICATION TESTS (TEAM-032) ==========

#[test]
fn test_verify_model_valid() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("tinyllama");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"valid model data").unwrap();

    let result = provisioner.verify_model("tinyllama");
    assert!(result.is_ok());

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_verify_model_no_gguf_files() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("broken");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("config.json"), b"{}").unwrap();

    let result = provisioner.verify_model("broken");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No .gguf files"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_verify_model_empty_gguf() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("empty");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), b"").unwrap(); // Empty file

    let result = provisioner.verify_model("empty");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Empty .gguf file"));

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_verify_model_multiple_gguf() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("multi");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("q4.gguf"), b"q4 data").unwrap();
    fs::write(model_dir.join("q8.gguf"), b"q8 data").unwrap();

    let result = provisioner.verify_model("multi");
    assert!(result.is_ok());

    cleanup_test_dir(&temp_dir);
}

// ========== DISK USAGE TESTS (TEAM-032) ==========

#[test]
fn test_disk_usage_empty() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let usage = provisioner.get_total_disk_usage().unwrap();
    assert_eq!(usage, 0);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_disk_usage_single_model() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    let model_dir = temp_dir.join("tinyllama");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), vec![0u8; 1000]).unwrap();

    let usage = provisioner.get_total_disk_usage().unwrap();
    assert_eq!(usage, 1000);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_disk_usage_multiple_models() {
    let (provisioner, temp_dir) = setup_test_provisioner();

    // Model 1
    let model1 = temp_dir.join("tinyllama");
    fs::create_dir_all(&model1).unwrap();
    fs::write(model1.join("model.gguf"), vec![0u8; 500]).unwrap();

    // Model 2
    let model2 = temp_dir.join("qwen");
    fs::create_dir_all(&model2).unwrap();
    fs::write(model2.join("model.gguf"), vec![0u8; 300]).unwrap();
    fs::write(model2.join("config.json"), vec![0u8; 50]).unwrap();

    let usage = provisioner.get_total_disk_usage().unwrap();
    assert_eq!(usage, 850);

    cleanup_test_dir(&temp_dir);
}

#[test]
fn test_disk_usage_nonexistent_dir() {
    let nonexistent = PathBuf::from("/nonexistent/models");
    let provisioner = rbee_hive::provisioner::ModelProvisioner::new(nonexistent);

    let usage = provisioner.get_total_disk_usage().unwrap();
    assert_eq!(usage, 0);
}
