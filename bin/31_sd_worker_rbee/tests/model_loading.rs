// TEAM-487: Model loading verification tests
//
// Tests that all 7 SD model variants can be loaded successfully

use sd_worker_rbee::backend::model_loader::load_model;
use sd_worker_rbee::backend::models::SDVersion;
use sd_worker_rbee::backend::traits::ImageModel;
#[cfg(feature = "cpu")]
use shared_worker_rbee::device::init_cpu_device;

mod fixtures;
use fixtures::models::{get_model_path, TEST_MODELS};

/// Test that all model variants can be loaded
///
/// TEAM-487: Smoke test for model loading
#[test]
#[cfg(feature = "cpu")]
fn test_all_models_load() {
    let mut loaded_count = 0;
    let mut skipped_count = 0;

    println!("\n🧪 Testing Model Loading");
    println!("========================\n");

    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Model {:?} not found, skipping", fixture.version);
                eprintln!(
                    "    Set SD_MODEL_{:?} or download to ~/.cache/rbee/models/{}",
                    format!("{:?}", fixture.version).to_uppercase(),
                    fixture.repo
                );
                skipped_count += 1;
                continue;
            }
        };

        println!("📦 Testing model: {:?}", fixture.version);
        println!("   Path: {}", model_path.display());

        // Initialize device (CPU for testing)
        let device = init_cpu_device().expect("Failed to initialize device");

        // Load model (model path is determined by version)
        let result = load_model(
            fixture.version,
            &device,
            false, // use_f16
            &[],   // TEAM-487: No LoRAs
            true,  // TEAM-483: Quantized
        );

        match result {
            Ok(model) => {
                println!("   ✅ Loaded successfully");

                // Verify model capabilities match expected fixture
                let capabilities = model.capabilities();
                assert_eq!(
                    capabilities.default_size, fixture.expected_size,
                    "Model {:?} has wrong default size",
                    fixture.version
                );

                println!(
                    "   ✓ Default size: {}x{}",
                    capabilities.default_size.0, capabilities.default_size.1
                );
                println!("   ✓ Is inpainting: {}", capabilities.inpainting);
                println!();

                loaded_count += 1;
            }
            Err(e) => {
                panic!("❌ Failed to load model {:?}: {}", fixture.version, e);
            }
        }
    }

    println!("📊 Summary:");
    println!("   Loaded: {}", loaded_count);
    println!("   Skipped: {}", skipped_count);
    println!("   Total: {}", TEST_MODELS.len());

    if loaded_count == 0 {
        panic!("❌ No models were loaded! Please download models first.");
    }
}

/// Test model loading with F16 precision (CUDA only)
///
/// TEAM-487: Verify F16 works on CUDA
#[test]
#[cfg(feature = "cuda")]
fn test_models_load_f16() {
    use shared_worker_rbee::device::init_cuda_device;

    println!("\n🧪 Testing F16 Model Loading (CUDA)");
    println!("====================================\n");

    let mut loaded_count = 0;

    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Model {:?} not found, skipping", fixture.version);
                continue;
            }
        };

        println!("📦 Testing {:?} with F16", fixture.version);

        let device = init_cuda_device(0).expect("Failed to initialize CUDA device");

        let result = load_model(
            fixture.version,
            &device,
            true,  // use_f16
            &[],   // TEAM-487: No LoRAs
            false, // Not quantized in this test
        );

        assert!(
            result.is_ok(),
            "Failed to load {:?} with F16: {:?}",
            fixture.version,
            result.err()
        );

        println!("   ✅ Loaded with F16 successfully\n");
        loaded_count += 1;
    }

    println!("📊 Loaded {} models with F16", loaded_count);

    if loaded_count == 0 {
        panic!("❌ No models were loaded with F16!");
    }
}

/// Test that model configs are correct
///
/// TEAM-487: Verify SDVersion enum configurations
#[test]
fn test_model_configs() {
    println!("\n🧪 Testing Model Configurations");
    println!("================================\n");

    // Test V1.5
    assert_eq!(SDVersion::V1_5.default_size(), (512, 512));
    assert_eq!(SDVersion::V1_5.default_steps(), 20);
    assert!(!SDVersion::V1_5.is_inpainting());
    println!("✅ V1.5 config correct");

    // Test V1.5 Inpaint
    assert_eq!(SDVersion::V1_5Inpaint.default_size(), (512, 512));
    assert!(SDVersion::V1_5Inpaint.is_inpainting());
    println!("✅ V1.5Inpaint config correct");

    // Test V2.1
    assert_eq!(SDVersion::V2_1.default_size(), (768, 768));
    assert_eq!(SDVersion::V2_1.default_steps(), 20);
    assert!(!SDVersion::V2_1.is_inpainting());
    println!("✅ V2.1 config correct");

    // Test V2 Inpaint
    assert_eq!(SDVersion::V2Inpaint.default_size(), (768, 768));
    assert!(SDVersion::V2Inpaint.is_inpainting());
    println!("✅ V2Inpaint config correct");

    // Test XL
    assert_eq!(SDVersion::XL.default_size(), (1024, 1024));
    assert_eq!(SDVersion::XL.default_steps(), 20);
    assert!(!SDVersion::XL.is_inpainting());
    println!("✅ XL config correct");

    // Test XL Inpaint
    assert_eq!(SDVersion::XLInpaint.default_size(), (1024, 1024));
    assert!(SDVersion::XLInpaint.is_inpainting());
    println!("✅ XLInpaint config correct");

    // Test Turbo
    assert_eq!(SDVersion::Turbo.default_size(), (1024, 1024));
    assert_eq!(SDVersion::Turbo.default_steps(), 4);
    assert!(!SDVersion::Turbo.is_inpainting());
    println!("✅ Turbo config correct");

    println!("\n📊 All 7 model configs verified");
}
