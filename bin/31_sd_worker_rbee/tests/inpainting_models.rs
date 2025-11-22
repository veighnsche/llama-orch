// TEAM-487: Inpainting model verification tests
//
// Tests that inpainting models are correctly identified and work

use sd_worker_rbee::backend::models::SDVersion;

mod fixtures;
use fixtures::models::get_model_path;

/// Test that inpainting models are correctly identified
///
/// TEAM-487: Verify is_inpainting() method
#[test]
fn test_inpainting_model_detection() {
    println!("\n🖌️  Testing Inpainting Model Detection");
    println!("======================================\n");

    // Inpainting models
    assert!(SDVersion::V1_5Inpaint.is_inpainting());
    println!("✅ V1_5Inpaint correctly identified as inpainting");

    assert!(SDVersion::V2Inpaint.is_inpainting());
    println!("✅ V2Inpaint correctly identified as inpainting");

    assert!(SDVersion::XLInpaint.is_inpainting());
    println!("✅ XLInpaint correctly identified as inpainting");

    // Non-inpainting models
    assert!(!SDVersion::V1_5.is_inpainting());
    println!("✅ V1_5 correctly identified as NOT inpainting");

    assert!(!SDVersion::V2_1.is_inpainting());
    println!("✅ V2_1 correctly identified as NOT inpainting");

    assert!(!SDVersion::XL.is_inpainting());
    println!("✅ XL correctly identified as NOT inpainting");

    assert!(!SDVersion::Turbo.is_inpainting());
    println!("✅ Turbo correctly identified as NOT inpainting");

    println!("\n📊 All 7 models correctly classified");
}

/// Test that inpainting models load correctly
///
/// TEAM-487: Verify inpainting models can be loaded
/// Run with: cargo test --test inpainting_models test_inpainting_models_load -- --ignored --nocapture
#[test]
#[cfg(feature = "cpu")]
#[ignore]
fn test_inpainting_models_load() {
    println!("\n🖌️  Testing Inpainting Model Loading");
    println!("====================================\n");

    let inpainting_models =
        vec![SDVersion::V1_5Inpaint, SDVersion::V2Inpaint, SDVersion::XLInpaint];

    let mut loaded_count = 0;

    for version in inpainting_models {
        let model_path = match get_model_path(version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Inpainting model {:?} not found, skipping", version);
                continue;
            }
        };

        println!("📦 Testing inpainting model: {:?}", version);
        println!("   Path: {}", model_path.display());

        let device = shared_worker_rbee::device::init_cpu_device().unwrap();
        let model = sd_worker_rbee::backend::model_loader::load_model(
            version,
            &device,
            false,
            &[],   // TEAM-487: No LoRAs
            false, // TEAM-483: Not quantized
        )
        .unwrap();

        // Verify capabilities match expected inpainting behavior
        let capabilities = model.capabilities();
        assert!(capabilities.inpainting, "Model {:?} should be inpainting", version);
        assert_eq!(
            capabilities.default_size,
            version.default_size(),
            "Model {:?} default size mismatch",
            version
        );
        println!("   ✅ Loaded and verified as inpainting model");
        println!("   ✓ Default size: {:?}", capabilities.default_size);
        println!();

        loaded_count += 1;
    }

    println!("📊 Loaded {} inpainting models", loaded_count);

    if loaded_count == 0 {
        panic!("❌ No inpainting models were loaded!");
    }
}

// TEAM_527: Removed legacy inpainting generation test that depended on old backend::generation API.

/// Test XL vs non-XL model detection
///
/// TEAM-487: Verify is_xl() method
#[test]
fn test_xl_model_detection() {
    println!("\n🔍 Testing XL Model Detection");
    println!("==============================\n");

    // XL models
    assert!(SDVersion::XL.is_xl());
    println!("✅ XL correctly identified as XL");

    assert!(SDVersion::XLInpaint.is_xl());
    println!("✅ XLInpaint correctly identified as XL");

    assert!(SDVersion::Turbo.is_xl());
    println!("✅ Turbo correctly identified as XL");

    // Non-XL models
    assert!(!SDVersion::V1_5.is_xl());
    println!("✅ V1_5 correctly identified as NOT XL");

    assert!(!SDVersion::V1_5Inpaint.is_xl());
    println!("✅ V1_5Inpaint correctly identified as NOT XL");

    assert!(!SDVersion::V2_1.is_xl());
    println!("✅ V2_1 correctly identified as NOT XL");

    assert!(!SDVersion::V2Inpaint.is_xl());
    println!("✅ V2Inpaint correctly identified as NOT XL");

    println!("\n📊 All XL classifications correct");
}
