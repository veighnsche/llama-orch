//! Generate proof bundle for proof-bundle crate
//!
//! This uses the capture_tests() feature to generate a comprehensive proof bundle
//! for the proof-bundle crate itself, demonstrating the feature in action.

use proof_bundle::{ProofBundle, TestType};
use std::path::PathBuf;

#[test]
#[ignore] // Run explicitly: cargo +nightly test -p proof-bundle generate_proof_bundle -- --ignored --nocapture
fn generate_proof_bundle() -> anyhow::Result<()> {
    // Set proof bundle to crate's own directory
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proof_bundle_dir = crate_root.join(".proof_bundle");
    std::env::set_var("LLORCH_PROOF_DIR", proof_bundle_dir.to_str().unwrap());
    
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    println!("\n📦 Generating proof bundle for proof-bundle crate...");
    println!("   Using capture_tests() to dogfood our own feature!\n");
    
    // Capture ALL tests automatically
    let summary = pb.capture_tests("proof-bundle")
        .lib()      // Unit tests
        .tests()    // Integration tests
        .run()?;
    
    println!("✅ Proof bundle generated!");
    println!("   Total tests: {}", summary.total);
    println!("   Passed: {} ({:.1}%)", summary.passed, summary.pass_rate);
    println!("   Failed: {}", summary.failed);
    println!("   Ignored: {}", summary.ignored);
    println!("   Duration: {:.2}s", summary.duration_secs);
    println!("   Location: test-harness/proof-bundle/.proof_bundle/unit/<timestamp>/\n");
    
    // Verify files were created
    let root = pb.root();
    assert!(root.join("test_results.ndjson").exists(), "test_results.ndjson should exist");
    assert!(root.join("summary.json").exists(), "summary.json should exist");
    assert!(root.join("test_report.md").exists(), "test_report.md should exist");
    
    println!("📄 Generated files:");
    println!("   - test_results.ndjson ({} tests)", summary.total);
    println!("   - summary.json");
    println!("   - test_report.md\n");
    
    // Show some test details
    if summary.failed > 0 {
        println!("❌ Failed tests:");
        for test in &summary.tests {
            if test.status == proof_bundle::TestStatus::Failed {
                println!("   - {}", test.name);
                if let Some(ref err) = test.error_message {
                    println!("     Error: {}", err);
                }
            }
        }
        println!();
    }
    
    println!("🎉 Proof bundle demonstrates:");
    println!("   ✅ capture_tests() works on itself (dogfooding)");
    println!("   ✅ Captures all tests automatically");
    println!("   ✅ Generates human-readable reports");
    println!("   ✅ Includes timing data");
    println!("   ✅ Captures failures (if any)");
    println!("   ✅ Creates crate-local proof bundles\n");
    
    Ok(())
}
