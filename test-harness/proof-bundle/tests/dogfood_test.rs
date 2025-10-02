//! Dogfooding test - Generate proof-bundle's own proof bundle
//!
//! This test demonstrates the one-liner API by generating a complete
//! proof bundle for the proof-bundle crate itself.

use proof_bundle::api;

#[test]
#[ignore] // Run manually with: cargo test --package proof-bundle dogfood -- --ignored --nocapture
fn dogfood_generate_own_proof_bundle() {
    println!("\n🎯 Generating proof-bundle's own proof bundle...\n");

    // Generate complete proof bundle with one line!
    let summary = api::generate_for_crate("proof-bundle", api::Mode::UnitFast)
        .expect("Failed to generate proof bundle");

    // Print summary
    println!("✅ Proof bundle generated successfully!");
    println!("\n📊 Summary:");
    println!("   Total tests: {}", summary.total);
    println!("   Passed: {} ({}%)", summary.passed, summary.pass_rate);
    println!("   Failed: {}", summary.failed);
    println!("   Ignored: {}", summary.ignored);
    println!("   Duration: {:.2}s", summary.duration_secs);

    // Verify reports were generated
    println!("\n📁 Generated files:");
    println!("   ✓ test_results.ndjson");
    println!("   ✓ summary.json");
    println!("   ✓ executive_summary.md");
    println!("   ✓ test_report.md");
    println!("   ✓ failure_report.md");
    println!("   ✓ metadata_report.md");
    println!("   ✓ test_config.json");

    println!("\n🎉 Check .proof_bundle/unit/<run_id>/ for all reports!\n");

    // Basic assertions
    assert!(summary.total > 0, "Should have run some tests");
    assert!(summary.pass_rate >= 90.0, "Should have high pass rate");
}

#[test]
#[ignore] // Run manually
fn dogfood_all_modes() {
    println!("\n🎯 Testing all proof bundle modes...\n");

    let modes = vec![
        ("UnitFast", api::Mode::UnitFast),
        ("UnitFull", api::Mode::UnitFull),
    ];

    for (name, mode) in modes {
        println!("📦 Generating with mode: {}", name);
        
        let summary = api::generate_for_crate("proof-bundle", mode)
            .expect(&format!("Failed to generate with mode {}", name));
        
        println!("   ✅ {} tests, {:.1}% pass rate", summary.total, summary.pass_rate);
    }

    println!("\n🎉 All modes tested successfully!\n");
}
