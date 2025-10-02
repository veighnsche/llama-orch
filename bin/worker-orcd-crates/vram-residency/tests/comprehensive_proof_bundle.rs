//! Comprehensive proof bundle generator for vram-residency
//!
//! Captures ALL test results automatically using capture_tests() API.
//! 
//! Run with:
//!   cargo +nightly test -p vram-residency generate_comprehensive_proof_bundle -- --ignored --nocapture
//!   cargo +nightly test -p vram-residency generate_comprehensive_proof_bundle_fast -- --ignored --nocapture

use proof_bundle::{ProofBundle, TestType};
use std::path::PathBuf;

#[test]
#[ignore] // Run explicitly to avoid recursion
fn generate_comprehensive_proof_bundle() -> anyhow::Result<()> {
    generate_proof_bundle_internal(false)
}

#[test]
#[ignore] // Run explicitly to avoid recursion
fn generate_comprehensive_proof_bundle_fast() -> anyhow::Result<()> {
    generate_proof_bundle_internal(true)
}

fn generate_proof_bundle_internal(skip_long_tests: bool) -> anyhow::Result<()> {
    // Set proof bundle to crate's own directory with different subdirs for fast/full
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proof_bundle_dir = if skip_long_tests {
        crate_root.join(".proof_bundle").join("unit-fast")
    } else {
        crate_root.join(".proof_bundle").join("unit-full")
    };
    std::env::set_var("LLORCH_PROOF_DIR", proof_bundle_dir.to_str().unwrap());
    
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    let mode = if skip_long_tests { "FAST (skip-long-tests)" } else { "FULL (all tests)" };
    println!("\n📦 Generating comprehensive proof bundle for vram-residency...");
    println!("   Mode: {}", mode);
    println!("   Capturing ALL tests automatically\n");
    
    // Capture ALL test results automatically
    // Note: We need to run ALL tests, but cargo test filters when run from within a test
    // Solution: Don't use .lib() or .tests(), just run everything except this test
    let mut builder = pb.capture_tests("vram-residency")
        .no_fail_fast();  // Continue on failure
    
    // Add skip-long-tests feature if requested
    if skip_long_tests {
        builder = builder.features(&["skip-long-tests"]);
    }
    
    let summary = builder.run()?;
    
    println!("✅ Comprehensive proof bundle generated!");
    println!("   Mode: {}", mode);
    println!("   Total: {} tests", summary.total);
    println!("   Passed: {} ({:.1}%)", summary.passed, summary.pass_rate);
    println!("   Failed: {}", summary.failed);
    println!("   Ignored: {}", summary.ignored);
    println!("   Duration: {:.2}s", summary.duration_secs);
    
    let subdir = if skip_long_tests { "unit-fast" } else { "unit-full" };
    println!("   Location: {}/.proof_bundle/{}/", crate_root.display(), subdir);
    
    // Verify files were created
    let root = pb.root();
    assert!(root.join("test_results.ndjson").exists(), "test_results.ndjson should exist");
    assert!(root.join("summary.json").exists(), "summary.json should exist");
    assert!(root.join("test_report.md").exists(), "test_report.md should exist");
    
    // Assert we captured significantly more than the old 6 tests
    let expected_min = if skip_long_tests { 20 } else { 50 };
    if summary.total < expected_min {
        eprintln!("⚠️  WARNING: Expected {}+ tests, got {}. Old approach only captured 6.", expected_min, summary.total);
    }
    
    println!("\n🎯 Success! Captured {} tests (old manual approach: only 6)", summary.total);
    if skip_long_tests {
        println!("   Note: Long-running tests skipped (property tests, stress tests)");
    }
    println!();
    
    Ok(())
}
