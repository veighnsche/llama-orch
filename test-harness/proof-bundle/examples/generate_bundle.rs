//! Example: Generate a proof bundle using the V2 one-liner API
//!
//! Run with: cargo run --example generate_bundle

use proof_bundle::api;

fn main() -> anyhow::Result<()> {
    println!("\n🎯 Generating proof bundle for proof-bundle crate...\n");

    // THE ONE-LINER - This does everything!
    let summary = api::generate_for_crate("proof-bundle", api::Mode::UnitFast)?;

    // Print results
    println!("✅ Proof bundle generated successfully!\n");
    println!("📊 Summary:");
    println!("   Total tests: {}", summary.total);
    println!("   Passed: {} ({:.1}%)", summary.passed, summary.pass_rate);
    println!("   Failed: {}", summary.failed);
    println!("   Ignored: {}", summary.ignored);
    println!("   Duration: {:.2}s\n", summary.duration_secs);

    println!("📁 Generated files in .proof_bundle/unit/<run_id>/:");
    println!("   ✓ test_results.ndjson      - All test results");
    println!("   ✓ summary.json             - Test summary");
    println!("   ✓ executive_summary.md     - For management");
    println!("   ✓ test_report.md           - For developers");
    println!("   ✓ failure_report.md        - For debugging");
    println!("   ✓ metadata_report.md       - For compliance");
    println!("   ✓ test_config.json         - Template used\n");

    if summary.failed > 0 {
        println!("⚠️  Some tests failed. Check failure_report.md for details.\n");
    } else {
        println!("🎉 All tests passed!\n");
    }

    Ok(())
}
