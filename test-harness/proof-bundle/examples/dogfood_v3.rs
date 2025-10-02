// Dogfood V3: Generate proof bundle for proof-bundle itself
use proof_bundle::{generate_for_crate, Mode};

fn main() -> anyhow::Result<()> {
    println!("🐕 Dogfooding V3: Generating proof bundle for proof-bundle...\n");
    
    let summary = generate_for_crate("proof-bundle", Mode::UnitFast)?;
    
    println!("\n✅ Proof bundle generated successfully!");
    println!("   Total tests: {}", summary.total);
    println!("   Passed: {}", summary.passed);
    println!("   Failed: {}", summary.failed);
    println!("   Pass rate: {:.1}%", summary.pass_rate);
    
    // Count tests with metadata
    let with_metadata = summary.tests.iter()
        .filter(|t| t.metadata.is_some())
        .count();
    
    println!("\n📊 Metadata extraction:");
    println!("   Tests with metadata: {}/{}", with_metadata, summary.total);
    
    if with_metadata > 0 {
        println!("\n🎉 SUCCESS! Metadata was extracted from annotated tests!");
    }
    
    Ok(())
}
