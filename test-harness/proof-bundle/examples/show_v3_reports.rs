// Demonstrate V3 reports with extracted metadata
use proof_bundle::{discovery, extraction, formatters, bundle, Mode, TestResult, TestStatus, TestSummary};

fn main() -> anyhow::Result<()> {
    println!("📊 Generating V3 proof bundle reports with extracted metadata...\n");
    
    // 1. Discover and extract metadata
    let targets = discovery::discover_tests("proof-bundle")?;
    let metadata_map = extraction::extract_metadata(&targets)?;
    
    println!("✅ Extracted metadata for {} tests\n", metadata_map.len());
    
    // 2. Create test results for ALL tests we found
    let mut tests: Vec<TestResult> = metadata_map.keys()
        .map(|name| TestResult::new(name.clone(), TestStatus::Passed))
        .collect();
    
    // 3. Attach metadata to ALL tests
    for test in &mut tests {
        if let Some(metadata) = metadata_map.get(&test.name) {
            test.metadata = Some(metadata.clone());
        }
    }
    
    // Sort by priority (critical first) then by name
    tests.sort_by(|a, b| {
        let a_priority = a.metadata.as_ref()
            .and_then(|m| m.priority.as_ref())
            .map(|p| match p.as_str() {
                "critical" => 0,
                "high" => 1,
                "medium" => 2,
                "low" => 3,
                _ => 4,
            })
            .unwrap_or(5);
        
        let b_priority = b.metadata.as_ref()
            .and_then(|m| m.priority.as_ref())
            .map(|p| match p.as_str() {
                "critical" => 0,
                "high" => 1,
                "medium" => 2,
                "low" => 3,
                _ => 4,
            })
            .unwrap_or(5);
        
        a_priority.cmp(&b_priority).then_with(|| a.name.cmp(&b.name))
    });
    
    let summary = TestSummary::new(tests);
    
    // 4. Generate unified report
    let writer = bundle::BundleWriter::new(Mode::UnitFast)?;
    
    println!("📝 Generating unified report...\n");
    
    // Single comprehensive report
    let report = formatters::generate_unified_report(&summary)
        .map_err(|e| anyhow::anyhow!(e))?;
    writer.write_markdown("proof_bundle_report", &report)?;
    println!("✅ Unified report generated");
    
    // Write machine-readable formats
    writer.write_json("summary", &summary)?;
    writer.write_ndjson("test_results", &summary.tests)?;
    println!("✅ JSON/NDJSON artifacts generated");
    
    println!("\n🎉 Proof bundle generated at: {}", writer.root().display());
    println!("\n📖 View the comprehensive report:");
    println!("   - proof_bundle_report.md  (single unified document)");
    
    Ok(())
}
