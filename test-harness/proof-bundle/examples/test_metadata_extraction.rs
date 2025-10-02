// Test metadata extraction on proof-bundle's own tests
use proof_bundle::discovery;
use proof_bundle::extraction;

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing metadata extraction on proof-bundle's own tests...\n");
    
    // 1. Discover test targets
    println!("Step 1: Discovering test targets...");
    let targets = discovery::discover_tests("proof-bundle")?;
    println!("   Found {} test targets", targets.len());
    for target in &targets {
        println!("   - {} ({})", target.name, target.src_path.display());
    }
    
    // 2. Extract metadata
    println!("\nStep 2: Extracting metadata from source files...");
    let metadata_map = extraction::extract_metadata(&targets)?;
    println!("   Extracted metadata for {} tests", metadata_map.len());
    
    // 3. Show ALL test names found
    println!("\nStep 3: All test names found:");
    for (i, test_name) in metadata_map.keys().enumerate() {
        println!("   {}. {}", i + 1, test_name);
        if i >= 19 {
            println!("   ... ({} more)", metadata_map.len() - 20);
            break;
        }
    }
    
    // 4. Show some examples with metadata
    println!("\nStep 4: Sample extracted metadata:");
    let mut count = 0;
    for (test_name, metadata) in metadata_map.iter() {
        if metadata.priority.is_some() || metadata.spec.is_some() || !metadata.tags.is_empty() {
            println!("\n   📝 {}", test_name);
            if let Some(priority) = &metadata.priority {
                println!("      Priority: {}", priority);
            }
            if let Some(spec) = &metadata.spec {
                println!("      Spec: {}", spec);
            }
            if let Some(team) = &metadata.team {
                println!("      Team: {}", team);
            }
            if !metadata.tags.is_empty() {
                println!("      Tags: {}", metadata.tags.join(", "));
            }
            count += 1;
            if count >= 5 {
                break;
            }
        }
    }
    
    println!("\n✅ Metadata extraction successful!");
    println!("   Total tests with metadata: {}", 
        metadata_map.values().filter(|m| 
            m.priority.is_some() || m.spec.is_some() || !m.tags.is_empty()
        ).count()
    );
    
    Ok(())
}
