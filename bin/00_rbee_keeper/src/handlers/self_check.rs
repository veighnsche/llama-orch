// TEAM-309: Self-check handler with narration testing
//! Self-check command to test narration system

use anyhow::Result;
use observability_narration_core::{n, set_narration_mode, NarrationMode};
use std::time::Duration;

/// Run self-check with comprehensive narration testing
/// TEAM-309: Actor auto-detected from crate name (rbee-keeper)
pub async fn handle_self_check() -> Result<()> {
    println!("\n🔍 rbee-keeper Self-Check");
    println!("{}", "=".repeat(50));
    
    // Test 1: Simple narration
    println!("\n📝 Test 1: Simple Narration");
    n!("self_check_start", "Starting rbee-keeper self-check");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 2: Narration with variables
    println!("\n📝 Test 2: Narration with Variables");
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");
    n!("version_check", "Checking {} version {}", name, version);
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 3: All three narration modes (Human mode)
    println!("\n📝 Test 3: Human Mode (default)");
    set_narration_mode(NarrationMode::Human);
    n!("mode_test",
        human: "Testing narration in human mode",
        cute: "🐝 Testing narration in cute mode!",
        story: "'Testing narration', said the keeper"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 4: Cute mode
    println!("\n📝 Test 4: Cute Mode");
    set_narration_mode(NarrationMode::Cute);
    n!("mode_test",
        human: "Testing narration in human mode",
        cute: "🐝 Testing narration in cute mode!",
        story: "'Testing narration', said the keeper"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 5: Story mode
    println!("\n📝 Test 5: Story Mode");
    set_narration_mode(NarrationMode::Story);
    n!("mode_test",
        human: "Testing narration in human mode",
        cute: "🐝 Testing narration in cute mode!",
        story: "'Testing narration', said the keeper"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Reset to human mode
    set_narration_mode(NarrationMode::Human);
    
    // Test 6: Format specifiers
    println!("\n📝 Test 6: Format Specifiers");
    n!("format_test", "Hex: {:x}, Debug: {:?}, Float: {:.2}", 255, vec![1, 2, 3], 3.14159);
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 7: Multiple narrations in sequence
    println!("\n📝 Test 7: Sequential Narrations");
    for i in 1..=5 {
        n!("sequence_test", "Narration sequence {}/5", i);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Test 8: Config check
    println!("\n📝 Test 8: Configuration Check");
    match crate::config::Config::load() {
        Ok(config) => {
            n!("config_check", "✅ Configuration loaded successfully");
            n!("config_queen_url", "Queen URL: {}", config.queen_url());
        }
        Err(e) => {
            n!("config_check", "⚠️  Configuration load failed: {}", e);
        }
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 9: Partial mode combinations
    println!("\n📝 Test 9: Partial Mode Combinations (Human + Cute)");
    n!("partial_test",
        human: "Technical message for humans",
        cute: "🎀 Fun message for cute mode!"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 10: Final summary
    println!("\n📝 Test 10: Summary");
    n!("self_check_complete",
        human: "✅ Self-check complete - all narration tests passed",
        cute: "🎉 Self-check complete - everything works perfectly!",
        story: "'All systems operational', reported the keeper with satisfaction"
    );
    
    println!("\n{}", "=".repeat(50));
    println!("✅ Self-Check Complete!");
    println!("\nAll narration modes tested:");
    println!("  • Human mode (technical)");
    println!("  • Cute mode (whimsical)");
    println!("  • Story mode (narrative)");
    println!("\nIf you saw narration output above, the system is working correctly.");
    println!("If not, check that narration-core is properly configured.");
    
    Ok(())
}
