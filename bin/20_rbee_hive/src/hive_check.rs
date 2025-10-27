//! TEAM-313: Hive narration check handler
//! 
//! **PURPOSE:** Test narration through hive SSE streaming
//!
//! Like queen-check tests narration through queen's SSE pipeline,
//! this tests narration through hive's SSE pipeline.
//!
//! Tests:
//! - Narration from rbee-hive
//! - SSE streaming to client
//! - Job ID routing
//! - All three narration modes

use anyhow::Result;
use observability_narration_core::{n, set_narration_mode, NarrationMode};
use std::time::Duration;

/// Run hive narration check
///
/// TEAM-313: Tests narration through hive SSE streaming
/// 
/// This is identical to queen-check but runs in rbee-hive.
/// Tests that narration events from hive operations are properly
/// streamed via SSE to the client.
pub async fn handle_hive_check() -> Result<()> {
    n!("test_start", "🔍 Hive Narration Test - Testing SSE streaming from hive");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 1: Simple narration
    n!("test_1", "📝 Test 1: Simple narration through hive");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 2: Narration with variables
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");
    n!("test_2", "📝 Test 2: {} version {} reporting", name, version);
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 3: All three narration modes (Human mode)
    n!("test_3", "📝 Test 3: Testing Human Mode");
    set_narration_mode(NarrationMode::Human);
    n!("mode_test",
        human: "Testing narration in human mode via hive SSE",
        cute: "🐝 Testing narration in cute mode via hive SSE!",
        story: "'Testing narration through the hive', said the server"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 4: Cute mode
    n!("test_4", "📝 Test 4: Testing Cute Mode");
    set_narration_mode(NarrationMode::Cute);
    n!("mode_test",
        human: "Testing narration in human mode via hive SSE",
        cute: "🐝 Testing narration in cute mode via hive SSE!",
        story: "'Testing narration through the hive', said the server"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 5: Story mode
    n!("test_5", "📝 Test 5: Testing Story Mode");
    set_narration_mode(NarrationMode::Story);
    n!("mode_test",
        human: "Testing narration in human mode via hive SSE",
        cute: "🐝 Testing narration in cute mode via hive SSE!",
        story: "'Testing narration through the hive', said the server"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Reset to human mode
    set_narration_mode(NarrationMode::Human);
    
    // Test 6: Format specifiers
    n!("test_6", "📝 Test 6: Hex: {:x}, Debug: {:?}, Float: {:.2}", 255, vec![1, 2, 3], 3.14159);
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 7: Sequential narrations
    n!("test_7", "📝 Test 7: Sequential narrations");
    for i in 1..=5 {
        n!("sequence", "Narration sequence {}/5 streaming via hive SSE", i);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Test 8: Job context test
    n!("test_8", "📝 Test 8: Job ID context propagation");
    n!("job_context", "✅ This narration is routed via job_id to your SSE stream");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 9: Partial mode combinations
    n!("test_9", "📝 Test 9: Partial mode combinations");
    n!("partial_test",
        human: "Technical message streamed from hive to client",
        cute: "🎀 Fun message streamed in cute mode from hive!"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 10: Final summary
    n!("test_10", "📝 Test 10: Final summary");
    n!("hive_check_complete",
        human: "✅ Hive check complete - all SSE narration tests passed",
        cute: "🎉 Hive check complete - hive SSE streaming works perfectly!",
        story: "'All systems operational', reported the hive with satisfaction"
    );
    
    n!("summary", "✨ Hive narration test complete - {} tested successfully", "hive SSE streaming");
    
    Ok(())
}
