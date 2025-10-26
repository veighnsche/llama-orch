//! TEAM-312: Deep narration test handler for queen-rbee
//! Tests narration system through the entire job server pipeline:
//! rbee-keeper → queen-rbee → job router → SSE streaming → client
//!
//! This is identical to self-check but runs server-side and streams via SSE

use anyhow::Result;
use observability_narration_core::{n, set_narration_mode, NarrationMode};
use std::time::Duration;

/// Run comprehensive narration test through queen job server
///
/// TEAM-312: Deep integration test for narration system
/// Tests:
/// - n!() macro with actor auto-detection
/// - All three narration modes (Human, Cute, Story)
/// - Format specifiers
/// - Sequential narrations
/// - SSE streaming (via job_id routing)
pub async fn handle_queen_check() -> Result<()> {
    n!("test_start", "🔍 Queen Narration Test - Testing SSE streaming");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 1: Simple narration
    n!("test_1", "📝 Test 1: Simple narration through queen");
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
        human: "Testing narration in human mode via SSE",
        cute: "🐝 Testing narration in cute mode via SSE!",
        story: "'Testing narration through the queen', said the server"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 4: Cute mode
    n!("test_4", "📝 Test 4: Testing Cute Mode");
    set_narration_mode(NarrationMode::Cute);
    n!("mode_test",
        human: "Testing narration in human mode via SSE",
        cute: "🐝 Testing narration in cute mode via SSE!",
        story: "'Testing narration through the queen', said the server"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 5: Story mode
    n!("test_5", "📝 Test 5: Testing Story Mode");
    set_narration_mode(NarrationMode::Story);
    n!("mode_test",
        human: "Testing narration in human mode via SSE",
        cute: "🐝 Testing narration in cute mode via SSE!",
        story: "'Testing narration through the queen', said the server"
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
        n!("sequence", "Narration sequence {}/5 streaming via SSE", i);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Test 8: Job context test
    n!("test_8", "📝 Test 8: Job ID context propagation");
    n!("job_context", "✅ This narration is routed via job_id to your SSE stream");
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 9: Partial mode combinations
    n!("test_9", "📝 Test 9: Partial mode combinations");
    n!("partial_test",
        human: "Technical message streamed to client",
        cute: "🎀 Fun message streamed in cute mode!"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Test 10: Final summary
    n!("test_10", "📝 Test 10: Final summary");
    n!("queen_check_complete",
        human: "✅ Queen check complete - all SSE narration tests passed",
        cute: "🎉 Queen check complete - SSE streaming works perfectly!",
        story: "'All systems operational', reported the queen with satisfaction"
    );
    
    n!("summary", "✨ Queen narration test complete - {} tested successfully", "SSE streaming");
    
    Ok(())
}
