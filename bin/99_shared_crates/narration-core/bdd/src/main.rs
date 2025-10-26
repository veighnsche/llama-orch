// BDD runner for narration-core

mod steps;

use cucumber::World as _;
use steps::world::World;

// ============================================================
// BUG FIX: TEAM-308 | Parallel execution race condition
// ============================================================
// SUSPICION (TEAM-307):
// - Thought async task overlap caused event leakage
// - Tried job_id filtering (FAILED - defeats test purpose)
//
// SUSPICION (TEAM-308):
// - Thought baseline tracking would solve it
// - Tried recording initial_event_count (FAILED - still got 25 events)
//
// INVESTIGATION:
// - Discovered Cucumber runs with --concurrency 64 by default
// - All 18+ scenarios run in parallel, sharing ONE global CaptureAdapter
// - Race condition: Scenario A emits → Scenario B emits → Scenario A asserts
// - Each scenario sees events from ALL OTHER concurrent scenarios
//
// ROOT CAUSE:
// - Global singleton CaptureAdapter (OnceLock) shared across all scenarios
// - Default concurrency=64 means ~18 scenarios run simultaneously
// - No isolation between concurrent scenarios
// - clear() works, but other scenarios add events immediately after
//
// FIX:
// - Force sequential execution with max_concurrent_scenarios(1)
// - Eliminates race conditions by running one scenario at a time
// - Each scenario completes before next one starts
//
// TESTING:
// - WITHOUT fix (parallel): 2 passed, 106 skipped, 18 failed
// - WITH fix (sequential):  17 passed, 107 skipped, 2 failed
// - VERIFIED: Toggled fix on/off, results reproducible
// - Result: 83% improvement (15/18 failures fixed)
//
// LONG-TERM:
// - TODO: Implement thread-local CaptureAdapter for proper parallel support
// - See: BUG_003_BREAKTHROUGH.md for full analysis
// ============================================================

#[tokio::main]
async fn main() {
    World::cucumber()
        .max_concurrent_scenarios(1)  // TEAM-308: Force sequential to fix race condition
        .run_and_exit("features").await;
}
