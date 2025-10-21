//! Demo of NarrationFactory pattern (v0.4.0)
//!
//! Run with: cargo run --example factory_demo --features test-support

use observability_narration_core::{
    NarrationFactory, ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL, ACTION_HIVE_START, ACTION_STATUS,
};

// Define factory at module level (compile-time constant)
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

fn main() {
    println!("🎀 Narration Factory Demo (v0.4.0)\n");
    println!("Notice how:");
    println!("1. Actor is defined ONCE at module level");
    println!("2. All messages start at the same column");
    println!("3. Easy to scan and read!\n");
    println!("Output:\n");

    // Example 1: Status check
    NARRATE.narrate(ACTION_STATUS, "registry").human("Found 2 hives").emit();

    // Example 2: Hive installation
    NARRATE
        .narrate(ACTION_HIVE_INSTALL, "hive-1")
        .human("🔧 Installing hive 'hive-1'")
        .emit();

    // Example 3: Hive start
    NARRATE.narrate(ACTION_HIVE_START, "hive-1").human("🚀 Starting hive 'hive-1'").emit();

    // Example 4: With correlation ID
    NARRATE
        .narrate(ACTION_STATUS, "registry")
        .human("Status check complete")
        .correlation_id("req-123")
        .duration_ms(50)
        .emit();

    println!("\n✨ Notice the consistent column alignment!");
    println!("✨ All messages start at position 23 (after the actor field)");
    println!("✨ Much easier to scan than the old format!");
}
