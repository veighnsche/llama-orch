// TEAM-199: Example disabled due to API changes - needs update to current NarrationFactory API
// //! Demo of NarrationFactory pattern (v0.4.0)
// //!
// //! Run with: cargo run --example factory_demo --features test-support

// use observability_narration_core::{
//     NarrationFactory, ACTION_HIVE_INSTALL, ACTION_HIVE_START, ACTION_STATUS,
// };

// // TEAM-199: Use short actor name (pre-existing constants exceed 10 char limit)
// // Define factory at module level (compile-time constant)
// const NARRATE: NarrationFactory = NarrationFactory::new("demo");

// fn main() {
//     println!("🎀 Narration Factory Demo (v0.4.0)\n");
//     println!("Notice how:");
//     println!("1. Actor is defined ONCE at module level");
//     println!("2. All messages start at the same column");
//     println!("3. Easy to scan and read!\n");
//     println!("Output:\n");

//     // Example 1: Status check
//     NARRATE.action(ACTION_STATUS).context("registry").human("Found 2 hives").emit();

//     // Example 2: Hive installation
//     NARRATE
//         .action(ACTION_HIVE_INSTALL)
//         .context("hive-1")
//         .human("🔧 Installing hive 'hive-1'")
//         .emit();

//     // Example 3: Hive start
//     NARRATE.action(ACTION_HIVE_START).context("hive-1").human("🚀 Starting hive 'hive-1'").emit();

//     // Example 4: With correlation ID
//     NARRATE
//         .action(ACTION_STATUS)
//         .context("registry")
//         .human("Status check complete")
//         .correlation_id("req-123")
//         .duration_ms(50)
//         .emit();

//     println!("\n✨ Notice the consistent column alignment!");
//     println!("✨ All messages start at position 23 (after the actor field)");
//     println!("✨ Much easier to scan than the old format!");
// }

fn main() {
    println!("Example disabled - needs API update");
}
