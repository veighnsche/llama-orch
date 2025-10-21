// //! Comparison: Macro vs Factory vs Direct
// //!
// //! Run with: cargo run --example macro_vs_factory --features test-support

// use observability_narration_core::{
//     narration_macro, NarrationFactory, Narration, ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL,
//     ACTION_HIVE_START, ACTION_STATUS,
// };

// // ============================================================================
// // Pattern 1: Direct (v0.3.0 style) - Most verbose
// // ============================================================================
// fn pattern_1_direct() {
//     println!("📝 Pattern 1: Direct (v0.3.0 style)\n");

//     Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
//         .human("Found 2 hives")
//         .emit();

//     Narration::new(ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL, "hive-1")
//         .human("🔧 Installing hive")
//         .emit();

//     Narration::new(ACTOR_QUEEN_ROUTER, ACTION_HIVE_START, "hive-1")
//         .human("🚀 Starting hive")
//         .emit();

//     println!("\n❌ Issues:");
//     println!("   - Repetitive ACTOR_QUEEN_ROUTER");
//     println!("   - Easy to use wrong actor by mistake");
//     println!("   - Most boilerplate\n");
// }

// // ============================================================================
// // Pattern 2: Factory (v0.4.0) - Less boilerplate
// // ============================================================================
// fn pattern_2_factory() {
//     println!("🏭 Pattern 2: Factory (v0.4.0)\n");

//     const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

//     NARRATE.narrate(ACTION_STATUS, "registry").human("Found 2 hives").emit();

//     NARRATE.narrate(ACTION_HIVE_INSTALL, "hive-1").human("🔧 Installing hive").emit();

//     NARRATE.narrate(ACTION_HIVE_START, "hive-1").human("🚀 Starting hive").emit();

//     println!("\n✅ Benefits:");
//     println!("   - Actor defined once");
//     println!("   - Type-safe (const fn)");
//     println!("   - IDE-friendly (autocomplete works)");
//     println!("   - Still need to type .narrate()\n");
// }

// // ============================================================================
// // Pattern 3: Macro (v0.4.0) - MOST ergonomic!
// // ============================================================================
// fn pattern_3_macro() {
//     println!("🎀 Pattern 3: Macro (v0.4.0) - MOST ERGONOMIC!\n");

//     // Create the macro with actor baked in!
//     narration_macro!(ACTOR_QUEEN_ROUTER);

//     // Now use it - looks just like println!
//     narrate!(ACTION_STATUS, "registry").human("Found 2 hives").emit();

//     narrate!(ACTION_HIVE_INSTALL, "hive-1").human("🔧 Installing hive").emit();

//     narrate!(ACTION_HIVE_START, "hive-1").human("🚀 Starting hive").emit();

//     println!("\n✨ Benefits:");
//     println!("   - Actor defined once");
//     println!("   - Shortest syntax: narrate!(action, target)");
//     println!("   - Rust-idiomatic (like println!)");
//     println!("   - Zero runtime overhead");
//     println!("   - MOST ergonomic!\n");
// }

// fn main() {
//     println!("🎀 Narration Patterns Comparison\n");
//     println!("=".repeat(60));
//     println!();

//     pattern_1_direct();
//     println!("=".repeat(60));
//     println!();

//     pattern_2_factory();
//     println!("=".repeat(60));
//     println!();

//     pattern_3_macro();
//     println!("=".repeat(60));
//     println!();

//     println!("🎯 Recommendation:");
//     println!();
//     println!("   Use MACRO for ultimate ergonomics!");
//     println!("   Use FACTORY if you need runtime flexibility");
//     println!("   Use DIRECT only for one-off narrations");
//     println!();
//     println!("🎀 The macro pattern is inspired by println! - define once, use everywhere!");
// }
