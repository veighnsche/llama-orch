//! Queen rebuild operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Added queen rebuild command for local-hive optimization
//! TEAM-263: Implemented actual build logic
//! TEAM-312: Migrated to n!() macro, added running check
//! TEAM-316: Refactored to use shared daemon-lifecycle rebuild functions

use anyhow::Result;
use observability_narration_core::n;
use daemon_lifecycle::rebuild::{check_not_running_before_rebuild, build_daemon_local, RebuildConfig};

/// Rebuild queen-rbee with optional features (update operation)
///
/// TEAM-296: This is the 'update' command - rebuilds from source
/// TEAM-312: Added check to prevent rebuilding while queen is running
///
/// Runs `cargo build --release --bin queen-rbee` with optional features.
///
/// # Arguments
/// * `with_local_hive` - Include local-hive feature for 50-100x faster localhost operations
///
/// # Returns
/// * `Ok(())` - Build successful
/// * `Err` - Build failed
pub async fn rebuild_queen(with_local_hive: bool) -> Result<()> {
    n!("start", "🔄 Updating queen-rbee (rebuilding from source)...");
    
    // TEAM-316: Use shared health check function
    let queen_url = "http://localhost:7833";
    check_not_running_before_rebuild("queen-rbee", queen_url, None).await?;

    // TEAM-316: Build configuration
    let mut config = RebuildConfig::new("queen-rbee");
    
    if with_local_hive {
        n!("build_mode", "✨ Building with integrated local hive (50-100x faster localhost)...");
        config = config.with_features(vec!["local-hive".to_string()]);
    } else {
        n!("build_mode", "📡 Building distributed queen (remote hives only)...");
    }

    // TEAM-316: Use shared build function
    let _binary_path = build_daemon_local(config).await?;

    if with_local_hive {
        n!("restart_hint", "💡 Restart queen to use the new binary with local-hive feature");
    }
    
    Ok(())
}
