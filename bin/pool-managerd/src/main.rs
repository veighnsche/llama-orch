//! pool-managerd daemon entrypoint.
//!
//! Standalone daemon that manages engine lifecycle (spawn, health, supervision).
//! Exposes HTTP API for orchestratord to call.
//!
//! Responsibilities:
//! 1. Spawn engines from PreparedEngine (via POST /pools/{id}/preload)
//! 2. Monitor health and update registry
//! 3. Supervise processes with backoff on crash
//! 4. Expose HTTP API for orchestratord
//! 5. (CLOUD_PROFILE) Watch handoff files and register with orchestratord

// Security-critical binary: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::arithmetic_side_effects)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

use pool_managerd::api::routes::{create_router, AppState};
use pool_managerd::config::Config;
use pool_managerd::core::registry::Registry;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Configuration loaded. Cloud profile: {}", config.cloud_profile);

    // Create shared registry
    let registry = Arc::new(Mutex::new(Registry::new()));

    // Create app state
    let state = AppState { registry: registry.clone() };

    // TODO: CLOUD_PROFILE features (handoff watcher, node registration) will be implemented
    // when the corresponding library crates are created
    if config.cloud_profile {
        tracing::warn!(
            "Cloud profile enabled but handoff watcher and node registration not yet implemented"
        );
    }

    // Create router
    let app = create_router(state);

    // Bind address
    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    tracing::info!("pool-managerd listening on {}", config.bind_addr);

    // Start server
    axum::serve(listener, app).await?;

    Ok(())
}
