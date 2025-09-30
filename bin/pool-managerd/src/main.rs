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

use pool_managerd::api::routes::{create_router, AppState};
use pool_managerd::config::Config;
use pool_managerd::core::registry::Registry;
use handoff_watcher::{HandoffWatcher, HandoffWatcherConfig};
use node_registration::{NodeRegistration, NodeRegistrationConfig};
use service_registry::HeartbeatPoolStatus;
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
    let state = AppState {
        registry: registry.clone(),
    };

    // CLOUD_PROFILE: Spawn handoff watcher
    if config.cloud_profile {
        let registry_clone = registry.clone();
        let watcher_config = HandoffWatcherConfig {
            runtime_dir: config.handoff_config.runtime_dir.clone(),
            poll_interval_ms: config.handoff_config.poll_interval_ms,
        };
        
        let callback = Box::new(move |payload: handoff_watcher::HandoffPayload| {
            tracing::info!(
                pool_id = %payload.pool_id,
                replica_id = %payload.replica_id,
                engine = %payload.engine,
                url = %payload.url,
                "Handoff detected, updating registry"
            );
            
            let mut reg = registry_clone.lock().unwrap();
            let handoff_json = serde_json::json!({
                "engine_version": payload.engine_version,
                "device_mask": payload.device_mask,
                "slots": payload.slots,
            });
            reg.register_ready_from_handoff(&payload.pool_id, &handoff_json);
            
            Ok(())
        });
        
        let watcher = HandoffWatcher::new(watcher_config, callback);
        let _watcher_handle = watcher.spawn();
        tracing::info!("Handoff watcher started");
    }

    // CLOUD_PROFILE: Register with orchestratord
    if config.cloud_profile {
        if let Some(node_config) = &config.node_config {
            let registration_config = NodeRegistrationConfig {
                node_id: node_config.node_id.clone(),
                machine_id: node_config.machine_id.clone(),
                address: node_config.address.clone(),
                orchestratord_url: node_config.orchestratord_url.clone(),
                pools: node_config.pools.clone(),
                capabilities: node_config.capabilities.clone(),
                heartbeat_interval_secs: node_config.heartbeat_interval_secs,
                api_token: node_config.api_token.clone(),
            };
            
            let registration = NodeRegistration::new(registration_config);
            
            // Register on startup
            if node_config.register_on_startup {
                match registration.register().await {
                    Ok(_) => tracing::info!("Successfully registered with orchestratord"),
                    Err(e) => tracing::error!("Failed to register: {}", e),
                }
            }
            
            // Spawn heartbeat task
            let registry_clone = registry.clone();
            let _heartbeat_handle = registration.spawn_heartbeat(move || {
                let reg = registry_clone.lock().unwrap();
                let snapshots = reg.snapshots();
                
                snapshots.into_iter().map(|snap| {
                    HeartbeatPoolStatus {
                        pool_id: snap.pool_id,
                        ready: snap.health.ready,
                        draining: snap.draining,
                        slots_free: snap.slots_free.unwrap_or(0) as u32,
                        slots_total: snap.slots_total.unwrap_or(0) as u32,
                        vram_free_bytes: snap.vram_free_bytes.unwrap_or(0),
                        engine: snap.engine_version,
                    }
                }).collect()
            });
            tracing::info!("Heartbeat task started");
        }
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
