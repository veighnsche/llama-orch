// TEAM-373: Created by TEAM-373
//! Queen subscribes to Hive SSE streams
//!
//! After discovery handshake, Queen connects to each hive's
//! GET /v1/heartbeats/stream and aggregates telemetry.

use anyhow::Result;
use futures::StreamExt;
use observability_narration_core::n;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Hive heartbeat event from SSE stream
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HiveHeartbeatEvent {
    Telemetry {
        hive_id: String,
        #[serde(rename = "hive_info")]
        _hive_info: serde_json::Value, // We only need workers
        timestamp: String,
        workers: Vec<serde_json::Value>, // ProcessStats as JSON
    },
}

/// Subscribe to a single hive's SSE stream
///
/// TEAM-373: Queen calls this after receiving discovery callback.
/// Runs continuously, forwarding events to Queen's broadcast channel.
pub async fn subscribe_to_hive(
    hive_url: String,
    hive_id: String,
    hive_registry: Arc<queen_rbee_telemetry_registry::TelemetryRegistry>, // TEAM-374
    queen_event_tx: broadcast::Sender<crate::http::heartbeat::HeartbeatEvent>,
) -> Result<()> {
    let stream_url = format!("{}/v1/heartbeats/stream", hive_url);
    
    n!("hive_subscribe_start", "📡 Subscribing to hive {} SSE stream: {}", hive_id, stream_url);
    
    loop {
        let mut event_source = EventSource::get(&stream_url);
        
        n!("hive_subscribe_connected", "✅ Connected to hive {} SSE stream", hive_id);
        
        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Message(msg)) => {
                    // Parse telemetry event
                    if let Ok(hive_event) = serde_json::from_str::<HiveHeartbeatEvent>(&msg.data) {
                        match hive_event {
                            HiveHeartbeatEvent::Telemetry { hive_id, workers, timestamp, .. } => {
                                tracing::trace!("Received telemetry from hive {}: {} workers", hive_id, workers.len());
                                
                                // Parse workers (they're already ProcessStats JSON)
                                let parsed_workers: Vec<_> = workers
                                    .into_iter()
                                    .filter_map(|w| serde_json::from_value(w).ok())
                                    .collect();
                                
                                // Store in HiveRegistry
                                hive_registry.update_workers(&hive_id, parsed_workers.clone());
                                
                                // Forward to Queen's SSE stream
                                let queen_event = crate::http::heartbeat::HeartbeatEvent::HiveTelemetry {
                                    hive_id,
                                    timestamp,
                                    workers: parsed_workers,
                                };
                                
                                let _ = queen_event_tx.send(queen_event);
                            }
                        }
                    }
                }
                Ok(Event::Open) => {
                    n!("hive_subscribe_open", "🔗 SSE connection opened for hive {}", hive_id);
                }
                Err(e) => {
                    n!("hive_subscribe_error", "❌ Hive {} SSE error: {}", hive_id, e);
                    break; // Reconnect
                }
            }
        }
        
        // Connection closed, retry after delay
        n!("hive_subscribe_reconnect", "🔄 Reconnecting to hive {} in 5s...", hive_id);
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

/// Start subscription task for a hive
///
/// TEAM-373: Spawns background task that runs forever.
/// Called when Queen receives discovery callback from hive.
pub fn start_hive_subscription(
    hive_url: String,
    hive_id: String,
    hive_registry: Arc<queen_rbee_telemetry_registry::TelemetryRegistry>, // TEAM-374
    queen_event_tx: broadcast::Sender<crate::http::heartbeat::HeartbeatEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(e) = subscribe_to_hive(hive_url.clone(), hive_id.clone(), hive_registry, queen_event_tx).await {
            n!("hive_subscribe_fatal", "❌ Fatal error subscribing to hive {}: {}", hive_id, e);
        }
    })
}
