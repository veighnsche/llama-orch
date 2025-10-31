// TEAM-372: Created by TEAM-372
//! Hive SSE heartbeat stream
//!
//! Exposes GET /v1/heartbeats/stream for Queen and Hive SDK to subscribe.
//! Broadcasts worker telemetry every 1 second after discovery completes.

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use hive_contract::HiveInfo;
use rbee_hive_monitor::ProcessStats;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::interval;

/// Heartbeat event for SSE stream
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HiveHeartbeatEvent {
    /// Worker telemetry (sent every 1s)
    Telemetry {
        hive_id: String,
        hive_info: HiveInfo,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
}

/// State for heartbeat stream
#[derive(Clone)]
pub struct HeartbeatStreamState {
    pub hive_info: HiveInfo,
    pub event_tx: broadcast::Sender<HiveHeartbeatEvent>,
}

/// GET /v1/heartbeats/stream - SSE endpoint for hive telemetry
///
/// TEAM-372: Queen subscribes to this after discovery handshake completes.
/// Broadcasts worker telemetry every 1 second.
pub async fn handle_heartbeat_stream(
    State(state): State<Arc<HeartbeatStreamState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("New SSE client connected to hive heartbeat stream");
    
    let mut event_rx = state.event_tx.subscribe();
    
    let stream = async_stream::stream! {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    let json = serde_json::to_string(&event)
                        .unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("heartbeat").data(json));
                }
                Err(e) => {
                    tracing::warn!("SSE broadcast error: {}", e);
                    break;
                }
            }
        }
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Background task to collect and broadcast telemetry
///
/// TEAM-372: Replaces `start_normal_telemetry_task()` POST loop.
/// Collects worker telemetry every 1s and broadcasts to SSE subscribers.
pub fn start_telemetry_broadcaster(
    hive_info: HiveInfo,
    event_tx: broadcast::Sender<HiveHeartbeatEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        
        tracing::info!("Telemetry broadcaster started (1s interval)");
        
        loop {
            interval.tick().await;
            
            // TEAM-361: Collect worker telemetry (same as before)
            let workers = rbee_hive_monitor::collect_all_workers()
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to collect worker telemetry: {}", e);
                    Vec::new()
                });
            
            tracing::trace!("Collected telemetry for {} workers", workers.len());
            
            // Broadcast to SSE subscribers
            let event = HiveHeartbeatEvent::Telemetry {
                hive_id: hive_info.id.clone(),
                hive_info: hive_info.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                workers,
            };
            
            // Send to all subscribers (if any)
            // Errors are fine - means no subscribers
            let _ = event_tx.send(event);
        }
    })
}
