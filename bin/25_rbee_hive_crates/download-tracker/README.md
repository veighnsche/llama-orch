# rbee-hive-download-tracker

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Real-time model download progress tracking via SSE  
**Location:** `bin/rbee-hive-crates/download-tracker/` (hive-specific, but designed for future sharing)

---

## Overview

The `rbee-hive-download-tracker` crate provides **real-time download progress tracking** for model downloads. It tracks download state (speed, bytes transferred, ETA) and streams progress updates via **Server-Sent Events (SSE)** through the orchestration chain.

### System Context

In the llama-orch architecture, download progress flows through **two SSE streams**:

```
┌─────────────────┐
│   rbee-hive     │  ← Downloads model (THIS CRATE)
│ (pool-managerd) │  ← Tracks download progress
└────────┬────────┘
         │
         │ SSE Stream #1: Hive → Queen
         │ GET /v2/downloads/{download_id}/stream
         │ data: {"bytes": 1024, "total": 10240, "speed_bps": 1000000, ...}
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Relays download progress
│ (orchestratord) │  ← Aggregates from multiple hives
└────────┬────────┘
         │
         │ SSE Stream #2: Queen → Client
         │ GET /v1/downloads/{download_id}/stream
         │ data: {"bytes": 1024, "total": 10240, "speed_bps": 1000000, ...}
         ↓
┌─────────────────┐
│     Client      │  ← Displays progress bar
│  (rbee-keeper)  │
└─────────────────┘
```

**Key Responsibilities:**
- Track download progress (bytes, speed, ETA)
- Stream progress updates via SSE
- Support multiple concurrent downloads
- Handle download failures and retries
- Provide download history and cleanup

---

## Why SSE Only?

**Communication channel: SSE (Server-Sent Events) ONLY**

### Why SSE?

**Real-time streaming:**
- ✅ Unidirectional (server → client) - perfect for progress updates
- ✅ Built-in reconnection - handles network hiccups
- ✅ Event-based - structured progress events
- ✅ HTTP-based - works through firewalls/proxies

**NOT WebSocket:**
- ❌ Bidirectional - unnecessary overhead (client doesn't send data)
- ❌ More complex - requires upgrade handshake
- ❌ Less firewall-friendly

**NOT polling:**
- ❌ High latency - delays in progress updates
- ❌ Server load - constant HTTP requests
- ❌ Inefficient - wastes bandwidth

### SSE Event Format

**Progress events:**
```
event: progress
data: {"download_id": "dl-123", "bytes": 1024, "total": 10240, "speed_bps": 1000000, "eta_secs": 9}

event: progress
data: {"download_id": "dl-123", "bytes": 5120, "total": 10240, "speed_bps": 1200000, "eta_secs": 4}

event: complete
data: {"download_id": "dl-123", "bytes": 10240, "total": 10240, "duration_secs": 10}

event: error
data: {"download_id": "dl-123", "error": "Network timeout", "code": "DOWNLOAD_FAILED"}
```

---

## Architecture Principles

### 1. Two-Level SSE Streaming

**Stream #1: Hive → Queen**
- rbee-hive tracks download progress locally
- Exposes SSE endpoint: `GET /v2/downloads/{download_id}/stream`
- Queen subscribes to hive's SSE stream
- Hive sends progress events as download progresses

**Stream #2: Queen → Client**
- Queen relays progress from hive to client
- Exposes SSE endpoint: `GET /v1/downloads/{download_id}/stream`
- Client (rbee-keeper) subscribes to queen's SSE stream
- Queen forwards events from hive (with optional metadata)

### 2. Download State Management

**Download lifecycle:**
1. **Queued:** Download requested, waiting to start
2. **Starting:** Initializing HTTP connection
3. **Downloading:** Actively downloading bytes
4. **Complete:** Download finished successfully
5. **Failed:** Download failed (with retry support)
6. **Cancelled:** Download cancelled by user

### 3. Concurrent Download Support

**Multiple downloads:**
- Track multiple downloads simultaneously
- Each download has unique `download_id`
- Independent SSE streams per download
- Shared bandwidth management (optional)

---

## API Design

### Core Types

```rust
/// Download progress tracker
pub struct DownloadTracker {
    downloads: Arc<RwLock<HashMap<String, DownloadState>>>,
}

/// Download state
pub struct DownloadState {
    pub download_id: String,
    pub model_ref: String,
    pub status: DownloadStatus,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub speed_bps: u64,
    pub eta_secs: Option<u64>,
    pub started_at: SystemTime,
    pub completed_at: Option<SystemTime>,
    pub error: Option<String>,
}

/// Download status
pub enum DownloadStatus {
    Queued,
    Starting,
    Downloading,
    Complete,
    Failed,
    Cancelled,
}

/// Progress event (SSE payload)
#[derive(Serialize)]
pub struct ProgressEvent {
    pub download_id: String,
    pub bytes: u64,
    pub total: Option<u64>,
    pub speed_bps: u64,
    pub eta_secs: Option<u64>,
    pub percent: Option<f32>,
}
```

### Tracker API

```rust
impl DownloadTracker {
    /// Create new tracker
    pub fn new() -> Self;
    
    /// Start tracking a download
    pub fn start_download(&self, download_id: String, model_ref: String, total_bytes: Option<u64>);
    
    /// Update download progress
    pub fn update_progress(&self, download_id: &str, bytes_downloaded: u64);
    
    /// Mark download as complete
    pub fn complete_download(&self, download_id: &str);
    
    /// Mark download as failed
    pub fn fail_download(&self, download_id: &str, error: String);
    
    /// Cancel download
    pub fn cancel_download(&self, download_id: &str);
    
    /// Get download state
    pub fn get_download(&self, download_id: &str) -> Option<DownloadState>;
    
    /// List all active downloads
    pub fn list_active(&self) -> Vec<DownloadState>;
    
    /// Stream progress events (SSE)
    pub async fn stream_progress(&self, download_id: String) -> impl Stream<Item = ProgressEvent>;
}
```

---

## Usage Examples

### Basic Download Tracking

```rust
use rbee_hive_download_tracker::{DownloadTracker, DownloadStatus};

// Create tracker
let tracker = DownloadTracker::new();

// Start tracking download
tracker.start_download(
    "dl-123".to_string(),
    "hf:meta-llama/Llama-3-8B@main::file=model.gguf".to_string(),
    Some(8_000_000_000), // 8GB
);

// Update progress as download progresses
tracker.update_progress("dl-123", 1_000_000_000); // 1GB
tracker.update_progress("dl-123", 2_000_000_000); // 2GB
// ...

// Mark as complete
tracker.complete_download("dl-123");
```

### SSE Streaming (Hive Side)

```rust
use axum::{
    response::sse::{Event, Sse},
    extract::{Path, State},
};
use rbee_hive_download_tracker::DownloadTracker;

// SSE endpoint: GET /v2/downloads/:download_id/stream
async fn stream_download_progress(
    Path(download_id): Path<String>,
    State(tracker): State<Arc<DownloadTracker>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = tracker.stream_progress(download_id)
        .map(|progress| {
            Event::default()
                .event("progress")
                .json_data(progress)
        });
    
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
}
```

### SSE Streaming (Queen Side - Relay)

```rust
use axum::{
    response::sse::{Event, Sse},
    extract::{Path, State},
};

// SSE endpoint: GET /v1/downloads/:download_id/stream
async fn relay_download_progress(
    Path(download_id): Path<String>,
    State(queen_state): State<Arc<QueenState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Find which hive is downloading this model
    let hive_url = queen_state.find_hive_for_download(&download_id)?;
    
    // Subscribe to hive's SSE stream
    let hive_stream_url = format!("{}/v2/downloads/{}/stream", hive_url, download_id);
    let hive_events = subscribe_to_sse(hive_stream_url).await?;
    
    // Relay events to client (with optional metadata)
    let relay_stream = hive_events.map(|event| {
        // Optionally add queen metadata
        let mut data = event.data;
        data["orchestrator_id"] = "queen-1".into();
        
        Event::default()
            .event(event.event_type)
            .json_data(data)
    });
    
    Sse::new(relay_stream)
        .keep_alive(KeepAlive::default())
}
```

### Client Consumption (rbee-keeper)

```rust
use eventsource_client::{Client, SSE};

// Subscribe to download progress
let client = Client::new(
    "http://queen-rbee:8080/v1/downloads/dl-123/stream"
)?;

for event in client {
    match event {
        SSE::Event(e) if e.event_type == "progress" => {
            let progress: ProgressEvent = serde_json::from_str(&e.data)?;
            println!("Download progress: {}%", progress.percent.unwrap_or(0.0));
        }
        SSE::Event(e) if e.event_type == "complete" => {
            println!("Download complete!");
            break;
        }
        SSE::Event(e) if e.event_type == "error" => {
            eprintln!("Download failed: {}", e.data);
            break;
        }
        _ => {}
    }
}
```

---

## SSE Event Types

### 1. progress

**Sent periodically during download (e.g., every 1 second or every 10MB)**

```json
{
  "download_id": "dl-123",
  "bytes": 1024000000,
  "total": 8000000000,
  "speed_bps": 10485760,
  "eta_secs": 665,
  "percent": 12.8
}
```

### 2. complete

**Sent when download finishes successfully**

```json
{
  "download_id": "dl-123",
  "bytes": 8000000000,
  "total": 8000000000,
  "duration_secs": 762
}
```

### 3. error

**Sent when download fails**

```json
{
  "download_id": "dl-123",
  "error": "Network timeout after 30s",
  "code": "DOWNLOAD_TIMEOUT",
  "bytes": 1024000000,
  "total": 8000000000
}
```

### 4. cancelled

**Sent when download is cancelled**

```json
{
  "download_id": "dl-123",
  "bytes": 2048000000,
  "total": 8000000000
}
```

---

## Progress Calculation

### Speed Calculation

**Exponential moving average (EMA):**
```rust
// Smooth speed calculation to avoid jitter
let alpha = 0.3; // Smoothing factor
speed_bps = alpha * instant_speed + (1.0 - alpha) * previous_speed;
```

### ETA Calculation

**Based on current speed:**
```rust
let remaining_bytes = total_bytes - bytes_downloaded;
let eta_secs = remaining_bytes / speed_bps;
```

### Percent Calculation

**Only if total size is known:**
```rust
let percent = if let Some(total) = total_bytes {
    (bytes_downloaded as f32 / total as f32) * 100.0
} else {
    None // Unknown total size
};
```

---

## Dependencies

### Required

- **`tokio`**: Async runtime for streaming
- **`axum`**: HTTP server with SSE support
- **`serde`**: JSON serialization for events
- **`tracing`**: Structured logging

### Optional

- **`eventsource-client`**: SSE client (for queen relay)
- **`reqwest`**: HTTP client with streaming (for downloads)

---

## Implementation Status

### Phase 1: Core Tracking (M1)
- [ ] `DownloadTracker` implementation
- [ ] Progress state management
- [ ] Speed/ETA calculation
- [ ] Unit tests

### Phase 2: SSE Streaming (M1)
- [ ] SSE endpoint in rbee-hive
- [ ] Stream progress events
- [ ] Handle reconnection
- [ ] Integration tests

### Phase 3: Queen Relay (M2)
- [ ] SSE client in queen-rbee
- [ ] Subscribe to hive streams
- [ ] Relay events to clients
- [ ] Add orchestrator metadata

### Phase 4: Advanced Features (M3+)
- [ ] Bandwidth throttling
- [ ] Concurrent download limits
- [ ] Download history/cleanup
- [ ] Resume support (HTTP Range)

---

## Future: Shared Crate

**Currently:** `bin/rbee-hive-crates/download-tracker/`  
**Future:** `bin/shared-crates/download-tracker/`

**Why future shared:**
- Queen might track downloads from multiple hives
- Workers might download model shards
- Keeper might download CLI updates

**For now:** Hive-specific, but designed with sharing in mind.

---

## Related Crates

### Used By
- **`rbee-hive`**: Tracks model downloads

### Integrates With
- **`rbee-hive-crates/model-provisioner`**: Initiates downloads
- **`rbee-hive-crates/http-server`**: Exposes SSE endpoint
- **`queen-rbee-crates/http-server`**: Relays SSE streams

---

## Specification References

- **SYS-5.4.1**: SSE Streaming Protocol
- **SYS-5.7.x**: Multi-Modality Streaming Protocols
- **SYS-6.2.x**: Pool Manager (model provisioning)

See: `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`

---

## Team History

- **TEAM-135**: Scaffolding for new crate-based architecture
- **2025-10-19**: Documented SSE-based streaming architecture

---

**Next Steps:**
1. Implement `DownloadTracker` core functionality
2. Add SSE streaming endpoint in rbee-hive
3. Implement queen relay (subscribe to hive SSE)
4. Add client consumption in rbee-keeper
5. Test end-to-end SSE flow (hive → queen → client)
