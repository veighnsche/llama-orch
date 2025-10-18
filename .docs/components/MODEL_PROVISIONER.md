# Component: Model Provisioner

**Location:** `bin/rbee-hive/src/provisioner/`  
**Type:** Download & management system  
**Language:** Rust  
**Created by:** TEAM-029  
**Status:** ✅ IMPLEMENTED

## Overview

Downloads GGUF models from HuggingFace, tracks progress via SSE, and registers in Model Catalog. Handles concurrent downloads with progress streaming.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Model Provisioner                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Downloader (download.rs)                         │  │
│  │  - Finds .gguf files on HuggingFace              │  │
│  │  - Downloads with progress tracking              │  │
│  │  - Stores in .rbee/models/                       │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Download Tracker (TEAM-034)                      │  │
│  │  - Tracks active downloads                       │  │
│  │  - Broadcasts progress via SSE                   │  │
│  │  - Multiple subscribers supported                │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Model Catalog                                    │  │
│  │  - Registers completed downloads                 │  │
│  │  - Tracks local paths                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Downloader (`provisioner/download.rs`)

**Purpose:** Download models from HuggingFace

```rust
pub struct ModelProvisioner {
    base_dir: PathBuf,              // .rbee/models/
    download_tracker: Arc<DownloadTracker>,
}

impl ModelProvisioner {
    // Download model from HuggingFace
    pub async fn download_model(
        &self,
        reference: &str,
        provider: &str,
    ) -> Result<PathBuf>
    
    // Find local model (check catalog first)
    pub async fn find_local_model(
        &self,
        model_ref: &str,
    ) -> Result<Option<PathBuf>>
    
    // Get model size
    pub fn get_model_size(&self, path: &Path) -> Result<u64>
}
```

**Download Flow:**
```rust
// 1. Parse model reference
let (provider, reference) = parse_model_ref("hf:TheBloke/TinyLlama")?;

// 2. Find .gguf file on HuggingFace
let gguf_url = find_gguf_file(&reference).await?;

// 3. Start download with progress tracking
let download_id = tracker.start_download(&reference).await;

// 4. Download file with progress updates
let mut response = reqwest::get(&gguf_url).await?;
let total_size = response.content_length().unwrap_or(0);

while let Some(chunk) = response.chunk().await? {
    file.write_all(&chunk)?;
    downloaded += chunk.len() as u64;
    
    // Broadcast progress
    tracker.send_progress(download_id, downloaded, total_size).await;
}

// 5. Complete download
tracker.complete_download(download_id).await;

// 6. Return local path
Ok(local_path)
```

### 2. Download Tracker (`download_tracker.rs` - TEAM-034)

**Purpose:** Track and broadcast download progress via SSE

```rust
pub struct DownloadTracker {
    active_downloads: Arc<RwLock<HashMap<String, DownloadState>>>,
    subscribers: Arc<RwLock<Vec<Sender<DownloadEvent>>>>,
}

pub struct DownloadState {
    pub model_ref: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub status: DownloadStatus,
}

pub enum DownloadEvent {
    Started { model_ref: String },
    Progress { model_ref: String, bytes: u64, total: u64 },
    Complete { model_ref: String },
    Error { model_ref: String, error: String },
}
```

**API:**
```rust
// Start tracking download
pub async fn start_download(&self, model_ref: &str) -> String

// Send progress update
pub async fn send_progress(&self, id: &str, bytes: u64, total: u64)

// Complete download
pub async fn complete_download(&self, id: &str)

// Subscribe to events (SSE)
pub async fn subscribe(&self) -> Receiver<DownloadEvent>

// List active downloads
pub async fn list_active(&self) -> Vec<DownloadState>
```

### 3. Catalog Integration (`provisioner/catalog.rs`)

**Purpose:** Check catalog before downloading

```rust
pub async fn find_local_model(&self, model_ref: &str) -> Result<Option<PathBuf>> {
    let (provider, reference) = parse_model_ref(model_ref)?;
    
    // Check catalog first
    if let Some(model) = self.catalog.find_model(&reference, &provider).await? {
        let path = PathBuf::from(model.local_path);
        if path.exists() {
            return Ok(Some(path));
        }
    }
    
    Ok(None)
}
```

## HTTP API

### Download Model
```http
POST /v1/models/download
Content-Type: application/json

{
    "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
}

Response:
{
    "download_id": "abc123",
    "status": "started"
}
```

### Progress Stream (SSE)
```http
GET /v1/models/download/progress
Accept: text/event-stream

Response (SSE stream):
event: started
data: {"model_ref": "TinyLlama"}

event: progress
data: {"model_ref": "TinyLlama", "bytes": 1048576, "total": 10485760}

event: progress
data: {"model_ref": "TinyLlama", "bytes": 2097152, "total": 10485760}

event: complete
data: {"model_ref": "TinyLlama"}
```

## File Organization

```
.rbee/models/
├── catalog.db                          # SQLite catalog
├── hf/                                 # HuggingFace models
│   ├── TheBloke/
│   │   └── TinyLlama-1.1B-Chat-v1.0-GGUF/
│   │       └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
│   └── meta-llama/
│       └── Llama-2-7b-chat-GGUF/
│           └── llama-2-7b-chat.Q4_K_M.gguf
└── local/                              # Local models
    └── custom-model.gguf
```

## Integration Points

### Worker Spawn
```rust
// bin/rbee-hive/src/http/workers.rs
let model_path = match provisioner.find_local_model(&request.model_ref).await? {
    Some(path) => path,
    None => {
        // Download if not found
        provisioner.download_model(&reference, &provider).await?
    }
};

// Spawn worker with model_path
spawn_worker(&model_path).await?;
```

### Frontend Progress Display
```javascript
// Subscribe to SSE stream
const eventSource = new EventSource('/v1/models/download/progress');

eventSource.addEventListener('progress', (event) => {
    const data = JSON.parse(event.data);
    updateProgressBar(data.bytes, data.total);
});

eventSource.addEventListener('complete', (event) => {
    showSuccess('Download complete!');
});
```

## Maturity Assessment

**Status:** ✅ **PRODUCTION READY**

**Strengths:**
- ✅ HuggingFace integration
- ✅ Progress tracking (TEAM-034)
- ✅ SSE streaming (TEAM-034)
- ✅ Catalog integration
- ✅ Concurrent download support
- ✅ Error handling
- ✅ File organization

**Limitations:**
- ⚠️ Only supports HuggingFace (no other providers)
- ⚠️ No resume capability (restart from beginning)
- ⚠️ No checksum verification
- ⚠️ No bandwidth limiting
- ⚠️ No parallel chunk downloads
- ⚠️ No model deletion API
- ⚠️ No disk space checks before download

**Recommended Improvements:**
1. Add resume capability (Range requests)
2. Add checksum verification (SHA256)
3. Add bandwidth limiting
4. Add parallel chunk downloads
5. Add disk space checks
6. Add model deletion endpoint
7. Add support for other providers (local, S3, etc.)
8. Add download queue (limit concurrent downloads)

## Testing

```bash
# Unit tests
cargo test -p rbee-hive provisioner

# Integration test
# 1. Start rbee-hive
cargo run -p rbee-hive -- daemon &

# 2. Download model
curl -X POST http://localhost:8081/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"}'

# 3. Watch progress (SSE)
curl -N http://localhost:8081/v1/models/download/progress
```

## Related Components

- **Model Catalog** - Registers downloaded models
- **Download Tracker** - Progress tracking (TEAM-034)
- **Worker Spawn** - Uses provisioned models
- **HTTP API** (`http/models.rs`) - Download endpoints

---

**Created by:** TEAM-029  
**Enhanced by:** TEAM-034 (SSE streaming)  
**Last Updated:** 2025-10-18  
**Maturity:** ✅ Production Ready (with noted limitations)
