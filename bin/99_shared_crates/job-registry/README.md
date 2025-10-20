# job-registry

**Shared job registry for dual-call inference pattern**

Created by: TEAM-154  
Used by: Worker, Queen, Hive

---

## Purpose

Provides a generic job registry that tracks inference jobs between the POST (create) and GET (stream) calls in the dual-call pattern from `a_human_wrote_this.md`.

## Pattern

```
POST /v1/inference → { "job_id": "job-uuid", "sse_url": "/v1/inference/job-uuid/stream" }
GET /v1/inference/{job_id}/stream → SSE stream
```

## Usage

### Worker Example

```rust
use job_registry::{JobRegistry, JobState};
use crate::backend::request_queue::TokenResponse;

// Create registry with worker's token type
let registry: JobRegistry<TokenResponse> = JobRegistry::new();

// POST handler: Create job
let job_id = registry.create_job();

// Store token sender
let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
registry.set_token_sender(&job_id, tx);

// Return to client
CreateJobResponse {
    job_id,
    sse_url: format!("/v1/inference/{}/stream", job_id),
}

// GET handler: Stream results
let sender = registry.get_token_sender(&job_id).unwrap();
// Use sender to stream tokens...
```

### Queen Example

```rust
use job_registry::{JobRegistry, JobState};

// Queen might use a different token type
enum QueenToken {
    WorkerResponse(String),
    Aggregated(Vec<String>),
}

let registry: JobRegistry<QueenToken> = JobRegistry::new();
// Same API, different token type
```

## Features

- **Generic over token type** - Worker, Queen, and Hive can use their own token formats
- **Thread-safe** - Uses Arc<Mutex<>> internally
- **Server-generated IDs** - Uses UUID v4 for job IDs
- **State tracking** - Queued, Running, Completed, Failed
- **Cleanup support** - Remove jobs after completion

## API

### Core Methods

- `create_job()` - Create new job, returns job_id
- `get_job(job_id)` - Retrieve job by ID
- `update_state(job_id, state)` - Update job state
- `set_token_sender(job_id, sender)` - Store token sender
- `get_token_sender(job_id)` - Retrieve token sender
- `remove_job(job_id)` - Remove job (cleanup)
- `job_count()` - Get total job count
- `job_ids()` - Get all job IDs

## Dependencies

- `tokio` - For mpsc channels
- `uuid` - For job ID generation
- `chrono` - For timestamps

## Testing

```bash
cargo test -p job-registry
```

## License

GPL-3.0-or-later
