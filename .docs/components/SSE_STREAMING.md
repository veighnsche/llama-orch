# Component: SSE Streaming (Server-Sent Events)

**Location:** Multiple locations  
**Type:** Real-time communication protocol  
**Language:** Rust  
**Created by:** TEAM-034  
**Status:** ✅ IMPLEMENTED

## Overview

Server-Sent Events (SSE) for real-time streaming of download progress and inference results. One-way server-to-client push over HTTP.

## Use Cases

### 1. Model Download Progress (TEAM-034)
**Endpoint:** `GET /v1/models/download/progress`

```http
GET /v1/models/download/progress HTTP/1.1
Accept: text/event-stream

HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

event: started
data: {"model_ref":"TinyLlama"}

event: progress
data: {"model_ref":"TinyLlama","bytes":1048576,"total":10485760}

event: complete
data: {"model_ref":"TinyLlama"}
```

### 2. Inference Streaming
**Endpoint:** `POST /v1/infer` (with `stream: true`)

```http
POST /v1/infer HTTP/1.1
Content-Type: application/json

{"prompt": "Hello", "stream": true}

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"token":"Hello"}

data: {"token":" world"}

data: {"token":"!"}

data: [DONE]
```

## Implementation

### Download Progress (rbee-hive)

```rust
// bin/rbee-hive/src/download_tracker.rs
pub struct DownloadTracker {
    subscribers: Arc<RwLock<Vec<Sender<DownloadEvent>>>>,
}

pub async fn subscribe(&self) -> Receiver<DownloadEvent> {
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    self.subscribers.write().await.push(tx);
    rx
}

pub async fn send_progress(&self, id: &str, bytes: u64, total: u64) {
    let event = DownloadEvent::Progress { bytes, total };
    for subscriber in self.subscribers.read().await.iter() {
        let _ = subscriber.send(event.clone()).await;
    }
}
```

```rust
// bin/rbee-hive/src/http/models.rs
pub async fn handle_download_progress(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut rx = state.download_tracker.subscribe().await;
    
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(Event::default()
                .event(event.event_type())
                .data(serde_json::to_string(&event).unwrap()));
        }
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

### Inference Streaming (llm-worker-rbee)

```rust
// bin/llm-worker-rbee/src/http/inference.rs
pub async fn handle_inference_stream(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    
    // Spawn inference task
    tokio::spawn(async move {
        for token in generate_tokens(&req.prompt) {
            let _ = tx.send(token).await;
        }
    });
    
    // Stream tokens
    let stream = async_stream::stream! {
        while let Some(token) = rx.recv().await {
            yield Ok(Event::default()
                .data(serde_json::to_string(&token).unwrap()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

## Client Usage

### JavaScript
```javascript
// Download progress
const eventSource = new EventSource('/v1/models/download/progress');

eventSource.addEventListener('progress', (event) => {
    const data = JSON.parse(event.data);
    console.log(`Progress: ${data.bytes}/${data.total}`);
});

eventSource.addEventListener('complete', (event) => {
    console.log('Download complete!');
    eventSource.close();
});

// Inference streaming
const eventSource = new EventSource('/v1/infer?prompt=Hello&stream=true');

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    const token = JSON.parse(event.data);
    console.log(token.token);
};
```

### Rust Client
```rust
use eventsource_client::Client;

let client = Client::for_url("http://localhost:8081/v1/models/download/progress")?
    .build();

client.stream()
    .for_each(|event| async move {
        match event {
            Ok(Event::Message(msg)) => {
                println!("Event: {}", msg.data);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    })
    .await;
```

## Maturity Assessment

**Status:** ✅ **PRODUCTION READY**

**Strengths:**
- ✅ Standard SSE protocol
- ✅ Multiple subscribers supported
- ✅ Automatic reconnection (browser)
- ✅ Keep-alive support
- ✅ Error handling

**Limitations:**
- ⚠️ One-way only (server → client)
- ⚠️ No binary data support
- ⚠️ No backpressure handling
- ⚠️ No subscriber limits (memory leak risk)
- ⚠️ No authentication/authorization

**Recommended Improvements:**
1. Add subscriber limits
2. Add authentication
3. Add backpressure handling
4. Add subscriber timeout
5. Add metrics (active subscribers)

## Related Components

- **Download Tracker** - Broadcasts download events
- **Inference Engine** - Streams tokens
- **HTTP API** - SSE endpoints

---

**Created by:** TEAM-034  
**Last Updated:** 2025-10-18  
**Maturity:** ✅ Production Ready
