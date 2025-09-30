//! llama.cpp HTTP adapter — MVP HTTP-backed implementation with buffered SSE decode.
//! TODOs are annotated for health/props wiring, incremental streaming, cancel, and version capture.

use contracts_api_types as api;
use async_stream::stream;
use futures::StreamExt;
use tracing::{debug, info, warn};
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};
use worker_adapters_http_util as http_util;

#[derive(Debug, Clone)]
pub struct LlamaCppHttpAdapter {
    pub base_url: String,
}

impl LlamaCppHttpAdapter {
    pub fn new(base_url: impl Into<String>) -> Self { Self { base_url: base_url.into() } }
}

impl WorkerAdapter for LlamaCppHttpAdapter {
    fn health(&self) -> Result<WorkerHealth, WorkerError> {
        // TODO(OwnerB-LLAMACPP-HEALTH): Implement real GET {base}/health mapping.
        // Why: MVP returns Ready always; orchestrator should target only Ready replicas once
        // health is wired to the engine. Map 200 -> ready:true, 503 -> ready:false.
        // See: libs/worker-adapters/llamacpp-http/implementation-hints-llamacpp.md
        Ok(WorkerHealth { live: true, ready: true })
    }

    fn props(&self) -> Result<WorkerProps, WorkerError> {
        // TODO(OwnerB-LLAMACPP-PROPS): Query {base}/props and/or {base}/slots to report slots_total/free.
        // Why: Orchestrator placement benefits from accurate slot availability. MVP leaves None
        // to avoid misleading placement until wired.
        Ok(WorkerProps { slots_total: None, slots_free: None })
    }

    fn submit(&self, req: api::TaskRequest) -> Result<TokenStream, WorkerError> {
        let base = self.base_url.clone();
        let engine = format!("{:?}", req.engine).to_lowercase();
        let task_id = req.task_id.clone();

        let s = stream! {
            // Build upstream payload (native llama.cpp server /completion shape)
            let payload = serde_json::json!({
                "prompt": req.prompt.unwrap_or_default(),
                // llama.cpp uses n_predict; accept both in stub; real server accepts either in versions
                "n_predict": req.max_tokens,
                "max_tokens": req.max_tokens,
                "seed": req.seed.unwrap_or(0),
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": true
            });

            let url = format!("{}/completion", base.trim_end_matches('/'));
            info!(target: "llamacpp-adapter", task_id=%task_id, engine=%engine, url=%url, "submit begin");
            let client = http_util::client().clone();
            let resp = match client.post(&url)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(&payload)
                .send()
                .await {
                Ok(r) => r,
                Err(e) => {
                    let msg = http_util::redact_secrets(&format!("request error: {}", e));
                    warn!(target: "llamacpp-adapter", task_id=%task_id, "{}", msg);
                    yield Err(WorkerError::Adapter(msg));
                    return;
                }
            };

            let status = resp.status();
            if !status.is_success() {
                let code = status.as_u16();
                let body_text = match resp.text().await { Ok(t) => t, Err(_) => String::new() };
                let red = http_util::redact_secrets(&body_text);
                let classify = if http_util::is_non_retriable_status(code) {
                    "non-retriable"
                } else if http_util::is_retriable_status(code) {
                    "retriable"
                } else { "unknown" };
                warn!(target: "llamacpp-adapter", task_id=%task_id, status=%code, class=%classify, "upstream error: {}", red);
                yield Err(WorkerError::Adapter(format!("upstream {}: {}", code, red)));
                return;
            }

            // TODO(OwnerB-LLAMACPP-STREAM): Switch to incremental streaming decode (do not buffer full body),
            // implement backpressure, and cancel-on-disconnect.
            // Why: MVP buffers entire SSE transcript to simplify correctness; production must stream incrementally
            // to reduce latency and memory use, and promptly react to client cancel.
            // See: .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md
            let body = match resp.text().await {
                Ok(b) => b,
                Err(e) => {
                    let msg = http_util::redact_secrets(&format!("read body error: {}", e));
                    yield Err(WorkerError::Adapter(msg));
                    return;
                }
            };
            debug!(target: "llamacpp-adapter", task_id=%task_id, bytes=body.len(), "received SSE body");

            // Decode into a buffer first (MVP); preserve order and index checks
            let mut buf: Vec<http_util::StreamEvent> = Vec::new();
            if let Err(e) = http_util::stream_decode(&body, |ev| buf.push(ev)) {
                yield Err(WorkerError::Adapter(e.to_string()));
                return;
            }
            let mut idx_last: Option<usize> = None;
            let mut started_seen = false;
            for ev in buf {
                match ev {
                    http_util::StreamEvent::Started(v) => {
                        started_seen = true;
                        yield Ok(TokenEvent { kind: "started".into(), data: v });
                    }
                    http_util::StreamEvent::Token { i, t } => {
                        if let Some(prev) = idx_last { if i <= prev { warn!(target: "llamacpp-adapter", task_id=%task_id, "non-monotonic token index: {} <= {}", i, prev); } }
                        idx_last = Some(i);
                        let data = serde_json::json!({"i": i, "t": t});
                        yield Ok(TokenEvent { kind: "token".into(), data });
                    }
                    http_util::StreamEvent::Metrics(v) => {
                        yield Ok(TokenEvent { kind: "metrics".into(), data: v });
                    }
                    http_util::StreamEvent::End(v) => {
                        yield Ok(TokenEvent { kind: "end".into(), data: v });
                    }
                }
            }
            if !started_seen { yield Err(WorkerError::Adapter("missing started event".into())); }
        };

        Ok(s.boxed())
    }

    fn cancel(&self, _task_id: &str) -> Result<(), WorkerError> {
        // TODO(OwnerB-LLAMACPP-CANCEL): Implement explicit cancel path when supported by upstream (or
        // ensure stream drop triggers server-side cancellation promptly).
        // Why: MVP relies on drop semantics; production must ensure no tokens after cancel and slot is freed.
        // See: .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md
        Ok(())
    }

    fn engine_version(&self) -> Result<String, WorkerError> {
        // TODO(OwnerB-LLAMACPP-VERSION): Capture engine version/build info from server (banner or build endpoint)
        // and include sampler profile version if available. MVP returns a static marker string.
        Ok("llamacpp-http-mvp".into())
    }
}
