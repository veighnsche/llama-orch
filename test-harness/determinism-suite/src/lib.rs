//! Determinism suite helpers: parse SSE token streams and serialize snapshots.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamEvent {
    pub kind: String, // started | token | metrics | end | error
    pub data: String, // raw JSON fragment as string for simplicity
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Snapshot {
    pub engine: String,
    pub seed: u64,
    pub tokens_first_32: Vec<String>,
    pub total_tokens: usize,
    pub first_token_ms: Option<u64>,
    pub decode_ms: Option<u64>,
}

pub fn parse_sse_transcript(s: &str) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let mut current_kind: Option<String> = None;
    for line in s.lines() {
        let line = line.trim_end();
        if line.starts_with("event:") {
            let kind = line.trim_start_matches("event:").trim().to_string();
            current_kind = Some(kind);
        } else if line.starts_with("data:") {
            let data = line.trim_start_matches("data:").trim().to_string();
            let kind = current_kind.clone().unwrap_or_else(|| "token".to_string());
            events.push(StreamEvent { kind, data });
        }
    }
    events
}

pub fn snapshot_from_events(engine: &str, seed: u64, events: &[StreamEvent]) -> Snapshot {
    let mut tokens = Vec::new();
    let mut first_token_ms = None;
    let mut decode_ms = None;
    for ev in events {
        match ev.kind.as_str() {
            "token" => {
                // very light-weight extract of t field
                if let Some(i) = ev.data.find("\"t\"") {
                    // naive parse: look for next colon and quoted token
                    let rest = &ev.data[i..];
                    if let Some(start) = rest.find('"') {
                        let rest2 = &rest[start + 1..];
                        if let Some(end) = rest2.find('"') {
                            let tok = &rest2[..end];
                            tokens.push(tok.to_string());
                        }
                    }
                }
            }
            "end" => {
                // naive parse of decode_ms
                if let Some(i) = ev.data.find("decode_ms") {
                    let rest = &ev.data[i..];
                    if let Some(colon) = rest.find(':') {
                        let num = rest[colon + 1..].trim_matches(|c: char| !c.is_ascii_digit());
                        if let Ok(v) = num.parse::<u64>() {
                            decode_ms = Some(v);
                        }
                    }
                }
            }
            "metrics" => {
                // ignore for now
            }
            "started" => {
                // naive parse of predicted_start_ms as first_token_ms proxy
                if let Some(i) = ev.data.find("predicted_start_ms") {
                    let rest = &ev.data[i..];
                    if let Some(colon) = rest.find(':') {
                        let num = rest[colon + 1..].trim_matches(|c: char| !c.is_ascii_digit());
                        if let Ok(v) = num.parse::<u64>() {
                            first_token_ms = Some(v);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Snapshot {
        engine: engine.to_string(),
        seed,
        tokens_first_32: tokens.into_iter().take(32).collect(),
        total_tokens: events.iter().filter(|e| e.kind == "token").count(),
        first_token_ms,
        decode_ms,
    }
}
