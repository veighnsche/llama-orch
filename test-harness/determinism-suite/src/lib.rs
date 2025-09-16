//! Determinism suite helpers: parse SSE token streams and serialize snapshots.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamEvent {
    pub kind: String, // started | token | metrics | end | error
    pub data: String, // raw JSON fragment as string for simplicity
}

/// Generate a deterministic SSE transcript given engine id, seed and token count.
/// This is a pure function using a tiny LCG PRNG; suitable for Stage 4 tests without real engines.
pub fn generate_deterministic_sse(engine: &str, seed: u64, n_tokens: usize) -> String {
    fn salt_engine(e: &str) -> u64 {
        // Stable, trivial hash (wrapping) to avoid std hasher instability across runs.
        let mut acc: u64 = 0x9e37_79b9_7f4a_7c15;
        for (i, b) in e.as_bytes().iter().enumerate() {
            let mul = (i as u64 + 1).wrapping_mul(0x85eb_ca6b_27d4_eb2f);
            acc = acc.wrapping_add((*b as u64).wrapping_mul(mul));
            acc ^= acc.rotate_left(13);
        }
        acc
    }

    // LCG parameters (Numerical Recipes)
    let a: u64 = 1664525;
    let c: u64 = 1013904223;
    let m: u64 = 1u64 << 32;
    let mut state: u64 = seed ^ salt_engine(engine);

    let mut out = String::new();
    let predicted_start_ms = (state ^ 0x1234_5678).wrapping_rem_euclid(1000) + 100; // 100..1099
    out.push_str("event: started\n");
    out.push_str(&format!(
        "data: {{\"queue_position\":0,\"predicted_start_ms\":{}}}\n\n",
        predicted_start_ms
    ));

    for i in 0..n_tokens {
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        // Map to a small vocabulary; guarantee ASCII-safe tokens
        let idx = (state & 0xffff) as u16;
        let tok = format!("T{:04x}", idx);
        out.push_str("event: token\n");
        out.push_str(&format!("data: {{\"t\":\"{}\",\"i\":{}}}\n\n", tok, i));
    }

    let decode_ms = (state ^ 0xdead_beef).wrapping_rem_euclid(2000) + 10; // 10..2009
    out.push_str("event: end\n");
    out.push_str(&format!(
        "data: {{\"tokens_out\":{},\"decode_ms\":{}}}\n\n",
        n_tokens, decode_ms
    ));
    out
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
                // Parse the JSON object and extract the 't' field accurately.
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                    if let Some(t) = v.get("t").and_then(|x| x.as_str()) {
                        tokens.push(t.to_string());
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
