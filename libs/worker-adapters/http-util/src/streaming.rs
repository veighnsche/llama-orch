use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Started(Value),
    Token { i: usize, t: String },
    Metrics(Value),
    End(Value),
}

/// Decode a simple SSE-like body into events, invoking the sink for each.
pub fn stream_decode<S: AsRef<str>, F: FnMut(StreamEvent)>(body: S, mut sink: F) -> anyhow::Result<()> {
    let mut current_event: Option<String> = None;
    for line in body.as_ref().lines() {
        let line = line.trim_end();
        if line.is_empty() { continue; }
        if let Some(rest) = line.strip_prefix("event: ") {
            current_event = Some(rest.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let ev = current_event.clone().unwrap_or_default();
            let v: Value = serde_json::from_str(data.trim()).unwrap_or(Value::Null);
            match ev.as_str() {
                "started" => sink(StreamEvent::Started(v)),
                "token" => {
                    let i = v.get("i").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let t = v.get("t").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    sink(StreamEvent::Token { i, t });
                }
                "metrics" => sink(StreamEvent::Metrics(v)),
                "end" => sink(StreamEvent::End(v)),
                _ => {}
            }
        }
    }
    Ok(())
}
