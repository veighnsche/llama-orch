#[derive(Debug, Clone)]
pub enum SseEvent {
    Started,
    Token { t: String, i: i32 },
    Metrics { json: String },
    End { tokens_out: i32, decode_ms: i32 },
}
