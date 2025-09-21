// Placeholder module for applet prompt/message.
use std::fs;
use std::io;
use std::path::Path;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Source {
    Text(String),
    Lines(Vec<String>),
    File(String),
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageIn {
    pub role: String,
    pub source: Source,
    pub dedent: bool,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub fn run(input: MessageIn) -> io::Result<Message> {
    let raw = match input.source {
        Source::Text(s) => s,
        Source::Lines(lines) => lines.join("\n"),
        Source::File(path) => {
            let bytes = fs::read(Path::new(&path))?;
            String::from_utf8_lossy(&bytes).to_string()
        }
    };
    let content = if input.dedent { dedent(&raw) } else { raw };
    Ok(Message { role: input.role, content })
}

fn dedent(s: &str) -> String {
    // Compute minimal leading whitespace across non-empty lines, then trim that amount.
    let lines: Vec<&str> = s.lines().collect();
    let mut min_ws: Option<usize> = None;
    for &line in &lines {
        if line.trim().is_empty() { continue; }
        let count = line.chars().take_while(|c| c.is_whitespace() && *c != '\n' && *c != '\r').count();
        min_ws = Some(match min_ws { Some(m) => m.min(count), None => count });
    }
    let n = min_ws.unwrap_or(0);
    lines
        .into_iter()
        .map(|line| {
            let mut chs = line.chars();
            let mut trimmed = String::new();
            for (i, c) in chs.by_ref().enumerate() {
                if i < n && c.is_whitespace() && c != '\n' && c != '\r' { continue; }
                trimmed.push(c);
                break;
            }
            // push rest of the iterator
            trimmed.push_str(chs.as_str());
            trimmed
        })
        .collect::<Vec<_>>()
        .join("\n")
}
