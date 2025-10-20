//! Stream narration events from queen-rbee SSE endpoint.

use anyhow::Result;
use futures::StreamExt;

/// Stream narration events from queen-rbee and display them.
///
/// This subscribes to the /narration/stream endpoint and displays
/// all narration events from queen-rbee operations in real-time.
pub async fn stream_narration_to_stdout(sse_url: String) -> Result<()> {
    let client = reqwest::Client::new();
    let mut stream = client.get(&sse_url).send().await?.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Parse SSE format: "data: {json}\n\n"
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "keepalive" {
                    continue;
                }
                
                // Parse narration event JSON
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                    if let (Some(actor), Some(human)) = (
                        event.get("actor").and_then(|v| v.as_str()),
                        event.get("human").and_then(|v| v.as_str()),
                    ) {
                        eprintln!("[{}]\n  {}", actor, human);
                    }
                }
            }
        }
    }
    
    Ok(())
}
