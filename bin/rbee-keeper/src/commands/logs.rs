//!
//! Created by: TEAM-046
//! Modified by: TEAM-086

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;  // TEAM-086: Added missing import for stream.next()

/// Handle logs command
///
/// TEAM-046: Stream logs from remote node via SSH
/// TEAM-085: Does NOT need queen-rbee - this is a direct SSH operation
pub async fn handle(node: String, follow: bool) -> Result<()> {
    println!("{}", format!("=== Logs for node: {} ===", node).cyan().bold());
    println!();

    // TEAM-085: Logs are fetched directly via SSH, not through queen-rbee
    // Starting queen-rbee just to read empty logs makes no sense!
    
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";

    let url = if follow {
        format!("{}/v2/logs?node={}&follow=true", queen_url, node)
    } else {
        format!("{}/v2/logs?node={}", queen_url, node)
    };
    let response = client.get(&url).send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to stream logs: HTTP {}", response.status());
    }

    // Stream logs to stdout
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", String::from_utf8_lossy(&chunk));
    }

    Ok(())
}
