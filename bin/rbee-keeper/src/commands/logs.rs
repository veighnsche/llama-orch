//! Log streaming command
//!
//! Created by: TEAM-046

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;

/// Handle logs command
///
/// TEAM-046: Stream logs from remote node via queen-rbee
pub async fn handle(node: String, follow: bool) -> Result<()> {
    println!("{}", format!("=== Logs from {} ===", node).cyan().bold());
    println!();

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
