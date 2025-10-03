//! Worker startup and pool manager callback

use anyhow::Result;
use serde::Serialize;

#[derive(Serialize)]
struct ReadyCallback {
    worker_id: String,
    vram_bytes: u64,
    uri: String,
}

/// Call back to pool manager to report worker ready
pub async fn callback_ready(
    callback_url: &str,
    worker_id: &str,
    vram_bytes: u64,
    port: u16,
) -> Result<()> {
    let uri = format!("http://localhost:{}", port);
    
    let payload = ReadyCallback {
        worker_id: worker_id.to_string(),
        vram_bytes,
        uri,
    };
    
    tracing::info!(
        callback_url = %callback_url,
        worker_id = %worker_id,
        vram_bytes,
        "Calling back to pool manager"
    );
    
    let client = reqwest::Client::new();
    let response = client
        .post(callback_url)
        .json(&payload)
        .send()
        .await?;
    
    if !response.status().is_success() {
        anyhow::bail!(
            "Pool manager callback failed: {}",
            response.status()
        );
    }
    
    tracing::info!("Pool manager callback successful");
    
    Ok(())
}
