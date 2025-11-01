//! Development proxy for Vite dev server
//!
//! TEAM-374: Proxy /dev/* to Vite dev server (port 7836) during development
//! TEAM-378: Support remote dev servers by extracting hostname from queen_url
//!
//! This allows rbee-keeper to load the hive UI in dev mode without rebuilding:
//! - Production: http://localhost:7835/ → Embedded static files
//! - Development (localhost): http://localhost:7835/dev → Proxy to http://localhost:7836
//! - Development (remote): http://workstation:7835/dev → Proxy to http://devbox:7836
//!
//! See: PORT_CONFIGURATION.md for port mapping

use axum::{
    body::Body,
    extract::{Request, State},
    http::{StatusCode, Uri},
    response::{IntoResponse, Response},
};
use std::sync::Arc;

/// State for dev proxy handler
/// TEAM-378: Contains Vite dev server URL (derived from queen_url)
#[derive(Clone)]
pub struct DevProxyState {
    pub vite_url: Arc<String>,
}

/// Proxy handler for /dev/* routes
///
/// TEAM-378: Forwards requests to Vite dev server (hostname from queen_url, port 7836)
/// Strips /dev prefix before forwarding
///
/// Example:
/// - Request: http://workstation:7835/dev/assets/main.js
/// - Proxied to: http://devbox:7836/assets/main.js (where devbox is from queen_url)
pub async fn dev_proxy_handler(
    State(state): State<DevProxyState>,
    uri: Uri,
    req: Request,
) -> impl IntoResponse {
    // Strip /dev prefix
    let path = uri.path().strip_prefix("/dev").unwrap_or(uri.path());
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();
    
    // Construct target URL for Vite dev server (using hostname from state)
    let vite_url = format!("{}{}{}", state.vite_url, path, query);
    
    eprintln!("[DEV PROXY] {} → {}", uri, vite_url);
    
    // Create HTTP client
    let client = reqwest::Client::new();
    
    // Forward the request
    let method = req.method().clone();
    let headers = req.headers().clone();
    let body = axum::body::to_bytes(req.into_body(), usize::MAX).await.unwrap_or_default();
    
    match client
        .request(method, &vite_url)
        .headers(headers)
        .body(body.to_vec())
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await.unwrap_or_default();
            
            let mut builder = Response::builder().status(status);
            
            // Copy headers from Vite response
            for (key, value) in headers.iter() {
                builder = builder.header(key, value);
            }
            
            builder.body(Body::from(body)).unwrap()
        }
        Err(e) => {
            eprintln!("[DEV PROXY] Error: {}", e);
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Dev proxy error: {}", e)))
                .unwrap()
        }
    }
}
