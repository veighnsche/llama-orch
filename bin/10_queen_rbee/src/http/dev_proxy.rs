//! Development proxy for Vite dev server
//!
//! TEAM-350: Proxy /dev/* to Vite dev server (port 7834) during development
//!
//! This allows rbee-keeper to load the queen UI in dev mode without rebuilding:
//! - Production: http://localhost:7833/ → Embedded static files
//! - Development: http://localhost:7833/dev → Proxy to http://localhost:7834
//!
//! See: PORT_CONFIGURATION.md for port mapping

use axum::{
    body::Body,
    extract::Request,
    http::{StatusCode, Uri},
    response::{IntoResponse, Response},
};

/// Proxy handler for /dev/* routes
///
/// TEAM-350: Forwards requests to Vite dev server at localhost:7834
/// Strips /dev prefix before forwarding
///
/// Example:
/// - Request: http://localhost:7833/dev/assets/main.js
/// - Proxied to: http://localhost:7834/assets/main.js
pub async fn dev_proxy_handler(uri: Uri, req: Request) -> impl IntoResponse {
    // Strip /dev prefix
    let path = uri.path().strip_prefix("/dev").unwrap_or(uri.path());
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();
    
    // Construct target URL for Vite dev server
    let vite_url = format!("http://localhost:7834{}{}", path, query);
    
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
