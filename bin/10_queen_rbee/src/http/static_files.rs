//! Static file serving for web UI
//!
//! ============================================================================
//! CRITICAL REQUIREMENT: UI MUST BE SERVED AT ROOT PATH (/)
//! ============================================================================
//! The UI is served at http://localhost:7833/ NOT http://localhost:7833/ui
//! 
//! This is a HARD REQUIREMENT. DO NOT CHANGE THIS.
//! DO NOT nest under /ui. DO NOT add prefixes.
//! 
//! Development: Proxies ROOT (/) to Vite dev server (port 7834)
//! Production: Serves embedded static files from ui/app/dist/ at ROOT (/)
//!
//! API routes (/health, /v1/*) take priority via router merge order.
//! Everything else falls through to UI (SPA routing).
//! ============================================================================
//!
//! See: PORT_CONFIGURATION.md for port mapping

use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    Router,
};
use rust_embed::RustEmbed;

/// Embedded static assets from queen-rbee UI production build
///
/// TEAM-295: Files are embedded at compile time from ui/app/dist/
/// This allows the binary to be distributed as a single executable
/// The build.rs script ensures UI is built before Rust compilation
#[derive(RustEmbed)]
#[folder = "ui/app/dist/"]
struct Assets;

/// Create static file serving router
///
/// TEAM-341: CRITICAL FIX - UI is served at ROOT (/), not /ui
/// 
/// The UI MUST be accessible at http://localhost:7833/ (root path).
/// This is a fallback router that catches all non-API routes.
/// 
/// Development: Proxies to Vite dev server at http://localhost:7834
/// Production: Serves embedded static files
///
/// Mode detection: Checks if compiled with --release flag
pub fn create_static_router() -> Router {
    #[cfg(debug_assertions)]
    {
        // TEAM-341: Development mode - Proxy ROOT to Vite dev server
        // NO /ui prefix! UI is at root path.
        // NOTE: Vite HMR WebSockets won't work through proxy - connect directly to :7834
        Router::new().fallback(dev_proxy_handler)
    }
    
    #[cfg(not(debug_assertions))]
    {
        // TEAM-341: Production mode - Serve embedded static files at ROOT
        // NO /ui prefix! UI is at root path.
        Router::new().fallback(static_handler)
    }
}

/// Development mode: Proxy requests to Vite dev server
///
/// TEAM-341: Proxies ROOT path to Vite (no /ui prefix stripping needed)
#[cfg(debug_assertions)]
async fn dev_proxy_handler(uri: Uri, req: axum::extract::Request) -> impl IntoResponse {
    use reqwest::Client;
    
    // TEAM-341: UI is at root, so pass path as-is to Vite
    let path = uri.path();
    let query = uri.query().unwrap_or("");
    let vite_url = if query.is_empty() {
        format!("http://localhost:7834{}", path)
    } else {
        format!("http://localhost:7834{}?{}", path, query)
    };
    
    // TEAM-341: Check if this is a WebSocket upgrade (Vite HMR)
    if req.headers().get("upgrade").and_then(|v| v.to_str().ok()) == Some("websocket") {
        // Return 404 for WebSocket - Vite HMR won't work through proxy
        // User should connect directly to Vite on port 7834
        eprintln!("[DEV PROXY] WebSocket request blocked: {}", vite_url);
        eprintln!("[DEV PROXY] Vite HMR: Use http://localhost:7834 directly for hot reload");
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("WebSocket not supported through proxy. Use Vite dev server directly on port 7834."))
            .unwrap();
    }
    
    eprintln!("[DEV PROXY] Request: {} -> {}", path, vite_url);
    
    match Client::new().get(&vite_url).send().await {
        Ok(response) => {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await.unwrap_or_default();
            
            let mut builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                builder = builder.header(key, value);
            }
            
            // TEAM-341: Fix WASM MIME type for ES module imports
            // Browser requires application/wasm for WASM module imports
            if path.ends_with(".wasm") {
                builder = builder.header(header::CONTENT_TYPE, "application/wasm");
            }
            
            builder.body(Body::from(body)).unwrap()
        }
        Err(e) => {
            eprintln!("[DEV] Failed to proxy to Vite dev server: {}", e);
            eprintln!("[DEV] Make sure Vite is running: cd bin/10_queen_rbee/ui/app && npm run dev");
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!(
                    "Dev server not available. Start it with: cd bin/10_queen_rbee/ui/app && npm run dev\n\nError: {}",
                    e
                )))
                .unwrap()
        }
    }
}

/// Production mode: Handler for serving embedded static files
///
/// TEAM-341: This should NEVER be called in debug mode!
/// Debug mode uses dev_proxy_handler instead.
#[cfg(not(debug_assertions))]
async fn static_handler(uri: Uri) -> impl IntoResponse {
    eprintln!("[STATIC] Serving: {}", uri.path());
    let path = uri.path().trim_start_matches('/');

    // Try to serve the requested file
    if let Some(content) = Assets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        eprintln!("[STATIC] Found file: {} (mime: {})", path, mime);

        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(content.data))
            .unwrap();
    }

    // Fallback to index.html for SPA routing (all non-asset paths)
    eprintln!("[STATIC] File not found, serving index.html for SPA routing");
    if let Some(content) = Assets::get("index.html") {
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html")
            .body(Body::from(content.data))
            .unwrap();
    }

    // If even index.html is missing, return 404
    eprintln!("[STATIC] ERROR: index.html not found in embedded assets!");
    Response::builder().status(StatusCode::NOT_FOUND).body(Body::from("404 - Not Found")).unwrap()
}
