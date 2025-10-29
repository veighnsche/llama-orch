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
//! TEAM-XXX: Always serves embedded static files from ui/app/dist/ at ROOT (/)
//! No dev proxy - UI changes require queen rebuild (cargo build -p queen-rbee)
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

/// Create router for static file serving
///
/// TEAM-XXX: Always serve embedded static files from dist/
/// No dev proxy - UI changes require queen rebuild
/// Simpler workflow, no port confusion
pub fn create_static_router() -> Router {
    // Always serve embedded static files, even in debug mode
    Router::new().fallback(static_handler)
}

/// Handler for serving embedded static files
///
/// TEAM-XXX: Used in both debug and release modes
/// UI changes require queen rebuild
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
