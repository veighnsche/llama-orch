//! Static file serving for web UI
//!
//! TEAM-293: Serves production build of web-ui at root path
//! Files are embedded in the binary at compile time for single-executable distribution

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
/// Serves the embedded production build of web-ui
/// Works with packaged executables (Windows, Mac, Linux, AUR, etc.)
pub fn create_static_router() -> Router {
    Router::new().fallback(static_handler)
}

/// Handler for serving embedded static files
async fn static_handler(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    // Try to serve the requested file
    if let Some(content) = Assets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();

        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(content.data))
            .unwrap();
    }

    // Fallback to index.html for SPA routing (all non-asset paths)
    if let Some(content) = Assets::get("index.html") {
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html")
            .body(Body::from(content.data))
            .unwrap();
    }

    // If even index.html is missing, return 404
    Response::builder().status(StatusCode::NOT_FOUND).body(Body::from("404 - Not Found")).unwrap()
}
