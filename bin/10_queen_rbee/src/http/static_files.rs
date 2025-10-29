//! Static file serving for web UI
//!
//! Development: Proxies /ui to Vite dev server (port 7834)
//! Production: Serves embedded static files from ui/app/dist/
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
/// Development: Proxies to Vite dev server at http://localhost:7834
/// Production: Serves embedded static files
///
/// Mode detection: Checks if compiled with --release flag
pub fn create_static_router() -> Router {
    #[cfg(debug_assertions)]
    {
        // Development mode: Proxy to Vite dev server
        Router::new().nest("/ui", Router::new().fallback(dev_proxy_handler))
    }
    
    #[cfg(not(debug_assertions))]
    {
        // Production mode: Serve embedded static files
        Router::new().nest("/ui", Router::new().fallback(static_handler))
    }
}

/// Development mode: Proxy requests to Vite dev server
#[cfg(debug_assertions)]
async fn dev_proxy_handler(uri: Uri) -> impl IntoResponse {
    use reqwest::Client;
    
    let path = uri.path().trim_start_matches("/ui");
    let vite_url = format!("http://localhost:7834{}", path);
    
    match Client::new().get(&vite_url).send().await {
        Ok(response) => {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await.unwrap_or_default();
            
            let mut builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                builder = builder.header(key, value);
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
#[cfg(not(debug_assertions))]
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
