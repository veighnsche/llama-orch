use axum::Router;
use crate::{app::router::build_router, state::AppState};

pub fn build_app() -> Router {
    let state = AppState::new();
    build_router(state)
}

pub fn init_observability() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).json().try_init();
}

pub fn start_server() {
    // Synchronous wrapper that spins up Tokio runtime and serves the app
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        init_observability();
        let app = build_app();
        let addr = std::env::var("ORCHD_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        if let Err(e) = crate::app::auth_min::enforce_startup_bind_policy(&addr) {
            eprintln!("orchestratord startup refused: {}", e);
            std::process::exit(1);
        }
        // Narration breadcrumb for startup
        observability_narration_core::human("orchestratord", "start", &addr, "listening");
        let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind ORCHD_ADDR");
        eprintln!("orchestratord listening on {}", addr);
        axum::serve(listener, app).await.unwrap();
    });
}
