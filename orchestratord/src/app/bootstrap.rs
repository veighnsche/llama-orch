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
    // TO DO: implement start_server function
}
