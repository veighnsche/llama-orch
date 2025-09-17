use axum::Router;
use crate::{app::router::build_router, state::AppState};

pub fn build_app() -> Router {
    let state = AppState::new();
    build_router(state)
}
