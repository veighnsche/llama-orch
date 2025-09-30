use crate::{app::router::build_router, state::AppState};
use axum::Router;

pub fn build_app() -> Router {
    let state = AppState::new();
    // Optionally bind llama.cpp adapter for MVP wiring when feature + env configured
    // TODO(OwnerB-ORCH-BINDING-SHIM): This is an MVP shim for binding a single adapter via
    // feature gate + environment variables. Replace with pool-manager driven registration and
    // config-schema backed sources, and ensure reload/drain lifecycle integrates with AdapterHost.
    // TODO[ORCHD-CONFIG-VALIDATE-0001]: Load and validate orchestrator config via
    // (e.g., missing GPU pools, malformed placement policy). Wire reload/drain.
    #[cfg(feature = "llamacpp-adapter")]
    {
        if let Ok(url) = std::env::var("ORCHD_LLAMACPP_URL") {
            let pool =
                std::env::var("ORCHD_LLAMACPP_POOL").unwrap_or_else(|_| "default".to_string());
            let replica =
                std::env::var("ORCHD_LLAMACPP_REPLICA").unwrap_or_else(|_| "r0".to_string());
            let adapter = worker_adapters_llamacpp_http::LlamaCppHttpAdapter::new(url);
            state.adapter_host.bind(pool.clone(), replica, Arc::new(adapter));
            if let Ok(mut s) = state.bound_pools.lock() {
                s.insert(format!("{}:{}", pool, replica));
            }
        }
    }
    // Spawn stale node checker if cloud profile enabled
    if state.cloud_profile_enabled() {
        let registry = state.service_registry().clone();
        let check_interval_secs = std::env::var("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        service_registry::heartbeat::spawn_stale_checker(registry, check_interval_secs);
        tracing::info!(
            interval_secs = check_interval_secs,
            "Started stale node checker for cloud profile"
        );
    }

    build_router(state)
}

#[cfg(feature = "metrics")]
pub fn init_observability() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).json().try_init();
}

#[cfg(not(feature = "metrics"))]
pub fn init_observability() {
    // No-op when metrics/advanced logging is disabled
}

pub fn start_server() {
    // Synchronous wrapper that spins up Tokio runtime and serves the app
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        init_observability();
        crate::metrics::pre_register();
        let app = build_app();
        let addr = std::env::var("ORCHD_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        if let Err(e) = crate::app::auth_min::enforce_startup_bind_policy(&addr) {
            eprintln!("orchestratord startup refused: {}", e);
            std::process::exit(1);
        }
        // Narration breadcrumb for startup
        observability_narration_core::human("orchestratord", "start", &addr, "listening");
        if std::env::var("ORCHD_PREFER_H2")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            observability_narration_core::human(
                "orchestratord",
                "http2",
                &addr,
                "preference set (h2/h2c when available)",
            );
        }
        let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind ORCHD_ADDR");
        eprintln!("orchestratord listening on {}", addr);
        axum::serve(listener, app).await.unwrap();
    });
}
