//! worker-adapters/http-util — shared HTTP client and streaming helpers for adapters.

use std::time::Duration;
use reqwest::Client;
use once_cell::sync::Lazy;

static DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

static CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .tcp_keepalive(Duration::from_secs(60))
        .timeout(DEFAULT_TIMEOUT)
        .build()
        .expect("http client")
});

pub fn client() -> &'static Client { &CLIENT }
