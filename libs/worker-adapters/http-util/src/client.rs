use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

static DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
static DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

static CLIENT: Lazy<Client> = Lazy::new(|| {
    let cfg = HttpClientConfig::default();
    make_client(&cfg).expect("http client")
});

/// Get a shared HTTP client instance. Safe to clone and use across threads.
pub fn client() -> &'static Client {
    &CLIENT
}

/// Configuration for building the shared HTTP client.
#[derive(Clone, Debug)]
pub struct HttpClientConfig {
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    /// When false, accept invalid TLS certs (test-only). Default true.
    pub tls_verify: bool,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_TIMEOUT,
            tls_verify: true,
        }
    }
}

/// Returns the default config (useful for verification in tests/BDD).
pub fn default_config() -> HttpClientConfig {
    HttpClientConfig::default()
}

/// Prefer HTTP/2 when ALPN supports it (hint for tests/BDD; reqwest/hyper handle this automatically).
pub fn h2_preference() -> bool {
    true
}

/// Build a reqwest Client from config.
pub fn make_client(cfg: &HttpClientConfig) -> anyhow::Result<Client> {
    let mut builder = Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .tcp_keepalive(Duration::from_secs(60))
        .timeout(cfg.request_timeout)
        .connect_timeout(cfg.connect_timeout);

    if !cfg.tls_verify {
        builder = builder.danger_accept_invalid_certs(true);
    }

    Ok(builder.build()?)
}
