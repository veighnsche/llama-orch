//! worker-adapters/http-util — shared HTTP client and streaming helpers for adapters.

use http::header::AUTHORIZATION;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use std::sync::Mutex;
use std::time::Duration;
use tokio::time::sleep;
use rand::{rngs::StdRng, SeedableRng, Rng};

static DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
static DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

static CLIENT: Lazy<Client> = Lazy::new(|| {
    let cfg = HttpClientConfig::default();
    make_client(&cfg).expect("http client")
});

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

/// Return an Authorization header value if AUTH_TOKEN is configured.
pub fn bearer_header_from_env() -> Option<String> {
    std::env::var("AUTH_TOKEN").ok().map(|t| format!("Bearer {}", t))
}

/// Apply Authorization header if available from env to a RequestBuilder.
pub fn with_bearer_if_configured(rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(v) = bearer_header_from_env() {
        rb.header(AUTHORIZATION, v)
    } else {
        rb
    }
}

/// Redact secrets in a log string, masking bearer tokens, X-API-Key, and common token/api_key patterns to fp6.
pub fn redact_secrets(s: &str) -> String {
    // Case-insensitive header patterns and common key/value token appearances.
    // Replace captures with masked fp6 suffix.
    static AUTH_BEARER_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)Authorization\s*:\s*Bearer\s+(?P<t>[A-Za-z0-9._\-]+)").unwrap()
    });
    static X_API_KEY_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)X-API-Key\s*:\s*(?P<t>[A-Za-z0-9._\-]+)").unwrap()
    });
    static KV_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
        // Matches token or api_key in simple JSON/text: token:"..." or api_key=...
        Regex::new(r"(?i)(token|api[_-]?key)\s*[:=]\s*\"?(?P<t>[A-Za-z0-9._\-]{8,})\"?").unwrap()
    });

    let mut out = AUTH_BEARER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            format!("Authorization: Bearer ****{}", fp6)
        })
        .into_owned();

    out = X_API_KEY_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            format!("X-API-Key: ****{}", fp6)
        })
        .into_owned();

    out = KV_TOKEN_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            // Preserve the key name as matched in group 1, normalize formatting
            format!("{}: \"****{}\"", &caps[1], fp6)
        })
        .into_owned();

    out
}

// ===== Retries (HTU-1002) =====
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub base: Duration,
    pub multiplier: f64,
    pub cap: Duration,
    pub max_attempts: u32,
    /// Optional RNG seed for deterministic jitter in tests
    pub seed: Option<u64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        let seed = std::env::var("HTTP_UTIL_TEST_SEED").ok().and_then(|s| s.parse().ok());
        Self {
            base: Duration::from_millis(100),
            multiplier: 2.0,
            cap: Duration::from_millis(2000),
            max_attempts: 4,
            seed,
        }
    }
}

/// Classification of errors for retry handling.
#[derive(thiserror::Error, Debug)]
pub enum RetryError {
    #[error("retriable: {0}")]
    Retriable(#[source] anyhow::Error),
    #[error("non-retriable: {0}")]
    NonRetriable(#[source] anyhow::Error),
}

static RETRY_TIMELINE_MS: Lazy<Mutex<Vec<u64>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Returns and clears the last retry timeline delays (ms) recorded by `with_retries`.
pub fn get_and_clear_retry_timeline() -> Vec<u64> {
    let mut g = RETRY_TIMELINE_MS.lock().unwrap();
    let v = g.clone();
    g.clear();
    v
}

/// Execute an async operation with capped full-jitter backoff for retriable errors.
pub async fn with_retries<F, Fut, T>(mut op: F, policy: RetryPolicy) -> Result<T, RetryError>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, RetryError>>,
{
    // RNG: seed when provided for deterministic tests
    let mut rng = if let Some(seed) = policy.seed {
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        // Use thread_rng-like API via closure
        Some(r)
    } else {
        None
    };

    let mut attempt: u32 = 0;
    let mut last_err: Option<RetryError> = None;
    while attempt < policy.max_attempts {
        attempt += 1;
        match op(attempt).await {
            Ok(v) => return Ok(v),
            Err(RetryError::NonRetriable(e)) => return Err(RetryError::NonRetriable(e)),
            Err(RetryError::Retriable(e)) => {
                last_err = Some(RetryError::Retriable(e));
                if attempt >= policy.max_attempts {
                    break;
                }
                // full jitter in [0, min(cap, base*multiplier^n)]
                let exp = policy.base.as_millis() as f64 * policy.multiplier.powi(attempt as i32);
                let max_delay_ms = std::cmp::min(policy.cap.as_millis() as u128, exp as u128) as u64;
                let delay_ms = if max_delay_ms == 0 { 0 } else {
                    if let Some(ref mut srng) = rng {
                        use rand::Rng;
                        srng.gen_range(0..=max_delay_ms)
                    } else {
                        rand::thread_rng().gen_range(0..=max_delay_ms)
                    }
                };
                RETRY_TIMELINE_MS.lock().unwrap().push(delay_ms);
                sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| RetryError::Retriable(anyhow::anyhow!("unknown failure"))))
}

// ===== Streaming decode (HTU-1003) =====
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Started(Value),
    Token { i: usize, t: String },
    Metrics(Value),
    End(Value),
}

/// Decode a simple SSE-like body into events, invoking the sink for each.
pub fn stream_decode<S: AsRef<str>, F: FnMut(StreamEvent)>(body: S, mut sink: F) -> anyhow::Result<()> {
    let mut current_event: Option<String> = None;
    for line in body.as_ref().lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event: ") {
            current_event = Some(rest.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let ev = current_event.clone().unwrap_or_default();
            let v: Value = serde_json::from_str(data.trim()).unwrap_or(Value::Null);
            match ev.as_str() {
                "started" => sink(StreamEvent::Started(v)),
                "token" => {
                    let i = v.get("i").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let t = v.get("t").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    sink(StreamEvent::Token { i, t });
                }
                "metrics" => sink(StreamEvent::Metrics(v)),
                "end" => sink(StreamEvent::End(v)),
                _ => {}
            }
        }
    }
    Ok(())
}
