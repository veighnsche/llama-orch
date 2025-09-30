use once_cell::sync::Lazy;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use std::time::Duration;
use tokio::time::sleep;

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
    // Ensure timeline is per-call isolated
    RETRY_TIMELINE_MS.lock().unwrap().clear();

    // RNG: seed when provided for deterministic tests
    let mut rng = if let Some(seed) = policy.seed {
        let r = rand::rngs::StdRng::seed_from_u64(seed);
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
                let max_delay_ms = std::cmp::min(policy.cap.as_millis(), exp as u128) as u64;
                let delay_ms = if max_delay_ms == 0 {
                    0
                } else if let Some(ref mut srng) = rng {
                    srng.gen_range(0..=max_delay_ms)
                } else {
                    rand::thread_rng().gen_range(0..=max_delay_ms)
                };
                RETRY_TIMELINE_MS.lock().unwrap().push(delay_ms);
                sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| RetryError::Retriable(anyhow::anyhow!("unknown failure"))))
}
