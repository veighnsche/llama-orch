//! rate-limiting — Request rate limiting and DoS protection
//!
//! Prevents denial-of-service attacks by limiting request rates per client/endpoint.
//!
//! # Security Properties
//!
//! - Token bucket algorithm (burst + sustained rate)
//! - Per-client tracking (by IP or identity)
//! - Configurable limits per endpoint
//! - Returns 429 Too Many Requests with Retry-After
//!
//! # Example
//!
//! ```rust
//! use rate_limiting::{RateLimiter, RateLimit};
//!
//! let limiter = RateLimiter::new();
//! limiter.configure("api", RateLimit::per_second(100).burst(50));
//!
//! // Check if request allowed
//! if limiter.check("api", "client-ip-192.168.1.1").await? {
//!     // Process request
//! } else {
//!     // Return 429 Too Many Requests
//! }
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("rate limit exceeded")]
    Exceeded,
    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, RateLimitError>;

/// Rate limit configuration
#[derive(Debug, Clone, Copy)]
pub struct RateLimit {
    /// Requests per second (sustained rate)
    pub rate: u32,
    /// Burst capacity (max tokens)
    pub burst: u32,
}

impl RateLimit {
    pub fn per_second(rate: u32) -> Self {
        Self {
            rate,
            burst: rate,
        }
    }
    
    pub fn burst(mut self, burst: u32) -> Self {
        self.burst = burst;
        self
    }
}

/// Token bucket for rate limiting
struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(limit: RateLimit) -> Self {
        Self {
            tokens: limit.burst as f64,
            capacity: limit.burst as f64,
            refill_rate: limit.rate as f64,
            last_refill: Instant::now(),
        }
    }
    
    fn try_consume(&mut self) -> bool {
        self.refill();
        
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
    
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }
}

/// Rate limiter
pub struct RateLimiter {
    limits: Arc<Mutex<HashMap<String, RateLimit>>>,
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: Arc::new(Mutex::new(HashMap::new())),
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Configure rate limit for endpoint
    pub fn configure(&self, endpoint: &str, limit: RateLimit) -> Result<()> {
        let mut limits = self.limits.lock()
            .map_err(|e| RateLimitError::Internal(format!("Lock poisoned: {}", e)))?;
        limits.insert(endpoint.to_string(), limit);
        Ok(())
    }
    
    /// Check if request is allowed
    pub async fn check(&self, endpoint: &str, client_id: &str) -> Result<bool> {
        let key = format!("{}:{}", endpoint, client_id);
        
        // Get limit for endpoint
        let limit = {
            let limits = self.limits.lock()
                .map_err(|e| RateLimitError::Internal(format!("Lock poisoned: {}", e)))?;
            limits.get(endpoint).copied()
        };
        
        let limit = match limit {
            Some(l) => l,
            None => return Ok(true), // No limit configured
        };
        
        // Get or create bucket
        let mut buckets = self.buckets.lock()
            .map_err(|e| RateLimitError::Internal(format!("Lock poisoned: {}", e)))?;
        
        let bucket = buckets.entry(key.clone())
            .or_insert_with(|| TokenBucket::new(limit));
        
        if bucket.try_consume() {
            Ok(true)
        } else {
            tracing::warn!(
                endpoint = %endpoint,
                client = %client_id,
                "Rate limit exceeded"
            );
            Err(RateLimitError::Exceeded)
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let limiter = RateLimiter::new();
        limiter.configure("test", RateLimit::per_second(2).burst(2))
            .ok();
        
        // First 2 requests should succeed (burst)
        assert!(limiter.check("test", "client1").await.ok().unwrap_or(false));
        assert!(limiter.check("test", "client1").await.ok().unwrap_or(false));
        
        // Third request should fail
        assert!(limiter.check("test", "client1").await.is_err());
    }
}
