//! JWT claims structures

use serde::{Deserialize, Serialize};

/// Standard JWT claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,

    /// Issuer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,

    /// Audience
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// JWT ID (unique identifier)
    pub jti: String,
}

impl Claims {
    /// Create new claims with subject
    pub fn new(subject: impl Into<String>) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            sub: subject.into(),
            iss: None,
            aud: None,
            exp: now + 900, // 15 minutes default
            iat: now,
            jti: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Set issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.iss = Some(issuer.into());
        self
    }

    /// Set audience
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.aud = Some(audience.into());
        self
    }

    /// Set expiration (minutes from now)
    pub fn with_expiration_minutes(mut self, minutes: i64) -> Self {
        self.exp = self.iat + (minutes * 60);
        self
    }

    /// Set expiration (Unix timestamp)
    pub fn with_expiration(mut self, exp: i64) -> Self {
        self.exp = exp;
        self
    }

    /// Set JWT ID
    pub fn with_jti(mut self, jti: impl Into<String>) -> Self {
        self.jti = jti.into();
        self
    }

    /// Check if token is expired (with optional clock skew)
    pub fn is_expired(&self, clock_skew_seconds: u64) -> bool {
        let now = chrono::Utc::now().timestamp();
        self.exp + (clock_skew_seconds as i64) < now
    }

    /// Get remaining time until expiration (seconds)
    pub fn remaining_seconds(&self) -> i64 {
        let now = chrono::Utc::now().timestamp();
        self.exp - now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claims_builder() {
        let claims = Claims::new("user-123")
            .with_issuer("llama-orch")
            .with_audience("api")
            .with_expiration_minutes(15);

        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.iss, Some("llama-orch".to_string()));
        assert_eq!(claims.aud, Some("api".to_string()));
        assert!(claims.exp > claims.iat);
    }

    #[test]
    fn test_is_expired() {
        let mut claims = Claims::new("user-123");
        
        // Not expired (future expiration)
        claims.exp = chrono::Utc::now().timestamp() + 600; // 10 minutes
        assert!(!claims.is_expired(0));

        // Expired (past expiration)
        claims.exp = chrono::Utc::now().timestamp() - 600; // 10 minutes ago
        assert!(claims.is_expired(0));

        // Within clock skew tolerance
        claims.exp = chrono::Utc::now().timestamp() - 60; // 1 minute ago
        assert!(!claims.is_expired(300)); // 5 minute tolerance
    }

    #[test]
    fn test_remaining_seconds() {
        let mut claims = Claims::new("user-123");
        claims.exp = chrono::Utc::now().timestamp() + 600; // 10 minutes

        let remaining = claims.remaining_seconds();
        assert!(remaining > 590 && remaining <= 600);
    }
}
