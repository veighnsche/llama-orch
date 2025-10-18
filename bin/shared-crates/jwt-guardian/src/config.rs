//! JWT validation configuration

use crate::Algorithm;

/// Configuration for JWT validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Expected issuer (iss claim)
    pub issuer: Option<String>,

    /// Expected audience (aud claim)
    pub audience: Option<String>,

    /// Clock skew tolerance in seconds (default: 300 = 5 minutes)
    pub clock_skew_seconds: u64,

    /// Allowed algorithms (default: RS256, ES256)
    pub algorithms: Vec<Algorithm>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            issuer: None,
            audience: None,
            clock_skew_seconds: 300, // 5 minutes
            algorithms: vec![Algorithm::RS256, Algorithm::ES256],
        }
    }
}

impl ValidationConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set expected issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(issuer.into());
        self
    }

    /// Set expected audience
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = Some(audience.into());
        self
    }

    /// Set clock skew tolerance (seconds)
    pub fn with_clock_skew(mut self, seconds: u64) -> Self {
        self.clock_skew_seconds = seconds;
        self
    }

    /// Set allowed algorithms
    pub fn with_algorithms(mut self, algorithms: Vec<Algorithm>) -> Self {
        self.algorithms = algorithms;
        self
    }

    /// Convert to jsonwebtoken Validation
    pub(crate) fn to_jsonwebtoken_validation(&self) -> jsonwebtoken::Validation {
        let mut validation = jsonwebtoken::Validation::default();

        // Set algorithms
        validation.algorithms = self.algorithms.iter().map(|a| a.to_jsonwebtoken()).collect();

        // Set issuer
        if let Some(ref issuer) = self.issuer {
            validation.iss = Some(std::collections::HashSet::from([issuer.clone()]));
        }

        // Set audience
        if let Some(ref audience) = self.audience {
            validation.aud = Some(std::collections::HashSet::from([audience.clone()]));
        }

        // Set clock skew
        validation.leeway = self.clock_skew_seconds;

        // Require exp claim
        validation.validate_exp = true;

        validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ValidationConfig::default();
        assert_eq!(config.clock_skew_seconds, 300);
        assert_eq!(config.algorithms.len(), 2);
        assert!(config.issuer.is_none());
        assert!(config.audience.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = ValidationConfig::default()
            .with_issuer("llama-orch")
            .with_audience("api")
            .with_clock_skew(600);

        assert_eq!(config.issuer, Some("llama-orch".to_string()));
        assert_eq!(config.audience, Some("api".to_string()));
        assert_eq!(config.clock_skew_seconds, 600);
    }

    #[test]
    fn test_to_jsonwebtoken_validation() {
        let config = ValidationConfig::default().with_issuer("llama-orch").with_audience("api");

        let validation = config.to_jsonwebtoken_validation();
        assert_eq!(validation.leeway, 300);
        assert!(validation.validate_exp);
    }
}
