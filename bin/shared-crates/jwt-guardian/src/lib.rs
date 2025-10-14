//! # jwt-guardian
//!
//! JWT Token Lifecycle Manager for llama-orch
//!
//! Enterprise-grade JWT validation, revocation, and lifecycle management.
//!
//! ## Features
//!
//! - **RS256/ES256 Validation** — Asymmetric signature verification
//! - **Clock-Skew Tolerance** — ±5 minute tolerance for time drift
//! - **Revocation Lists** — Redis-backed token revocation (optional)
//! - **Secure Defaults** — No algorithm confusion attacks
//!
//! ## Example
//!
//! ```rust,no_run
//! use jwt_guardian::{JwtValidator, ValidationConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create validator with public key
//! let config = ValidationConfig::default()
//!     .with_issuer("llama-orch")
//!     .with_audience("api");
//!
//! let validator = JwtValidator::new("public_key_pem", config)?;
//!
//! // Validate token
//! let claims = validator.validate("token")?;
//! println!("Valid token for user: {}", claims.sub);
//! # Ok(())
//! # }
//! ```

mod error;
mod validator;
mod claims;
mod config;

#[cfg(feature = "revocation")]
mod revocation;

pub use error::{JwtError, Result};
pub use validator::JwtValidator;
pub use claims::Claims;
pub use config::ValidationConfig;

#[cfg(feature = "revocation")]
pub use revocation::RevocationList;

/// JWT algorithm types supported by jwt-guardian
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// RSA with SHA-256 (recommended)
    RS256,
    /// ECDSA with SHA-256 (recommended)
    ES256,
}

impl Algorithm {
    /// Convert to jsonwebtoken Algorithm
    pub(crate) fn to_jsonwebtoken(&self) -> jsonwebtoken::Algorithm {
        match self {
            Algorithm::RS256 => jsonwebtoken::Algorithm::RS256,
            Algorithm::ES256 => jsonwebtoken::Algorithm::ES256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_conversion() {
        assert_eq!(
            Algorithm::RS256.to_jsonwebtoken(),
            jsonwebtoken::Algorithm::RS256
        );
        assert_eq!(
            Algorithm::ES256.to_jsonwebtoken(),
            jsonwebtoken::Algorithm::ES256
        );
    }
}
