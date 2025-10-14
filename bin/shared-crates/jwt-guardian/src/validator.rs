//! JWT validation logic

use crate::{Claims, JwtError, Result, ValidationConfig};
use jsonwebtoken::{decode, DecodingKey};

/// JWT validator with configurable policies
pub struct JwtValidator {
    decoding_key: DecodingKey,
    validation: jsonwebtoken::Validation,
}

impl JwtValidator {
    /// Create validator with public key (PEM format)
    ///
    /// # Arguments
    ///
    /// * `public_key_pem` - RSA/ECDSA public key in PEM format
    /// * `config` - Validation configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use jwt_guardian::{JwtValidator, ValidationConfig};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = ValidationConfig::default()
    ///     .with_issuer("llama-orch");
    ///
    /// let validator = JwtValidator::new("public_key_pem", config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(public_key_pem: &str, config: ValidationConfig) -> Result<Self> {
        let decoding_key = DecodingKey::from_rsa_pem(public_key_pem.as_bytes())
            .or_else(|_| DecodingKey::from_ec_pem(public_key_pem.as_bytes()))
            .map_err(|e| JwtError::InvalidFormat(format!("Invalid public key: {}", e)))?;

        let validation = config.to_jsonwebtoken_validation();

        Ok(Self {
            decoding_key,
            validation,
        })
    }

    /// Validate token and extract claims
    ///
    /// # Arguments
    ///
    /// * `token` - JWT token string
    ///
    /// # Returns
    ///
    /// Validated claims if token is valid
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Signature is invalid
    /// - Token is expired
    /// - Claims don't match configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use jwt_guardian::{JwtValidator, ValidationConfig};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let validator = JwtValidator::new("key", ValidationConfig::default())?;
    /// let claims = validator.validate("token")?;
    /// println!("User: {}", claims.sub);
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self, token: &str) -> Result<Claims> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)?;
        Ok(token_data.claims)
    }

    /// Validate with custom validation time (for testing)
    #[cfg(test)]
    pub fn validate_at(&self, token: &str, _validation_time: i64) -> Result<Claims> {
        // For testing purposes - in production this would use the validation_time
        self.validate(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require actual RSA keys
    // These are unit tests for the structure

    #[test]
    fn test_validator_creation_invalid_key() {
        let config = ValidationConfig::default();
        let result = JwtValidator::new("invalid-key", config);
        assert!(result.is_err());
    }
}
