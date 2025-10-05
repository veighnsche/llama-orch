//! BDD World for secrets-management tests

use cucumber::World;
use secrets_management::{Result, Secret, SecretError};
use std::path::PathBuf;
use tempfile::TempDir;

#[derive(World)]
pub struct BddWorld {
    /// Last operation result
    pub last_result: Option<Result<()>>,

    /// Last error (if any)
    pub last_error: Option<SecretError>,

    /// Secret file path
    pub secret_path: Option<PathBuf>,

    /// Secret file permissions (Unix mode)
    pub file_mode: Option<u32>,

    /// Secret value for verification
    pub secret_value: Option<String>,

    /// Loaded secret (actual Secret type)
    pub secret_loaded: Option<Secret>,

    /// Input for verification
    pub verify_input: Option<String>,

    /// Verification result
    pub verify_result: Option<bool>,

    /// Token for key derivation
    pub token: Option<String>,

    /// Domain for key derivation
    pub domain: Option<Vec<u8>>,

    /// Derived key (stored as hex)
    pub derived_key: Option<String>,

    /// Systemd credential name
    pub credential_name: Option<String>,

    /// Temporary directory (kept alive during test)
    pub temp_dir: Option<TempDir>,

    /// Flag to track if CREDENTIALS_DIRECTORY was set (for cleanup)
    pub credentials_dir_set: bool,
}

impl Default for BddWorld {
    fn default() -> Self {
        Self {
            last_result: None,
            last_error: None,
            secret_path: None,
            file_mode: None,
            secret_value: None,
            secret_loaded: None,
            verify_input: None,
            verify_result: None,
            token: None,
            domain: None,
            derived_key: None,
            credential_name: None,
            temp_dir: None,
            credentials_dir_set: false,
        }
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        // Clean up CREDENTIALS_DIRECTORY if we set it
        if self.credentials_dir_set {
            std::env::remove_var("CREDENTIALS_DIRECTORY");
        }
    }
}

impl std::fmt::Debug for BddWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BddWorld")
            .field("last_result", &self.last_result.as_ref().map(|r| r.is_ok()))
            .field("last_error", &self.last_error)
            .field("secret_path", &self.secret_path)
            .field("file_mode", &self.file_mode)
            .field("secret_value", &self.secret_value.as_ref().map(|_| "[REDACTED]"))
            .field("secret_loaded", &self.secret_loaded.as_ref().map(|_| "[REDACTED]"))
            .field("verify_input", &self.verify_input)
            .field("verify_result", &self.verify_result)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("domain", &self.domain)
            .field("derived_key", &self.derived_key.as_ref().map(|_| "[REDACTED]"))
            .field("credential_name", &self.credential_name)
            .field("temp_dir", &self.temp_dir.as_ref().map(|_| "[TempDir]"))
            .field("credentials_dir_set", &self.credentials_dir_set)
            .finish()
    }
}

impl BddWorld {
    /// Store operation result
    pub fn store_result(&mut self, result: Result<()>) {
        match result {
            Ok(()) => {
                self.last_result = Some(Ok(()));
                self.last_error = None;
            }
            Err(e) => {
                // Store error without cloning (move it)
                self.last_error = Some(e);
                self.last_result =
                    Some(Err(secrets_management::SecretError::InvalidFormat("error".to_string())));
            }
        }
    }

    /// Check if last operation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last operation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }

    /// Get last error
    pub fn get_last_error(&self) -> Option<&SecretError> {
        self.last_error.as_ref()
    }
}
