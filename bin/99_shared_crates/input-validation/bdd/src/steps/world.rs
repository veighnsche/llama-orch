//! BDD World for input-validation tests

use cucumber::World;
use input_validation::{Result, ValidationError};

#[derive(Debug, Default, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<()>>,

    /// Last error (if any)
    pub last_error: Option<ValidationError>,

    /// Input string being validated
    pub input: String,

    /// Max length for validation
    pub max_len: usize,

    /// Expected length for hex strings
    pub expected_len: usize,

    /// Min value for range validation
    pub min_value: i64,

    /// Max value for range validation
    pub max_value: i64,

    /// Value being validated
    pub value: i64,
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<()>) {
        match result {
            Ok(()) => {
                self.last_result = Some(Ok(()));
                self.last_error = None;
            }
            Err(e) => {
                self.last_result = Some(Err(e.clone()));
                self.last_error = Some(e);
            }
        }
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }

    /// Get last error
    pub fn get_last_error(&self) -> Option<&ValidationError> {
        self.last_error.as_ref()
    }
}
