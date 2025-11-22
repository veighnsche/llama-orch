// TEAM-481: Newtype pattern for type-safe IDs
//
// Provides compile-time type safety for different ID types.
// Prevents mixing up request_id, job_id, etc.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Request ID for generation requests
///
/// TEAM-481: Newtype pattern prevents mixing up with `JobId`
/// TEAM-482: AGGRESSIVE - Store Uuid directly, convert to string only when needed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(uuid::Uuid);

impl RequestId {
    /// Create a new random request ID
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    /// Create from an existing string
    #[must_use]
    #[inline]
    pub fn from_string(s: String) -> Self {
        Self(uuid::Uuid::parse_str(&s).expect("Invalid UUID"))
    }

    /// Get the ID as a string (allocates)
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }

    /// Convert to owned String
    #[must_use]
    #[inline]
    pub fn into_string(self) -> String {
        self.0.to_string()
    }

    /// Get raw UUID (zero-cost)
    #[must_use]
    #[inline(always)]
    pub fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for RequestId {
    #[inline]
    fn from(s: String) -> Self {
        Self::from_string(s)
    }
}

impl From<RequestId> for String {
    #[inline]
    fn from(id: RequestId) -> Self {
        id.into_string()
    }
}

/// Job ID for HTTP job tracking
///
/// TEAM-481: Newtype pattern prevents mixing up with `RequestId`
/// TEAM-482: AGGRESSIVE - Store Uuid directly, convert to string only when needed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(uuid::Uuid);

impl JobId {
    /// Create a new random job ID
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    /// Create from an existing string
    #[must_use]
    #[inline]
    pub fn from_string(s: String) -> Self {
        Self(uuid::Uuid::parse_str(&s).expect("Invalid UUID"))
    }

    /// Get the ID as a string (allocates)
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }

    /// Convert to owned String
    #[must_use]
    #[inline]
    pub fn into_string(self) -> String {
        self.0.to_string()
    }

    /// Get raw UUID (zero-cost)
    #[must_use]
    #[inline(always)]
    pub fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for JobId {
    #[inline]
    fn from(s: String) -> Self {
        Self::from_string(s)
    }
}

impl From<JobId> for String {
    #[inline]
    fn from(id: JobId) -> Self {
        id.into_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_creation() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1, id2); // Should be unique
    }

    #[test]
    fn test_job_id_creation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2); // Should be unique
    }

    #[test]
    fn test_request_id_display() {
        // TEAM_527: Round-trip a valid UUID string instead of hardcoded non-UUID literal
        let original = RequestId::new();
        let s = original.to_string();
        let parsed = RequestId::from_string(s.clone());
        assert_eq!(parsed.to_string(), s);
        assert_eq!(parsed.as_str(), s);
    }

    #[test]
    fn test_job_id_display() {
        // TEAM_527: Round-trip a valid UUID string instead of hardcoded non-UUID literal
        let original = JobId::new();
        let s = original.to_string();
        let parsed = JobId::from_string(s.clone());
        assert_eq!(parsed.to_string(), s);
        assert_eq!(parsed.as_str(), s);
    }

    #[test]
    fn test_ids_are_different_types() {
        // This test verifies compile-time type safety
        let request_id = RequestId::new();
        let job_id = JobId::new();

        // These are different types - can't mix them up!
        assert_ne!(request_id.as_str(), job_id.as_str()); // Different UUIDs
    }
}
