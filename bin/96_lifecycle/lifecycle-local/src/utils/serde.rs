//! Serde utilities for custom serialization
//!
//! TEAM-329: Extracted from types/install.rs (serialization helpers are utilities, not types)

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::time::{SystemTime, UNIX_EPOCH};

/// Serialize SystemTime as Unix timestamp (seconds since epoch)
pub fn serialize_systemtime<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let duration = time.duration_since(UNIX_EPOCH).unwrap();
    duration.as_secs().serialize(serializer)
}

/// Deserialize SystemTime from Unix timestamp (seconds since epoch)
pub fn deserialize_systemtime<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_systemtime_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct TestStruct {
            #[serde(
                serialize_with = "serialize_systemtime",
                deserialize_with = "deserialize_systemtime"
            )]
            time: SystemTime,
        }

        let original = TestStruct { time: SystemTime::now() };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: TestStruct = serde_json::from_str(&json).unwrap();

        // Compare as Unix timestamps (within 1 second tolerance)
        let orig_secs = original.time.duration_since(UNIX_EPOCH).unwrap().as_secs();
        let deser_secs = deserialized.time.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(orig_secs, deser_secs);
    }
}
