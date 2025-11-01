//! Generic heartbeat registry
//!
//! TEAM-285: Generic registry for tracking component state via heartbeats
//!
//! **Purpose:**
//! Eliminate duplication between worker-registry and hive-registry by providing
//! a generic implementation that works with any heartbeat type.
//!
//! **Pattern:**
//! ```text
//! Component → Heartbeat → Registry → Query API
//! ```
//!
//! # Example
//!
//! ```rust
//! use heartbeat_registry::{HeartbeatRegistry, HeartbeatItem};
//!
//! // Define your heartbeat type
//! #[derive(Clone)]
//! struct MyHeartbeat {
//!     id: String,
//!     timestamp: std::time::SystemTime,
//!     // ... other fields
//! }
//!
//! // Implement the trait
//! impl HeartbeatItem for MyHeartbeat {
//!     type Info = String;
//!     
//!     fn id(&self) -> &str {
//!         &self.id
//!     }
//!     
//!     fn info(&self) -> Self::Info {
//!         self.id.clone()
//!     }
//!     
//!     fn is_recent(&self) -> bool {
//!         // Check if timestamp is recent
//!         true
//!     }
//!     
//!     fn is_available(&self) -> bool {
//!         // Check if component is available for work
//!         true
//!     }
//! }
//!
//! // Use the registry
//! let registry = HeartbeatRegistry::<MyHeartbeat>::new();
//! assert_eq!(registry.count_total(), 0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

use std::collections::HashMap;
use std::sync::RwLock;

/// Trait for heartbeat items that can be stored in the registry
///
/// Implement this trait for your heartbeat type to use the generic registry.
pub trait HeartbeatItem: Clone + Send + Sync {
    /// The info type extracted from this heartbeat
    type Info: Clone;

    /// Get the unique ID of this heartbeat item
    fn id(&self) -> &str;

    /// Extract the info from this heartbeat
    fn info(&self) -> Self::Info;

    /// Check if this heartbeat is recent (within timeout window)
    fn is_recent(&self) -> bool;

    /// Check if this item is available for work
    fn is_available(&self) -> bool;
}

/// Generic heartbeat registry
///
/// Thread-safe registry using RwLock for concurrent access.
/// Components send heartbeats which are stored and can be queried.
///
/// # Type Parameters
///
/// * `T` - The heartbeat type (must implement `HeartbeatItem`)
///
/// # Example
///
/// See module-level documentation for usage example.
#[derive(Debug)]
pub struct HeartbeatRegistry<T: HeartbeatItem> {
    items: RwLock<HashMap<String, T>>,
}

impl<T: HeartbeatItem> HeartbeatRegistry<T> {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self { items: RwLock::new(HashMap::new()) }
    }

    /// Update item from heartbeat
    ///
    /// Upserts item - creates if new, updates if exists.
    pub fn update(&self, heartbeat: T) {
        let mut items = self.items.write().unwrap();
        items.insert(heartbeat.id().to_string(), heartbeat);
    }

    /// Get item by ID
    pub fn get(&self, id: &str) -> Option<T::Info> {
        let items = self.items.read().unwrap();
        items.get(id).map(|hb| hb.info())
    }

    /// Remove item from registry
    ///
    /// Returns true if item was removed, false if not found.
    pub fn remove(&self, id: &str) -> bool {
        let mut items = self.items.write().unwrap();
        items.remove(id).is_some()
    }

    /// List all items (including stale ones)
    pub fn list_all(&self) -> Vec<T::Info> {
        let items = self.items.read().unwrap();
        items.values().map(|hb| hb.info()).collect()
    }

    /// List items with recent heartbeats
    ///
    /// Only returns items that sent heartbeat within timeout window.
    pub fn list_online(&self) -> Vec<T::Info> {
        let items = self.items.read().unwrap();
        items.values().filter(|hb| hb.is_recent()).map(|hb| hb.info()).collect()
    }

    /// List available items (online + ready status)
    ///
    /// Returns items that are:
    /// 1. Online (recent heartbeat)
    /// 2. Available (ready for work)
    pub fn list_available(&self) -> Vec<T::Info> {
        let items = self.items.read().unwrap();
        items
            .values()
            .filter(|hb| hb.is_recent() && hb.is_available())
            .map(|hb| hb.info())
            .collect()
    }

    /// Get count of online items
    pub fn count_online(&self) -> usize {
        let items = self.items.read().unwrap();
        items.values().filter(|hb| hb.is_recent()).count()
    }

    /// Get count of available items
    pub fn count_available(&self) -> usize {
        let items = self.items.read().unwrap();
        items.values().filter(|hb| hb.is_recent() && hb.is_available()).count()
    }

    /// Get total count (including stale items)
    pub fn count_total(&self) -> usize {
        let items = self.items.read().unwrap();
        items.len()
    }

    /// Check if item is online
    pub fn is_online(&self, id: &str) -> bool {
        let items = self.items.read().unwrap();
        items.get(id).map(|hb| hb.is_recent()).unwrap_or(false)
    }

    /// Cleanup stale items
    ///
    /// Removes items that haven't sent heartbeat within timeout window.
    /// Returns number of items removed.
    pub fn cleanup_stale(&self) -> usize {
        let mut items = self.items.write().unwrap();
        let before_count = items.len();
        items.retain(|_, hb| hb.is_recent());
        before_count - items.len()
    }

    /// Get all heartbeats (for advanced queries)
    ///
    /// Returns a snapshot of all heartbeats for custom filtering.
    pub fn get_all_heartbeats(&self) -> Vec<T> {
        let items = self.items.read().unwrap();
        items.values().cloned().collect()
    }
}

impl<T: HeartbeatItem> Default for HeartbeatRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    // Test heartbeat implementation
    #[derive(Clone)]
    struct TestHeartbeat {
        id: String,
        timestamp: SystemTime,
        available: bool,
    }

    impl TestHeartbeat {
        fn new(id: &str, available: bool) -> Self {
            Self { id: id.to_string(), timestamp: SystemTime::now(), available }
        }

        fn with_old_timestamp(id: &str, available: bool) -> Self {
            Self {
                id: id.to_string(),
                timestamp: SystemTime::now() - Duration::from_secs(120),
                available,
            }
        }
    }

    impl HeartbeatItem for TestHeartbeat {
        type Info = String;

        fn id(&self) -> &str {
            &self.id
        }

        fn info(&self) -> Self::Info {
            self.id.clone()
        }

        fn is_recent(&self) -> bool {
            // Consider recent if within 90 seconds
            self.timestamp.elapsed().unwrap_or(Duration::from_secs(999)) < Duration::from_secs(90)
        }

        fn is_available(&self) -> bool {
            self.available
        }
    }

    #[test]
    fn test_new_registry() {
        let registry = HeartbeatRegistry::<TestHeartbeat>::new();
        assert_eq!(registry.count_total(), 0);
    }

    #[test]
    fn test_update() {
        let registry = HeartbeatRegistry::new();
        let heartbeat = TestHeartbeat::new("item-1", true);

        registry.update(heartbeat);

        assert_eq!(registry.count_total(), 1);
        assert!(registry.get("item-1").is_some());
    }

    #[test]
    fn test_get() {
        let registry = HeartbeatRegistry::new();
        let heartbeat = TestHeartbeat::new("item-1", true);
        registry.update(heartbeat);

        let info = registry.get("item-1").unwrap();
        assert_eq!(info, "item-1");

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_remove() {
        let registry = HeartbeatRegistry::new();
        let heartbeat = TestHeartbeat::new("item-1", true);
        registry.update(heartbeat);

        assert!(registry.remove("item-1"));
        assert!(!registry.remove("item-1")); // Already removed
        assert_eq!(registry.count_total(), 0);
    }

    #[test]
    fn test_list_all() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", true));
        registry.update(TestHeartbeat::new("item-2", false));

        let all = registry.list_all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_list_online() {
        let registry = HeartbeatRegistry::new();

        // Recent heartbeat
        registry.update(TestHeartbeat::new("item-1", true));

        // Old heartbeat
        registry.update(TestHeartbeat::with_old_timestamp("item-2", true));

        let online = registry.list_online();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0], "item-1");
    }

    #[test]
    fn test_list_available() {
        let registry = HeartbeatRegistry::new();

        // Recent + available
        registry.update(TestHeartbeat::new("item-1", true));

        // Recent + not available
        registry.update(TestHeartbeat::new("item-2", false));

        // Old + available
        registry.update(TestHeartbeat::with_old_timestamp("item-3", true));

        let available = registry.list_available();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0], "item-1");
    }

    #[test]
    fn test_count_online() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", true));
        registry.update(TestHeartbeat::new("item-2", false));
        registry.update(TestHeartbeat::with_old_timestamp("item-3", true));

        assert_eq!(registry.count_online(), 2);
    }

    #[test]
    fn test_count_available() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", true));
        registry.update(TestHeartbeat::new("item-2", false));
        registry.update(TestHeartbeat::with_old_timestamp("item-3", true));

        assert_eq!(registry.count_available(), 1);
    }

    #[test]
    fn test_is_online() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", true));
        registry.update(TestHeartbeat::with_old_timestamp("item-2", true));

        assert!(registry.is_online("item-1"));
        assert!(!registry.is_online("item-2"));
        assert!(!registry.is_online("nonexistent"));
    }

    #[test]
    fn test_cleanup_stale() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", true));
        registry.update(TestHeartbeat::with_old_timestamp("item-2", true));
        registry.update(TestHeartbeat::with_old_timestamp("item-3", true));

        let removed = registry.cleanup_stale();
        assert_eq!(removed, 2);
        assert_eq!(registry.count_total(), 1);
    }

    #[test]
    fn test_update_existing() {
        let registry = HeartbeatRegistry::new();
        registry.update(TestHeartbeat::new("item-1", false));
        registry.update(TestHeartbeat::new("item-1", true));

        assert_eq!(registry.count_total(), 1);
        let heartbeats = registry.get_all_heartbeats();
        assert!(heartbeats[0].is_available());
    }
}
