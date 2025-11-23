//! Worker Port Assigner - Dynamic port allocation for workers
//!
//! TEAM-XXX: Port assignment component for rbee-hive
//!
//! CRITICAL: Workers do NOT have fixed ports
//! - Hive assigns ports dynamically starting from 8080
//! - First worker: 8080, second: 8081, third: 8082, etc.
//! - Port assignment is managed centrally to avoid conflicts
//!
//! See: /PORT_CONFIGURATION.md for canonical port allocation strategy

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

mod worker_registry;
pub use worker_registry::WorkerRegistry;

/// Starting port for worker assignment
const WORKER_PORT_START: u16 = 8080;

/// Maximum port number (inclusive)
const WORKER_PORT_END: u16 = 9999;

/// Port assigner for worker processes
///
/// Manages dynamic port allocation for workers spawned by the hive.
/// Ensures no port conflicts and provides sequential allocation.
///
/// # Thread Safety
/// This component is thread-safe and can be shared across async tasks.
///
/// # Example
/// ```
/// use port_assigner::PortAssigner;
///
/// let assigner = PortAssigner::new();
///
/// // Assign ports to workers
/// let port1 = assigner.assign().unwrap(); // 8080
/// let port2 = assigner.assign().unwrap(); // 8081
/// let port3 = assigner.assign().unwrap(); // 8082
///
/// // Release a port when worker stops
/// assigner.release(port1);
///
/// // Next assignment reuses the released port
/// let port4 = assigner.assign().unwrap(); // 8080 (reused)
/// ```
#[derive(Debug, Clone)]
pub struct PortAssigner {
    state: Arc<Mutex<PortState>>,
}

#[derive(Debug)]
struct PortState {
    /// Currently assigned ports
    assigned: HashSet<u16>,
    /// Next port to try (for sequential allocation)
    next_port: u16,
}

impl Default for PortAssigner {
    fn default() -> Self {
        Self::new()
    }
}

impl PortAssigner {
    /// Create a new port assigner
    ///
    /// Starts with port 8080 and increments sequentially.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(PortState {
                assigned: HashSet::new(),
                next_port: WORKER_PORT_START,
            })),
        }
    }

    /// Assign the next available port
    ///
    /// Returns the assigned port number, or `None` if all ports are exhausted.
    ///
    /// # Returns
    /// - `Some(port)` - Successfully assigned port
    /// - `None` - No ports available (all 8080-9999 are in use)
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// let port = assigner.assign().expect("Port assignment failed");
    /// assert_eq!(port, 8080);
    /// ```
    pub fn assign(&self) -> Option<u16> {
        let mut state = self.state.lock().unwrap();

        // Try to find an available port starting from next_port
        let start_port = state.next_port;
        let mut current_port = start_port;

        loop {
            if !state.assigned.contains(&current_port) {
                // Found an available port
                state.assigned.insert(current_port);
                state.next_port = if current_port == WORKER_PORT_END {
                    WORKER_PORT_START
                } else {
                    current_port + 1
                };
                return Some(current_port);
            }

            // Try next port
            current_port =
                if current_port == WORKER_PORT_END { WORKER_PORT_START } else { current_port + 1 };

            // If we've wrapped around to where we started, all ports are taken
            if current_port == start_port {
                return None;
            }
        }
    }

    /// Release a previously assigned port
    ///
    /// Makes the port available for future assignments.
    /// If the port was not assigned, this is a no-op.
    ///
    /// # Arguments
    /// * `port` - Port number to release
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// let port = assigner.assign().unwrap();
    /// assigner.release(port);
    ///
    /// // Port is now available for reuse
    /// let reused_port = assigner.assign().unwrap();
    /// assert_eq!(port, reused_port);
    /// ```
    pub fn release(&self, port: u16) {
        let mut state = self.state.lock().unwrap();
        state.assigned.remove(&port);
    }

    /// Check if a specific port is currently assigned
    ///
    /// # Arguments
    /// * `port` - Port number to check
    ///
    /// # Returns
    /// `true` if the port is currently assigned, `false` otherwise
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// let port = assigner.assign().unwrap();
    ///
    /// assert!(assigner.is_assigned(port));
    /// assert!(!assigner.is_assigned(9999));
    /// ```
    #[must_use]
    pub fn is_assigned(&self, port: u16) -> bool {
        let state = self.state.lock().unwrap();
        state.assigned.contains(&port)
    }

    /// Get the count of currently assigned ports
    ///
    /// # Returns
    /// Number of ports currently in use
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// assert_eq!(assigner.assigned_count(), 0);
    ///
    /// assigner.assign();
    /// assigner.assign();
    /// assert_eq!(assigner.assigned_count(), 2);
    /// ```
    #[must_use]
    pub fn assigned_count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.assigned.len()
    }

    /// Get all currently assigned ports
    ///
    /// Returns a sorted vector of assigned port numbers.
    ///
    /// # Returns
    /// Vector of assigned ports in ascending order
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// assigner.assign(); // 8080
    /// assigner.assign(); // 8081
    ///
    /// let ports = assigner.assigned_ports();
    /// assert_eq!(ports, vec![8080, 8081]);
    /// ```
    #[must_use]
    pub fn assigned_ports(&self) -> Vec<u16> {
        let state = self.state.lock().unwrap();
        let mut ports: Vec<u16> = state.assigned.iter().copied().collect();
        ports.sort_unstable();
        ports
    }

    /// Reset the port assigner
    ///
    /// Releases all assigned ports and resets to initial state.
    /// Useful for testing or when restarting the hive.
    ///
    /// # Example
    /// ```
    /// use port_assigner::PortAssigner;
    ///
    /// let assigner = PortAssigner::new();
    /// assigner.assign();
    /// assigner.assign();
    ///
    /// assigner.reset();
    /// assert_eq!(assigner.assigned_count(), 0);
    /// ```
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.assigned.clear();
        state.next_port = WORKER_PORT_START;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_assignment() {
        let assigner = PortAssigner::new();

        assert_eq!(assigner.assign(), Some(8080));
        assert_eq!(assigner.assign(), Some(8081));
        assert_eq!(assigner.assign(), Some(8082));
    }

    #[test]
    fn test_release_and_reuse() {
        let assigner = PortAssigner::new();

        let port1 = assigner.assign().unwrap();
        let port2 = assigner.assign().unwrap();
        let port3 = assigner.assign().unwrap();

        assert_eq!(port1, 8080);
        assert_eq!(port2, 8081);
        assert_eq!(port3, 8082);

        // Release middle port
        assigner.release(port2);

        // Next assignment should continue sequentially (not reuse released ports)
        let port4 = assigner.assign().unwrap();
        assert_eq!(port4, 8083);
    }

    #[test]
    fn test_is_assigned() {
        let assigner = PortAssigner::new();

        let port = assigner.assign().unwrap();
        assert!(assigner.is_assigned(port));
        assert!(!assigner.is_assigned(9999));

        assigner.release(port);
        assert!(!assigner.is_assigned(port));
    }

    #[test]
    fn test_assigned_count() {
        let assigner = PortAssigner::new();

        assert_eq!(assigner.assigned_count(), 0);

        assigner.assign();
        assert_eq!(assigner.assigned_count(), 1);

        assigner.assign();
        assert_eq!(assigner.assigned_count(), 2);

        assigner.release(8080);
        assert_eq!(assigner.assigned_count(), 1);
    }

    #[test]
    fn test_assigned_ports() {
        let assigner = PortAssigner::new();

        assigner.assign(); // 8080
        assigner.assign(); // 8081
        assigner.assign(); // 8082

        let ports = assigner.assigned_ports();
        assert_eq!(ports, vec![8080, 8081, 8082]);
    }

    #[test]
    fn test_reset() {
        let assigner = PortAssigner::new();

        assigner.assign();
        assigner.assign();
        assert_eq!(assigner.assigned_count(), 2);

        assigner.reset();
        assert_eq!(assigner.assigned_count(), 0);

        // Should start from 8080 again
        assert_eq!(assigner.assign(), Some(8080));
    }

    #[test]
    fn test_wraparound() {
        let assigner = PortAssigner::new();

        // Manually set next_port to near the end
        {
            let mut state = assigner.state.lock().unwrap();
            state.next_port = 9998;
        }

        assert_eq!(assigner.assign(), Some(9998));
        assert_eq!(assigner.assign(), Some(9999));
        // Should wrap around to 8080
        assert_eq!(assigner.assign(), Some(8080));
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let assigner = PortAssigner::new();
        let mut handles = vec![];

        // Spawn 10 threads, each assigning 10 ports
        for _ in 0..10 {
            let assigner_clone = assigner.clone();
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    assigner_clone.assign();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 100 assigned ports
        assert_eq!(assigner.assigned_count(), 100);
    }

    #[test]
    fn test_exhaustion() {
        let assigner = PortAssigner::new();

        // Assign all possible ports (8080-9999 = 1920 ports)
        let max_ports = (WORKER_PORT_END - WORKER_PORT_START + 1) as usize;

        for _ in 0..max_ports {
            assert!(assigner.assign().is_some());
        }

        // Next assignment should fail
        assert_eq!(assigner.assign(), None);

        // Release one port
        assigner.release(8080);

        // Should be able to assign again
        assert_eq!(assigner.assign(), Some(8080));
    }
}
