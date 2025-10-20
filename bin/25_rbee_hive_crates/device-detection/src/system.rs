//! System information detection
//!
//! Created by: TEAM-159
//!
//! Detects CPU cores and system RAM for device capabilities reporting.

/// Get the number of CPU cores
///
/// Returns the number of logical CPU cores available on the system.
///
/// # Example
///
/// ```rust
/// use rbee_hive_device_detection::get_cpu_cores;
///
/// let cores = get_cpu_cores();
/// println!("System has {} CPU cores", cores);
/// ```
pub fn get_cpu_cores() -> u32 {
    num_cpus::get() as u32
}

/// Get system RAM in GB
///
/// Returns the total system RAM in gigabytes.
///
/// # Example
///
/// ```rust
/// use rbee_hive_device_detection::get_system_ram_gb;
///
/// let ram_gb = get_system_ram_gb();
/// println!("System has {} GB RAM", ram_gb);
/// ```
pub fn get_system_ram_gb() -> u32 {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory_bytes = sys.total_memory();
    // sysinfo 0.32+ returns bytes
    // Convert bytes to GB (1 GB = 1024^3 bytes)
    (total_memory_bytes / (1024 * 1024 * 1024)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cpu_cores() {
        let cores = get_cpu_cores();
        assert!(cores > 0, "Should detect at least 1 CPU core");
        assert!(cores <= 256, "CPU core count should be reasonable");
    }

    #[test]
    fn test_get_system_ram_gb() {
        let ram_gb = get_system_ram_gb();
        assert!(ram_gb > 0, "Should detect at least 1 GB RAM");
        assert!(ram_gb <= 2048, "RAM size should be reasonable (max 2TB)");
    }
}
