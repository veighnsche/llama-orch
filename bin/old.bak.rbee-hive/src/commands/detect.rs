//! Backend detection command
//!
//! Created by: TEAM-052
//!
//! Detects available compute backends (CUDA, Metal, CPU) on this machine

use anyhow::Result;

pub fn handle() -> Result<()> {
    // Detect backends
    let caps = gpu_info::detect_backends();

    // Print results
    println!("Backend Detection Results:");
    println!("==========================");
    println!();
    println!("Available backends: {}", caps.backends.len());
    for backend in &caps.backends {
        let count = caps.device_count(*backend);
        println!("  - {}: {} device(s)", backend, count);
    }
    println!();
    println!("Total devices: {}", caps.total_devices());
    println!();

    // Print JSON format for registry
    let (backends_json, devices_json) = caps.to_json_strings();
    println!("Registry format:");
    println!("  backends: {}", backends_json);
    println!("  devices:  {}", devices_json);

    Ok(())
}
