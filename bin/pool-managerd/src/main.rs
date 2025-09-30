//! pool-managerd daemon entrypoint.
//!
//! STATUS: STUB (Home profile embeds as library; Cloud profile will use standalone daemon)
//!
//! Home Profile (Current):
//! - orchestratord embeds pool-managerd::registry as library (bin/orchestratord/src/state.rs:6)
//! - Single binary, single workstation, on-demand engine startup
//!
//! Cloud Profile (Future, per .specs/proposals/CLOUD_PROFILE.md):
//! - pool-managerd as standalone daemon (DaemonSet on k8s GPU nodes)
//! - orchestratord queries via HTTP control API for health/placement
//! - Multi-tenant: separate instances per namespace/VPC
//! - On-demand startup (home) OR preload (CLOUD-810 dedicated baseline)
//!
//! Responsibilities (when implemented):
//! 1. GPU Discovery — enumerate NVIDIA devices, collect compute_capability, VRAM
//! 2. On-Demand Startup — start engines when needed (or preload for dedicated)
//! 3. Engine Supervision — spawn/monitor processes, restart with backoff on crash
//! 4. Health Monitoring — periodic checks, update registry, detect driver errors
//! 5. Control API — HTTP endpoints for orchestratord (drain/reload, health, placement)
//!
//! Implementation trigger: When implementing CLOUD_PROFILE.md (k8s deployment)
//! Until then: Home profile continues to embed as library (no changes needed)

fn main() {
    println!("pool-managerd stub");
    eprintln!("Home profile: orchestratord embeds pool-managerd as library");
    eprintln!("Cloud profile: implement standalone daemon per CLOUD_PROFILE.md");
}
