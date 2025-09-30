// Pool Manager library - spawn and supervise engine processes
//
// Organized by domain:
// - core: fundamental types and state (health, registry)
// - lifecycle: spawn, drain, reload, supervision
// - placement: device masks, GPU split planning
// - validation: preflight checks
// - api: HTTP API for daemon mode

pub mod api;
pub mod config;
pub mod core;
pub mod lifecycle;
pub mod placement;
pub mod validation;

// Re-export for backward compatibility and convenience
pub use core::{health, registry};
pub use lifecycle::{drain, preload, supervision};
pub use placement::{devicemasks, hetero_split};
pub use validation::preflight;
