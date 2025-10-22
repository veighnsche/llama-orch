// TEAM-259: Created by TEAM-259 (extracted from rbee-keeper)
// Purpose: Queen-rbee lifecycle management

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-lifecycle
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Lifecycle management for queen-rbee daemon from rbee-keeper CLI.
//!
//! # Module Structure
//!
//! - `types` - QueenHandle type
//! - `health` - Health checking functions
//! - `ensure` - Ensure queen running pattern
//!
//! # Interface
//!
//! ## Ensure Queen Running
//! ```rust,no_run
//! use queen_lifecycle::ensure_queen_running;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let handle = ensure_queen_running("http://localhost:8500").await?;
//!
//! // Use queen...
//!
//! // Keep queen alive for future tasks
//! handle.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Health Checking
//! ```rust,no_run
//! use queen_lifecycle::is_queen_healthy;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let is_healthy = is_queen_healthy("http://localhost:8500").await?;
//! # Ok(())
//! # }
//! ```

// TEAM-259: Module declarations
pub mod ensure;
pub mod health;
pub mod types;

// TEAM-259: Re-export main types and functions
pub use ensure::ensure_queen_running;
pub use health::{is_queen_healthy, poll_until_healthy};
pub use types::QueenHandle;
