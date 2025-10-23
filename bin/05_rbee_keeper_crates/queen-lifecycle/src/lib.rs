// TEAM-259: Created by TEAM-259 (extracted from rbee-keeper)
// TEAM-276: Enhanced with all queen lifecycle operations
// Purpose: Queen-rbee lifecycle management

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-lifecycle
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Complete lifecycle management for queen-rbee daemon from rbee-keeper CLI.
//!
//! # Module Structure
//!
//! - `types` - QueenHandle type
//! - `health` - Health checking functions
//! - `ensure` - Ensure queen running pattern
//! - `start` - Start queen daemon
//! - `stop` - Stop queen daemon
//! - `status` - Check queen status
//! - `rebuild` - Rebuild queen binary
//! - `info` - Get queen build info
//! - `install` - Install queen binary
//! - `uninstall` - Uninstall queen binary
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
//! ## Start/Stop Queen
//! ```rust,no_run
//! use queen_lifecycle::{start_queen, stop_queen};
//!
//! # async fn example() -> anyhow::Result<()> {
//! start_queen("http://localhost:8500").await?;
//! // ... use queen ...
//! stop_queen("http://localhost:8500").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Install/Uninstall Queen
//! ```rust,no_run
//! use queen_lifecycle::{install_queen, uninstall_queen};
//!
//! # async fn example() -> anyhow::Result<()> {
//! install_queen(None).await?; // Auto-detect binary
//! // ... use queen ...
//! uninstall_queen("http://localhost:8500").await?;
//! # Ok(())
//! # }
//! ```

// TEAM-259: Module declarations
// TEAM-276: Added all lifecycle operations
pub mod ensure;
pub mod health;
pub mod info;
pub mod install;
pub mod rebuild;
pub mod start;
pub mod status;
pub mod stop;
pub mod types;
pub mod uninstall;

// TEAM-259: Re-export main types and functions
// TEAM-276: Added all lifecycle operations
pub use ensure::ensure_queen_running;
pub use health::{is_queen_healthy, poll_until_healthy};
pub use info::get_queen_info;
pub use install::install_queen;
pub use rebuild::rebuild_queen;
pub use start::start_queen;
pub use status::check_queen_status;
pub use stop::stop_queen;
pub use types::QueenHandle;
pub use uninstall::uninstall_queen;
