// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-218: Investigated Oct 22, 2025 - STUB with no implementation
// TEAM-284: Added hive heartbeat to queen
// Purpose: rbee-hive library code
// Status: STUB - Awaiting implementation

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive library
//!
//! Core library code for rbee-hive daemon

pub mod heartbeat; // TEAM-284: Hive heartbeat to queen
pub mod hive_check; // TEAM-313: Hive narration check (tests SSE streaming)
pub mod pkgbuild_parser; // PKGBUILD parser for worker installation
pub mod pkgbuild_executor; // PKGBUILD executor for worker installation
pub mod source_fetcher; // TEAM-378: Source fetcher for PKGBUILD (git clone, etc.)
pub mod worker_install; // TEAM-378: Worker binary installation handler
