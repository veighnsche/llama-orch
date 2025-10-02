//! Report formatters
//!
//! All formatters validate input and return Result to prevent garbage output.

pub mod executive;
pub mod developer;
pub mod failure;
pub mod metadata_report;
pub mod unified;

pub use executive::generate_executive_summary;
pub use developer::generate_developer_report;
pub use failure::generate_failure_report;
pub use metadata_report::generate_metadata_report;
pub use unified::generate_unified_report;
