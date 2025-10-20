//! End-to-end integration tests
//!
//! Created by: TEAM-160
//!
//! Tests real daemon orchestration with no mocks:
//! - Queen lifecycle (start/stop)
//! - Hive lifecycle (start/stop)
//! - Cascading shutdown

pub mod queen_lifecycle;
pub mod hive_lifecycle;
pub mod cascade_shutdown;
pub mod helpers;

pub use queen_lifecycle::test_queen_lifecycle;
pub use hive_lifecycle::test_hive_lifecycle;
pub use cascade_shutdown::test_cascade_shutdown;
