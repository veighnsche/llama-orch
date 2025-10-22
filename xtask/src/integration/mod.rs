// TEAM-251: Integration testing infrastructure
// Purpose: Test all commands with actual binaries in different states
// Architecture: Custom test harness + state machine + chaos testing

pub mod assertions;
pub mod harness;

#[cfg(test)]
pub mod commands;

#[cfg(test)]
pub mod state_machine; // TEAM-252: State machine tests

pub use assertions::*;
pub use harness::TestHarness;
