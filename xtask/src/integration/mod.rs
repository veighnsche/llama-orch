// TEAM-251: Integration testing infrastructure
// Purpose: Test all commands with actual binaries in different states
// Architecture: Custom test harness + state machine + chaos testing
//
// TEAM-282: Removed docker_harness (wrong architecture)
// See tests/docker/ARCHITECTURE_FIX.md for details

pub mod assertions;
pub mod harness; // Docker-based network testing

#[cfg(test)]
pub mod commands;

#[cfg(test)]
pub mod state_machine; // TEAM-252: State machine tests
