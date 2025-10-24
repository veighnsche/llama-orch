// SSH communication tests
// Purpose: Test SSH operations between queen-rbee and rbee-hive
// Uses: RbeeSSHClient (russh library) for SSH connections
// Run with: cargo test --package xtask --test docker_ssh_tests --ignored

#[path = "docker/ssh_tests.rs"]
mod ssh_tests;
