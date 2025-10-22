#!/bin/bash
# TEAM-135: Create BDD scaffolding for all new crates

set -e

cd /home/vince/Projects/llama-orch/bin

# Function to create BDD structure for a crate
create_bdd() {
    local crate_path=$1
    local crate_name=$2
    local parent_crate=$3
    
    echo "Creating BDD for $crate_name..."
    
    # Create BDD directory structure
    mkdir -p "$crate_path/bdd/src/steps"
    mkdir -p "$crate_path/bdd/tests/features"
    
    # Create Cargo.toml
    cat > "$crate_path/bdd/Cargo.toml" <<EOF
# TEAM-135: Created by TEAM-135 (BDD scaffolding)

[package]
name = "${crate_name}-bdd"
version = "0.0.0"
edition = "2021"
license = "GPL-3.0-or-later"

[features]
default = ["bdd-cucumber"]
bdd-cucumber = []

[dependencies]
anyhow = { workspace = true }
cucumber = { version = "0.20", features = ["macros"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = { workspace = true }
${parent_crate} = { path = ".." }

[[bin]]
name = "bdd-runner"
path = "src/main.rs"
EOF

    # Create main.rs
    cat > "$crate_path/bdd/src/main.rs" <<EOF
// TEAM-135: Created by TEAM-135 (BDD scaffolding)

mod steps;

use cucumber::World as _;
use steps::world::BddWorld;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features_env = std::env::var("LLORCH_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            root.join(pb)
        }
    } else {
        root.join("tests/features")
    };

    BddWorld::cucumber().run_and_exit(features).await;
}
EOF

    # Create steps/mod.rs
    cat > "$crate_path/bdd/src/steps/mod.rs" <<EOF
// TEAM-135: Created by TEAM-135 (BDD scaffolding)

pub mod world;
// TODO: Add step modules here
// pub mod basic_steps;
EOF

    # Create steps/world.rs
    cat > "$crate_path/bdd/src/steps/world.rs" <<EOF
// TEAM-135: Created by TEAM-135 (BDD scaffolding)
//! BDD World for ${crate_name} tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,
    
    // TODO: Add test state fields here
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<(), String>) {
        self.last_result = Some(result);
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }
}
EOF

    # Create README.md
    cat > "$crate_path/bdd/README.md" <<EOF
# ${crate_name}-bdd

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** BDD test harness for ${crate_name}

## Overview

Behavior-Driven Development test harness for ${crate_name}.

## Running Tests

\`\`\`bash
# Run all features
cargo run --bin bdd-runner

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/example.feature cargo run --bin bdd-runner
\`\`\`

## Structure

- \`src/main.rs\` - BDD runner entry point
- \`src/steps/world.rs\` - Shared test state (World)
- \`src/steps/\` - Step definitions
- \`tests/features/\` - Gherkin feature files

## Adding Tests

1. Create \`.feature\` file in \`tests/features/\`
2. Implement steps in \`src/steps/\`
3. Export step module from \`src/steps/mod.rs\`

## Status

- [ ] Feature files created
- [ ] Step definitions implemented
- [ ] Tests passing

**Created by TEAM-135**
EOF

    # Create a placeholder feature file
    cat > "$crate_path/bdd/tests/features/placeholder.feature" <<EOF
# TEAM-135: Placeholder feature for ${crate_name}
# TODO: Replace with actual feature files

Feature: ${crate_name} placeholder
  As a developer
  I want placeholder tests
  So that the BDD structure is ready

  Scenario: Placeholder test
    Given the BDD harness is set up
    When I run the tests
    Then the structure should be ready
EOF

    echo "✅ Created BDD for $crate_name"
}

# Shared crates
create_bdd "shared-crates/daemon-lifecycle" "daemon-lifecycle" "daemon-lifecycle"
create_bdd "shared-crates/rbee-http-client" "rbee-http-client" "rbee-http-client"
create_bdd "shared-crates/rbee-types" "rbee-types" "rbee-types"

# rbee-keeper crates
create_bdd "rbee-keeper-crates/config" "rbee-keeper-config" "rbee-keeper-config"
create_bdd "rbee-keeper-crates/cli" "rbee-keeper-cli" "rbee-keeper-cli"
create_bdd "rbee-keeper-crates/commands" "rbee-keeper-commands" "rbee-keeper-commands"

# queen-rbee crates
create_bdd "queen-rbee-crates/ssh-client" "queen-rbee-ssh-client" "queen-rbee-ssh-client"
create_bdd "queen-rbee-crates/hive-registry" "queen-rbee-hive-registry" "queen-rbee-hive-registry"
create_bdd "queen-rbee-crates/worker-registry" "queen-rbee-worker-registry" "queen-rbee-worker-registry"
create_bdd "queen-rbee-crates/hive-lifecycle" "queen-rbee-hive-lifecycle" "queen-rbee-hive-lifecycle"
create_bdd "queen-rbee-crates/http-server" "queen-rbee-http-server" "queen-rbee-http-server"
create_bdd "queen-rbee-crates/preflight" "queen-rbee-preflight" "queen-rbee-preflight"

# rbee-hive crates
create_bdd "rbee-hive-crates/worker-lifecycle" "rbee-hive-worker-lifecycle" "rbee-hive-worker-lifecycle"
create_bdd "rbee-hive-crates/worker-registry" "rbee-hive-worker-registry" "rbee-hive-worker-registry"
create_bdd "rbee-hive-crates/model-catalog" "rbee-hive-model-catalog" "rbee-hive-model-catalog"
create_bdd "rbee-hive-crates/model-provisioner" "rbee-hive-model-provisioner" "rbee-hive-model-provisioner"
create_bdd "rbee-hive-crates/monitor" "rbee-hive-monitor" "rbee-hive-monitor"
create_bdd "rbee-hive-crates/http-server" "rbee-hive-http-server" "rbee-hive-http-server"
create_bdd "rbee-hive-crates/download-tracker" "rbee-hive-download-tracker" "rbee-hive-download-tracker"
create_bdd "rbee-hive-crates/device-detection" "rbee-hive-device-detection" "rbee-hive-device-detection"

# worker-rbee crates
create_bdd "worker-rbee-crates/http-server" "worker-rbee-http-server" "worker-rbee-http-server"
create_bdd "worker-rbee-crates/heartbeat" "worker-rbee-heartbeat" "worker-rbee-heartbeat"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ BDD SCAFFOLDING COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Created BDD harnesses for 22 crates:"
echo "  - 3 shared crates"
echo "  - 3 rbee-keeper crates"
echo "  - 6 queen-rbee crates"
echo "  - 8 rbee-hive crates"
echo "  - 2 worker-rbee crates"
echo ""
echo "Next: Update workspace Cargo.toml to include BDD crates"
