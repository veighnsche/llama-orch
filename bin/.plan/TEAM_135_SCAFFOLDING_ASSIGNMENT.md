# TEAM-135: SCAFFOLDING ASSIGNMENT

**Date:** 2025-10-19  
**Team:** TEAM-135  
**Phase:** Phase 3 - Crate Scaffolding  
**Status:** 🚧 IN PROGRESS

---

## 🎯 MISSION

Create the complete directory structure and scaffolding for the new crate-based architecture.

**Current State:** All binaries renamed to `*.bak`  
**Target State:** New crate structure with proper Cargo.toml files and empty lib.rs stubs

**Reference Document:** `TEAM_130G_FINAL_ARCHITECTURE.md`

---

## 📋 DELIVERABLES

### 1. Shared Crates (3 crates in `bin/shared-crates/`)

Create these NEW shared crates:

```
bin/shared-crates/
├─ daemon-lifecycle/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ rbee-http-client/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
└─ rbee-types/
   ├─ src/lib.rs
   ├─ Cargo.toml
   └─ README.md
```

**NOTE:** Only 3 shared crates! SSH is NOT shared.

---

### 2. rbee-keeper Binary + Crates

```
bin/rbee-keeper/
├─ src/
│  ├─ main.rs                     (~12 LOC stub)
│  └─ lib.rs                      (~10 LOC stub)
├─ Cargo.toml
└─ README.md

bin/rbee-keeper-crates/
├─ config/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ cli/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
└─ commands/
   ├─ src/
   │  ├─ lib.rs
   │  ├─ infer.rs
   │  ├─ setup.rs
   │  ├─ workers.rs
   │  ├─ logs.rs
   │  └─ install.rs
   ├─ Cargo.toml
   └─ README.md
```

---

### 3. queen-rbee Binary + Crates

```
bin/queen-rbee/
├─ src/
│  ├─ main.rs                     (~50 LOC stub)
│  └─ lib.rs                      (~30 LOC stub)
├─ Cargo.toml
└─ README.md

bin/queen-rbee-crates/
├─ ssh-client/                    ← ONLY queen uses SSH!
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ hive-registry/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ worker-registry/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ hive-lifecycle/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ http-server/
│  ├─ src/
│  │  ├─ lib.rs
│  │  ├─ routes.rs
│  │  ├─ health.rs
│  │  ├─ beehives.rs
│  │  ├─ workers.rs
│  │  └─ inference.rs
│  ├─ Cargo.toml
│  └─ README.md
└─ preflight/
   ├─ src/lib.rs
   ├─ Cargo.toml
   └─ README.md
```

---

### 4. rbee-hive Binary + Crates

```
bin/rbee-hive/
├─ src/
│  ├─ main.rs                     (~50 LOC stub - daemon args only!)
│  └─ lib.rs                      (~30 LOC stub)
├─ Cargo.toml
└─ README.md

bin/rbee-hive-crates/              ← NO CLI crate!
├─ worker-lifecycle/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ worker-registry/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ model-catalog/                 (MOVED from shared-crates)
│  ├─ src/
│  │  ├─ lib.rs
│  │  ├─ catalog.rs
│  │  └─ types.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ model-provisioner/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ monitor/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ http-server/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
└─ download-tracker/
   ├─ src/lib.rs
   ├─ Cargo.toml
   └─ README.md
```

**CRITICAL:** NO `cli/` crate in rbee-hive! Daemon only!

---

### 5. llm-worker-rbee Binary + Crates

```
bin/llm-worker-rbee/
├─ src/
│  ├─ main.rs                     (~50 LOC stub)
│  ├─ lib.rs                      (~80 LOC stub)
│  ├─ bin/
│  │  ├─ cpu.rs                   (~30 LOC stub)
│  │  ├─ cuda.rs                  (~30 LOC stub)
│  │  └─ metal.rs                 (~30 LOC stub)
│  └─ backend/                    ← EXCEPTION: Stays in binary!
│     ├─ mod.rs
│     ├─ inference.rs
│     ├─ sampling.rs
│     ├─ tokenizer_loader.rs
│     ├─ gguf_tokenizer.rs
│     └─ models/
│        ├─ mod.rs
│        ├─ llama.rs
│        ├─ mistral.rs
│        ├─ phi.rs
│        └─ qwen.rs
├─ Cargo.toml
└─ README.md

bin/worker-rbee-crates/
├─ http-server/
│  ├─ src/
│  │  ├─ lib.rs
│  │  ├─ validation.rs
│  │  ├─ execute.rs
│  │  └─ health.rs
│  ├─ Cargo.toml
│  └─ README.md
├─ device-detection/
│  ├─ src/lib.rs
│  ├─ Cargo.toml
│  └─ README.md
└─ heartbeat/
   ├─ src/lib.rs
   ├─ Cargo.toml
   └─ README.md
```

**EXCEPTION:** `src/backend/` stays in binary (LLM-specific inference)

---

## 📝 SCAFFOLDING REQUIREMENTS

### For Each Crate

#### 1. Cargo.toml Template

```toml
[package]
name = "crate-name"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[lib]
name = "crate_name"
path = "src/lib.rs"

[dependencies]
# Add dependencies as needed

[dev-dependencies]
# Add dev dependencies as needed
```

#### 2. lib.rs Stub Template

```rust
// TEAM-135: Created by TEAM-135 (scaffolding)
// Purpose: [Brief description]
// Status: STUB - Awaiting implementation

#![warn(missing_docs)]
#![warn(clippy::all)]

//! [Crate name]
//!
//! [Brief description of what this crate does]

// TODO: Implement crate functionality
```

#### 3. README.md Template

```markdown
# [Crate Name]

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** [Brief description]

## Overview

[What this crate does]

## Dependencies

- [List dependencies]

## Usage

```rust
// TODO: Add usage examples
```

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

[Any important notes]
```

---

## 🔧 IMPLEMENTATION STEPS

### Step 1: Create Shared Crates (30 min)

```bash
cd /home/vince/Projects/llama-orch/bin/shared-crates

# Create daemon-lifecycle
mkdir -p daemon-lifecycle/src
touch daemon-lifecycle/src/lib.rs
touch daemon-lifecycle/Cargo.toml
touch daemon-lifecycle/README.md

# Create rbee-http-client
mkdir -p rbee-http-client/src
touch rbee-http-client/src/lib.rs
touch rbee-http-client/Cargo.toml
touch rbee-http-client/README.md

# Create rbee-types
mkdir -p rbee-types/src
touch rbee-types/src/lib.rs
touch rbee-types/Cargo.toml
touch rbee-types/README.md
```

### Step 2: Create rbee-keeper Structure (30 min)

```bash
cd /home/vince/Projects/llama-orch/bin

# Create binary
mkdir -p rbee-keeper/src
touch rbee-keeper/src/main.rs
touch rbee-keeper/src/lib.rs
touch rbee-keeper/Cargo.toml
touch rbee-keeper/README.md

# Create crates
mkdir -p rbee-keeper-crates/config/src
mkdir -p rbee-keeper-crates/cli/src
mkdir -p rbee-keeper-crates/commands/src

# Create Cargo.toml and README.md for each crate
for crate in config cli commands; do
  touch rbee-keeper-crates/$crate/Cargo.toml
  touch rbee-keeper-crates/$crate/README.md
  touch rbee-keeper-crates/$crate/src/lib.rs
done

# Create command module files
touch rbee-keeper-crates/commands/src/infer.rs
touch rbee-keeper-crates/commands/src/setup.rs
touch rbee-keeper-crates/commands/src/workers.rs
touch rbee-keeper-crates/commands/src/logs.rs
touch rbee-keeper-crates/commands/src/install.rs
```

### Step 3: Create queen-rbee Structure (45 min)

```bash
cd /home/vince/Projects/llama-orch/bin

# Create binary
mkdir -p queen-rbee/src
touch queen-rbee/src/main.rs
touch queen-rbee/src/lib.rs
touch queen-rbee/Cargo.toml
touch queen-rbee/README.md

# Create crates
mkdir -p queen-rbee-crates/ssh-client/src
mkdir -p queen-rbee-crates/hive-registry/src
mkdir -p queen-rbee-crates/worker-registry/src
mkdir -p queen-rbee-crates/hive-lifecycle/src
mkdir -p queen-rbee-crates/http-server/src
mkdir -p queen-rbee-crates/preflight/src

# Create Cargo.toml and README.md for each crate
for crate in ssh-client hive-registry worker-registry hive-lifecycle http-server preflight; do
  touch queen-rbee-crates/$crate/Cargo.toml
  touch queen-rbee-crates/$crate/README.md
  touch queen-rbee-crates/$crate/src/lib.rs
done

# Create http-server module files
touch queen-rbee-crates/http-server/src/routes.rs
touch queen-rbee-crates/http-server/src/health.rs
touch queen-rbee-crates/http-server/src/beehives.rs
touch queen-rbee-crates/http-server/src/workers.rs
touch queen-rbee-crates/http-server/src/inference.rs
```

### Step 4: Create rbee-hive Structure (45 min)

```bash
cd /home/vince/Projects/llama-orch/bin

# Create binary
mkdir -p rbee-hive/src
touch rbee-hive/src/main.rs
touch rbee-hive/src/lib.rs
touch rbee-hive/Cargo.toml
touch rbee-hive/README.md

# Create crates (NO CLI!)
mkdir -p rbee-hive-crates/worker-lifecycle/src
mkdir -p rbee-hive-crates/worker-registry/src
mkdir -p rbee-hive-crates/model-catalog/src
mkdir -p rbee-hive-crates/model-provisioner/src
mkdir -p rbee-hive-crates/monitor/src
mkdir -p rbee-hive-crates/http-server/src
mkdir -p rbee-hive-crates/download-tracker/src

# Create Cargo.toml and README.md for each crate
for crate in worker-lifecycle worker-registry model-catalog model-provisioner monitor http-server download-tracker; do
  touch rbee-hive-crates/$crate/Cargo.toml
  touch rbee-hive-crates/$crate/README.md
  touch rbee-hive-crates/$crate/src/lib.rs
done

# Create model-catalog module files
touch rbee-hive-crates/model-catalog/src/catalog.rs
touch rbee-hive-crates/model-catalog/src/types.rs
```

### Step 5: Create llm-worker-rbee Structure (45 min)

```bash
cd /home/vince/Projects/llama-orch/bin

# Create binary
mkdir -p llm-worker-rbee/src/bin
mkdir -p llm-worker-rbee/src/backend/models
touch llm-worker-rbee/src/main.rs
touch llm-worker-rbee/src/lib.rs
touch llm-worker-rbee/Cargo.toml
touch llm-worker-rbee/README.md

# Create bin files
touch llm-worker-rbee/src/bin/cpu.rs
touch llm-worker-rbee/src/bin/cuda.rs
touch llm-worker-rbee/src/bin/metal.rs

# Create backend files (stays in binary!)
touch llm-worker-rbee/src/backend/mod.rs
touch llm-worker-rbee/src/backend/inference.rs
touch llm-worker-rbee/src/backend/sampling.rs
touch llm-worker-rbee/src/backend/tokenizer_loader.rs
touch llm-worker-rbee/src/backend/gguf_tokenizer.rs
touch llm-worker-rbee/src/backend/models/mod.rs
touch llm-worker-rbee/src/backend/models/llama.rs
touch llm-worker-rbee/src/backend/models/mistral.rs
touch llm-worker-rbee/src/backend/models/phi.rs
touch llm-worker-rbee/src/backend/models/qwen.rs

# Create crates
mkdir -p llm-worker-rbee-crates/http-server/src
mkdir -p llm-worker-rbee-crates/device-detection/src
mkdir -p llm-worker-rbee-crates/heartbeat/src

# Create Cargo.toml and README.md for each crate
for crate in http-server device-detection heartbeat; do
  touch llm-worker-rbee-crates/$crate/Cargo.toml
  touch llm-worker-rbee-crates/$crate/README.md
  touch llm-worker-rbee-crates/$crate/src/lib.rs
done

# Create http-server module files
touch llm-worker-rbee-crates/http-server/src/validation.rs
touch llm-worker-rbee-crates/http-server/src/execute.rs
touch llm-worker-rbee-crates/http-server/src/health.rs
```

### Step 6: Populate Cargo.toml Files (1 hour)

For each crate, create a proper Cargo.toml using the template above.

**CRITICAL:** Use correct crate names (snake_case for lib names, kebab-case for package names)

### Step 7: Populate lib.rs Stubs (30 min)

For each crate, create a minimal lib.rs stub with:
- TEAM-135 signature
- Purpose comment
- Status: STUB
- Basic module structure

### Step 8: Populate README.md Files (30 min)

For each crate, create a README.md with:
- Crate name
- Status: STUB
- Purpose
- Basic structure

---

## ✅ ACCEPTANCE CRITERIA

### Structure
- [ ] All 3 shared crates created
- [ ] All 4 binaries created with src/main.rs and src/lib.rs
- [ ] All binary-specific crate directories created
- [ ] NO `rbee-hive-crates/cli/` directory (daemon only!)
- [ ] `llm-worker-rbee/src/backend/` exists (exception)

### Files
- [ ] Every crate has Cargo.toml
- [ ] Every crate has README.md
- [ ] Every crate has src/lib.rs (or src/main.rs for binaries)
- [ ] All module files created (empty is OK)

### Compilation
- [ ] `cargo check --workspace` runs without errors
- [ ] All Cargo.toml files are valid
- [ ] All crate names follow conventions (snake_case lib, kebab-case package)

### Documentation
- [ ] Every Cargo.toml has TEAM-135 comment
- [ ] Every lib.rs has TEAM-135 signature
- [ ] Every README.md has status: STUB

---

## 🚨 CRITICAL RULES

### 1. NO SSH in shared-crates!
- ✅ `queen-rbee-crates/ssh-client/` (correct)
- ❌ `shared-crates/rbee-ssh-client/` (WRONG!)

### 2. NO CLI in rbee-hive!
- ✅ `rbee-hive/src/main.rs` with daemon args only
- ❌ `rbee-hive-crates/cli/` (WRONG!)

### 3. backend/ Stays in llm-worker Binary!
- ✅ `llm-worker-rbee/src/backend/` (correct)
- ❌ `llm-worker-rbee-crates/backend/` (WRONG!)

### 4. Crate Naming
- Package name: `kebab-case` (e.g., `daemon-lifecycle`)
- Lib name: `snake_case` (e.g., `daemon_lifecycle`)
- Directory: `kebab-case` (e.g., `daemon-lifecycle/`)

---

## 📊 EXPECTED DIRECTORY COUNT

```
bin/
├─ shared-crates/           (3 crates)
├─ rbee-keeper/             (1 binary)
├─ rbee-keeper-crates/      (3 crates)
├─ queen-rbee/              (1 binary)
├─ queen-rbee-crates/       (6 crates)
├─ rbee-hive/               (1 binary)
├─ rbee-hive-crates/        (7 crates)
├─ llm-worker-rbee/         (1 binary)
└─ llm-worker-rbee-crates/  (3 crates)

Total: 4 binaries + 22 crates = 26 directories
```

---

## 📝 VERIFICATION SCRIPT

```bash
#!/bin/bash
# verify_scaffolding.sh

echo "Verifying scaffolding..."

# Check shared crates
for crate in daemon-lifecycle rbee-http-client rbee-types; do
  if [ ! -d "bin/shared-crates/$crate" ]; then
    echo "❌ Missing: bin/shared-crates/$crate"
  else
    echo "✅ Found: bin/shared-crates/$crate"
  fi
done

# Check binaries
for binary in rbee-keeper queen-rbee rbee-hive llm-worker-rbee; do
  if [ ! -f "bin/$binary/src/main.rs" ]; then
    echo "❌ Missing: bin/$binary/src/main.rs"
  else
    echo "✅ Found: bin/$binary/src/main.rs"
  fi
done

# Check NO CLI in rbee-hive
if [ -d "bin/rbee-hive-crates/cli" ]; then
  echo "❌ VIOLATION: bin/rbee-hive-crates/cli/ should NOT exist!"
else
  echo "✅ Correct: No CLI in rbee-hive"
fi

# Check SSH in queen-rbee-crates (NOT shared-crates)
if [ -d "bin/queen-rbee-crates/ssh-client" ]; then
  echo "✅ Correct: SSH in queen-rbee-crates"
else
  echo "❌ Missing: bin/queen-rbee-crates/ssh-client"
fi

if [ -d "bin/shared-crates/rbee-ssh-client" ]; then
  echo "❌ VIOLATION: SSH should NOT be in shared-crates!"
else
  echo "✅ Correct: No SSH in shared-crates"
fi

# Check backend in llm-worker binary
if [ -d "bin/llm-worker-rbee/src/backend" ]; then
  echo "✅ Correct: backend in llm-worker binary"
else
  echo "❌ Missing: bin/llm-worker-rbee/src/backend"
fi

echo ""
echo "Running cargo check..."
cargo check --workspace
```

---

## 🎯 TIME ESTIMATE

- Step 1: Create shared crates (30 min)
- Step 2: Create rbee-keeper (30 min)
- Step 3: Create queen-rbee (45 min)
- Step 4: Create rbee-hive (45 min)
- Step 5: Create llm-worker-rbee (45 min)
- Step 6: Populate Cargo.toml (1 hour)
- Step 7: Populate lib.rs stubs (30 min)
- Step 8: Populate README.md (30 min)
- Step 9: Verification (30 min)

**Total: ~5 hours**

---

## 📋 HANDOFF TO NEXT TEAM

After scaffolding is complete, TEAM-136 will:
1. Migrate code from `*.bak` binaries to new crate structure
2. Implement shared crate functionality
3. Remove violations (SSH, CLI, etc.)
4. Test compilation

**TEAM-135 delivers:** Empty scaffolding with proper structure  
**TEAM-136 receives:** Ready-to-populate crate structure

---

**Status:** 🚧 READY TO START  
**Team:** TEAM-135  
**Estimated Time:** 5 hours  
**Next Team:** TEAM-136 (Migration)

---

**END OF TEAM-135 ASSIGNMENT**
