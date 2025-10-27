# TEAM-314: Search Instructions for Contract Implementation

**Purpose:** How to search for implementations to ensure parity

---

## Overview

When implementing contracts, you need to find ALL places where types are used to ensure nothing is missed.

---

## General Search Strategy

### 1. Find Type Definitions

```bash
# Find struct definitions
rg "pub struct TypeName" --type rust

# Find enum definitions
rg "pub enum TypeName" --type rust

# Find type aliases
rg "pub type TypeName" --type rust
```

### 2. Find All Usages

```bash
# Find all uses of a type
rg "TypeName" --type rust

# Find imports
rg "use.*TypeName" --type rust

# Find in specific directory
rg "TypeName" bin/05_rbee_keeper_crates/ --type rust
```

### 3. Find Serialization

```bash
# Find types with Serialize/Deserialize
rg "#\[derive.*Serialize.*Deserialize" --type rust

# Find serde attributes
rg "#\[serde" --type rust
```

---

## daemon-contract Searches

### Finding DaemonHandle

```bash
# Current implementation
rg "pub struct QueenHandle" -A 30 bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs

# All usages
rg "QueenHandle" bin/05_rbee_keeper_crates/queen-lifecycle/ --type rust
rg "QueenHandle" bin/00_rbee_keeper/ --type rust

# Check if HiveHandle exists
rg "HiveHandle" --type rust

# Check if WorkerHandle exists
rg "WorkerHandle" --type rust
```

### Finding Status Types

```bash
# Current implementation
rg "pub struct StatusRequest" -A 10 bin/99_shared_crates/daemon-lifecycle/src/status.rs
rg "pub struct StatusResponse" -A 10 bin/99_shared_crates/daemon-lifecycle/src/status.rs

# All usages
rg "StatusRequest|StatusResponse" --type rust

# Check for similar types
rg "struct.*Status.*Request" --type rust
rg "struct.*Status.*Response" --type rust
```

### Finding Install Types

```bash
# Current implementation
rg "pub struct InstallConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/install.rs
rg "pub struct InstallResult" -A 10 bin/99_shared_crates/daemon-lifecycle/src/install.rs
rg "pub struct UninstallConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/install.rs

# All usages
rg "InstallConfig|InstallResult|UninstallConfig" --type rust

# Check for similar types
rg "struct.*Install" --type rust
rg "struct.*Uninstall" --type rust
```

### Finding Lifecycle Types

```bash
# Current implementation
rg "pub struct HttpDaemonConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/lifecycle.rs

# All usages
rg "HttpDaemonConfig" --type rust

# Check for similar types
rg "struct.*DaemonConfig" --type rust
```

### Finding Shutdown Types

```bash
# Current implementation
rg "pub struct ShutdownConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/shutdown.rs

# All usages
rg "ShutdownConfig" --type rust

# Check for similar types
rg "struct.*Shutdown" --type rust
```

---

## ssh-contract Searches

### Finding SshTarget

```bash
# Current implementations (should find 2)
rg "pub struct SshTarget" -A 15 --type rust

# Specific locations
rg "pub struct SshTarget" bin/05_rbee_keeper_crates/ssh-config/src/lib.rs -A 15
rg "pub struct SshTarget" bin/00_rbee_keeper/src/tauri_commands.rs -A 15

# All usages
rg "SshTarget" --type rust

# Check for imports
rg "use.*SshTarget" --type rust
```

### Finding SshTargetStatus

```bash
# Current implementation
rg "pub enum SshTargetStatus" -A 5 bin/05_rbee_keeper_crates/ssh-config/src/lib.rs

# All usages
rg "SshTargetStatus" --type rust

# Check for similar types
rg "enum.*Ssh.*Status" --type rust
```

---

## keeper-config-contract Searches

### Finding Config

```bash
# Current implementation
rg "pub struct Config" bin/00_rbee_keeper/src/config.rs -A 20

# All usages
rg "Config::" bin/00_rbee_keeper/ --type rust

# Check for similar types
rg "struct.*Config" bin/00_rbee_keeper/ --type rust
```

---

## Checking for Parity

### 1. Compare Implementations

```bash
# Find all struct definitions in a file
rg "pub struct" file1.rs > /tmp/file1_structs.txt
rg "pub struct" file2.rs > /tmp/file2_structs.txt
diff /tmp/file1_structs.txt /tmp/file2_structs.txt
```

### 2. Find Missing Derives

```bash
# Find all derives for a type
rg "struct TypeName" -B 5 --type rust | rg "#\[derive"

# Compare derives between files
rg "struct SshTarget" -B 5 bin/05_rbee_keeper_crates/ssh-config/src/lib.rs
rg "struct SshTarget" -B 5 bin/00_rbee_keeper/src/tauri_commands.rs
```

### 3. Find All Methods

```bash
# Find all impl blocks for a type
rg "impl TypeName" -A 50 --type rust

# Find all public methods
rg "pub fn" file.rs
```

### 4. Find All Tests

```bash
# Find all tests for a type
rg "#\[test\]" -A 10 --type rust | rg "TypeName"

# Find test modules
rg "mod tests" -A 50 --type rust
```

---

## Advanced Searches

### Find Duplicate Types

```bash
# Find all structs with same name
rg "pub struct TypeName" --type rust --count-matches

# If count > 1, find all locations
rg "pub struct TypeName" --type rust --files-with-matches
```

### Find Related Types

```bash
# Find types that contain a field
rg "field_name:" --type rust

# Find types that use another type
rg "TypeA.*TypeB|TypeB.*TypeA" --type rust
```

### Find Serialization Patterns

```bash
# Find custom serde attributes
rg "#\[serde\(" --type rust

# Find skip_serializing_if
rg "skip_serializing_if" --type rust

# Find rename patterns
rg "serde.*rename" --type rust
```

---

## Verification Searches

### After Creating Contract

```bash
# 1. Verify contract compiles
cd bin/97_contracts/new-contract
cargo build

# 2. Find all consumers
rg "OldTypeName" --type rust

# 3. Verify no duplicates remain
rg "pub struct OldTypeName" --type rust --count-matches

# 4. Check imports updated
rg "use.*new_contract" --type rust

# 5. Verify tests pass
cargo test -p new-contract
```

---

## Common Patterns

### Pattern 1: Type with Builder

```bash
# Find builder methods
rg "pub fn with_" file.rs

# Find new() constructors
rg "pub fn new\(" file.rs
```

### Pattern 2: Type with Conversions

```bash
# Find From/Into implementations
rg "impl From<.*> for TypeName" --type rust
rg "impl Into<.*> for TypeName" --type rust

# Find TryFrom/TryInto
rg "impl TryFrom" --type rust
```

### Pattern 3: Type with Serialization

```bash
# Find custom serialization
rg "impl Serialize for TypeName" --type rust
rg "impl Deserialize for TypeName" --type rust

# Find serde modules
rg "mod.*serde" --type rust
```

---

## Checklist for Each Contract

### Before Implementation

- [ ] Find all current implementations
- [ ] Find all usages
- [ ] Find all derives
- [ ] Find all methods
- [ ] Find all tests
- [ ] Check for duplicates

### During Implementation

- [ ] Copy all fields
- [ ] Copy all derives
- [ ] Copy all methods
- [ ] Copy all tests
- [ ] Add documentation
- [ ] Add examples

### After Implementation

- [ ] Verify contract compiles
- [ ] Update all consumers
- [ ] Remove old implementations
- [ ] Verify no duplicates
- [ ] Run all tests
- [ ] Check documentation

---

## Example: Complete Search for SshTarget

```bash
# Step 1: Find all definitions
echo "=== Finding all SshTarget definitions ==="
rg "pub struct SshTarget" --type rust

# Step 2: Find all usages
echo "=== Finding all SshTarget usages ==="
rg "SshTarget" --type rust --count

# Step 3: Find derives
echo "=== Finding derives ==="
rg "struct SshTarget" -B 5 --type rust | rg "#\[derive"

# Step 4: Find methods
echo "=== Finding methods ==="
rg "impl SshTarget" -A 50 --type rust

# Step 5: Find tests
echo "=== Finding tests ==="
rg "#\[test\]" -A 10 --type rust | rg "SshTarget"

# Step 6: Find imports
echo "=== Finding imports ==="
rg "use.*SshTarget" --type rust

# Step 7: Verify no duplicates after migration
echo "=== Verifying no duplicates ==="
rg "pub struct SshTarget" --type rust --count-matches
# Should be 0 after migration (all use contract)
```

---

## Tools

### ripgrep (rg)

```bash
# Install
cargo install ripgrep

# Common flags
rg "pattern" --type rust      # Search only Rust files
rg "pattern" -A 10            # Show 10 lines after match
rg "pattern" -B 5             # Show 5 lines before match
rg "pattern" -C 5             # Show 5 lines context
rg "pattern" --count          # Count matches per file
rg "pattern" --files-with-matches  # Show only filenames
```

### fd (find alternative)

```bash
# Install
cargo install fd-find

# Find files
fd "pattern" --type f         # Find files
fd "pattern" --extension rs   # Find .rs files
```

### ast-grep (structural search)

```bash
# Install
cargo install ast-grep

# Search by AST pattern
ast-grep --pattern 'struct $NAME { $$$FIELDS }'
```

---

## Summary

**Key Commands:**

```bash
# Find type definitions
rg "pub struct TypeName" --type rust

# Find all usages
rg "TypeName" --type rust

# Find duplicates
rg "pub struct TypeName" --type rust --count-matches

# Verify migration
rg "OldTypeName" --type rust  # Should be 0 after migration
```

**Workflow:**

1. Search for current implementation
2. Find all usages
3. Create contract
4. Update consumers
5. Verify no duplicates
6. Run tests

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** GUIDE 📖
