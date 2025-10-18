# Port BDD Test Runner to Rust Plan

**TEAM-111** - Porting bash script to Rust xtask  
**Date:** 2025-10-18  
**Status:** Ready to start

---

## 🎯 Overview

Port the BDD test runner from bash script to Rust xtask, preserving all features while leveraging Rust's type safety and the existing xtask infrastructure.

---

## 📋 Current State

### Bash Script
- ✅ **42 functions**, ~650 lines
- ✅ **8 major features** (live output, failure reporting, rerun scripts, etc.)
- ✅ **Comprehensive documentation** (~3,500 lines)
- ❌ **Wrong language** for this Rust project

### xtask
- ✅ **Cleaned up** (removed 4 stubs)
- ✅ **12 working commands**
- ✅ **Ready for new tasks**

---

## 🎨 Features to Port

### Core Features
1. ✅ **Live output mode** (default) - Stream all stdout/stderr
2. ✅ **Quiet mode** (`--quiet`) - Summary only with spinner
3. ✅ **Tag filtering** (`--tags @auth`)
4. ✅ **Feature filtering** (`--feature lifecycle`)
5. ✅ **Compilation check** - Pre-flight cargo check
6. ✅ **Test discovery** - Count scenarios in feature files
7. ✅ **Result parsing** - Extract passed/failed/skipped counts
8. ✅ **Failure-focused reporting** - Show ONLY failures at end

### Advanced Features
9. ✅ **Failure extraction** - Multiple patterns (FAILED, Error, assertion, panic, stack traces)
10. ✅ **Dedicated failures file** - Organized sections
11. ✅ **Auto-generated rerun command** - Instant retry of failed tests
12. ✅ **Timestamped logs** - All output preserved
13. ✅ **Summary generation** - Human-readable results
14. ✅ **Warning detection** - Compilation warnings
15. ✅ **Exit codes** - 0=pass, 1=fail, 2=error

---

## 🏗️ Rust Architecture

### Module Structure
```
xtask/src/tasks/
├── bdd.rs          # Main BDD test task
└── bdd/
    ├── mod.rs      # Public API
    ├── runner.rs   # Test execution
    ├── parser.rs   # Result parsing
    ├── reporter.rs # Output formatting
    ├── files.rs    # File generation
    └── types.rs    # Data structures
```

### Key Types
```rust
pub struct BddConfig {
    pub tags: Option<String>,
    pub feature: Option<String>,
    pub quiet: bool,
}

pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub status: i32,
}

pub struct FailureInfo {
    pub test_name: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
}
```

### Command Structure
```rust
// In cli.rs
#[derive(Subcommand)]
pub enum Cmd {
    // ... existing commands ...
    
    #[command(name = "bdd:test")]
    BddTest {
        /// Run tests with specific tag
        #[arg(long)]
        tags: Option<String>,
        
        /// Run specific feature file
        #[arg(long)]
        feature: Option<String>,
        
        /// Suppress live output (only show summary)
        #[arg(long, short)]
        quiet: bool,
    },
}
```

---

## 📝 Implementation Plan

### Phase 1: Basic Structure (1-2 hours)
- [ ] Create `src/tasks/bdd.rs`
- [ ] Add `BddTest` command to `cli.rs`
- [ ] Add handler to `main.rs`
- [ ] Implement basic test execution
- [ ] Test with `cargo xtask bdd:test`

### Phase 2: Core Features (2-3 hours)
- [ ] Implement tag filtering
- [ ] Implement feature filtering
- [ ] Implement quiet mode with spinner
- [ ] Implement live output mode
- [ ] Add compilation check

### Phase 3: Result Parsing (1-2 hours)
- [ ] Parse test output
- [ ] Extract passed/failed/skipped counts
- [ ] Detect test status

### Phase 4: Failure Reporting (2-3 hours)
- [ ] Extract failure patterns
- [ ] Generate failures file
- [ ] Display failure details
- [ ] Implement failure-focused output

### Phase 5: Rerun Feature (1-2 hours)
- [ ] Extract failed test names
- [ ] Generate rerun command
- [ ] Save to file

### Phase 6: Polish (1-2 hours)
- [ ] Timestamped log files
- [ ] Summary generation
- [ ] Warning detection
- [ ] Error handling
- [ ] Documentation

**Total Estimated Time: 8-14 hours**

---

## 🔧 Rust Advantages

### Type Safety
```rust
// Bash: PASSED=$(grep -o "[0-9]*" file)
// Rust: Type-safe parsing
let passed: usize = parse_count(&output, "passed")?;
```

### Error Handling
```rust
// Bash: if [[ $? -ne 0 ]]; then
// Rust: Result type
fn run_tests() -> Result<TestResults> {
    let output = Command::new("cargo")
        .arg("test")
        .output()
        .context("running tests")?;
    // ...
}
```

### Pattern Matching
```rust
// Bash: case $1 in ... esac
// Rust: Exhaustive matching
match cmd {
    Cmd::BddTest { tags, feature, quiet } => {
        bdd::run_tests(tags, feature, quiet)?
    }
}
```

### Structured Data
```rust
// Bash: Multiple variables
// Rust: Structured types
struct TestResults {
    passed: usize,
    failed: usize,
    skipped: usize,
}
```

---

## 📚 Crates to Use

### Already in xtask
- ✅ `anyhow` - Error handling
- ✅ `clap` - CLI parsing

### To Add
- `indicatif` - Progress bars and spinners
- `colored` - Colored output
- `regex` - Pattern matching
- `chrono` - Timestamps

---

## 🎯 Success Criteria

### Functionality
- [ ] All bash script features work
- [ ] Tag filtering works
- [ ] Feature filtering works
- [ ] Quiet mode works
- [ ] Live mode works
- [ ] Failure extraction works
- [ ] Rerun generation works

### Quality
- [ ] Type-safe implementation
- [ ] Proper error handling
- [ ] Clean code structure
- [ ] Good test coverage
- [ ] Clear documentation

### Integration
- [ ] Works with existing xtask
- [ ] Follows xtask patterns
- [ ] Uses workspace dependencies
- [ ] CI integration ready

---

## 📖 Documentation Plan

### Update Existing
- [ ] `xtask/README.md` - Add BDD commands
- [ ] `test-harness/bdd/README.md` - Update with xtask usage

### Create New
- [ ] `test-harness/bdd/USAGE.md` - How to use xtask commands
- [ ] `xtask/src/tasks/bdd/README.md` - Implementation notes

### Archive
- [ ] Move bash docs to `.archive/bash-script/`
- [ ] Keep as reference during port

---

## 🚀 Quick Start

### 1. Cleanup Bash Docs
```bash
cd test-harness/bdd
./cleanup-bash-docs.sh
```

### 2. Start Rust Implementation
```bash
cd xtask
# Create bdd task module
mkdir -p src/tasks/bdd
touch src/tasks/bdd/mod.rs
```

### 3. Add Dependencies
```toml
# xtask/Cargo.toml
[dependencies]
indicatif = "0.17"
colored = "2.0"
regex = "1.10"
chrono = "0.4"
```

### 4. Implement & Test
```bash
# Implement features
# Test as you go
cargo xtask bdd:test
```

---

## 💡 Key Decisions

### Command Names
- `bdd:test` - Run all tests (live output)
- Use flags for options (`--quiet`, `--tags`, `--feature`)

### Output Files
- Same structure as bash: `.test-logs/` directory
- Timestamped files
- failures, summary, full log

### Rerun Strategy
- Generate command file (not executable script)
- User copies command to run
- Simpler and safer

---

## ✅ Checklist

### Pre-Port
- [x] Inventory bash documentation
- [x] Create cleanup script
- [x] Create port plan
- [ ] Run cleanup script
- [ ] Review archived docs

### During Port
- [ ] Implement basic structure
- [ ] Port core features
- [ ] Port advanced features
- [ ] Test thoroughly
- [ ] Update documentation

### Post-Port
- [ ] Delete bash script (if not needed)
- [ ] Update all references
- [ ] CI integration
- [ ] Team review

---

## 🎉 Expected Outcome

**A world-class Rust-based BDD test runner that:**
- ✅ Matches all bash script features
- ✅ Leverages Rust's type safety
- ✅ Integrates with existing xtask
- ✅ Is maintainable and extensible
- ✅ Follows Rust best practices
- ✅ Has clear documentation

**And eliminates:**
- ❌ Bash-specific quirks
- ❌ Pipeline anti-patterns
- ❌ String parsing fragility
- ❌ Language mismatch

---

**TEAM-111** - Ready to build the Rust version! 🦀🚀
