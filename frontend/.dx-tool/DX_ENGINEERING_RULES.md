# 🛠️ DX ENGINEERING RULES

> **UNIVERSAL RULES FOR BUILDING DEVELOPER TOOLS**

**Version:** 1.0  
**Date:** 2025-10-12  
**Status:** MANDATORY for all DX tool development  
**Scope:** Global - applies to any developer tooling project

---

## ⚠️ CRITICAL: READ THIS FIRST

These rules apply to **ANY tool built for developers**, not just this project.

**Good DX tools are:**
- Fast (< 2 seconds)
- Reliable (deterministic, no flakiness)
- Clear (actionable output, not noise)
- Composable (works with other tools)
- Self-documenting (helpful errors)

**Bad DX tools:**
- Slow (developers won't use them)
- Flaky (breaks trust)
- Noisy (too much output)
- Isolated (can't integrate)
- Cryptic (unhelpful errors)

**Build good tools. Follow these rules.**

---

## 1. Performance Rules

### ⚠️ MANDATORY: Sub-2-Second Response Time

**Every command MUST complete in < 2 seconds for typical use cases.**

✅ **VALID:**
- Simple queries: < 500ms
- Complex analysis: < 2s
- Network requests: < 2s (with timeout)

❌ **INVALID:**
- Slow parsing (optimize algorithms)
- Unnecessary work (cache results)
- No timeout (hangs indefinitely)

### Why This Matters

Developers run tools hundreds of times per day. If your tool takes 5 seconds, that's:
- **5 seconds × 100 runs = 8 minutes wasted per day**
- **8 minutes × 5 days = 40 minutes wasted per week**
- **40 minutes × 50 weeks = 33 hours wasted per year**

**Per developer. Make it fast.**

### Implementation

```rust
// ✅ GOOD: Fast with timeout
#[tokio::main]
async fn main() {
    let timeout = Duration::from_secs(2);
    let result = tokio::time::timeout(timeout, fetch_data()).await;
}

// ❌ BAD: Slow, no timeout
fn main() {
    let result = fetch_data(); // Could hang forever
}
```

---

## 2. Reliability Rules

### ⚠️ MANDATORY: Deterministic Output

**Same input MUST produce same output. Every time. No exceptions.**

✅ **VALID:**
- Deterministic algorithms
- Stable sorting
- Consistent formatting
- Reproducible results

❌ **INVALID:**
- Random ordering
- Timestamp-dependent output
- Non-deterministic tests
- Flaky behavior

### Why This Matters

Developers need to trust your tool. If it gives different results for the same input:
- They won't trust it
- They won't use it
- They'll build their own (badly)

**Make it deterministic.**

### Implementation

```rust
// ✅ GOOD: Deterministic
let mut results = vec!["c", "a", "b"];
results.sort(); // Always: ["a", "b", "c"]

// ❌ BAD: Non-deterministic
let results: HashSet<_> = vec!["a", "b", "c"].into_iter().collect();
// Order undefined, changes between runs
```

---

## 3. Output Rules

### ⚠️ MANDATORY: Signal-to-Noise Ratio

**Output MUST be actionable. No noise. No clutter.**

✅ **VALID:**
- Clear success/failure indicators
- Relevant information only
- Structured output
- Helpful suggestions

❌ **INVALID:**
- Debug logs in production
- Verbose stack traces
- Irrelevant warnings
- Raw data dumps

### Output Hierarchy

1. **Errors** - Red, clear, actionable
2. **Warnings** - Yellow, important but not blocking
3. **Success** - Green, concise
4. **Info** - Gray, minimal

### Implementation

```rust
// ✅ GOOD: Clear, actionable
println!("✓ Class 'cursor-pointer' found in stylesheet");

// ❌ BAD: Noisy, unclear
println!("DEBUG: Parsing CSS...");
println!("DEBUG: Found 1234 rules...");
println!("DEBUG: Searching for class...");
println!("INFO: Class found at position 5678");
println!("cursor-pointer { cursor: pointer; }");
```

### Error Messages

```rust
// ✅ GOOD: Helpful error
eprintln!("✗ Error: Class 'cursor-pointer' not found in stylesheet");
eprintln!("  Possible causes:");
eprintln!("    - Class not used in any component");
eprintln!("    - Tailwind not scanning source files");
eprintln!("  Suggestion: Run 'dx css --unused' to find unused classes");

// ❌ BAD: Cryptic error
eprintln!("Error: NotFound");
```

---

## 4. Composability Rules

### ⚠️ MANDATORY: Unix Philosophy

**Tools MUST be composable with other tools.**

✅ **VALID:**
- Pipe-friendly output
- Exit codes (0 = success, non-zero = failure)
- JSON output option
- Stdin/stdout support

❌ **INVALID:**
- Interactive prompts in scripts
- Hardcoded output format
- No exit codes
- Can't pipe output

### Implementation

```rust
// ✅ GOOD: Composable
#[derive(Parser)]
struct Cli {
    #[arg(long)]
    format: Option<Format>, // text, json, csv
    
    #[arg(long)]
    quiet: bool, // Suppress non-error output
}

fn main() -> ExitCode {
    match run() {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

// ❌ BAD: Not composable
fn main() {
    println!("Press Enter to continue...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap(); // Blocks in scripts!
}
```

### Exit Codes

```rust
pub enum ExitCode {
    Success = 0,
    GeneralError = 1,
    NetworkError = 2,
    ParseError = 3,
    AssertionFailed = 4,
    Timeout = 5,
}
```

---

## 5. Self-Documentation Rules

### ⚠️ MANDATORY: Helpful Errors

**Errors MUST suggest solutions, not just describe problems.**

✅ **VALID:**
- Clear error message
- Possible causes
- Suggested fixes
- Links to docs

❌ **INVALID:**
- Generic error message
- No context
- No suggestions
- Stack trace only

### Implementation

```rust
// ✅ GOOD: Self-documenting error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Server startup timeout ({timeout}s)\n\
             Server process is running but not responding\n\
             \n\
             Suggestions:\n\
             - Increase timeout: --timeout 60\n\
             - Check server logs: {log_file}\n\
             - Verify port is correct: --port {port}")]
    ServerTimeout {
        timeout: u64,
        log_file: String,
        port: u16,
    },
}

// ❌ BAD: Cryptic error
#[derive(Debug, thiserror::Error)]
#[error("Timeout")]
pub struct TimeoutError;
```

---

## 6. Configuration Rules

### ⚠️ MANDATORY: Sensible Defaults

**Tools MUST work out-of-the-box with zero configuration.**

✅ **VALID:**
- Sensible defaults
- Auto-detection
- Optional config file
- Environment variables

❌ **INVALID:**
- Required config file
- No defaults
- Complex setup
- Mandatory flags

### Configuration Hierarchy

1. **Command-line flags** (highest priority)
2. **Environment variables**
3. **Config file** (`.dxrc.json`)
4. **Sensible defaults** (lowest priority)

### Implementation

```rust
// ✅ GOOD: Sensible defaults
#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "http://localhost:3000")]
    url: String,
    
    #[arg(long, default_value = "30")]
    timeout: u64,
    
    #[arg(long, env = "DX_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

// Load config with fallbacks
let config = Config::load()
    .or_else(|_| Config::from_env())
    .unwrap_or_default();

// ❌ BAD: No defaults
#[derive(Parser)]
struct Cli {
    #[arg(long)] // Required, no default
    url: String,
    
    #[arg(long)] // Required, no default
    timeout: u64,
}
```

---

## 7. Testing Rules

### ⚠️ MANDATORY: Test the Happy Path AND Edge Cases

**Every feature MUST have tests for:**
1. Happy path (normal usage)
2. Error cases (invalid input)
3. Edge cases (empty, null, large)
4. Integration (works with other tools)

✅ **VALID:**
- Unit tests for core logic
- Integration tests for workflows
- Property tests for algorithms
- Snapshot tests for output

❌ **INVALID:**
- No tests
- Only happy path tests
- Flaky tests
- Tests that don't run in CI

### Implementation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Happy path
    #[test]
    fn test_class_exists() {
        let css = ".cursor-pointer { cursor: pointer; }";
        assert!(find_class(css, "cursor-pointer").is_some());
    }
    
    // Error case
    #[test]
    fn test_class_not_found() {
        let css = ".other-class { }";
        assert!(find_class(css, "cursor-pointer").is_none());
    }
    
    // Edge case
    #[test]
    fn test_empty_css() {
        let css = "";
        assert!(find_class(css, "cursor-pointer").is_none());
    }
    
    // Integration
    #[tokio::test]
    async fn test_full_workflow() {
        let server = start_test_server().await;
        let result = check_class_exists(&server.url(), "test-class").await;
        assert!(result.is_ok());
    }
}
```

---

## 8. Distribution Rules

### ⚠️ MANDATORY: Single Binary, No Runtime Dependencies

**Tools MUST be distributed as a single binary with no runtime dependencies.**

✅ **VALID:**
- Statically linked binary
- No external dependencies
- Cross-platform (Linux, macOS, Windows)
- Easy installation

❌ **INVALID:**
- Requires Python/Node/Ruby
- Requires system libraries
- Complex installation
- Platform-specific

### Why This Matters

Developers work in different environments:
- CI/CD containers
- SSH servers
- Docker images
- Air-gapped systems

**If your tool requires dependencies, it won't work everywhere.**

### Implementation (Rust)

```toml
# Cargo.toml
[profile.release]
lto = true              # Link-time optimization
codegen-units = 1       # Better optimization
strip = true            # Smaller binary
opt-level = "z"         # Optimize for size

[dependencies]
# Use pure Rust dependencies, avoid C bindings when possible
reqwest = { version = "0.11", default-features = false, features = ["rustls-tls"] }
```

---

## 9. Documentation Rules

### ⚠️ MANDATORY: Examples Over Explanations

**Documentation MUST show examples, not just describe features.**

✅ **VALID:**
- Copy-pastable examples
- Real-world use cases
- Common workflows
- Troubleshooting guide

❌ **INVALID:**
- API reference only
- No examples
- Theoretical explanations
- Outdated docs

### Documentation Structure

```markdown
# Command: dx css

## Description
Brief one-line description.

## Usage
```bash
dx css [OPTIONS] <URL>
```

## Examples

### Check if class exists
```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
```

Output:
```
✓ Class 'cursor-pointer' found in stylesheet
```

### Inspect element styles
```bash
dx css --selector ".theme-toggle" http://localhost:3000
```

Output:
```
✓ Selector: .theme-toggle
  Computed Styles:
    cursor: pointer
    color: rgb(148, 163, 184)
```

## Common Issues

### "Class not found"
Possible causes:
- Class not used in any component
- Tailwind not scanning source files

Solution: Check Tailwind config...
```

---

## 10. Code Ownership & History Rules

### ⚠️ MANDATORY: Team Signatures

**All code MUST be signed by the team that wrote it.**

✅ **VALID:**
```rust
// Created by: TEAM-DX-001
// TEAM-DX-002: Added timeout handling
// TEAM-DX-003: Optimized for performance

pub async fn fetch_page(url: &str) -> Result<String> {
    // Implementation
}
```

❌ **INVALID:**
```rust
// No signature
pub async fn fetch_page(url: &str) -> Result<String> {
    // Implementation
}
```

### Signature Format

**New files:**
```rust
// Created by: TEAM-DX-XXX
```

**Modifications:**
```rust
// TEAM-DX-XXX: Brief description of change
```

**Documentation:**
```markdown
**Created by:** TEAM-DX-XXX
**Modified by:** TEAM-DX-YYY (description)
```

### ⚠️ CRITICAL: Never Remove Historical Signatures

**NEVER remove other teams' signatures. They provide historical context.**

✅ **VALID:**
- Add your signature when modifying code
- Keep all previous signatures
- Add context for why you changed something

❌ **INVALID:**
- Removing old signatures
- Replacing signatures with your own
- Deleting historical context

**Exception:** Only remove signatures if the entire surrounding code is deleted and keeping them would cause confusion.

### Why This Matters

Team signatures provide:
- **Accountability** - Know who to ask about code
- **History** - Understand evolution of the codebase
- **Context** - Why decisions were made
- **Attribution** - Credit for work done

**Without signatures, the codebase becomes a mystery.**

---

## 11. Versioning Rules

### ⚠️ MANDATORY: Semantic Versioning

**Tools MUST follow semantic versioning: MAJOR.MINOR.PATCH**

- **MAJOR:** Breaking changes (incompatible CLI changes)
- **MINOR:** New features (backwards compatible)
- **PATCH:** Bug fixes (backwards compatible)

✅ **VALID:**
- `1.0.0` → `1.1.0` (new feature)
- `1.1.0` → `1.1.1` (bug fix)
- `1.1.1` → `2.0.0` (breaking change)

❌ **INVALID:**
- No version numbers
- Random version numbers
- Breaking changes in patch releases

### Implementation

```rust
// Show version in CLI
#[derive(Parser)]
#[command(version)] // Automatically uses Cargo.toml version
struct Cli {
    // ...
}

// Check version compatibility
const MIN_SUPPORTED_VERSION: &str = "1.0.0";

fn check_compatibility(version: &str) -> Result<()> {
    if version < MIN_SUPPORTED_VERSION {
        return Err(Error::IncompatibleVersion {
            current: version.to_string(),
            minimum: MIN_SUPPORTED_VERSION.to_string(),
        });
    }
    Ok(())
}
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ DX ENGINEERING RULES QUICK REFERENCE                    │
├─────────────────────────────────────────────────────────┤
│ Performance:                                            │
│   ✅ < 2 seconds response time                         │
│   ✅ Timeouts on network requests                      │
│   ✅ Cache expensive operations                        │
│                                                         │
│ Reliability:                                            │
│   ✅ Deterministic output                              │
│   ✅ Stable sorting                                    │
│   ✅ No flaky behavior                                 │
│                                                         │
│ Output:                                                 │
│   ✅ Clear success/failure indicators                  │
│   ✅ Actionable error messages                         │
│   ✅ Minimal noise                                     │
│                                                         │
│ Composability:                                          │
│   ✅ Pipe-friendly output                              │
│   ✅ Exit codes (0 = success)                          │
│   ✅ JSON output option                                │
│                                                         │
│ Self-Documentation:                                     │
│   ✅ Helpful error messages                            │
│   ✅ Suggest solutions                                 │
│   ✅ Link to docs                                      │
│                                                         │
│ Configuration:                                          │
│   ✅ Sensible defaults                                 │
│   ✅ Works out-of-the-box                              │
│   ✅ Optional config file                              │
│                                                         │
│ Testing:                                                │
│   ✅ Unit + integration tests                          │
│   ✅ Test happy path + edge cases                      │
│   ✅ No flaky tests                                    │
│                                                         │
│ Distribution:                                           │
│   ✅ Single binary                                     │
│   ✅ No runtime dependencies                           │
│   ✅ Cross-platform                                    │
│                                                         │
│ Documentation:                                          │
│   ✅ Examples over explanations                        │
│   ✅ Copy-pastable code                                │
│   ✅ Troubleshooting guide                             │
│                                                         │
│ Code Ownership:                                         │
│   ✅ Add TEAM-DX-XXX signatures                        │
│   ✅ Never remove old signatures                       │
│   ✅ Preserve historical context                       │
│                                                         │
│ Versioning:                                             │
│   ✅ Semantic versioning                               │
│   ✅ Changelog                                         │
│   ✅ Compatibility checks                              │
└─────────────────────────────────────────────────────────┘
```

---

## 12. Frontend Verification Rules (MANDATORY)

### ⚠️ CRITICAL: ALWAYS VERIFY WITH DX TOOL

**All frontend changes MUST be verified using the DX tool before handoff.**

### Why This Matters

Frontend engineers without browser access need a reliable way to verify:
- CSS classes are generated correctly
- HTML structure is correct
- Components render properly
- No regressions introduced

**The DX tool solves this. Use it. Don't guess. Don't skip verification.**

---

### Required Verification Commands

#### 1. Verify CSS Classes Exist

```bash
# Check if Tailwind generated your class
dx css --class-exists "cursor-pointer" http://localhost:6007

# Check multiple classes
dx css --class-exists "h-11" http://localhost:6007
dx css --class-exists "bg-primary" http://localhost:6007
```

**When to use:**
- After adding new Tailwind classes to components
- When debugging missing styles
- Before committing CSS changes
- Before every handoff

#### 2. Inspect Element Styles

```bash
# Get computed styles for an element
dx css --selector ".theme-toggle" http://localhost:6007

# List all classes on an element
dx css --list-classes --selector "button" http://localhost:6007
```

**When to use:**
- Debugging style issues
- Verifying design tokens
- Checking hover/dark mode styles

#### 3. Query DOM Structure

```bash
# Check if component renders
dx html --selector "nav" http://localhost:6007

# Get element tree
dx html --selector "nav" --tree http://localhost:6007

# Get element attributes
dx html --attributes "button" http://localhost:6007
```

**When to use:**
- After adding new components
- Verifying conditional rendering
- Checking component hierarchy

#### 4. Inspect Element (All-in-One)

```bash
# Get HTML + all CSS in one command
dx inspect button http://localhost:6007
```

**When to use:**
- Quick verification of component rendering
- Debugging both HTML and CSS issues
- Before handoff verification

---

### Verification Checklist (MANDATORY BEFORE HANDOFF)

**Before handing off frontend work, you MUST verify:**

- [ ] **CSS classes exist:** `dx css --class-exists "your-class" http://localhost:6007`
- [ ] **Component renders:** `dx html --selector ".your-component" http://localhost:6007`
- [ ] **All variants work:** Test each variant with `dx inspect`
- [ ] **No console errors:** Check terminal output for errors
- [ ] **Accessibility:** `dx html --selector "button" --a11y http://localhost:6007`

**If ANY check fails, fix it before handoff.**

---

### Handoff Requirements (MANDATORY)

Your handoff MUST include DX tool verification output:

```markdown
## Verification

✅ CSS classes verified:
```bash
dx css --class-exists "cursor-pointer" http://localhost:6007
# Output: ✓ Class 'cursor-pointer' found in stylesheet
```

✅ Component rendering verified:
```bash
dx html --selector ".theme-toggle" http://localhost:6007
# Output: ✓ Found 1 element matching '.theme-toggle'
```

✅ All variants tested:
```bash
dx inspect button http://localhost:6007/story/button-primary
dx inspect button http://localhost:6007/story/button-secondary
dx inspect button http://localhost:6007/story/button-disabled
```
```

**Without DX tool verification, your handoff is INCOMPLETE.**

---

### Common Workflows

#### After Adding New Component

```bash
# 1. Verify component renders
dx html --selector ".new-component" http://localhost:6007

# 2. Check CSS classes
dx css --list-classes --selector ".new-component" http://localhost:6007

# 3. Inspect full output
dx inspect ".new-component" http://localhost:6007
```

#### After Changing Styles

```bash
# 1. Verify class exists
dx css --class-exists "new-class" http://localhost:6007

# 2. Check computed styles
dx css --selector ".affected-element" http://localhost:6007

# 3. Verify no regressions
dx inspect ".affected-element" http://localhost:6007
```

#### Before Handoff (MANDATORY)

```bash
# Run full verification suite
dx css --class-exists "cursor-pointer" http://localhost:6007
dx html --selector "nav" http://localhost:6007
dx inspect "button" http://localhost:6007
```

---

### Troubleshooting

#### "Class not found in stylesheet"

```bash
dx css --class-exists "cursor-pointer" http://localhost:6007
# ✗ Error: Class 'cursor-pointer' not found in stylesheet
```

**Possible causes:**
1. Tailwind not scanning source files
2. Class not used in any component
3. PostCSS not configured in Histoire
4. Class tree-shaken by build tool

**Solution:** 
- Check `histoire.config.ts` has PostCSS with `@tailwindcss/postcss`
- Check `tailwind.config.js` scans the right paths
- Verify class is actually used in a component

#### "Selector not found"

```bash
dx html --selector ".nonexistent" http://localhost:6007
# ✗ Error: Selector '.nonexistent' not found
```

**Solution:** 
- Use `dx html --tree` to explore DOM structure
- Verify component is rendering
- Check if element is in an iframe (Histoire uses iframes)

#### "Server not responding"

```bash
dx css --class-exists "test" http://localhost:6007
# ✗ Error: Failed to fetch page
```

**Solution:**
- Verify server is running: `curl http://localhost:6007`
- Check port number is correct
- Increase timeout if server is slow to start

---

### ⚠️ BANNED PRACTICES

❌ **DO NOT create ad-hoc scripts** to inspect HTML/CSS
- Use `dx` commands instead
- If a feature is missing, document it in `frontend/.dx-tool/FEATURE_REQUESTS.md`

❌ **DO NOT use `curl` + manual parsing**
- `dx` handles SPA rendering, iframe detection, CSS extraction
- Manual parsing is error-prone and doesn't handle SPAs

❌ **DO NOT skip verification**
- Every story variant MUST be verified with `dx inspect`
- Document verification results in your handoff

❌ **DO NOT guess if something works**
- Use the DX tool to verify
- Don't rely on "it compiles so it must work"

---

## The Bottom Line

**Good DX tools:**
- Are fast (< 2s)
- Are reliable (deterministic)
- Have clear output (signal, not noise)
- Are composable (Unix philosophy)
- Are self-documenting (helpful errors)
- Have sensible defaults (zero config)
- Are well-tested (happy path + edge cases)
- Are easy to distribute (single binary)
- Have great docs (examples first)
- Have clear ownership (team signatures)
- Follow semver (predictable updates)
- **Are actually used (mandatory verification)**

**Bad DX tools:**
- Are slow (developers won't use them)
- Are flaky (breaks trust)
- Are noisy (too much output)
- Are isolated (can't integrate)
- Are cryptic (unhelpful errors)
- Require config (friction)
- Are untested (bugs everywhere)
- Are hard to install (dependencies)
- Have poor docs (no examples)
- Break randomly (no versioning)
- **Are optional (developers skip them)**

**Build good tools. Follow these rules. Make developers' lives better.**

**ALWAYS verify with the DX tool. No exceptions. No excuses.**

---

## Examples of Good DX Tools

- **ripgrep** - Fast, reliable, clear output
- **fd** - Simple, composable, sensible defaults
- **exa** - Beautiful output, helpful errors
- **bat** - Self-documenting, great defaults
- **hyperfine** - Deterministic, clear benchmarks
- **dx (this tool)** - Frontend verification without browser access

**Study these tools. Learn from them. Build tools like them.**

---

**This is not optional. This is mandatory for all DX tool development.**

**Build tools that developers love to use. Not tools they tolerate.**

**Verify your work. Don't guess. Use the DX tool.**
