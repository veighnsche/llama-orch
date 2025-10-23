# TEAM-216: rbee-keeper Behavior Investigation

**Phase:** 1 (Main Binaries)  
**Component:** `00_rbee_keeper` - CLI client  
**Duration:** 1 day  
**Output:** `TEAM_216_RBEE_KEEPER_BEHAVIORS.md`

---

## Mission

Inventory ALL behaviors in `rbee-keeper` CLI to enable comprehensive test coverage.

---

## Investigation Areas

### 1. CLI Command Structure

**File:** `bin/00_rbee_keeper/src/main.rs`

**Tasks:**
- Document ALL commands and subcommands
- Document ALL flags and arguments
- Document default values
- Document validation rules
- Document help text patterns

**Example Questions:**
- What happens with invalid arguments?
- What are the exit codes?
- How does it handle missing config?
- How does it handle network failures?

### 2. Operation Enum

**File:** `bin/00_rbee_keeper/src/main.rs` (Operation enum)

**Tasks:**
- List ALL operation variants
- Document parameters for each operation
- Document expected queen-rbee responses
- Document error handling per operation

**Focus Areas:**
- Hive operations (list, get, start, stop, install, uninstall, status, capabilities)
- Worker operations (if any)
- Model operations (if any)
- Job operations (if any)

### 3. HTTP Client Integration

**Files:**
- `bin/00_rbee_keeper/src/main.rs` (HTTP calls)

**Tasks:**
- Document ALL HTTP requests made
- Document request serialization
- Document response deserialization
- Document timeout handling
- Document retry logic (if any)

**Critical Questions:**
- How does it connect to queen-rbee?
- What happens if queen-rbee is unreachable?
- What happens on HTTP errors?
- What happens on network timeouts?

### 4. SSE Stream Handling

**Files:**
- Look for SSE/EventSource usage
- Look for `/jobs/{job_id}/stream` endpoint usage

**Tasks:**
- Document how SSE streams are consumed
- Document narration event handling
- Document stream completion detection
- Document stream error handling

**Edge Cases:**
- What if stream closes early?
- What if stream never starts?
- What if narration events are malformed?

### 5. Configuration

**Files:**
- Look for config loading
- Look for environment variables
- Look for default values

**Tasks:**
- Document ALL configuration sources
- Document configuration precedence
- Document required vs optional config
- Document config validation

### 6. Output Formatting

**Tasks:**
- Document stdout patterns
- Document stderr patterns
- Document JSON output (if any)
- Document table formatting (if any)
- Document progress indicators (if any)

**Examples:**
- How are hive lists formatted?
- How are errors displayed?
- How is progress shown?

### 7. Error Handling

**Tasks:**
- Document ALL error types
- Document error display patterns
- Document exit codes
- Document error recovery (if any)

**Scenarios:**
- Network errors
- HTTP errors
- SSE errors
- Validation errors
- Configuration errors

### 8. State Management

**Tasks:**
- Document any local state
- Document any caching
- Document any persistence

**Questions:**
- Does it store anything locally?
- Does it cache responses?
- Does it maintain session state?

---

## Investigation Methodology

### Step 1: Read Main Entry Point
```bash
# Read main.rs to understand command structure
cat bin/00_rbee_keeper/src/main.rs | head -500
```

### Step 2: Identify All Files
```bash
# List all source files
find bin/00_rbee_keeper/src -name "*.rs"
```

### Step 3: Read Each Module
- Systematically read every source file
- Take notes on all behaviors
- Identify test gaps

### Step 4: Check Existing Tests
```bash
# Check for existing tests
find bin/00_rbee_keeper -name "*test*.rs"
find bin/00_rbee_keeper/bdd -name "*.feature"
```

### Step 5: Document Everything
- Use the template from master plan
- Include code examples
- Cite line numbers
- Identify coverage gaps

---

## Key Files to Investigate

1. `bin/00_rbee_keeper/src/main.rs` - Main entry point
2. `bin/00_rbee_keeper/Cargo.toml` - Dependencies
3. `bin/00_rbee_keeper/bdd/` - Existing BDD tests
4. Any other `*.rs` files in `src/`

---

## Expected Behaviors to Document

### CLI Behaviors
- [ ] Command parsing
- [ ] Argument validation
- [ ] Help text generation
- [ ] Version display
- [ ] Config loading

### Network Behaviors
- [ ] HTTP request construction
- [ ] Response parsing
- [ ] Error handling
- [ ] Timeout handling
- [ ] SSE stream consumption

### Output Behaviors
- [ ] Stdout formatting
- [ ] Stderr formatting
- [ ] Progress indicators
- [ ] Error messages
- [ ] Success messages

### Error Behaviors
- [ ] Network errors
- [ ] HTTP errors
- [ ] Validation errors
- [ ] Config errors
- [ ] SSE errors

---

## Deliverables Checklist

- [ ] All CLI commands documented
- [ ] All HTTP interactions documented
- [ ] All error paths documented
- [ ] All output patterns documented
- [ ] All configuration sources documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
- [ ] Code signatures added (`// TEAM-216: Investigated`)
- [ ] Document follows template
- [ ] Document ≤3 pages
- [ ] Examples include line numbers

---

## Success Criteria

1. ✅ Complete behavior inventory document
2. ✅ All public APIs documented
3. ✅ All error paths documented
4. ✅ All edge cases identified
5. ✅ Test coverage gaps identified
6. ✅ Code signatures added
7. ✅ No TODO markers in document

---

## Next Steps After Completion

1. Hand off to TEAM-242 for test plan creation
2. Document will be used to create:
   - Unit test plan
   - BDD test plan
   - Integration test plan
   - E2E test plan

---

**Status:** READY  
**Blocked By:** None (can start immediately)  
**Blocks:** TEAM-242 (test planning)
