# Frontend Engineering Rules - DX Tool Addendum

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)  
**Date:** 2025-10-12  
**Status:** MANDATORY for all frontend teams

---

## 8. Frontend Testing & Verification Rules

### ⚠️ MANDATORY: Use DX Tool for Verification

**All frontend changes MUST be verified using the DX tool before handoff.**

### Why This Matters

Frontend engineers without browser access need a reliable way to verify:
- CSS classes are generated correctly
- HTML structure is correct
- Components render properly
- No regressions introduced

**The DX tool solves this. Use it.**

---

## DX Tool Usage

### Installation

```bash
# Build the DX tool (one-time setup)
cd frontend/.dx-tool
cargo build --release

# Add to PATH (optional)
export PATH="$PATH:$(pwd)/target/release"
```

### Basic Commands

#### 1. Verify CSS Classes

**Problem:** "Did Tailwind generate my class?"

```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
```

**When to use:**
- After adding new Tailwind classes to components
- When debugging missing styles
- Before committing CSS changes

#### 2. Inspect Element Styles

**Problem:** "What styles are applied to this element?"

```bash
dx css --selector ".theme-toggle" http://localhost:3000
```

**When to use:**
- Debugging style issues
- Verifying design tokens
- Checking hover/dark mode styles

#### 3. Query DOM Structure

**Problem:** "Is my component rendering correctly?"

```bash
dx html --selector "nav" --tree http://localhost:3000
```

**When to use:**
- After adding new components
- Verifying conditional rendering
- Checking component hierarchy

#### 4. Snapshot Testing

**Problem:** "Did I break anything?"

```bash
# Create baseline (first time)
dx snapshot --create --name "homepage" http://localhost:3000

# Compare against baseline (every change)
dx snapshot --compare --name "homepage" http://localhost:3000
```

**When to use:**
- Before every handoff
- After major refactoring
- When changing shared components

---

## Verification Checklist

**Before handing off frontend work, you MUST verify:**

- [ ] **CSS classes exist:** `dx css --class-exists "your-class" http://localhost:3000`
- [ ] **Component renders:** `dx html --selector ".your-component" http://localhost:3000`
- [ ] **No regressions:** `dx snapshot --compare --name "page-name" http://localhost:3000`
- [ ] **Accessibility:** `dx html --selector "button" --a11y http://localhost:3000`

**If ANY check fails, fix it before handoff.**

---

## Dev Server Lifecycle

### ⚠️ CRITICAL: Proper Server Management

**The DX tool manages dev server lifecycle automatically. Use it correctly.**

### Integrated Testing (Recommended)

```bash
# DX tool starts server, runs test, keeps server running
dx test --start-server --cwd frontend/bin/commercial \
  css --class-exists "cursor-pointer" http://localhost:3000
```

**Benefits:**
- Automatic server startup
- Waits for server readiness
- Handles crashes gracefully
- Cleans up on exit

### Manual Server Management (Advanced)

```bash
# Start server and wait for readiness
dx server --start --wait --cwd frontend/bin/commercial

# Run your tests
dx css --class-exists "cursor-pointer" http://localhost:3000
dx html --selector "nav" --tree http://localhost:3000

# Stop server when done
dx server --stop
```

### Server Health Checks

```bash
# Verify server is responding
dx server --health http://localhost:3000
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Frontend Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Build DX tool
        run: cargo build --release --manifest-path frontend/.dx-tool/Cargo.toml
      
      - name: Verify CSS classes
        run: |
          cd frontend/.dx-tool
          cargo run --release -- test \
            --start-server \
            --cwd ../bin/commercial \
            css --class-exists "cursor-pointer" http://localhost:3000
      
      - name: Run snapshot tests
        run: |
          cd frontend/.dx-tool
          cargo run --release -- test \
            --start-server \
            --cwd ../bin/commercial \
            snapshot --compare --name "homepage" http://localhost:3000
```

### pnpm Scripts

Add to `frontend/bin/commercial/package.json`:

```json
{
  "scripts": {
    "verify:css": "dx test --start-server --cwd . css --class-exists cursor-pointer http://localhost:3000",
    "verify:snapshot": "dx test --start-server --cwd . snapshot --compare --name homepage http://localhost:3000",
    "verify:all": "pnpm verify:css && pnpm verify:snapshot"
  }
}
```

---

## Common Workflows

### After Adding New Component

```bash
# 1. Verify component renders
dx html --selector ".new-component" http://localhost:3000

# 2. Check CSS classes
dx css --list-classes --selector ".new-component" http://localhost:3000

# 3. Create snapshot baseline
dx snapshot --create --name "page-with-new-component" http://localhost:3000
```

### After Changing Styles

```bash
# 1. Verify class exists
dx css --class-exists "new-class" http://localhost:3000

# 2. Check computed styles
dx css --selector ".affected-element" http://localhost:3000

# 3. Compare snapshot
dx snapshot --compare --name "affected-page" http://localhost:3000
```

### Before Handoff

```bash
# Run full verification suite
dx test --start-server --cwd frontend/bin/commercial \
  snapshot --compare --name "homepage" http://localhost:3000

dx test --start-server --cwd frontend/bin/commercial \
  css --class-exists "cursor-pointer" http://localhost:3000

dx test --start-server --cwd frontend/bin/commercial \
  html --selector "nav" --a11y http://localhost:3000
```

---

## Handoff Requirements (Updated)

### Must Include DX Tool Verification

Your handoff MUST include:

```markdown
## Verification

✅ CSS classes verified:
```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
# Output: ✓ Class 'cursor-pointer' found in stylesheet
```

✅ Component rendering verified:
```bash
dx html --selector ".theme-toggle" http://localhost:3000
# Output: ✓ Found 1 element matching '.theme-toggle'
```

✅ Snapshot test passed:
```bash
dx snapshot --compare --name "homepage" http://localhost:3000
# Output: ✓ No changes detected
```
```

**Without DX tool verification, your handoff is INCOMPLETE.**

---

## Troubleshooting

### "Class not found in stylesheet"

```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
# ✗ Error: Class 'cursor-pointer' not found in stylesheet
```

**Possible causes:**
1. Tailwind not scanning source files (see TEAM-FE-012 handoff)
2. Class not used in any component
3. Class tree-shaken by build tool

**Solution:** Use `dx css --unused` to find unused classes, or check Tailwind config.

### "Server startup timeout"

```bash
dx server --start --wait
# ✗ Error: Server startup timeout (30s)
```

**Possible causes:**
1. Dependencies not installed
2. Build errors in code
3. Port already in use

**Solution:** Check logs with `--log-file server.log` or increase timeout with `--timeout 60`.

### "Selector not found"

```bash
dx html --selector ".nonexistent" http://localhost:3000
# ✗ Error: Selector '.nonexistent' not found
```

**Solution:** Use `dx html --tree` to explore DOM structure, or verify component is rendering.

---

## The Bottom Line

- **Always verify with DX tool** before handoff
- **Use integrated testing** (`dx test --start-server`) for reliability
- **Create snapshots** for regression detection
- **Include verification output** in handoff documents

**This is not optional. This is mandatory for all frontend work.**

---

## Resources

- **DX Tool Docs:** `frontend/.dx-tool/README.md`
- **CLI Design:** `frontend/.dx-tool/01_CLI_DESIGN.md`
- **Server Lifecycle:** `frontend/.dx-tool/07_DEV_SERVER_LIFECYCLE.md`
- **Kickoff Guide:** `frontend/.dx-tool/TEAM_DX_001_KICKOFF.md`

---

**TEAM-FE-011 (aka TEAM-DX-000) OUT.**

**Use the DX tool. Verify your work. Don't guess. Don't skip verification.**
