# CLI Design & User Experience

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## Command Structure

```bash
dx [COMMAND] [OPTIONS] [URL]
```

## Core Commands

### 1. `dx css` - CSS Verification

Query and verify CSS styles for elements.

```bash
# Get computed styles for a selector
dx css --selector ".theme-toggle" http://localhost:3000

# Check if specific classes are present
dx css --has-class "cursor-pointer" --selector "button" http://localhost:3000

# Verify color token
dx css --property "color" --selector ".nav-link" http://localhost:3000

# Get all Tailwind classes on an element
dx css --list-classes --selector ".pricing-card" http://localhost:3000

# Check if class is generated in stylesheet
dx css --class-exists "cursor-pointer" http://localhost:3000
```

**Output Example:**
```
✓ Selector: .theme-toggle
  Classes: relative overflow-hidden text-muted-foreground hover:text-foreground
  Computed Styles:
    cursor: pointer
    color: rgb(148, 163, 184)
    position: relative
    overflow: hidden

✓ Class 'cursor-pointer' found in stylesheet
```

### 2. `dx html` - HTML Structure Queries

Query DOM structure like browser DevTools.

```bash
# Get element attributes
dx html --selector ".theme-toggle" --attrs http://localhost:3000

# Get text content
dx html --selector "h1" --text http://localhost:3000

# Count elements
dx html --selector "button" --count http://localhost:3000

# Get element tree
dx html --selector "nav" --tree http://localhost:3000

# Check accessibility
dx html --selector "button" --a11y http://localhost:3000
```

**Output Example:**
```
✓ Selector: button.theme-toggle
  Attributes:
    aria-label: "Toggle theme"
    class: "relative overflow-hidden text-muted-foreground..."
    type: "button"
  
  Accessibility:
    ✓ Has aria-label
    ✓ Has focusable role
    ✗ Missing keyboard handler (needs tabindex)
```

### 3. `dx snapshot` - Visual Regression

Capture and compare snapshots.

```bash
# Create baseline snapshot
dx snapshot --create --name "homepage" http://localhost:3000

# Compare against baseline
dx snapshot --compare --name "homepage" http://localhost:3000

# Update baseline
dx snapshot --update --name "homepage" http://localhost:3000

# List all snapshots
dx snapshot --list
```

**Output Example:**
```
✓ Snapshot comparison: homepage
  Changes detected:
    - .theme-toggle: cursor changed from 'default' to 'pointer'
    - .nav-link: color changed from '#f1f5f9' to '#94a3b8'
  
  Summary: 2 style changes, 0 structural changes
```

### 4. `dx component` - Component Testing

Test individual components.

```bash
# Test component in isolation
dx component --file "ThemeToggle.vue" --props '{"size":"icon"}'

# Verify component output
dx component --file "Button.vue" --props '{"variant":"default"}' --assert-class "cursor-pointer"

# Test responsive behavior
dx component --file "Navigation.vue" --viewport "mobile"
```

### 5. `dx diff` - Compare Versions

Compare two versions of the same page.

```bash
# Compare local vs production
dx diff http://localhost:3000 https://rbee.app

# Compare two local ports (before/after)
dx diff http://localhost:3000 http://localhost:3001

# Focus on specific selector
dx diff --selector ".pricing-card" http://localhost:3000 https://rbee.app
```

## Global Options

```bash
--format json|text|table     # Output format (default: text)
--output FILE                # Write to file instead of stdout
--verbose                    # Show detailed information
--quiet                      # Suppress non-error output
--timeout SECONDS            # Request timeout (default: 30)
--follow-redirects           # Follow HTTP redirects
--auth TOKEN                 # Bearer token for authenticated requests
```

## Configuration File

`.dxrc.json` in project root:

```json
{
  "baseUrl": "http://localhost:3000",
  "timeout": 30,
  "snapshots": {
    "dir": ".dx-snapshots",
    "ignore": [".timestamp", "[data-testid]"]
  },
  "css": {
    "ignoredClasses": ["hmr-*", "nuxt-*"]
  }
}
```

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Network error
- `3` - Parse error
- `4` - Assertion failed
- `5` - Snapshot mismatch

## Integration with pnpm Scripts

```json
{
  "scripts": {
    "dx:verify": "dx css --class-exists cursor-pointer http://localhost:3000",
    "dx:snapshot": "dx snapshot --compare --name homepage http://localhost:3000",
    "dx:test": "dx component --file Button.vue --assert-class cursor-pointer"
  }
}
```

## CI/CD Integration

```yaml
# .github/workflows/frontend.yml
- name: Start dev server
  run: pnpm dev &
  
- name: Wait for server
  run: sleep 5

- name: Verify CSS changes
  run: dx css --class-exists cursor-pointer http://localhost:3000

- name: Run snapshot tests
  run: dx snapshot --compare --name homepage http://localhost:3000
```

## Error Handling

```bash
$ dx css --selector ".nonexistent" http://localhost:3000
✗ Error: Selector '.nonexistent' not found
  Suggestions:
    - Check if element exists in DOM
    - Try a more general selector
    - Use 'dx html --tree' to explore structure

$ dx css --class-exists "cursor-pointer" http://localhost:3000
✗ Error: Class 'cursor-pointer' not found in stylesheet
  Possible causes:
    - Class not used in any component
    - Tailwind not scanning source files
    - Class tree-shaken by build tool
```

## User Experience Principles

1. **Fast** - Results in <2 seconds
2. **Clear** - Actionable output, not raw dumps
3. **Helpful** - Suggest fixes for common errors
4. **Composable** - Pipe-friendly output
5. **Consistent** - Same UX across all commands

---

**Next:** See `02_CSS_VERIFICATION.md` for detailed CSS analysis features.
