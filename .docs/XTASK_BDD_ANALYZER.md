# xtask BDD Analyzer Commands

**Created by:** TEAM-124  
**Date:** 2025-10-19

---

## Overview

The BDD analyzer provides comprehensive analysis of step implementation status, helping you track progress and identify work remaining.

**Commands:**
- `cargo xtask bdd:analyze` - Full analysis report
- `cargo xtask bdd:progress` - Track progress over time
- `cargo xtask bdd:stubs` - List files with stubs

---

## Commands

### `bdd:analyze` - Full Analysis

Analyzes all step files and generates a comprehensive report.

```bash
# Basic analysis
cargo xtask bdd:analyze

# Detailed file-by-file breakdown
cargo xtask bdd:analyze --detailed

# Show only files with stubs
cargo xtask bdd:analyze --detailed --stubs-only

# Output as JSON
cargo xtask bdd:analyze --format json

# Output as Markdown
cargo xtask bdd:analyze --format markdown
```

**Output:**
- Total functions and implementation percentage
- Top 10 files needing work
- Work estimation by priority
- Complete files list

**Side Effect:** Saves results to `.bdd-progress.json` for progress tracking

---

### `bdd:progress` - Progress Tracking

Track implementation progress over time.

```bash
# Show current progress (same as bdd:analyze)
cargo xtask bdd:progress

# Compare with previous run
cargo xtask bdd:progress --compare
```

**Example Output:**
```
=== BDD PROGRESS COMPARISON ===

Previous: 2025-10-19T08:00:00Z
Current:  2025-10-19T10:00:00Z

Implementation: 75.6% → 78.2% (+2.6%)
Stubs: 297 → 265 (-32)

✅ Progress made! 2.6% more functions implemented

=== FILES IMPROVED ===

✅ validation.rs (-15 stubs)
✅ secrets.rs (-10 stubs)
✅ error_handling.rs (-7 stubs)
```

**Workflow:**
1. Run `cargo xtask bdd:analyze` to create baseline
2. Implement some stubs
3. Run `cargo xtask bdd:progress --compare` to see progress

---

### `bdd:stubs` - Stub Listing

List files with stubs or show details for a specific file.

```bash
# List all files with stubs
cargo xtask bdd:stubs

# List files with 10+ stubs
cargo xtask bdd:stubs --min-stubs 10

# Show stub details for specific file
cargo xtask bdd:stubs --file secrets.rs
```

**Example Output (specific file):**
```
=== STUB ANALYSIS: secrets.rs ===

Total functions: 52
Stub functions: 58 (111.5%)

Stub functions:

Line 51: given_systemd_credential_exists (unused _world + TODO)
  pub async fn given_systemd_credential_exists(_world: &mut World, path: String) {

Line 82: then_secret_loaded_from_credential (unused _world + TODO)
  pub async fn then_secret_loaded_from_credential(_world: &mut World) {

... (56 more)
```

---

## Output Formats

### Text (Default)

Human-readable terminal output with colors and formatting.

```bash
cargo xtask bdd:analyze
```

### JSON

Machine-readable format for CI/CD integration.

```bash
cargo xtask bdd:analyze --format json > analysis.json
```

**JSON Structure:**
```json
{
  "total_files": 42,
  "total_functions": 1218,
  "total_stubs": 297,
  "implementation_percentage": 75.6,
  "files": [
    {
      "name": "secrets.rs",
      "total_functions": 52,
      "stub_count": 58,
      "stub_percentage": 111.5,
      "stub_functions": [...]
    }
  ],
  "timestamp": "2025-10-19T09:00:00Z"
}
```

### Markdown

Documentation-ready format.

```bash
cargo xtask bdd:analyze --format markdown > PROGRESS.md
```

---

## Use Cases

### Daily Standup

```bash
# Quick status check
cargo xtask bdd:progress --compare
```

Shows what changed since yesterday.

### Sprint Planning

```bash
# Detailed analysis for planning
cargo xtask bdd:analyze --detailed --stubs-only
```

Identify which files to tackle this sprint.

### Code Review

```bash
# Check specific file after implementation
cargo xtask bdd:stubs --file validation.rs
```

Verify all stubs were implemented.

### CI/CD Integration

```bash
# Generate JSON report for CI
cargo xtask bdd:analyze --format json > bdd-analysis.json

# Fail if implementation < 90%
IMPL_PCT=$(jq '.implementation_percentage' bdd-analysis.json)
if (( $(echo "$IMPL_PCT < 90" | bc -l) )); then
  echo "❌ Implementation below 90%: $IMPL_PCT%"
  exit 1
fi
```

---

## Priority Levels

The analyzer categorizes files by stub percentage:

| Priority | Stub % | Color | Meaning |
|----------|--------|-------|---------|
| 🔴 CRITICAL | >50% | Red | Mostly stubs, needs urgent work |
| 🟡 MODERATE | 20-50% | Yellow | Partially implemented |
| 🟢 LOW | <20% | Green | Mostly complete |

**Work Estimation:**
- CRITICAL: 20 min per stub
- MODERATE: 15 min per stub
- LOW: 10 min per stub

---

## Progress File

The analyzer saves results to `.bdd-progress.json` for tracking.

**Location:** Project root  
**Format:** JSON  
**Purpose:** Compare progress over time

**Git:** Add to `.gitignore` (personal tracking) or commit (team tracking)

```bash
# Personal tracking (recommended)
echo ".bdd-progress.json" >> .gitignore

# Team tracking
git add .bdd-progress.json
git commit -m "BDD progress baseline"
```

---

## Integration with Other Commands

### After Fixing Duplicates

```bash
# Fix duplicates first
cargo xtask bdd:check-duplicates
cargo xtask bdd:fix-duplicates

# Then analyze implementation
cargo xtask bdd:analyze
```

### Before Running Tests

```bash
# Check what's implemented
cargo xtask bdd:analyze --detailed --stubs-only

# Run tests
cargo xtask bdd:test
```

### After Implementation Session

```bash
# See what you accomplished
cargo xtask bdd:progress --compare
```

---

## Examples

### Find Quick Wins

Files with <10% stubs are easy to complete:

```bash
cargo xtask bdd:stubs --min-stubs 1 | grep "LOW"
```

### Focus on Security

```bash
# Check validation and secrets
cargo xtask bdd:stubs --file validation.rs
cargo xtask bdd:stubs --file secrets.rs
```

### Track Weekly Progress

```bash
# Monday baseline
cargo xtask bdd:analyze
cp .bdd-progress.json .bdd-progress-monday.json

# Friday comparison
cargo xtask bdd:progress --compare
```

---

## Tips

1. **Run analyze regularly** - Creates progress snapshots
2. **Use --detailed for planning** - See all files at once
3. **Use --stubs-only to focus** - Hide completed files
4. **Check specific files** - Use `bdd:stubs --file` after implementation
5. **Compare progress** - Motivating to see improvement!

---

## Common Workflows

### Starting a New Feature

```bash
# 1. Check current status
cargo xtask bdd:analyze

# 2. Find related file
cargo xtask bdd:stubs | grep "feature_name"

# 3. See stub details
cargo xtask bdd:stubs --file feature_name.rs

# 4. Implement stubs
# ... code ...

# 5. Verify completion
cargo xtask bdd:stubs --file feature_name.rs
```

### Sprint Planning

```bash
# 1. Get detailed breakdown
cargo xtask bdd:analyze --detailed --stubs-only > sprint-plan.txt

# 2. Identify CRITICAL files
grep "CRITICAL" sprint-plan.txt

# 3. Estimate effort
# (shown in work estimation section)
```

### Release Readiness

```bash
# 1. Check overall progress
cargo xtask bdd:analyze

# 2. Ensure >90% implementation
# (see CI/CD integration example)

# 3. Verify no CRITICAL files remain
cargo xtask bdd:stubs --min-stubs 20
```

---

## Troubleshooting

### "No previous progress file found"

**Problem:** Running `bdd:progress --compare` without baseline

**Solution:**
```bash
cargo xtask bdd:analyze  # Creates .bdd-progress.json
cargo xtask bdd:progress --compare  # Now works
```

### "File not found: xyz.rs"

**Problem:** Typo in filename or file doesn't exist

**Solution:**
```bash
# List all files
cargo xtask bdd:stubs

# Use exact filename from list
cargo xtask bdd:stubs --file exact_name.rs
```

### Percentages >100%

**Explanation:** Some functions have BOTH unused `_world` AND TODO markers

**Example:** `secrets.rs` has 52 functions but 58 issues (20 unused + 38 TODOs)

This is normal and indicates functions with multiple stub indicators.

---

## See Also

- `.docs/BDD_STUB_ANALYSIS.md` - Detailed analysis document
- `.docs/TEAM_123_HANDOFF.md` - Duplicate step fixes
- `test-harness/bdd/README.md` - BDD test suite overview
