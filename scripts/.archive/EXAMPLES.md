# Dependency Analysis Examples

Real-world examples of using the dependency analysis tools.

## Example 1: Quick Overview

Get a quick overview of the monorepo structure:

```bash
python scripts/dependency-graph.py
```

**Output:**
```
✅ Found 71 total packages
   - Cargo crates: 45
   - pnpm packages: 26

## Most Connected Packages (by dependencies)
- rbee-hive (cargo): 13 dependencies
- queen-rbee (cargo): 11 dependencies
- @rbee/llm-worker-ui (pnpm): 9 dependencies

## Most Depended Upon (reverse dependencies)
- observability-narration-core (cargo): used by 17 packages
- operations-contract (cargo): used by 6 packages
- @rbee/ui (pnpm): used by 6 packages
```

**Insight:** `observability-narration-core` is critical infrastructure - changes affect 24% of packages!

## Example 2: Impact Analysis Before Refactoring

You want to refactor `observability-narration-core`. How many packages will be affected?

```bash
# Check reverse dependencies
python scripts/dependency-graph.py | grep "observability-narration-core"
```

**Output:**
```
- observability-narration-core (cargo): used by 17 packages
```

**Next step:** Find which packages depend on it:

```bash
python scripts/dependency-graph.py --format json | \
  jq -r '.dependencies[] | select(.to == "observability-narration-core") | .from' | \
  sort -u
```

**Output:**
```
health-poll
job-server
lifecycle-local
lifecycle-ssh
llm-worker-rbee
queen-rbee
rbee-hive
rbee-keeper
sd-worker-rbee
timeout-enforcer
... (17 total)
```

**Conclusion:** Breaking changes require updating 17 packages. Plan accordingly!

## Example 3: Finding Dependency Paths

How does `rbee-keeper` (CLI) depend on `observability-narration-core`?

```bash
python scripts/find-dependency-path.py rbee-keeper observability-narration-core
```

**Output:**
```
✅ Found 12 dependency path(s)

## Shortest Path
**Length:** 1 hop(s)

rbee-keeper (cargo)
    📁 bin/00_rbee_keeper
  └─> observability-narration-core (cargo)
      📁 bin/99_shared_crates/narration-core
```

**Insight:** Direct dependency! Changes to narration-core immediately affect rbee-keeper.

## Example 4: Circular Dependency Detection

Check if two packages have circular dependencies:

```bash
python scripts/find-dependency-path.py package-a package-b
```

If circular dependency exists:
```
⚠️  Circular dependency detected!
'package-b' also depends on 'package-a' (directly or transitively)
```

**Action:** Refactor to break the cycle!

## Example 5: Generate Complete Documentation

Generate all formats for documentation:

```bash
./scripts/generate-all-deps.sh .docs/architecture/dependencies/
```

**Output:**
```
✅ Generated all formats:
   - .docs/architecture/dependencies/stats.md
   - .docs/architecture/dependencies/dependencies.json
   - .docs/architecture/dependencies/dependencies.mmd
   - .docs/architecture/dependencies/dependencies.dot
   - .docs/architecture/dependencies/dependencies.png
   - .docs/architecture/dependencies/dependencies.svg
```

**Use cases:**
- `stats.md` - Include in README
- `dependencies.png` - Add to presentations
- `dependencies.mmd` - Embed in Markdown docs
- `dependencies.json` - Custom analysis scripts

## Example 6: Find Packages with No Dependencies

Find leaf packages (no dependencies):

```bash
python scripts/dependency-graph.py --format json | \
  jq -r '.packages | to_entries | 
    map(select((.value.dependencies | length) == 0 and 
               (.value.devDependencies | length) == 0)) | 
    .[].key'
```

**Output:**
```
artifacts-contract
shared-contract
worker-contract
...
```

**Insight:** These are pure contract/type packages - good architecture!

## Example 7: Find Packages with No Reverse Dependencies

Find potentially unused packages:

```bash
python scripts/dependency-graph.py --format json > /tmp/deps.json

# Get all package names
jq -r '.packages | keys[]' /tmp/deps.json > /tmp/all-packages.txt

# Get all dependencies
jq -r '.dependencies[].to' /tmp/deps.json | sort -u > /tmp/used-packages.txt

# Find difference
comm -23 /tmp/all-packages.txt /tmp/used-packages.txt
```

**Output:**
```
rbee-keeper
queen-rbee
rbee-hive
llm-worker-rbee
```

**Insight:** These are top-level binaries (expected to have no reverse deps).

## Example 8: Dependency Chain Analysis

Find the longest dependency chain:

```bash
python scripts/dependency-graph.py --format json | \
  jq -r '.packages | to_entries | 
    map({name: .key, deps: (.value.dependencies | length)}) | 
    sort_by(.deps) | 
    reverse | 
    .[] | 
    "\(.name): \(.deps) dependencies"' | \
  head -10
```

**Output:**
```
rbee-hive: 13 dependencies
queen-rbee: 11 dependencies
@rbee/llm-worker-ui: 9 dependencies
rbee-keeper: 8 dependencies
llm-worker-rbee: 7 dependencies
```

**Insight:** `rbee-hive` has the deepest dependency tree - potential refactoring target.

## Example 9: Cross-Workspace Dependencies

Find Cargo crates that depend on pnpm packages (shouldn't happen):

```bash
python scripts/dependency-graph.py --format json | \
  jq -r '.dependencies[] | 
    select((.from | . as $f | 
      ($f | IN(["rbee-keeper", "queen-rbee", "rbee-hive"]))) and 
      (.to | startswith("@rbee/")))' 
```

**Expected:** Empty (Cargo shouldn't depend on pnpm)

## Example 10: Generate Mermaid for Specific Subsystem

Generate dependency graph for just the hive subsystem:

```bash
# First, get full graph
python scripts/dependency-graph.py --format json > /tmp/deps.json

# Filter to hive-related packages
jq '.packages | to_entries | 
    map(select(.key | contains("hive"))) | 
    from_entries' /tmp/deps.json > /tmp/hive-deps.json

# Generate custom Mermaid (manual or script)
```

## Example 11: Audit External Dependencies

Find packages with the most external (non-workspace) dependencies:

```bash
# This requires parsing Cargo.toml files directly
for crate in bin/*/Cargo.toml; do
  name=$(grep '^name = ' "$crate" | cut -d'"' -f2)
  deps=$(grep -A100 '^\[dependencies\]' "$crate" | \
         grep -v '^\[' | \
         grep '=' | \
         wc -l)
  echo "$name: $deps external dependencies"
done | sort -t: -k2 -rn | head -10
```

## Example 12: Visualize with GraphViz

Generate and view interactive graph:

```bash
# Generate DOT file
python scripts/dependency-graph.py --format dot --output deps.dot

# View interactively (requires xdot)
xdot deps.dot

# Or render to PNG
dot -Tpng deps.dot -o deps.png
open deps.png  # macOS
xdg-open deps.png  # Linux
```

## Example 13: CI/CD Integration

Add to CI pipeline to detect circular dependencies:

```bash
#!/bin/bash
# .github/workflows/check-deps.yml

set -e

echo "🔍 Checking for circular dependencies..."

# Generate dependency graph
python scripts/dependency-graph.py --format json --output deps.json

# Custom check (you'd need to implement this)
if python scripts/check-circular-deps.py deps.json; then
  echo "✅ No circular dependencies found"
else
  echo "❌ Circular dependencies detected!"
  exit 1
fi
```

## Example 14: Compare Before/After Refactoring

Before refactoring:

```bash
python scripts/dependency-graph.py > before-refactor.txt
```

After refactoring:

```bash
python scripts/dependency-graph.py > after-refactor.txt
diff before-refactor.txt after-refactor.txt
```

**Shows:** Changes in dependency counts and relationships.

## Example 15: Find Transitive Dependencies

How many packages does `rbee-keeper` depend on (transitively)?

```bash
python scripts/dependency-graph.py --format json | \
  jq -r '
    .packages["rbee-keeper"].dependencies as $direct |
    .dependencies | 
    map(select(.from == "rbee-keeper" or 
               (.from | IN($direct[])))) | 
    map(.to) | 
    unique | 
    length
  '
```

## Tips & Tricks

### 1. Save Output for Later Analysis

```bash
# Generate once, analyze multiple times
python scripts/dependency-graph.py --format json --output deps.json

# Then use jq for various queries
jq '.packages | length' deps.json
jq '.dependencies | length' deps.json
```

### 2. Combine with Other Tools

```bash
# Find packages with most lines of code AND most dependencies
python scripts/dependency-graph.py --format json | \
  jq -r '.packages | to_entries | .[].key' | \
  while read pkg; do
    path=$(jq -r ".packages[\"$pkg\"].path" deps.json)
    loc=$(find "$path" -name "*.rs" -o -name "*.ts" -o -name "*.tsx" | \
          xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
    deps=$(jq -r ".packages[\"$pkg\"].dependencies | length" deps.json)
    echo "$pkg: $loc LOC, $deps deps"
  done | sort -t: -k2 -rn
```

### 3. Watch for Changes

```bash
# Monitor dependency changes
watch -n 60 'python scripts/dependency-graph.py | head -20'
```

## Common Patterns

### Pattern 1: Core Infrastructure

Packages used by many others (high reverse dependency count):
- Keep stable APIs
- Semantic versioning critical
- Comprehensive tests required

### Pattern 2: Service Binaries

Top-level packages with many dependencies but no reverse deps:
- Main entry points
- Integration testing focus
- Monitor dependency count

### Pattern 3: Contract Packages

Pure type definitions with no dependencies:
- Shared across services
- Breaking changes affect many packages
- Version carefully

### Pattern 4: Utility Packages

Moderate dependencies, moderate reverse dependencies:
- Reusable components
- Good candidates for open-sourcing
- Keep focused and single-purpose

## Troubleshooting

### "Package not found"

Check package name spelling:

```bash
# List all package names
python scripts/dependency-graph.py --format json | jq -r '.packages | keys[]' | sort
```

### "No path found"

Packages might not be connected:

```bash
# Check if package exists
python scripts/dependency-graph.py --format json | jq '.packages["package-name"]'
```

### Slow performance

Use cached JSON:

```bash
# Generate once
python scripts/dependency-graph.py --format json --output deps.json

# Use cached version
jq '...' deps.json
```

## See Also

- [DEPENDENCY_GRAPH_USAGE.md](./DEPENDENCY_GRAPH_USAGE.md) - Full documentation
- [README.md](./README.md) - Scripts overview
- [../.docs/DEPENDENCY_GRAPH_SUMMARY.md](../.docs/DEPENDENCY_GRAPH_SUMMARY.md) - Analysis summary
