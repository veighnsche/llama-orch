# rename-template.sh Script Improvements

**Date:** 2025-10-17  
**Reason:** Script was incomplete and couldn't handle real-world scenarios

## Problems with Original Script

### 1. **Required Stories Files That Didn't Exist**
- Script failed if `.stories.tsx` file was missing
- Many templates were created without stories initially
- This blocked the rename operation entirely

### 2. **No Non-Interactive Mode**
- Script always required manual confirmation
- Impossible to use in batch operations
- Couldn't be automated in CI/CD pipelines

### 3. **Git Check Was Too Strict**
- Always required clean git working directory
- No way to skip this check
- Blocked legitimate use cases where uncommitted changes were acceptable

## Improvements Made

### 1. **Made Stories Files Optional**
```bash
# Before: REQUIRED
REQUIRED_FILES=(
    "$TEMPLATE_NAME.tsx"
    "$TEMPLATE_NAME.stories.tsx"  # ❌ Blocked if missing
    "index.ts"
)

# After: OPTIONAL
REQUIRED_FILES=(
    "$TEMPLATE_NAME.tsx"
    "index.ts"
)

OPTIONAL_FILES=(
    "$TEMPLATE_NAME.stories.tsx"  # ✅ Warns but continues
)
```

**Result:** Script now warns if stories are missing but continues the rename operation.

### 2. **Added `--force` Flag**
```bash
# Usage
./scripts/rename-template.sh EnterpriseHeroTemplate --force
```

**What it does:**
- Skips all confirmation prompts
- Continues despite uncommitted git changes
- Perfect for batch operations

**Example batch rename:**
```bash
for template in EnterpriseHeroTemplate EnterpriseCTATemplate EnterpriseComparisonTemplate; do
  ./scripts/rename-template.sh "$template" --force --skip-git-check
done
```

### 3. **Added `--skip-git-check` Flag**
```bash
# Usage
./scripts/rename-template.sh EnterpriseHeroTemplate --skip-git-check
```

**What it does:**
- Skips git working directory check entirely
- Useful when you know you have uncommitted changes
- Allows rename operations in dirty working directories

### 4. **Improved Error Handling**
- Stories file rename: checks if file exists before renaming
- Stories update: skips if file doesn't exist
- Post-checks: distinguishes between required and optional files
- Better warning messages for missing optional files

### 5. **Better User Feedback**
```bash
# Before
✗ Required file missing: EnterpriseHeroTemplate.stories.tsx

# After
⚠ Optional file missing: EnterpriseHeroTemplate.stories.tsx (will be skipped)
✓ Found: EnterpriseHeroTemplate.tsx
✓ Found: index.ts
```

## Updated Usage

### Interactive Mode (Default)
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate
```
- Prompts for confirmation
- Checks git status
- Safest option

### Force Mode (No Prompts)
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate --force
```
- No confirmation prompts
- Continues despite git changes
- Best for batch operations

### Skip Git Check
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate --skip-git-check
```
- Skips git working directory check
- Still prompts for confirmation
- Useful when you have uncommitted changes

### Combined Flags
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate --force --skip-git-check
```
- No prompts, no git check
- Fully automated
- Perfect for CI/CD or batch operations

## Real-World Use Case

**Scenario:** Rename all 10 Enterprise templates that were created without stories files.

**Before (Failed):**
```bash
./scripts/rename-template.sh EnterpriseHeroTemplate
# ✗ Required file missing: EnterpriseHeroTemplate.stories.tsx
# Script exits with error
```

**After (Success):**
```bash
# Create stories first
for template in Enterprise*Template; do
  # Create stories file
  # ...
done

# Then rename all templates in batch
for template in Enterprise*Template; do
  ./scripts/rename-template.sh "$template" --force --skip-git-check
done
# ✓ All 10 templates renamed successfully
```

## Benefits

1. **✅ Handles Missing Stories** - Warns but continues
2. **✅ Batch Operations** - Can rename multiple templates
3. **✅ CI/CD Ready** - Can be automated with `--force`
4. **✅ Flexible Git Handling** - Can skip git checks when needed
5. **✅ Better Error Messages** - Clear distinction between required and optional files

## Backward Compatibility

**100% backward compatible** - existing usage still works:
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate
```

New flags are optional and don't break existing workflows.

---

**Status:** ✅ Script is now production-ready and handles real-world scenarios
