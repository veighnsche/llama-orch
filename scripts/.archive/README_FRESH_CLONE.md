# Fresh Clone Script

**Location:** `scripts/fresh-clone.sh`

## What It Does

Completely deletes the repository directory and reclones it fresh from the remote.

**Use cases:**
- Clean up all build artifacts (`target/`, `node_modules/`, `.next/`, etc.)
- Reset to a completely clean state
- Fix corrupted git state
- Start fresh after major changes

## Safety Features

### ✅ REQUIRED: Everything Must Be Committed

The script **WILL NOT RUN** unless:
1. ✅ All changes are committed
2. ✅ No untracked files exist
3. ✅ Working directory is clean

**This prevents accidental data loss!**

### What It Checks

1. **Uncommitted changes:**
   ```bash
   git diff-index --quiet HEAD --
   ```
   - Checks for modified files
   - Checks for staged changes

2. **Untracked files:**
   ```bash
   git ls-files --others --exclude-standard
   ```
   - Checks for new files not in `.gitignore`

3. **Git repository:**
   - Verifies `.git/` directory exists
   - Gets remote URL
   - Gets current branch

### Final Confirmation

Even if everything is committed, the script asks:
```
Are you ABSOLUTELY SURE? Type 'yes' to continue:
```

**Only typing `yes` (exactly) will proceed.**

## Usage

### Basic Usage

```bash
# From repository root
./scripts/fresh-clone.sh
```

### If You Have Uncommitted Changes

**Option 1: Commit them**
```bash
git add .
git commit -m "Your commit message"
./scripts/fresh-clone.sh
```

**Option 2: Stash them** (NOT RECOMMENDED - you'll lose them!)
```bash
git stash
./scripts/fresh-clone.sh
# WARNING: Stashed changes will be lost!
```

### If You Have Untracked Files

**Add and commit them:**
```bash
git add .
git commit -m "Add untracked files"
./scripts/fresh-clone.sh
```

## What Gets Deleted

**EVERYTHING in the repository directory:**
- ✅ All source code (recloned fresh)
- ✅ All build artifacts (`target/`, `node_modules/`, `.next/`)
- ✅ All generated files
- ✅ All local configuration (`.env`, etc.)
- ✅ Git history (recloned fresh)
- ✅ **EVERYTHING**

**What's preserved:**
- ✅ Committed changes (in remote)
- ✅ Remote branches
- ✅ Nothing else!

## Process Flow

```
1. Check if in git repository
   ↓
2. Check for uncommitted changes → ABORT if found
   ↓
3. Check for untracked files → ABORT if found
   ↓
4. Get remote URL and current branch
   ↓
5. Ask for confirmation → ABORT if not "yes"
   ↓
6. Delete entire repository directory
   ↓
7. Clone fresh from remote
   ↓
8. Checkout original branch
   ↓
9. Done!
```

## Example Output

### Success Case

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FRESH CLONE SCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Repository: /home/vince/Projects/llama-orch
Parent dir: /home/vince/Projects

✓ Git repository detected

Checking for uncommitted changes...
✓ No uncommitted changes
✓ No untracked files
✓ Remote URL: git@github.com:rbee-keeper/rbee.git
✓ Current branch: main

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WARNING: THIS WILL DELETE EVERYTHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script will:
  1. Delete the entire directory: /home/vince/Projects/llama-orch
  2. Clone fresh from: git@github.com:rbee-keeper/rbee.git
  3. Checkout branch: main

ALL local files will be PERMANENTLY DELETED!

Are you ABSOLUTELY SURE? Type 'yes' to continue: yes

Starting fresh clone...

Deleting /home/vince/Projects/llama-orch...
✓ Deleted

Cloning from git@github.com:rbee-keeper/rbee.git...
✓ Cloned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FRESH CLONE COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Repository: /home/vince/Projects/llama-orch
Branch: main

Next steps:
  cd /home/vince/Projects/llama-orch
  cargo build --release
  # or
  cd frontend && pnpm install && pnpm build
```

### Error Case (Uncommitted Changes)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ERROR: UNCOMMITTED CHANGES DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have uncommitted changes. Please commit or stash them first:

 M frontend/apps/user-docs/package.json
 M scripts/fresh-clone.sh

To commit:
  git add .
  git commit -m "Your commit message"

To stash:
  git stash
```

## When to Use

### ✅ Good Use Cases

- **Clean build artifacts:** Remove all `target/`, `node_modules/`, `.next/` directories
- **Fix corrupted state:** Git issues, broken builds, weird errors
- **Start fresh:** After major refactoring or restructuring
- **Test clean install:** Verify build works from scratch

### ❌ Bad Use Cases

- **Don't use to discard changes** - Commit them first or use `git reset`
- **Don't use to switch branches** - Use `git checkout` instead
- **Don't use to update** - Use `git pull` instead

## Alternatives

### Just Clean Build Artifacts

```bash
# Rust
cargo clean

# Frontend
cd frontend && pnpm clean
# or
rm -rf frontend/node_modules frontend/.next
```

### Just Reset Git State

```bash
# Discard uncommitted changes (DANGEROUS!)
git reset --hard HEAD

# Clean untracked files (DANGEROUS!)
git clean -fdx
```

### Just Update

```bash
git pull --rebase
```

## Safety Checklist

Before running, ask yourself:

- [ ] Are all my changes committed?
- [ ] Are all my changes pushed to remote?
- [ ] Do I have any untracked files I want to keep?
- [ ] Do I have any local configuration I need to backup?
- [ ] Am I sure I want to delete EVERYTHING?

**If any answer is "no", DON'T RUN THE SCRIPT!**

## Recovery

**If you run the script by accident:**

1. **Committed changes:** ✅ Safe - they're in the remote
2. **Uncommitted changes:** ❌ LOST FOREVER
3. **Untracked files:** ❌ LOST FOREVER
4. **Local config:** ❌ LOST FOREVER

**There is NO UNDO!**

## Technical Details

### What It Does Under the Hood

```bash
# 1. Safety checks
git diff-index --quiet HEAD --  # Check uncommitted changes
git ls-files --others --exclude-standard  # Check untracked files

# 2. Get info
REMOTE_URL=$(git config --get remote.origin.url)
CURRENT_BRANCH=$(git branch --show-current)

# 3. Delete and reclone
cd ..
rm -rf llama-orch
git clone "$REMOTE_URL" llama-orch
cd llama-orch
git checkout "$CURRENT_BRANCH"
```

### Exit Codes

- `0` - Success or user aborted
- `1` - Error (uncommitted changes, not a git repo, etc.)

### Dependencies

- `git` - Required
- `bash` - Required
- `rm` - Required (standard Unix tool)

## Troubleshooting

### "Not in a git repository"

**Cause:** Script not run from repository root

**Fix:**
```bash
cd /home/vince/Projects/llama-orch
./scripts/fresh-clone.sh
```

### "No remote origin URL found"

**Cause:** Repository has no remote

**Fix:**
```bash
git remote add origin git@github.com:rbee-keeper/rbee.git
```

### "Could not determine current branch"

**Cause:** Detached HEAD state

**Fix:**
```bash
git checkout main
./scripts/fresh-clone.sh
```

---

**Created by:** TEAM-427  
**Date:** 2025-11-08  
**Status:** Production-ready with multiple safety checks
