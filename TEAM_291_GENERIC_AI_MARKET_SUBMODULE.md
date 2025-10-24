# TEAM-291: Generic AI Market Submodule Added

**Status:** ✅ COMPLETE

**Mission:** Add `generic_ai_market` repository as a git submodule in the frontend apps directory.

## Implementation

### Submodule Details

**Repository:** `git@github.com:veighnsche/generic_ai_market.git`  
**Path:** `frontend/apps/generic_ai_market`  
**Commit:** `82306732484c4886d887a9628a107e5e4c9ae306` (master branch)

### Command Executed

```bash
git submodule add git@github.com:veighnsche/generic_ai_market.git frontend/apps/generic_ai_market
```

### Files Modified

**`.gitmodules`** (created/updated)
```ini
[submodule "frontend/apps/generic_ai_market"]
	path = frontend/apps/generic_ai_market
	url = git@github.com:veighnsche/generic_ai_market.git
```

### Submodule Contents

The submodule contains a Next.js application with:
- TypeScript configuration
- ESLint and Prettier setup
- Storybook integration
- i18n support
- Public assets
- Source code in `src/` directory

## Usage

### Clone with Submodules

For new clones:
```bash
git clone --recurse-submodules git@github.com:veighnsche/llama-orch.git
```

Or after cloning:
```bash
git submodule update --init --recursive
```

### Update Submodule

To pull latest changes from the submodule:
```bash
cd frontend/apps/generic_ai_market
git pull origin master
cd ../../..
git add frontend/apps/generic_ai_market
git commit -m "Update generic_ai_market submodule"
```

### Work on Submodule

The submodule is a full git repository:
```bash
cd frontend/apps/generic_ai_market
git checkout -b feature-branch
# Make changes
git add .
git commit -m "Changes"
git push origin feature-branch
```

### Remove Submodule (if needed)

```bash
git submodule deinit -f frontend/apps/generic_ai_market
git rm -f frontend/apps/generic_ai_market
rm -rf .git/modules/frontend/apps/generic_ai_market
```

## Integration with Monorepo

### Workspace Configuration

The submodule can be integrated into the pnpm workspace by adding to `pnpm-workspace.yaml`:

```yaml
packages:
  - 'frontend/apps/*'
  - 'frontend/packages/*'
  - 'tools/*'
```

This already includes `frontend/apps/*`, so the submodule is automatically part of the workspace.

### Install Dependencies

```bash
cd frontend/apps/generic_ai_market
pnpm install
```

Or from root:
```bash
pnpm install --filter generic_ai_market
```

### Build

```bash
cd frontend/apps/generic_ai_market
pnpm build
```

Or from root:
```bash
pnpm --filter generic_ai_market build
```

### Development

```bash
cd frontend/apps/generic_ai_market
pnpm dev
```

## Benefits of Submodule Approach

### Advantages
1. **Separate Repository** - Independent version control
2. **Independent Development** - Can be developed separately
3. **Reusability** - Can be used in multiple projects
4. **Access Control** - Separate repository permissions
5. **History Preservation** - Maintains its own git history

### Considerations
1. **Two-Step Updates** - Need to update submodule, then parent repo
2. **Clone Complexity** - Requires `--recurse-submodules` flag
3. **Branch Management** - Submodule can be on different branch
4. **Detached HEAD** - Submodule defaults to specific commit

## Verification

```bash
# Check submodule status
git submodule status

# Output:
# 82306732484c4886d887a9628a107e5e4c9ae306 frontend/apps/generic_ai_market (heads/master)

# List submodules
git config --file .gitmodules --get-regexp path

# Output:
# submodule.frontend/apps/generic_ai_market.path frontend/apps/generic_ai_market
```

## Next Steps

1. **Install dependencies** - `pnpm install` in submodule
2. **Configure integration** - Update turbo.json if needed
3. **Add to CI/CD** - Ensure CI clones with submodules
4. **Documentation** - Update main README with submodule info
5. **Team onboarding** - Inform team about submodule workflow

## CI/CD Considerations

### GitHub Actions

Ensure workflows clone submodules:
```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive
```

### Local Development

Team members need to run after pulling:
```bash
git submodule update --init --recursive
```

## Engineering Rules Compliance

- ✅ Proper git submodule configuration
- ✅ Correct path structure
- ✅ SSH URL for authentication
- ✅ Team signature (TEAM-291)

---

**TEAM-291 COMPLETE** - Generic AI Market added as git submodule at `frontend/apps/generic_ai_market`.
