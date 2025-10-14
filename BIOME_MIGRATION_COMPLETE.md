# ✅ Biome Migration Complete

Replaced Prettier and ESLint with Biome following the official Turborepo pattern.

## Changes Made

### 1. Installed Biome
```bash
pnpm add -D -w @biomejs/biome
```

### 2. Initialized Configuration
- **`biome.json`**: Auto-generated configuration migrated from Prettier settings
  - Single quotes, no semicolons, 120 char line width
  - Trailing commas, arrow parentheses
  - Import sorting enabled
- **`.biomeignore`**: Ignore patterns for node_modules, build outputs, generated files

### 3. Added Root Scripts
```json
{
  "scripts": {
    "format-and-lint": "biome check .",
    "format-and-lint:fix": "biome check . --write"
  }
}
```

### 4. Added Turborepo Root Tasks
```json
{
  "tasks": {
    "//#format-and-lint": {},
    "//#format-and-lint:fix": {
      "cache": false
    }
  }
}
```

### 5. Removed Prettier and ESLint
**From `@rbee/commercial`:**
- ❌ Removed `prettier` dependency
- ❌ Removed `@eslint/eslintrc` dependency  
- ❌ Removed `prettier.config.cjs`
- ❌ Removed `.prettierignore`
- ❌ Removed `eslint.config.mjs`
- ❌ Removed `format` and `format:check` scripts
- ✅ Kept `eslint` and `eslint-config-next` (needed by Next.js)
- ✅ Kept `lint` script (uses Next.js built-in linting)

**From `@rbee/user-docs`:**
- ❌ Removed `eslint.config.mjs`

## Usage

### Check formatting and linting (no changes)
```bash
turbo run //#format-and-lint
# or
pnpm run format-and-lint
```

### Fix formatting and linting issues
```bash
turbo run //#format-and-lint:fix
# or
pnpm run format-and-lint:fix
```

## Why Biome as a Root Task?

Following Turborepo's recommendation:
> Biome is a rare exception to most tools that are used with Turborepo because it is **so extraordinarily fast**. For this reason, we recommend using a Root Task rather than creating separate scripts in each of your packages.

Benefits:
- ⚡ **Fast**: Rust-based, processes entire monorepo quickly
- 🎯 **Simple**: Single configuration, single command
- 🔧 **Unified**: Format + Lint in one tool
- 📦 **Less config**: No per-package setup needed

## Configuration Details

### Formatter Settings (matches Prettier)
- **Indent**: 2 spaces (tabs in config converted to spaces by Biome)
- **Line width**: 120 characters
- **Quotes**: Single quotes
- **Semicolons**: Optional (ASI)
- **Trailing commas**: Always
- **Arrow parentheses**: Always

### Linter Settings
- **Rules**: Recommended set enabled
- **Import sorting**: Automatic
- **Next.js ESLint**: Still available via `pnpm --filter @rbee/commercial lint`

## Next Steps

1. **Run auto-fix** to format all code:
   ```bash
   turbo run //#format-and-lint:fix
   ```

2. **Commit changes**: The auto-fix will format all files consistently

3. **IDE setup**: Install Biome extension for your editor:
   - [VS Code](https://marketplace.visualstudio.com/items?itemName=biomejs.biome)
   - [Other editors](https://biomejs.dev/guides/editors/first-party-extensions/)

4. **CI integration**: Add to your CI pipeline:
   ```yaml
   - name: Lint and format
     run: turbo run //#format-and-lint
   ```

## Troubleshooting

### "Property ignore is not allowed" error
✅ **Fixed**: Using `.biomeignore` file instead of `ignore` property in `biome.json`

### Lots of lint errors on first run
✅ **Expected**: Run `turbo run //#format-and-lint:fix` to auto-fix most issues

### Next.js lint still needed?
✅ **Yes**: Keep Next.js's built-in ESLint for framework-specific rules. Biome handles general formatting/linting.
