# Linting & Formatting Audit

This audit captures the current ESLint/Prettier/Editor/VSCode configuration across the repo, highlights any contradictions or duplication, and provides recommendations so that everything Prettier touches stays aligned.

## Sources of Truth

- Prettier: Per-workspace via the package.json `prettier` field pointing to `orchyra-frontend-tooling/prettier.config.cjs`.
- ESLint: Shared flat config in `frontend/libs/frontend-tooling/eslint.config.js` (consumed by projects).
- Editor/VSCode: Workspace settings in `.vscode/settings.json`.
- EditorConfig: `.editorconfig` at repo root.
- Ignores: `.prettierignore` and `.eslintignore` at repo root (added).

## Inventory (by path)

- package.json `prettier` (per workspace):
  - `frontend/bin/commercial-frontend/package.json` → `"prettier": "orchyra-frontend-tooling/prettier.config.cjs"`
  - `frontend/libs/storybook/package.json` → `"prettier": "orchyra-frontend-tooling/prettier.config.cjs"`
- `frontend/libs/frontend-tooling/prettier.config.cjs`: the canonical shared Prettier settings.
  - Key options: `semi: false`, `singleQuote: true`, `trailingComma: 'all'`, `arrowParens: 'always'`, `printWidth: 100`, `vueIndentScriptAndStyle: true`.
  - Overrides: (none) — indentation for stylesheets is governed by `.editorconfig`.
- `frontend/libs/frontend-tooling/eslint.config.js`: shared ESLint flat config for Vue + TS.
  - Includes `@eslint/js` recommended, `eslint-plugin-vue` flat recommended, `typescript-eslint` recommended.
  - Adds `eslint-config-prettier` as the last item to disable any rules that conflict with Prettier formatting.
- Consumers import ESLint config:
  - `frontend/bin/commercial-frontend/eslint.config.js` → `import shared from 'orchyra-frontend-tooling/eslint.config.js'`.
  - `frontend/libs/storybook/eslint.config.js` → same import pattern.
- VS Code: `.vscode/settings.json`
  - Global: `formatOnSave: true`, Prettier as formatter for JS/TS/Vue/CSS/SCSS/LESS, default 4-space tabs except JS family uses 2.
  - `prettier.useEditorConfig: true` (Prettier respects `.editorconfig`).
  - `eslint.format.enable: false` to avoid format overlap; ESLint fixes only via explicit code action.
- EditorConfig: `.editorconfig`
  - Default 4 spaces.
  - JS/TS/JSX/TSX/Vue → 2 spaces.
  - CSS/SCSS/LESS → 4 spaces.
  - Trim trailing whitespace (except Markdown) and newline at EOF.

## Duplications & Potential Contradictions

1. Consumer-level Prettier configs: REMOVED ✅
   - Deleted files:
     - `frontend/bin/commercial-frontend/prettier.config.cjs`
     - `frontend/libs/storybook/prettier.config.cjs`
   - Status: Single source of truth at root; no shadowing.

2. CSS/SCSS/LESS indentation defined in two places: RESOLVED ✅
   - Removed Prettier overrides; `.editorconfig` remains the single authority.

3. ESLint vs Prettier formatting authority:
   - `.vscode/settings.json` sets `eslint.format.enable: false` and Prettier as default formatter for JS-family and stylesheets.
   - Shared ESLint config includes `eslint-config-prettier` last.
   - Status: Aligned; no conflict. Formatting is owned by Prettier, lint is owned by ESLint.

4. Vue formatting and linting:
   - Prettier option `vueIndentScriptAndStyle: true` controls formatting of `<script>` and `<style>` blocks.
   - ESLint Vue rules that could conflict (e.g., indentation-related) are disabled by `eslint-config-prettier`.
   - Status: Aligned.

## Recommendations

- Keep consumer-level Prettier config files removed (enforced single source of truth at root).

- Rely on `.editorconfig` for indentation rules (CSS/SCSS/LESS overrides removed from Prettier config).

- Keep ESLint→Prettier alignment as-is:
  - Continue to include `eslint-config-prettier` as the last item in the shared ESLint flat config.
  - Continue to disable `eslint.format.enable` in VS Code to avoid format ownership conflicts.

## Proposed Commands

Prettier CLI should be invoked from each package (the monorepo root has no package.json). Examples:

```bash
# Commercial frontend
pnpm --dir frontend/bin/commercial-frontend prettier --write .
pnpm --dir frontend/bin/commercial-frontend eslint . --fix
pnpm --dir frontend/bin/commercial-frontend prettier --check .

# Storybook library
pnpm --dir frontend/libs/storybook prettier --write .
pnpm --dir frontend/libs/storybook eslint . --fix
pnpm --dir frontend/libs/storybook prettier --check .
```

Optional: Add a top-level convenience script runner later if a root `package.json` is introduced.

## Current Alignment Status

- Prettier single source of truth: ✅ (root forwards to tooling)
- ESLint formatting: ✅ disabled where it could conflict (via config and VS Code)
- Editor tab sizing and Prettier: ✅ consistent via `.editorconfig` and Prettier options
- Duplications: ✅ none (consumer configs removed; stylesheet overrides removed)

## Next Steps

1) Use the package-local commands above to verify no flip-flops: format → lint:fix → format:check.
2) If desired, we can add lint-staged pre-commit hooks later to enforce both locally.
