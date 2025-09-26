# Linting & Formatting Audit

This audit captures the current ESLint/Prettier/Editor/VSCode configuration across the repo, highlights any contradictions or duplication, and provides recommendations so that everything Prettier touches stays aligned.

## Sources of Truth

- Prettier: `prettier.config.cjs` at repo root → forwards to `frontend/libs/frontend-tooling/prettier.config.cjs`.
- ESLint: Shared flat config in `frontend/libs/frontend-tooling/eslint.config.js` (consumed by projects).
- Editor/VSCode: Workspace settings in `.vscode/settings.json`.
- EditorConfig: `.editorconfig` at repo root.

## Inventory (by path)

- `prettier.config.cjs` (root): forwards to shared tooling config so editors/CLIs at the repo root resolve a single config.
- `frontend/libs/frontend-tooling/prettier.config.cjs`: the canonical shared Prettier settings.
  - Key options: `semi: false`, `singleQuote: true`, `trailingComma: 'all'`, `arrowParens: 'always'`, `printWidth: 100`, `vueIndentScriptAndStyle: true`.
  - Overrides: `{ files: ['**/*.css','**/*.scss','**/*.less'], options: { tabWidth: 4 } }`.
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

1. Consumer-level Prettier configs:
   - `frontend/bin/commercial-frontend/prettier.config.cjs`
   - `frontend/libs/storybook/prettier.config.cjs`
   - Status: These re-export the shared config and will work now that exports are fixed, but they shadow the root config due to Prettier’s nearest-config resolution. This creates duplication and a risk of drift. It also contradicts the repo rule to keep config out of consuming libs/binaries.

2. CSS/SCSS/LESS indentation defined in two places:
   - Prettier overrides in `frontend/libs/frontend-tooling/prettier.config.cjs` set `tabWidth: 4` for CSS/SCSS/LESS.
   - `.editorconfig` also sets `indent_size = 4` for these file types.
   - Status: Not a behavior conflict (values match), but duplication. Prettier is explicitly configured to read `.editorconfig` in `.vscode/settings.json` (`prettier.useEditorConfig: true`), so the overrides are redundant.

3. ESLint vs Prettier formatting authority:
   - `.vscode/settings.json` sets `eslint.format.enable: false` and Prettier as default formatter for JS-family and stylesheets.
   - Shared ESLint config includes `eslint-config-prettier` last.
   - Status: Aligned; no conflict. Formatting is owned by Prettier, lint is owned by ESLint.

4. Vue formatting and linting:
   - Prettier option `vueIndentScriptAndStyle: true` controls formatting of `<script>` and `<style>` blocks.
   - ESLint Vue rules that could conflict (e.g., indentation-related) are disabled by `eslint-config-prettier`.
   - Status: Aligned.

## Recommendations

- Remove consumer-level Prettier config files to enforce a single source of truth:
  - Delete:
    - `frontend/bin/commercial-frontend/prettier.config.cjs`
    - `frontend/libs/storybook/prettier.config.cjs`
  - Rationale: Ensures Prettier uses the root config, avoids shadowing and drift, and meets the policy of no configs in consumers.

- De-duplicate indentation rules for stylesheets by relying on `.editorconfig` only:
  - Option A (recommended for simplicity): Remove the CSS/SCSS/LESS overrides from `frontend/libs/frontend-tooling/prettier.config.cjs`. Prettier will respect `.editorconfig` because `.vscode/settings.json` enables `prettier.useEditorConfig`.
  - Option B: Keep the overrides and also keep `.editorconfig` entries. This is harmless but redundant. If kept, document the redundancy here.

- Keep ESLint→Prettier alignment as-is:
  - Continue to include `eslint-config-prettier` as the last item in the shared ESLint flat config.
  - Continue to disable `eslint.format.enable` in VS Code to avoid format ownership conflicts.

## Proposed Commands

Run from repo root to remove consumer-level Prettier configs:

```bash
rm frontend/bin/commercial-frontend/prettier.config.cjs \
   frontend/libs/storybook/prettier.config.cjs
```

Optional: If you accept Recommendation A (remove redundant stylesheet overrides), I will open a PR/commit to update `frontend/libs/frontend-tooling/prettier.config.cjs` accordingly.

## Current Alignment Status

- Prettier single source of truth: ✅ (root forwards to tooling)
- ESLint formatting: ✅ disabled where it could conflict (via config and VS Code)
- Editor tab sizing and Prettier: ✅ consistent via `.editorconfig` and Prettier options
- Duplications: ⚠️ consumer Prettier configs present; stylesheet indent rules duplicated (but consistent)

## Next Steps

1) Approve deletion of consumer-level Prettier configs (command above).
2) Choose Option A or B for stylesheet indentation duplication and I will implement.
