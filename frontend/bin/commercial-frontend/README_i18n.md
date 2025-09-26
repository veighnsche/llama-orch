# i18n Workflow (Commercial Frontend)

This package ships with a spec-first, code-driven i18n workflow. Keys are discovered from source code, merged into locale JSON files, checked for issues, and optionally pruned.

- Locale store: `src/i18n/en.json`, `src/i18n/nl.json`
- Extracted artifacts: `src/i18n/extracted/`
- Scripts: `frontend/bin/commercial-frontend/scripts/*.mjs`

Prerequisites
- Node 20+ (see `package.json` engines)
- pnpm
- Install deps once: `pnpm install`

## Quick start

- Add keys directly in code using `$t('path.to.key', 'Default English text')` or `t('path.to.key', 'Default English text')`.
- Sync locales:
  - `pnpm run i18n:extract`
  - `pnpm run i18n:merge`
  - Optionally review: `src/i18n/extracted/messages.base.json` and `src/i18n/extracted/messages.keys.json`
- Check for problems:
  - `pnpm run i18n:report` (summarizes dynamic usages, missing/unused keys)
  - `pnpm run i18n:check` can be used in CI
- Prune unused keys (dry-run by default):
  - `pnpm run i18n:prune`
  - To apply safely with backups: `pnpm run i18n:prune:write`

## Commands

- `pnpm run i18n:extract`
  - Scans source for i18n usages and produces extracted artifacts under `src/i18n/extracted/`:
    - `messages.keys.json`: sorted list of used keys
    - `messages.base.json`: key -> default string (from second argument of `$t/t` if present)
    - `messages.catalog.json`: key metadata (files, counts, defaults)
    - `messages.warnings.json`: dynamic or non-literal usages that cannot be auto-extracted

- `pnpm run i18n:merge`
  - Merges `messages.base.json` into `en.json` and `nl.json`, adding any missing keys while preserving existing translations. Missing values are initialized to the discovered default string (if any) or `""`.

- `pnpm run i18n:check` (and variants)
  - `i18n:check`: runs extract + check
  - `i18n:report`: runs check with an extended, human-readable summary
  - `i18n:missing`: prints missing keys per locale
  - `i18n:unused`: prints unused keys per locale
  - `i18n:unused:strict`: same as above but fails the process if unused keys exist

  Flags for `i18n-check.mjs` (you can also call `node scripts/i18n-check.mjs` directly):
  - `--fail-on-unused`: cause a non-zero exit when unused keys are detected
  - `--print-unused`: print unused keys list (max 50 per locale)
  - `--print-missing`: print missing keys list (max 50 per locale)
  - `--report`: print both missing/unused summaries

  Outputs written by check:
  - `src/i18n/extracted/messages.unused.json`
  - `src/i18n/extracted/messages.warnings.json`

- `pnpm run i18n:prune` and `pnpm run i18n:prune:write`
  - Safe pruning of unused keys from locales based on extracted usage.
  - Dry-run (default): prints summary and writes `messages.prune.dryrun.json` under `src/i18n/extracted/`.
  - Write mode: removes unused keys from locales and writes backups alongside the JSON files.

  Flags for `i18n-prune.mjs`:
  - `--write`: actually apply deletions (otherwise DRY-RUN)
  - `--prune-empty-objects`: remove empty objects left after deletions
  - `--no-backup`: skip creating timestamped `.bak-*` backups in write mode
  - `--print` (or `--print-unused`): print list of keys to delete (limited)
  - `--locale en|nl`: operate on one locale (default: all)
  - `--keys <path>`: override path to extracted keys JSON (default: `src/i18n/extracted/messages.keys.json`)

- `pnpm run i18n:migrate`
  - One-time helper to convert legacy `src/i18n/en.ts`/`nl.ts` modules to JSON. Not needed for fresh projects.

## What the extractor recognizes

The custom extractor is regex-based and covers common Vue i18n usage patterns:

- Function calls with literal keys:
  - `$t('key', 'Default')`, `t('key', 'Default')`
  - `$tc('key')`, `tc('key')`
- Vue templates:
  - `v-t="'key.path'"`
  - `v-t="{ path: 'key.path', ... }"`
- Composition API patterns:
  - `const i18n = useI18n(); i18n.t('key') / i18n.tc('key')`
  - Destructured (including aliasing):
    - `const { t, tc } = useI18n()`
    - `const { t: translate, tc: pluralize } = useI18n()`
  - Bare `t()` / `tc()` are only considered in files that actually call `useI18n()` and destructure these functions (reduces false positives).

Dynamic usages are not auto-extracted and are reported to `messages.warnings.json`:
- `t(` with non-literal first argument
- Template literals with interpolation inside the key, e.g. `` t(`user.${id}.name`) ``

Prefer literal keys. If you truly need dynamic keys, manually ensure those keys exist in the locales.

## Adding or changing translations

1. In code, call `$t('path.to.key', 'Default English text')` where the second arg is the default/base string (optional but recommended).
2. Run `pnpm run i18n:extract && pnpm run i18n:merge`.
3. Open `src/i18n/en.json` and `src/i18n/nl.json` to translate missing entries.
4. Run `pnpm run i18n:report` to confirm no missing or dynamic usage remains.
5. Commit locale changes along with any code that added keys.

## CI recommendations

- Add `pnpm run i18n:check` to your CI. For strict hygiene, add `--fail-on-unused`.
- For release branches, consider running `i18n:prune:write` periodically to keep locale files clean.

## Multiple locales

- The current setup targets `en` and `nl`.
- To add locales, extend the following scripts/files:
  - `scripts/i18n-merge.mjs`: update the `locales` array
  - `scripts/i18n-check.mjs`: add the new locale file(s) to the loaded set
  - `scripts/i18n-prune.mjs`: allow `--locale <id>` for the new locale and include it by default
  - Create `src/i18n/<id>.json`
  - Update `src/i18n/index.ts` to include the new locale in `messages`

## Troubleshooting

- TypeScript errors importing JSON: Make sure `tsconfig.app.json` has `"resolveJsonModule": true`.
- Missing keys keep reappearing: re-run `i18n:merge` after `i18n:extract`, then translate in the JSON files.
- Many dynamic warnings: refactor to literal keys or whitelist necessary ones by ensuring they exist in locale JSONs.

## Notes

- We removed `vue-i18n-extract` and replaced it with our own scripts. `i18n:unused` and `i18n:report` map to our checker.
- The extractor is intentionally lightweight and fast. If we need AST-level guarantees, we can switch to parsers later.
