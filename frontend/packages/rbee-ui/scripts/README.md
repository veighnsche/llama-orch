# Frontend Scripts

## rename-template.sh

Removes the "Template" suffix from a template directory and updates all references throughout the codebase.

### Usage

```bash
./scripts/rename-template.sh <TemplateName> [--force] [--skip-git-check]
```

### Options

- `--force` - Skip all confirmation prompts (useful for batch operations)
- `--skip-git-check` - Skip git working directory check

### Examples

**Interactive mode (default):**
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate
```

**Force mode (no prompts):**
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate --force
```

**Skip git check:**
```bash
./scripts/rename-template.sh ProvidersSecurityTemplate --skip-git-check
```

**Batch rename multiple templates:**
```bash
for template in EnterpriseHeroTemplate EnterpriseCTATemplate; do
  ./scripts/rename-template.sh "$template" --force --skip-git-check
done
```

This will:
- Rename `ProvidersSecurityTemplate/` → `ProvidersSecurity/`
- Rename `ProvidersSecurityTemplate.tsx` → `ProvidersSecurity.tsx`
- Update component name: `ProvidersSecurityTemplate` → `ProvidersSecurity`
- Update type: `ProvidersSecurityTemplateProps` → `ProvidersSecurityProps`
- Update all imports and usages across the codebase

### Features

**Preflight Checks:**
- ✓ Template directory exists
- ✓ Target directory doesn't already exist
- ✓ Required files exist (`.tsx`, `index.ts`)
- ⚠ Optional files (`.stories.tsx`) - warns if missing but continues
- ✓ Git working directory status (can be skipped with `--skip-git-check`)

**Operations:**
1. Rename directory
2. Rename component and story files
3. Update component name and props type
4. Update JSDoc comments
5. Update story imports and metadata
6. Update barrel exports in `templates/index.ts`
7. Update all page imports and usages

**Post-Rename Checks:**
- ✓ New directory exists
- ✓ Old directory removed
- ✓ New files exist
- ✓ No old references in renamed files
- ✓ Templates index updated correctly
- ✓ TypeScript compilation (if available)

### Requirements

- Bash 4.0+
- Git
- sed

### Notes

- The script will prompt for confirmation before making changes (unless `--force` is used)
- It's recommended to have a clean git working directory (or use `--skip-git-check`)
- TypeScript type checking is optional but recommended
- Stories files are now optional - the script will warn but continue if missing
- Use `--force` for batch operations or CI/CD pipelines
