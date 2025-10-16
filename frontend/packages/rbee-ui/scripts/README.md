# Frontend Scripts

## rename-template.sh

Removes the "Template" suffix from a template directory and updates all references throughout the codebase.

### Usage

```bash
./scripts/rename-template.sh <TemplateName>
```

### Example

```bash
./scripts/rename-template.sh ProvidersSecurityTemplate
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
- ✓ Required files exist (`.tsx`, `.stories.tsx`, `index.ts`)
- ✓ Git working directory status

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

- The script will prompt for confirmation before making changes
- It's recommended to have a clean git working directory
- TypeScript type checking is optional but recommended
