# Migration: Molecules to Organisms

**Date:** 2025-01-17  
**Status:** ✅ Complete

## Summary

Successfully moved 10 components from `molecules/` to `organisms/` and updated all imports across the codebase.

## Components Moved

1. **AudienceCard** - Audience segmentation cards with icon, features, and CTA
2. **BeeArchitecture** - System architecture diagram visualization
3. **CTAOptionCard** - Enterprise-grade CTA option cards
4. **EarningsBreakdownCard** - Detailed earnings calculation breakdown
5. **EarningsCard** - GPU earnings display card
6. **GPUSelector** - GPU model selection component
7. **IndustryCaseCard** - Industry-specific use case cards
8. **ProvidersCaseCard** - Provider profile cards
9. **SectionContainer** - Foundational layout container (formerly TemplateContainer)
10. **SecurityCrate** - Security crate capability display

## Consolidation

- **SecurityCrateCard** was consolidated into **SecurityCrate** (removed duplicate)

## Changes Made

### 1. Directory Structure
```bash
mv src/molecules/AudienceCard → src/organisms/AudienceCard
mv src/molecules/BeeArchitecture → src/organisms/BeeArchitecture
mv src/molecules/CTAOptionCard → src/organisms/CTAOptionCard
mv src/molecules/EarningsBreakdownCard → src/organisms/EarningsBreakdownCard
mv src/molecules/EarningsCard → src/organisms/EarningsCard
mv src/molecules/GPUSelector → src/organisms/GPUSelector
mv src/molecules/IndustryCaseCard → src/organisms/IndustryCaseCard
mv src/molecules/ProvidersCaseCard → src/organisms/ProvidersCaseCard
mv src/molecules/SectionContainer → src/organisms/SectionContainer
mv src/molecules/SecurityCrate → src/organisms/SecurityCrate
rm -rf src/molecules/SecurityCrateCard
```

### 2. Index Files Updated

**`src/molecules/index.ts`**
- Removed exports for all moved components

**`src/organisms/index.ts`**
- Added exports for all moved components under "Card organisms" section

### 3. Story Files Updated

All `.stories.tsx` files updated from:
```typescript
title: 'Molecules/ComponentName'
```
to:
```typescript
title: 'Organisms/ComponentName'
```

### 4. Import Updates

Updated imports across the codebase:

**Pages:**
- `src/pages/UseCasesPage/`
- `src/pages/PricingPage/`
- `src/pages/EnterprisePage/`
- `src/pages/ProvidersPage/`

**Templates:**
- `src/templates/SolutionTemplate/`
- `src/templates/EnterpriseSecurityTemplate/`
- `src/templates/AudienceSelector/`
- `src/templates/EnterpriseUseCasesTemplate/`
- `src/templates/CardGridTemplate/`
- `src/templates/EnterpriseCTATemplate/`
- `src/templates/EnterpriseSolutionTemplate/`
- `src/templates/ProvidersEarnings/`

### 5. Type Renames

- `TemplateContainerProps` → `SectionContainerProps`
- `TemplateContainer` → `SectionContainer`

### 6. Props Cleanup

Removed unsupported props from `SectionContainerProps` usage:
- `ctas` (not supported by SectionContainer)
- `disclaimer` (not supported by SectionContainer)
- `ribbon` (not supported by SectionContainer)

## Verification

✅ Build successful: `pnpm run build`
✅ All TypeScript errors resolved
✅ All imports updated
✅ All story files categorized correctly

## Files Modified

- 10 component directories moved
- 1 component directory removed (SecurityCrateCard)
- 2 index files updated (molecules, organisms)
- 10 story files updated
- 20+ import statements updated across pages and templates
- 4 page props files updated

## Breaking Changes

None - all imports use the barrel exports from `@rbee/ui/organisms` and `@rbee/ui/molecules`, so external consumers are not affected.
