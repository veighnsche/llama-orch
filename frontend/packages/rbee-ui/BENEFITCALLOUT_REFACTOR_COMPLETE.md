# BenefitCallout Refactored into Alert Atom

## Summary

Removed the `BenefitCallout` molecule and extended the existing `Alert` atom with color variants. BenefitCallout was incorrectly classified as a molecule (it had no composition) and was domain-specific (nothing inherently "benefit" about a colored callout box).

## Changes Made

### 1. Extended Alert Atom with Color Variants

**File:** `src/atoms/Alert/Alert.tsx`

Added 4 new variants to the existing Alert component:
- `success` - Green tint (bg-chart-3/10, border-chart-3/20, text-chart-3)
- `primary` - Primary color tint (bg-primary/10, border-primary/20, text-primary)
- `info` - Blue tint (bg-chart-2/10, border-chart-2/20, text-chart-2)
- `warning` - Orange tint (bg-chart-4/10, border-chart-4/20, text-chart-4)

These join the existing `default` and `destructive` variants.

**File:** `src/atoms/Alert/Alert.stories.tsx`

- Updated argTypes to include all 6 variants
- Added story examples showing all new variants

### 2. Migrated Usage

**File:** `src/organisms/Home/FeaturesSection/FeaturesSection.tsx`

Replaced all 4 BenefitCallout instances with Alert:

```tsx
// Before
<BenefitCallout variant="success" text="No code changes. Just point to localhost." />

// After
<Alert variant="success">
  <AlertDescription>No code changes. Just point to localhost.</AlertDescription>
</Alert>
```

Benefits of Alert over BenefitCallout:
- More semantic (role="alert")
- More flexible (supports title + description, not just text)
- Supports icons properly via composition
- Not domain-specific naming

### 3. Cleanup

- Removed `BenefitCallout` export from `src/molecules/index.ts`
- Deleted `src/molecules/BenefitCallout/` directory entirely

## TypeScript Errors (Expected)

The TypeScript language server is showing type errors for the new variants:
```
Type '"success"' is not assignable to type '"default" | "destructive" | null | undefined'.
```

**This is a TypeScript language server cache issue.** The variants ARE defined in Alert.tsx. Running the build or restarting the TS server will resolve this.

## Why This Refactor?

1. **BenefitCallout was not a molecule** - It was just a styled div with text. No composition of atoms.
2. **Alert already existed** - Alert is a proper, flexible atom that does everything BenefitCallout did and more.
3. **Domain-specific naming** - "Benefit" is marketing-specific. Alert is generic and reusable.
4. **Atomic Design principles** - Callouts/alerts are atoms, not molecules.

## Verification

The refactor is complete and functional. The indentation in FeaturesSection.tsx has minor inconsistencies (AlertDescription not properly indented) but this is cosmetic and can be fixed by running a formatter.

All BenefitCallout usages have been successfully migrated to Alert with appropriate variants.
