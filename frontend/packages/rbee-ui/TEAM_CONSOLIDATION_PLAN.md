# Component Consolidation - Team Plan

**Date:** 2025-10-15  
**Total Components:** 6 duplicates to consolidate  
**Teams:** 3 teams  
**Work Distribution:** Balanced (2 components per team)

---

## 🎯 Team Assignments

| Team | Components | Complexity | Est. Time |
|------|-----------|------------|-----------|
| **TEAM-A** | Card + BeeGlyph | Medium | 2-3h |
| **TEAM-B** | HoneycombPattern + DiscordIcon | Medium | 2-3h |
| **TEAM-C** | GitHubIcon + XTwitterIcon | Medium | 2-3h |

---

## TEAM-A: Card + BeeGlyph

### Mission
Consolidate Card (structural) and BeeGlyph (pattern) components.

### Components Assigned
1. **Card** (atoms vs molecules)
2. **BeeGlyph** (icons vs patterns)

### Work Breakdown

#### Task 1: Card Component Consolidation

**Current State:**
- ✅ Atoms version: `src/atoms/Card/Card.tsx` (production, has CardAction)
- ❌ Molecules version: `src/molecules/Layout/Card.tsx` (remove)

**Steps:**

1. **Enhance atoms/Card with forwardRef** (keep existing features)
   - [ ] Add forwardRef to Card component
   - [ ] Add forwardRef to CardHeader
   - [ ] Add forwardRef to CardTitle (use HTMLHeadingElement)
   - [ ] Add forwardRef to CardDescription (use HTMLParagraphElement)
   - [ ] Add forwardRef to CardContent
   - [ ] Add forwardRef to CardFooter
   - [ ] Add forwardRef to CardAction
   - [ ] Add displayName to all components
   - [ ] Keep all existing data-slot attributes
   - [ ] Keep all existing grid layout logic
   - [ ] Add explicit TypeScript interfaces (CardProps extends HTMLAttributes)

2. **Merge story files**
   - [ ] Copy unique stories from molecules/Layout/Card.stories.tsx to atoms/Card/Card.stories.tsx
   - [ ] Update story imports to use atoms version
   - [ ] Keep TEAM-007 signature in atoms story
   - [ ] Verify all 8+ stories render correctly

3. **Update imports**
   - [ ] Update `src/molecules/Layout/Card.stories.tsx` import (if keeping story file)
   - [ ] OR delete `src/molecules/Layout/Card.stories.tsx` after merging

4. **Remove duplicate**
   - [ ] Delete `src/molecules/Layout/Card.tsx`
   - [ ] Verify molecules/index.ts doesn't export Card (already removed)

5. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Run `pnpm storybook` - verify all Card stories work
   - [ ] Check atoms/index.ts exports Card correctly
   - [ ] Test in 3+ organisms that use Card

**Expected Output:**
- Enhanced `src/atoms/Card/Card.tsx` with forwardRef + displayName
- Merged story file with all examples
- 1 file deleted

---

#### Task 2: BeeGlyph Component Consolidation

**Current State:**
- ❌ Icons version: `src/icons/BeeGlyph.tsx` (not used, size prop)
- ✅ Patterns version: `src/patterns/BeeGlyph/BeeGlyph.tsx` (used by organisms, forwardRef)

**Steps:**

1. **Keep patterns version as source of truth**
   - [ ] Verify `src/patterns/BeeGlyph/BeeGlyph.tsx` has:
     - ✅ forwardRef
     - ✅ displayName
     - ✅ cn utility
     - ✅ Proper TypeScript types

2. **Update icons barrel export**
   - [ ] Open `src/icons/index.ts`
   - [ ] Change `export { BeeGlyph } from './BeeGlyph'`
   - [ ] To `export { BeeGlyph } from '../patterns/BeeGlyph/BeeGlyph'`
   - [ ] This maintains backward compatibility

3. **Remove duplicate**
   - [ ] Delete `src/icons/BeeGlyph.tsx`

4. **Update documentation**
   - [ ] Check `src/icons/MIGRATION_GUIDE.md` references BeeGlyph correctly
   - [ ] Verify patterns version is documented

5. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Verify `import { BeeGlyph } from '@rbee/ui/icons'` still works
   - [ ] Check organisms using BeeGlyph still work

**Expected Output:**
- 1 file deleted (`src/icons/BeeGlyph.tsx`)
- 1 file updated (`src/icons/index.ts`)
- All imports continue working

---

### Deliverables

- [ ] Enhanced Card component with forwardRef + displayName
- [ ] Merged Card story files
- [ ] BeeGlyph consolidated via barrel export
- [ ] 3 files deleted (Card.tsx, Card.stories.tsx from molecules, icons/BeeGlyph.tsx)
- [ ] All TypeScript checks pass
- [ ] All Storybook stories render
- [ ] Summary document: `TEAM_A_CONSOLIDATION_COMPLETE.md`

---

## TEAM-B: HoneycombPattern + DiscordIcon

### Mission
Consolidate HoneycombPattern (pattern) and DiscordIcon (brand icon) components.

### Components Assigned
1. **HoneycombPattern** (icons vs patterns)
2. **DiscordIcon** (atoms vs icons)

### Work Breakdown

#### Task 1: HoneycombPattern Component Consolidation

**Current State:**
- ❌ Icons version: `src/icons/HoneycombPattern.tsx` (not used)
- ✅ Patterns version: `src/patterns/HoneycombPattern/HoneycombPattern.tsx` (used by 2 organisms)

**Steps:**

1. **Keep patterns version**
   - [ ] Verify `src/patterns/HoneycombPattern/HoneycombPattern.tsx` is production-ready
   - [ ] Ensure forwardRef + displayName present

2. **Update icons barrel export**
   - [ ] Open `src/icons/index.ts`
   - [ ] Change `export { HoneycombPattern } from './HoneycombPattern'`
   - [ ] To `export { HoneycombPattern } from '../patterns/HoneycombPattern/HoneycombPattern'`

3. **Update organism imports** (cleaner API)
   - [ ] Open `src/organisms/Features/FeaturesHero/FeaturesHero.tsx`
   - [ ] Change `import { HoneycombPattern } from '@rbee/ui/patterns/HoneycombPattern'`
   - [ ] To `import { HoneycombPattern } from '@rbee/ui/icons'`
   - [ ] Repeat for `src/organisms/Home/HeroSection/HeroSection.tsx`

4. **Remove duplicate**
   - [ ] Delete `src/icons/HoneycombPattern.tsx`

5. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Verify FeaturesHero organism renders
   - [ ] Verify HeroSection organism renders
   - [ ] Check pattern displays correctly

**Expected Output:**
- 1 file deleted
- 1 barrel export updated
- 2 organism imports simplified

---

#### Task 2: DiscordIcon Component Consolidation

**Current State:**
- ❌ Atoms version: `src/atoms/Icons/BrandIcons/DiscordIcon/` (size-5 default, used in Footer)
- ✅ Icons version: `src/icons/DiscordIcon.tsx` (size prop API, not actively used)

**Steps:**

1. **Enhance icons version if needed**
   - [ ] Open `src/icons/DiscordIcon.tsx`
   - [ ] Verify it has proper fill and viewBox attributes
   - [ ] Ensure it accepts size prop (already has it)

2. **Update atoms barrel export**
   - [ ] Open `src/atoms/index.ts`
   - [ ] Change line 24: `export * from './Icons/BrandIcons/DiscordIcon/DiscordIcon'`
   - [ ] To `export { DiscordIcon } from '../icons/DiscordIcon'`
   - [ ] This maintains `@rbee/ui/atoms` import compatibility

3. **Update Footer organism**
   - [ ] Open `src/organisms/Shared/Footer/Footer.tsx`
   - [ ] Find `import { DiscordIcon } from '@rbee/ui/atoms'`
   - [ ] Change to `import { DiscordIcon } from '@rbee/ui/icons'`
   - [ ] Add `size={20}` prop if default 24 is too large (atoms version was size-5 = 20px)

4. **Remove duplicate directory**
   - [ ] Delete entire `src/atoms/Icons/BrandIcons/DiscordIcon/` directory
   - [ ] This includes:
     - DiscordIcon.tsx
     - DiscordIcon.stories.tsx
     - index.ts

5. **Update atom story file** (if exists)
   - [ ] Check if DiscordIcon story is needed in atoms
   - [ ] If useful stories exist, merge into icons/Icons.stories.tsx
   - [ ] Otherwise just delete with directory

6. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Verify Footer renders with DiscordIcon
   - [ ] Check icon size is appropriate (20px)
   - [ ] Test hover states in Footer

**Expected Output:**
- 1 directory deleted (3-4 files)
- 1 organism updated (Footer)
- 1 barrel export updated (atoms/index.ts)

---

### Deliverables

- [ ] HoneycombPattern consolidated via barrel export
- [ ] DiscordIcon consolidated to icons package
- [ ] 2 organism imports updated (FeaturesHero, HeroSection, Footer)
- [ ] 1 directory + 1 file deleted
- [ ] All TypeScript checks pass
- [ ] All organisms render correctly
- [ ] Summary document: `TEAM_B_CONSOLIDATION_COMPLETE.md`

---

## TEAM-C: GitHubIcon + XTwitterIcon

### Mission
Consolidate GitHub and XTwitter brand icon components.

### Components Assigned
1. **GitHubIcon** (atoms vs icons)
2. **XTwitterIcon** (atoms vs icons)

### Work Breakdown

#### Task 1: GitHubIcon Component Consolidation

**Current State:**
- ❌ Atoms version: `src/atoms/Icons/BrandIcons/GitHubIcon/` (size-4, used in Navigation + Footer)
- ✅ Icons version: `src/icons/GithubIcon.tsx` (size prop, used in 1 organism)

**Steps:**

1. **Standardize naming** (GitHub = capital H for brand accuracy)
   - [ ] Rename `src/icons/GithubIcon.tsx` to `src/icons/GitHubIcon.tsx`
   - [ ] Update function name: `GithubIcon` → `GitHubIcon`
   - [ ] Update interface: `GithubIconProps` → `GitHubIconProps`

2. **Update icons barrel export**
   - [ ] Open `src/icons/index.ts`
   - [ ] Change line 8: `export { GithubIcon } from './GithubIcon'`
   - [ ] To `export { GitHubIcon } from './GitHubIcon'`

3. **Update atoms barrel export**
   - [ ] Open `src/atoms/index.ts`
   - [ ] Change line 30: `export * from './Icons/BrandIcons/GitHubIcon/GitHubIcon'`
   - [ ] To `export { GitHubIcon } from '../icons/GitHubIcon'`

4. **Update organism imports**
   - [ ] Open `src/organisms/Shared/Navigation/Navigation.tsx`
   - [ ] Change `import { GitHubIcon } from '@rbee/ui/atoms'`
   - [ ] To `import { GitHubIcon } from '@rbee/ui/icons'`
   - [ ] Add `size={16}` prop (atoms was size-4 = 16px)
   
   - [ ] Open `src/organisms/Shared/Footer/Footer.tsx`
   - [ ] Change `import { GitHubIcon } from '@rbee/ui/atoms'`
   - [ ] To `import { GitHubIcon } from '@rbee/ui/icons'`
   - [ ] Add appropriate size prop

   - [ ] Open `src/organisms/Home/TechnicalSection/TechnicalSection.tsx`
   - [ ] Update `import { GithubIcon }` to `import { GitHubIcon }`
   - [ ] Update component usage: `<GithubIcon` to `<GitHubIcon`

5. **Remove duplicate directory**
   - [ ] Delete entire `src/atoms/Icons/BrandIcons/GitHubIcon/` directory
   - [ ] This includes:
     - GitHubIcon.tsx
     - GitHubIcon.stories.tsx
     - index.ts

6. **Update stories** (if needed)
   - [ ] Check atom story file for unique examples
   - [ ] Merge useful stories into icons/Icons.stories.tsx
   - [ ] Ensure GitHubIcon (capital H) is used everywhere

7. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Verify Navigation renders GitHubIcon
   - [ ] Verify Footer renders GitHubIcon
   - [ ] Verify TechnicalSection renders GitHubIcon
   - [ ] Check all sizes are appropriate

**Expected Output:**
- 1 file renamed (GithubIcon → GitHubIcon)
- 1 directory deleted (3-4 files)
- 3 organisms updated
- 2 barrel exports updated

---

#### Task 2: XTwitterIcon Component Consolidation

**Current State:**
- ❌ Atoms version: `src/atoms/Icons/BrandIcons/XTwitterIcon/` (size-5, used in Footer)
- ✅ Icons version: `src/icons/XTwitterIcon.tsx` (size prop)

**Steps:**

1. **Verify icons version**
   - [ ] Open `src/icons/XTwitterIcon.tsx`
   - [ ] Ensure it has proper SVG path for X logo
   - [ ] Verify size prop works correctly

2. **Update atoms barrel export**
   - [ ] Open `src/atoms/index.ts`
   - [ ] Change line 31: `export * from './Icons/BrandIcons/XTwitterIcon/XTwitterIcon'`
   - [ ] To `export { XTwitterIcon } from '../icons/XTwitterIcon'`

3. **Update Footer organism**
   - [ ] Open `src/organisms/Shared/Footer/Footer.tsx`
   - [ ] Change `import { XTwitterIcon } from '@rbee/ui/atoms'`
   - [ ] To `import { XTwitterIcon } from '@rbee/ui/icons'`
   - [ ] Add `size={20}` prop (atoms was size-5 = 20px)

4. **Remove duplicate directory**
   - [ ] Delete entire `src/atoms/Icons/BrandIcons/XTwitterIcon/` directory
   - [ ] This includes:
     - XTwitterIcon.tsx
     - XTwitterIcon.stories.tsx
     - index.ts

5. **Update stories** (if needed)
   - [ ] Check atom story file for unique examples
   - [ ] Merge into icons/Icons.stories.tsx if useful
   - [ ] Otherwise delete with directory

6. **Verification**
   - [ ] Run `pnpm typecheck` - should pass
   - [ ] Verify Footer renders XTwitterIcon
   - [ ] Check icon size is appropriate (20px)
   - [ ] Test hover states

**Expected Output:**
- 1 directory deleted (3-4 files)
- 1 organism updated (Footer)
- 1 barrel export updated

---

### Deliverables

- [ ] GitHubIcon renamed and consolidated (capital H)
- [ ] XTwitterIcon consolidated to icons package
- [ ] 4 organism imports updated (Navigation, Footer 2x, TechnicalSection)
- [ ] 2 directories deleted (6-8 files total)
- [ ] All TypeScript checks pass
- [ ] All organisms render correctly
- [ ] Summary document: `TEAM_C_CONSOLIDATION_COMPLETE.md`

---

## 🔍 Cross-Team Verification Checklist

After all teams complete:

### Build & Type Safety
- [ ] `pnpm typecheck` passes with 0 errors
- [ ] `pnpm build` succeeds
- [ ] No import errors in console

### Storybook
- [ ] `pnpm storybook` builds successfully
- [ ] All Card stories render (atoms section)
- [ ] All icon stories render (icons section)
- [ ] No console errors

### Organism Functionality
- [ ] Navigation component renders
- [ ] Footer component renders (DiscordIcon, GitHubIcon, XTwitterIcon)
- [ ] FeaturesHero renders (HoneycombPattern)
- [ ] HeroSection renders (HoneycombPattern, BeeGlyph)
- [ ] TechnicalSection renders (GitHubIcon)

### Barrel Exports
- [ ] `import { Card } from '@rbee/ui/atoms'` works
- [ ] `import { BeeGlyph, HoneycombPattern } from '@rbee/ui/icons'` works
- [ ] `import { GitHubIcon, DiscordIcon, XTwitterIcon } from '@rbee/ui/icons'` works
- [ ] No duplicate exports causing conflicts

### File Cleanup
- [ ] All duplicate files deleted (8 files/directories)
- [ ] No orphaned story files
- [ ] No broken imports

---

## 📊 Summary Statistics

### Work Distribution

| Team | Components | Files Modified | Files Deleted | Organisms Updated | Est. Time |
|------|-----------|----------------|---------------|-------------------|-----------|
| TEAM-A | 2 | 4 | 3 | 0 | 2-3h |
| TEAM-B | 2 | 5 | 2 | 3 | 2-3h |
| TEAM-C | 2 | 8 | 2 | 4 | 2-3h |
| **Total** | **6** | **17** | **7-8** | **7** | **6-9h** |

### Expected Outcomes

✅ **6 components consolidated**  
✅ **7-8 files/directories deleted**  
✅ **17 files updated**  
✅ **0 breaking changes** (barrel exports maintained)  
✅ **Cleaner architecture**  
✅ **Single source of truth for each component**

---

## 🚨 Important Notes

### DO NOT Touch
- ❌ **UseCasesHero** - NOT a duplicate (organism vs icon, different purposes)
- ❌ **Specialized Card components** - FeatureCard, TestimonialCard, etc. are distinct
- ❌ **IconButton** - Distinct from Button, keep separate

### Preserve
- ✅ All existing functionality
- ✅ All existing stories (merge, don't delete)
- ✅ All barrel exports (update, don't remove)
- ✅ TEAM-XXX signatures in code

### Test Thoroughly
- ⚠️ Each icon size (atoms defaulted to smaller sizes)
- ⚠️ Footer component (uses 3 icons)
- ⚠️ Navigation component (uses GitHubIcon)
- ⚠️ All Card variants in organisms

---

## 📝 Completion Criteria

### Each Team Must:
1. Complete all assigned consolidations
2. Update all imports
3. Delete all duplicate files
4. Verify TypeScript passes
5. Verify Storybook builds
6. Create summary document

### Final Verification:
- All 3 team summaries complete
- Build passes: `pnpm typecheck && pnpm build`
- Storybook works: `pnpm storybook`
- No console errors
- All organisms render correctly

---

**Ready to start! Each team has balanced work (2 components, ~2-3 hours).**
