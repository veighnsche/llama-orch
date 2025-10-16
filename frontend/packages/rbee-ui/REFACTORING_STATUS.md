# Refactoring Status - Templates & Stories

## ✅ COMPLETED

### All Template Stories Fixed (NO DUPLICATION)
**Status:** 100% Complete  
**Pattern:** All stories import props from page files

**Fixed Templates:**
1. ✅ PricingTemplate (3 stories)
2. ✅ PricingHeroTemplate
3. ✅ PricingComparisonTemplate
4. ✅ UseCasesHeroTemplate
5. ✅ UseCasesPrimaryTemplate
6. ✅ UseCasesIndustryTemplate
7. ✅ UseCasesTemplate
8. ✅ CTATemplate
9. ✅ ComparisonTemplate
10. ✅ TechnicalTemplate
11. ✅ TestimonialsTemplate
12. ✅ EmailCapture (already had imports)
13. ✅ FAQTemplate (already had imports)
14. ✅ All other templates

**Lines Saved:** ~1,500+ lines of duplicated props eliminated

### Orphaned Organisms Removed
1. ✅ **PricingSection** - Deleted (replaced by PricingTemplate)

### Commercial App Pages Updated
1. ✅ **Home** - Uses `HomePage` from `@rbee/ui/pages`
2. ✅ **Features** - Uses `FeaturesPage` from `@rbee/ui/pages`
3. ✅ **Use Cases** - Uses `UseCasesPage` from `@rbee/ui/pages`
4. ✅ **Pricing** - Uses `PricingPage` from `@rbee/ui/pages`
5. ✅ **Developers** - Uses `DevelopersPage` from `@rbee/ui/pages`

---

## 🔄 IN PROGRESS

### Pages Not Yet Refactored
These pages still use organisms directly:

1. **Enterprise Page** (`apps/commercial/app/enterprise/page.tsx`)
   - Still imports from `@rbee/ui/organisms`
   - Uses: `ProblemSection`, `EmailCapture`, `EnterpriseHero`, etc.
   - **Action:** Create `EnterprisePage` in `/pages/EnterprisePage/`

2. **GPU Providers Page** (`apps/commercial/app/gpu-providers/page.tsx`)
   - Still imports from `@rbee/ui/organisms`
   - Uses: `ProvidersHero`, `ProvidersProblem`, `ProvidersSolution`, etc.
   - **Action:** Create `ProvidersPage` in `/pages/ProvidersPage/`

---

## 📋 NEXT STEPS

### Phase 1: Refactor Remaining Pages (Priority)

#### 1. Enterprise Page
**Create:** `/packages/rbee-ui/src/pages/EnterprisePage/`

**Files to create:**
- `EnterprisePage.tsx` - Main page component
- `EnterprisePageProps.tsx` - All props objects
- `index.ts` - Exports

**Templates to use:**
- `ProblemTemplate` (already exists)
- `SolutionTemplate` (already exists)
- `EmailCapture` (already exists)
- Create new templates for Enterprise-specific sections

**Benefit:** Can remove `ProblemSection` organism after this

#### 2. Providers Page
**Create:** `/packages/rbee-ui/src/pages/ProvidersPage/`

**Files to create:**
- `ProvidersPage.tsx` - Main page component
- `ProvidersPageProps.tsx` - All props objects
- `index.ts` - Exports

**Templates to use:**
- Existing templates where possible
- Create new templates for Providers-specific sections

**Benefit:** Can remove more organisms after this

### Phase 2: Clean Up Organism Wrappers

After Enterprise and Providers pages are refactored:

**Directories to remove:**
- `/organisms/Developers/` - Wrappers around base organisms
- `/organisms/Enterprise/` - Wrappers around base organisms
- `/organisms/Providers/` - Wrappers around base organisms

**Organisms to potentially remove:**
- `ProblemSection` (if fully migrated to `ProblemTemplate`)
- `SolutionSection` (if fully migrated to `SolutionTemplate`)
- `HowItWorksSection` (if fully migrated to `HowItWorks` template)
- `UseCasesSection` (if fully migrated to templates)
- `FaqSection` (already have `FAQTemplate`)

### Phase 3: Verify All Stories Import Props

**Check remaining organism stories:**
- `src/organisms/FaqSection/FaqSection.stories.tsx`
- `src/organisms/ProblemSection/ProblemSection.stories.tsx`
- `src/organisms/EmailCapture/EmailCapture.stories.tsx`
- Others in `/organisms/Developers/`, `/organisms/Providers/`, etc.

**Action:** Update any that still have inline props to import from page files

---

## 📊 Progress Metrics

### Pages Refactored
- **Complete:** 5/7 (71%)
- **Remaining:** 2 (Enterprise, Providers)

### Template Stories Fixed
- **Complete:** 100% (all templates)
- **Pattern established:** Import props, never duplicate

### Orphaned Organisms Removed
- **Complete:** 1 (PricingSection)
- **Pending:** 4-5 (after page refactoring)

### Commercial App Integration
- **Complete:** 5/7 pages use page pattern
- **Remaining:** 2 pages still use organisms directly

---

## 🎯 Success Criteria

### For "Refactoring Complete"
- [ ] All commercial app pages use page pattern
- [ ] All template stories import props (no duplication)
- [ ] All orphaned organisms removed
- [ ] All organism wrapper directories cleaned up
- [ ] Documentation updated

### Current Status
- [x] All template stories import props ✅
- [x] 5/7 pages refactored ✅
- [x] 1 orphaned organism removed ✅
- [ ] 2 pages remaining (Enterprise, Providers)
- [ ] 4-5 organisms pending removal
- [ ] Wrapper directories pending cleanup

---

## 🔧 Refactoring Pattern (Established)

### For New Pages
1. Create `/pages/[PageName]/` directory
2. Create `[PageName].tsx` with `'use client'` if needed
3. Create `[PageName]Props.tsx` with all props objects
4. Import templates from `@rbee/ui/templates`
5. Wrap templates with `<TemplateContainer>`
6. Export props from `index.ts`
7. Update commercial app to import page

### For Stories
1. **NEVER duplicate props inline**
2. Import props from page files: `import { xProps } from '@rbee/ui/pages'`
3. Use imported props: `args: xProps`
4. Add JSDoc comment describing usage
5. See `EmailCapture.stories.tsx` for reference

---

**Last Updated:** After fixing all template stories and removing PricingSection
