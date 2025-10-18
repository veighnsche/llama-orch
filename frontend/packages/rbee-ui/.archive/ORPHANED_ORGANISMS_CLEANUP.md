# Orphaned Organisms Cleanup

## Completed Removals

### ✅ PricingSection (Organism)
**Status:** DELETED  
**Reason:** Replaced by `PricingTemplate` in `/templates/PricingTemplate/`

**Replaced in:**
- ✅ Pricing Page - Now uses `PricingTemplate`
- ✅ Developers Page - Now uses `PricingTemplate`  
- ✅ Home Page - Now uses `PricingTemplate`

**Files removed:**
- `src/organisms/PricingSection/PricingSection.tsx`
- `src/organisms/PricingSection/PricingSection.stories.tsx`
- `src/organisms/PricingSection/index.ts`
- `src/organisms/PricingSection/REFACTOR_SUMMARY.md`

**Export removed from:**
- `src/organisms/index.ts`

---

## Still In Use (Not Orphaned)

### EmailCapture
**Status:** MIGRATED TO TEMPLATES  
**Location:** `src/templates/EmailCapture/`  
**Used in:** All pages (Home, Features, UseCases, Pricing, Developers, Enterprise)  
**Action:** None - already a template

### ProblemSection
**Status:** STILL USED AS ORGANISM  
**Used in:** Enterprise Page (commercial app)  
**Action:** Keep until Enterprise Page is refactored

### SolutionSection  
**Status:** STILL USED AS ORGANISM  
**Used in:** Providers, Enterprise, Developers organism wrappers  
**Action:** Keep until those pages are refactored

### HowItWorksSection
**Status:** STILL USED AS ORGANISM  
**Used in:** Developers organism wrappers  
**Action:** Keep until those pages are refactored

### CoreFeaturesTabs
**Status:** STILL USED AS ORGANISM  
**Used in:** Developers Page  
**Action:** Keep - actively used

### UseCasesSection
**Status:** STILL USED AS ORGANISM  
**Used in:** Multiple organism wrappers  
**Action:** Keep until those pages are refactored

### FaqSection
**Status:** MIGRATED TO TEMPLATES  
**Location:** `src/templates/FAQTemplate/`  
**Action:** Organism version can be removed once all references updated

---

## Next Candidates for Removal

Once the following pages are refactored to use the page pattern:

1. **Enterprise Page** → Can remove `ProblemSection` organism
2. **Providers Page** → Can remove `SolutionSection`, `HowItWorksSection` organisms  
3. **Update organism wrappers** → Remove wrapper organisms that just call the base organisms

---

## Cleanup Strategy

### Phase 1: Remove Duplicates (DONE)
- ✅ `PricingSection` - Replaced by `PricingTemplate`

### Phase 2: Migrate Remaining Pages
- [ ] Enterprise Page → `EnterprisePage` in `/pages/`
- [ ] Providers Page → `ProvidersPage` in `/pages/`

### Phase 3: Remove Orphaned Organisms
After Phase 2 completes:
- [ ] `ProblemSection` (if fully migrated to template)
- [ ] `SolutionSection` (if fully migrated to template)
- [ ] `HowItWorksSection` (if fully migrated to template)
- [ ] `FaqSection` (already have `FAQTemplate`)

### Phase 4: Remove Organism Wrappers
- [ ] `Developers/` organism wrappers
- [ ] `Enterprise/` organism wrappers  
- [ ] `Providers/` organism wrappers

---

## Benefits of Cleanup

✅ **Eliminated duplication** - `PricingSection` vs `PricingTemplate`  
✅ **Single source of truth** - Props in page files, not organisms  
✅ **Cleaner architecture** - Templates (reusable) vs Organisms (page-specific wrappers)  
✅ **Easier maintenance** - Fewer files to update  
✅ **Better discoverability** - Clear separation between templates and organisms

---

**Last updated:** After PricingSection removal
