# Features Page - Units 05-01 through 05-09

**Total Units:** 9 organisms + 1 page assembly  
**Estimated Time:** 10-13 hours  
**React Reference:** `/frontend/reference/v0/app/features/page.tsx`

---

## 📋 Units

### Organisms (9)
- `05-01-FeaturesHero.md`
- `05-02-CoreFeaturesTabs.md` - Tabbed features
- `05-03-MultiBackendGPU.md` - Multi-backend GPU feature
- `05-04-CrossNodeOrchestration.md` - Cross-node feature
- `05-05-IntelligentModelManagement.md` - Model management
- `05-06-RealTimeProgress.md` - Real-time progress tracking
- `05-07-ErrorHandling.md` - Error handling feature
- `05-08-SecurityIsolation.md` - Security feature
- `05-09-AdditionalFeaturesGrid.md` - Additional features grid

### Page Assembly
- `07-05-FeaturesView.md` - Assemble page + route `/features`

---

## 📦 React Component Locations

All in `/frontend/reference/v0/components/features/`

---

**Assigned To:** TEAM-FE-008

---

## 📚 Required Reading

**BEFORE starting this unit, read:**

1. **Design Tokens (CRITICAL):** `00-DESIGN-TOKENS-CRITICAL.md`
   - DO NOT copy colors from React reference
   - Use design tokens: `bg-primary`, `text-foreground`, etc.
   - Translation guide: React colors → Vue tokens

2. **Engineering Rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Section 2: Design tokens requirement
   - Section 3: Histoire `.story.vue` format
   - Section 8: Port vs create distinction

3. **Examples:** Look at completed components
   - HeroSection: `/frontend/libs/storybook/stories/organisms/HeroSection/`
   - WhatIsRbee: `/frontend/libs/storybook/stories/organisms/WhatIsRbee/`
   - ProblemSection: `/frontend/libs/storybook/stories/organisms/ProblemSection/`

**Key Rules:**
- ✅ Use `.story.vue` format (NOT `.story.ts`)
- ✅ Use design tokens (NOT hardcoded colors like `bg-amber-500`)
- ✅ Import from workspace: `import { Button } from 'rbee-storybook/stories'`
- ✅ Add team signature: `<!-- TEAM-FE-XXX: Implemented ComponentName -->`
- ✅ Export in `stories/index.ts`

---

