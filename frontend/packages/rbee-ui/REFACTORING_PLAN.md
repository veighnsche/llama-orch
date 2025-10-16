# Page & Template Refactoring Plan

## Goal
**ALL content must be props-driven for i18n and CMS integration.**

Pages = props objects + composition. Templates = reusable UI sections with ZERO hardcoded content.

## Critical Rule: NO HARDCODED CONTENT
**Every single piece of text, image, icon, link, or data MUST be passed as props.**

**Why:** 
- i18n (internationalization) requires all text to be translatable
- CMS integration requires all content to be editable
- NO EXCEPTIONS - even single-use components must accept props

**This means:**
- ❌ NO `const data = [...]` inside components
- ❌ NO hardcoded strings in JSX
- ❌ NO hardcoded image paths
- ❌ NO hardcoded links
- ✅ ALL content passed as props
- ✅ ALL data defined in page props files
- ✅ Templates are pure presentation

## Components to Migrate

**✅ Home Page (Phase A - COMPLETE):**
- ✅ All 12 sections refactored and integrated
- ✅ Commercial app replaced with clean `<HomePage />` import
- ✅ 1,324 lines of well-documented props objects
- ✅ 5-line commercial app implementation

**✅ Features Page (Phase B - COMPLETE):**
- ✅ Page structure created with `'use client'` directive
- ✅ All 8 templates created and integrated:
  - ✅ FeaturesHero
  - ✅ CrossNodeOrchestrationTemplate
  - ✅ IntelligentModelManagementTemplate
  - ✅ MultiBackendGpuTemplate
  - ✅ ErrorHandlingTemplate
  - ✅ RealTimeProgressTemplate
  - ✅ SecurityIsolationTemplate
  - ✅ AdditionalFeaturesGridTemplate
- ✅ FeaturesTabs props migrated (reused from HomePage pattern)
- ✅ EmailCapture props migrated
- ✅ Props organized in 3 files: featuresPageProps.tsx, featuresPagePropsExtended.tsx, errorAndProgressProps.tsx
- ✅ Commercial app replaced with clean `<FeaturesPage />` import (265 lines → 10 lines)

**✅ Use Cases Page (Phase C - COMPLETE):**
- ✅ Page structure created with `'use client'` directive
- ✅ All 3 templates created and integrated:
  - ✅ UseCasesHeroTemplate
  - ✅ UseCasesPrimaryTemplate
  - ✅ UseCasesIndustryTemplate
- ✅ EmailCapture props migrated
- ✅ Props organized in single file: UseCasesPageProps.tsx
- ✅ Commercial app replaced with clean `<UseCasesPage />` import (13 lines → 6 lines)
- ✅ Storybook stories created for all templates
- ✅ Templates exported from barrel file

**✅ Pricing Page (Phase D - COMPLETE):**
- ✅ Page structure created with `'use client'` directive
- ✅ All 2 new templates created and integrated:
  - ✅ PricingHeroTemplate
  - ✅ PricingComparisonTemplate
- ✅ PricingTemplate migrated from PricingSection organism (removed hardcoded content)
- ✅ FAQTemplate reused with pricing-specific data
- ✅ EmailCapture props migrated
- ✅ Props organized in single file: PricingPageProps.tsx
- ✅ Commercial app replaced with clean `<PricingPage />` import (30 lines → 6 lines)
- ✅ Storybook stories created for all templates (OnHomePage, OnPricingPage, OnDevelopersPage)
- ✅ Templates exported from barrel file

**✅ Developers Page (Phase E - COMPLETE):**
- ✅ Page structure created with `'use client'` directive
- ✅ All 2 new templates created and integrated:
  - ✅ DevelopersHeroTemplate
  - ✅ DevelopersCodeExamplesTemplate
- ✅ PricingTemplate migrated from PricingSection organism (with TemplateContainer wrapper)
- ✅ All other templates reused (ProblemTemplate, SolutionTemplate, HowItWorks, CoreFeaturesTabs, UseCasesTemplate, TestimonialsTemplate, CTATemplate)
- ✅ EmailCapture props migrated
- ✅ Props organized in single file: DevelopersPageProps.tsx
- ✅ Storybook stories created for all templates
- ✅ Templates exported from barrel file

**🔄 Enterprise Page (Phase F - IN PROGRESS):**
- ⚠️ ALL Enterprise organisms must be converted to templates with props
- ⚠️ NO hardcoded content allowed (i18n + CMS requirement)
- Templates to create:
  - [ ] EnterpriseHeroTemplate
  - [ ] EnterpriseSolutionTemplate
  - [ ] EnterpriseComplianceTemplate
  - [ ] EnterpriseSecurityTemplate
  - [ ] EnterpriseHowItWorksTemplate
  - [ ] EnterpriseUseCasesTemplate
  - [ ] EnterpriseComparisonTemplate
  - [ ] EnterpriseFeaturesTemplate
  - [ ] EnterpriseTestimonialsTemplate
  - [ ] EnterpriseCTATemplate

**🔄 Providers Page (Phase G - PENDING):**
- Same rule: ALL organisms → templates with props
- NO hardcoded content

**Shared Components:**
- CoreFeaturesTabs (used across multiple pages)

## Rules

**Naming:**
- `XxxSection` → `XxxTemplate` (e.g., `ProblemSection` → `ProblemTemplate`)
- No "Section" suffix → keep name (e.g., `HomeHero`, `EmailCapture`, `CoreFeaturesTabs` stay as-is)

**Templates:**
- Remove `SectionContainer` wrapper
- **Accept ALL content as props - ZERO hardcoded strings/data/images**
- Export typed Props interface
- Pure presentation, no business logic
- Add JSDoc comment with `@example` showing usage
- Use separator comments (`// ────────────────`) to organize Types, Main Component sections
- **NO EXCEPTIONS:** Even single-use templates must accept all content as props (i18n + CMS requirement)

**Pages:**
- Props objects at top (in visual order)
- Use `Omit<TemplateContainerProps, "children">` for container props
- Wrap templates with `<TemplateContainer>` (except self-contained ones like CTASection)
- Clean composition only

**Icons:**
- **Templates that use `FeatureInfoCard` or `UseCaseCard`** → pass rendered: `icon: <AlertTriangle className="h-6 w-6" />`
- **Templates that use `IconPlate`/`FeatureListItem`** → pass component: `icon: Zap`
- **Consistency rule**: All icons in props objects must be rendered React components (`<Icon />`) not component references (`Icon`)

**Stories:**
- **CRITICAL: ALWAYS IMPORT PROPS FROM PAGE FILES - NEVER DUPLICATE**
- Import props from `@rbee/ui/pages` or specific page exports
- One story per page usage (e.g., `OnHomePage`, `OnDevelopersPage`)
- NO variant stories
- Story args = imported props object: `args: importedPropsName`
- NO inline prop objects in stories
- See `EmailCapture.stories.tsx` for the correct pattern

## Migration Steps (7 Steps)

**1. Create folder:** `mkdir -p src/templates/[Name]` (apply naming rule)

**2. Copy & clean:**
- Copy from `/organisms/` to `/templates/`
- Remove `SectionContainer` wrapper
- Define typed Props interface
- Remove `SectionContainer` import

**3. Create props in page:**
```typescript
// At top of HomePage.tsx, after other props (maintain visual order)
// Add section comment: // === X Template ===
export const xContainerProps: Omit<TemplateContainerProps, "children"> = { 
  title: "...",
  description: "...",
  bgVariant: "secondary",  // or "default"
  paddingY: "2xl",
  maxWidth: "7xl",  // or "6xl", "5xl"
  align: "center",
}

export const xProps: XProps = { 
  // IMPORTANT: Icons must be RENDERED components, not references
  icon: <AlertTriangle className="h-6 w-6" />,  // ✅ Correct
  // NOT: icon: AlertTriangle,  // ❌ Wrong
}
```

**4. Add to page composition:**
```typescript
// 1. Remove old organism import from organisms section
// 2. Add template import to templates section (alphabetical order)
import { X, type XProps } from "@rbee/ui/templates"

// 3. Add any new icon imports needed (alphabetical order)
import { Check, X as XIcon } from "lucide-react"

// 4. In page component (maintain visual order matching props)
<TemplateContainer {...xContainerProps}>
  <X {...xProps} />
</TemplateContainer>
```

**5. Create story:**
```typescript
// Create [Name].stories.tsx in template folder
import type { Meta, StoryObj } from '@storybook/react'
import { xTemplateProps } from '@rbee/ui/pages'  // IMPORT PROPS - DO NOT DUPLICATE
import { XTemplate } from './XTemplate'

const meta = {
  title: 'Templates/XTemplate',
  component: XTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof XTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * XTemplate as used on the Home page
 * - Brief description of what's unique about this usage
 */
export const OnHomePage: Story = {
  args: xTemplateProps,  // IMPORTED - NOT INLINE
}
```

**6. Export:** 
- Create `/templates/[Name]/index.ts`: `export * from './[Name]Template'`
- Update `/templates/index.ts`: Add `export * from './[Name]Template'` (alphabetical)

**7. Update page index:** 
- Export both container and template props from `/pages/HomePage/index.ts` (alphabetical order)

**Special cases:**
- Self-contained (CTASection): No TemplateContainer wrapper, keep internal `<section>`
- Shared templates: Each page defines own props
- Complex nested: Keep together if tightly coupled, split if reusable
- Client components: Add `'use client'` directive at top if component uses hooks (useState, etc.)

## Best Practices Being Applied

**Type Safety:**
- Always export typed Props interfaces from templates
- Use `type` imports where possible: `import type { XProps }`
- Define sub-types for complex nested structures (e.g., `PricingKickerBadge`, `ComparisonLegendItem`)

**Import Organization:**
- Group imports: React/Next → UI components → Assets → Icons
- Keep imports alphabetical within each group
- Import icons needed for props at top of page file

**Props Organization:**
- Add section comments before each template's props: `// === X Template ===`
- Define container props first, then template props
- Maintain visual order (props appear in same order as components in page)

**Data Mirroring:**
- When copying from commercial app, preserve exact data (testimonials, stats, pricing, etc.)
- Copy emoji avatars, pricing tiers, feature lists verbatim
- Maintain consistency between commercial app and rbee-ui package

**Documentation:**
- Add JSDoc comments to all exported types with `@example`
- Use separator comments for visual organization
- Include usage examples in JSDoc showing rendered icons

**Refactoring Plan Updates:**
- Mark completed components with ✅ immediately after finishing
- Update phase completion status (move from pending to completed list)

## Lessons Learned from HomePage Refactoring

### ✅ What Went Well

**1. Systematic Approach:**
- Following the 7-step migration process for each section kept work organized
- Refactoring one section at a time prevented overwhelming changes
- Marking sections as complete (✅) provided clear progress tracking

**2. Props Object Organization:**
- Adding descriptive JSDoc comments above each props object improved readability
- Maintaining visual order (props match page composition order) made navigation intuitive
- Section separators (`// === X Template ===`) created clear boundaries
- The header comment block helped orient developers quickly

**3. Icon Handling:**
- Converting icon references to rendered components (`<Icon className="h-6 w-6" />`) early prevented serialization issues
- Documenting this in props comments saved debugging time
- Consistent icon sizing (h-6 w-6 for most, h-3.5 w-3.5 for badges) maintained visual harmony

**4. Type Safety:**
- Exporting both container and template props from page index enabled reuse
- Using `Omit<TemplateContainerProps, "children">` for container props prevented type errors
- Sub-types for complex structures (e.g., `ComparisonLegendItem`) improved maintainability

**5. Template Reusability:**
- **Removing ALL hardcoded content made templates truly reusable (CRITICAL for i18n + CMS)**
- Accepting images, icons, and complex React nodes as props provided maximum flexibility
- Self-contained templates (FAQ, CTA) with internal `<section>` tags worked well for full-width layouts
- **NO EXCEPTIONS:** Even single-use components must accept all content as props

**6. Commercial App Integration:**
- Final replacement was trivial: 642 lines → 5 lines
- Single source of truth eliminated duplication
- Changes to HomePage automatically propagate to commercial app

### ⚠️ Critical Requirements for Next Pages

**0. NEVER DUPLICATE PROPS IN STORIES:**
```typescript
// ❌ WRONG - Duplicating props inline
export const OnHomePage: Story = {
  args: {
    title: 'Some Title',
    items: [...],  // Hundreds of lines duplicated
  }
}

// ✅ CORRECT - Import from props file
import { homeTemplateProps } from '@rbee/ui/pages'
export const OnHomePage: Story = {
  args: homeTemplateProps,  // Single source of truth
}
```
**WHY:** Props files are the single source of truth. Duplicating in stories creates:
- Maintenance nightmare (update in 2 places)
- Drift between page and story
- Wasted lines of code
- See `EmailCapture.stories.tsx` for correct pattern

**1. Client Component Directive:**
```typescript
'use client'  // MUST be first line if page uses icon component references
```
- Required when passing `LucideIcon` components (not rendered) to templates
- Next.js Server Components cannot serialize component classes
- Add immediately to avoid serialization errors

**2. Icon Consistency:**
- **Templates expecting `LucideIcon`** (WhatIsRbee, AudienceSelector): Pass component reference `icon: Zap`
- **Templates expecting `React.ReactNode`** (ProblemTemplate, UseCaseCard): Pass rendered `icon: <Zap className="h-6 w-6" />`
- Check template type definitions before creating props
- Document icon expectations in template JSDoc

**3. Data Migration:**
- Copy exact content from commercial app (don't paraphrase)
- Preserve emoji avatars, pricing, testimonials verbatim
- Maintain URLs, GitHub links, email addresses
- Keep code examples and terminal commands identical

**4. Props Comments:**
- Add descriptive comment above EVERY props object
- Format: `/** [Section] container/content - [Brief description] */`
- Include key features in comment (e.g., "Six persona cards with decision paths")
- Note special cases (self-contained, no container, client-side state)

**5. Import Organization:**
```typescript
'use client'  // If needed

// 1. UI Components
import { Badge } from "@rbee/ui/atoms"
import { TemplateContainer } from "@rbee/ui/molecules"

// 2. Assets
import { image1, image2 } from "@rbee/ui/assets"

// 3. Templates (alphabetical)
import {
  Template1,
  type Template1Props,
  Template2,
  type Template2Props,
} from "@rbee/ui/templates"

// 4. Icons (alphabetical)
import { Icon1, Icon2, Icon3 } from "lucide-react"
```

**6. Export Pattern:**
```typescript
// In /pages/[PageName]/index.ts - ALWAYS export both container and template props
export {
  section1ContainerProps,
  section1Props,
  section2ContainerProps,
  section2Props,
  // ... alphabetical order
} from "./[PageName]"
export { default as [PageName] } from './[PageName]'
```

**7. Refactoring Order:**
- Start with sections that have NO dependencies
- Refactor templates BEFORE creating page props
- Test each template in Storybook before integration
- Update refactoring plan after EACH section completion

### 📋 Page Creation Checklist

For each new page, verify:

- [ ] `'use client'` directive added (if using icon references)
- [ ] All imports organized by category (UI → Assets → Templates → Icons)
- [ ] Props objects have descriptive JSDoc comments
- [ ] Props objects in visual order matching page composition
- [ ] Section separators added (`// === X Template ===`)
- [ ] Icons rendered correctly (check template expectations)
- [ ] Container props use `Omit<TemplateContainerProps, "children">`
- [ ] Template props match data from commercial app exactly
- [ ] All props exported from page index (alphabetical)
- [ ] Page composition uses `<TemplateContainer>` wrappers (except self-contained)
- [ ] **Storybook stories IMPORT props from page files (NO DUPLICATION)**
- [ ] Commercial app file updated to import page component
- [ ] Refactoring plan updated with ✅ completion markers

### 🚀 Starting a New Page

1. **Identify sections** in commercial app page
2. **Check if templates exist** - if not, create them first (follow 7-step process)
3. **Create page file** at `/packages/rbee-ui/src/pages/[PageName]/[PageName].tsx`
4. **Add `'use client'`** if needed (check icon usage)
5. **Copy imports** from HomePage as starting point
6. **Create props objects** in visual order with comments
7. **Build composition** with TemplateContainer wrappers
8. **Create index.ts** with exports
9. **Update commercial app** to use new page
10. **Mark complete** in refactoring plan

### 🎯 Success Metrics

A page is complete when:
- Commercial app file is ≤10 lines (just import + export)
- All content comes from props (zero hardcoded strings)
- Props objects have descriptive comments
- All templates have Storybook stories
- TypeScript has zero errors
- Page renders identically to original commercial app version

## Commercial App Integration Pattern

**After page refactoring is complete:**
```typescript
// apps/commercial/app/[page]/page.tsx
import { [PageName] } from '@rbee/ui/pages'

export default function [Page]() {
  return <[PageName] />
}
```

**✅ Completed:** HomePage, FeaturesPage, UseCasesPage, PricingPage
**🔄 Next:** Developers Page, Enterprise Page

## Reused Templates - Multi-Page Stories

The following templates are reused across multiple pages and have stories showing each usage:

**EmailCapture** (4 pages):
- `OnHomePage` - Badge with dev status, community focus, GitHub/Discord links
- `OnFeaturesPage` - Feature updates and performance improvements
- `OnUseCasesPage` - Use cases and best practices focus
- `OnPricingPage` - Pricing updates and billing notifications

**FAQTemplate** (2 pages):
- `OnHomePage` - 8 general questions, 6 categories (Setup, Models, Performance, etc.), support card with beehive illustration
- `OnPricingPage` - 6 pricing questions, 4 categories (Licensing, Plans, Billing, Trials), no support card

**FeaturesTabs** (2 pages):
- `OnHomePage` - Core capabilities section mid-page
- `OnFeaturesPage` - Core capabilities after hero for deep dive
