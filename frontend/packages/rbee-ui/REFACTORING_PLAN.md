# Page & Template Refactoring Plan

## Goal
Pages = props objects + composition. Templates = reusable UI sections from `/organisms/`.

## Components to Migrate

**Home Page (Phase A - High Priority):**
- ✅ AudienceSelector, ProblemTemplate, HowItWorks, SolutionTemplate, EmailCapture, UseCasesSection
- TestimonialsSection, TechnicalSection, ComparisonSection, FeaturesSection, CTASection

**Shared (Phase B):**
- PricingSection, CoreFeaturesTabs

**Features Page (Phase C):**
- MultiBackendGpu, CrossNodeOrchestration, IntelligentModelManagement, RealTimeProgress, ErrorHandling, SecurityIsolation, AdditionalFeaturesGrid

**Use Cases (Phase D):**
- UseCasesIndustry, UseCasesPrimary

## Rules

**Naming:**
- `XxxSection` → `XxxTemplate` (e.g., `ProblemSection` → `ProblemTemplate`)
- No "Section" suffix → keep name (e.g., `HomeHero`, `EmailCapture`, `CoreFeaturesTabs` stay as-is)

**Templates:**
- Remove `SectionContainer` wrapper
- Accept ALL content as props
- Export typed Props interface
- Pure presentation, no business logic

**Pages:**
- Props objects at top (in visual order)
- Use `Omit<TemplateContainerProps, "children">` for container props
- Wrap templates with `<TemplateContainer>` (except self-contained ones like CTASection)
- Clean composition only

**Icons:**
- `FeatureInfoCard` → pass rendered: `icon: <AlertTriangle className="h-6 w-6" />`
- `IconPlate`/`FeatureListItem` → pass component: `icon: Zap`

**Stories:**
- Import props from page files
- One story per page usage (e.g., `OnHomePage`, `OnDevelopersPage`)
- NO variant stories

## Migration Steps (7 Steps)

**1. Create folder:** `mkdir -p src/templates/[Name]` (apply naming rule)

**2. Copy & clean:**
- Copy from `/organisms/` to `/templates/`
- Remove `SectionContainer` wrapper
- Define typed Props interface
- Remove `SectionContainer` import

**3. Create props in page:**
```typescript
// At top of HomePage.tsx
export const xContainerProps: Omit<TemplateContainerProps, "children"> = { /* layout */ }
export const xProps: XProps = { /* content */ }
```

**4. Add to page composition:**
```typescript
// Import template at top
import { X } from "@rbee/ui/templates"

// In page component
<TemplateContainer {...xContainerProps}>
  <X {...xProps} />
</TemplateContainer>
```

**5. Create story:**
```typescript
import { xProps } from '@rbee/ui/pages/HomePage'
export const OnHomePage: Story = { args: xProps }
```

**6. Export:** Update `/templates/[Name]/index.ts` and `/templates/index.ts`

**7. Update page index:** Export props from `/pages/HomePage/index.ts`

**Special cases:**
- Self-contained (CTASection): No TemplateContainer wrapper, keep internal `<section>`
- Shared templates: Each page defines own props
- Complex nested: Keep together if tightly coupled, split if reusable

## Commercial App Integration

**CRITICAL**: Do NOT touch commercial app until HomePage refactoring is complete.

When ready:
```typescript
import { HomePage } from '@rbee/ui/pages'  // ✅ Correct
// NOT: import { AudienceSelector } from '@rbee/ui/templates'  ❌ Wrong
```
