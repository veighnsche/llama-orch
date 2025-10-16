# Page & Template Refactoring Plan

## Goal
Pages = props objects + composition. Templates = reusable UI sections from `/organisms/`.

## Components to Migrate

**Home Page (Phase A - High Priority):**
- ✅ AudienceSelector, ProblemTemplate, HowItWorks, SolutionTemplate, EmailCapture, UseCasesSection, ComparisonSection, PricingSection, TestimonialsSection, TechnicalSection, FAQSection, CTASection
- FeaturesSection

**Shared (Phase B):**
- CoreFeaturesTabs

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
- Add JSDoc comment with `@example` showing usage
- Use separator comments (`// ────────────────`) to organize Types, Main Component sections

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
- Import props from page files when available
- One story per page usage (e.g., `OnHomePage`, `OnDevelopersPage`)
- NO variant stories
- Story args should match the page props exactly (copy from commercial app if needed)
- Import any required assets (images, icons) at top of story file

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
import { Icon1, Icon2 } from 'lucide-react'  // Import icons if needed
import { asset } from '@rbee/ui/assets'  // Import assets if needed
import { XTemplate } from './XTemplate'

const meta = {
  title: 'Templates/XTemplate',
  component: XTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof XTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    // Copy exact props from HomePage or commercial app
  },
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

## Commercial App Integration

**CRITICAL**: Do NOT touch commercial app until HomePage refactoring is complete.

When ready:
```typescript
import { HomePage } from '@rbee/ui/pages'  // ✅ Correct
// NOT: import { AudienceSelector } from '@rbee/ui/templates'  ❌ Wrong
```
