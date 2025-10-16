# Page & Template Refactoring Plan

## Goal
Transform pages into CMS-like copy editing interfaces where content is defined as props objects, making templates reusable and composable.

## Current State (As of Audit)

### What We Have Now
- **33+ section components in `/organisms/`** that use `SectionContainer`
- **2 templates in `/templates/`**: `HomeHero`, `WhatIsRbee` (already migrated)
- **1 page in `/pages/`**: `HomePage` (partially migrated)
- **`SectionContainer`** in `/molecules/` - used by all section components
- **`TemplateContainer`** in `/molecules/` - new wrapper for templates (created)

### Components Using SectionContainer (Need Migration)
Based on grep audit, these components wrap content with `SectionContainer`:

**Home Page Sections:**
- `AudienceSelector`
- `ComparisonSection`
- `FeaturesSection`
- `SocialProofSection`
- `TechnicalSection`
- `WhatIsRbee` (in `/organisms/`, duplicate of template version)

**Shared Sections:**
- `HowItWorksSection`
- `ProblemSection`
- `PricingSection`
- `HomeSolutionSection` / `SolutionSection`
- `UseCasesSection`

**Feature Page Sections:**
- `AdditionalFeaturesGrid`
- `CrossNodeOrchestration`
- `ErrorHandling`
- `IntelligentModelManagement`
- `MultiBackendGpu`
- `RealTimeProgress`
- `SecurityIsolation`

**Use Case Sections:**
- `UseCasesIndustry`
- `UseCasesPrimary`

### Components That DON'T Use SectionContainer
These are self-contained and may not need `TemplateContainer`:
- `CTASection` - Has its own section wrapper with custom styling
- `EmailCapture` - Likely self-contained
- `CoreFeaturesTabs` - Likely self-contained

## Core Principles

### 1. **Pages are Content Configuration Files**
- Pages (e.g., `HomePage.tsx`) should contain only:
  - Props objects for each template (in visual order)
  - Template composition with TemplateContainer wrappers
  - NO business logic or complex JSX
  - Think of it like a CMS page editor

### 2. **Templates are Reusable UI Sections**
- Move sections from `/organisms/` to `/templates/`
- **NAMING RULE**: If component name ends with "Section", rename to "Template" (e.g., `ProblemSection` → `ProblemTemplate`)
- **DO NOT** add "Template" suffix to components without "Section" (e.g., `HomeHero`, `EmailCapture` stay as-is)
- Templates should:
  - Accept props for ALL content (text, CTAs, images, etc.)
  - NOT include `SectionContainer` wrapper
  - Be pure presentation components
  - Export typed Props interfaces (e.g., `ProblemTemplateProps`, `HomeHeroProps`)
  - Be reusable across multiple pages

### 3. **TemplateContainer Manages Layout**
- `TemplateContainer` (renamed from `SectionContainer`) handles:
  - Section title, eyebrow, description
  - Background variants
  - Padding, max-width, alignment
  - Wraps templates at the page level
- **Note**: `SectionContainer` still exists for backward compatibility during migration

### 4. **Organisms vs Templates**
- **Organisms** = Composite components (multiple molecules) but NOT full page sections
- **Templates** = Full page sections that were previously in `/organisms/`
- After migration, `/organisms/` should only contain true composite components, not sections

### 5. **Icon Props Pattern**
Icons should be passed based on the component that will render them:

**For components using `FeatureInfoCard` (e.g., ProblemSection, SolutionSection):**
- ✅ **Correct**: `icon: <AlertTriangle className="h-6 w-6" />`
- ❌ **Incorrect**: `icon: AlertTriangle`
- Pass as **rendered React elements** with className

**For components using `IconPlate` or `FeatureListItem` (e.g., WhatIsRbee):**
- ✅ **Correct**: `icon: Zap`
- ❌ **Incorrect**: `icon: <Zap className="h-6 w-6" />`
- Pass as **component references** (LucideIcon type)

**Why the difference?**
- `FeatureInfoCard` accepts `React.ReactNode` and handles both patterns
- `IconPlate`/`FeatureListItem` expect `LucideIcon` component types and render them internally
- Check the component's TypeScript interface to determine which pattern to use

## File Structure

**NAMING CONVENTION**: Only rename "Section" to "Template" (e.g., `ProblemSection` → `ProblemTemplate`). Don't add suffixes to other components.

```
/templates/
  /HomeHero/               # No suffix - stays as-is
    HomeHero.tsx
    HomeHero.stories.tsx
    index.ts
  /EmailCapture/           # No suffix - stays as-is
    EmailCapture.tsx
    EmailCapture.stories.tsx
    index.ts
  /ProblemTemplate/        # "Section" → "Template"
    ProblemTemplate.tsx
    ProblemTemplate.stories.tsx
    index.ts
  /SolutionTemplate/       # "Section" → "Template"
    SolutionTemplate.tsx
    SolutionTemplate.stories.tsx
    index.ts

/pages/
  /HomePage/
    HomePage.tsx           # Props objects + composition
    HomePage.stories.tsx   # Page-level stories
    index.ts
  /[PageName]/
    [PageName].tsx
    [PageName].stories.tsx
    index.ts

/molecules/
  /TemplateContainer/      # Layout wrapper
    TemplateContainer.tsx
    index.ts
```

## Refactoring Steps

### Phase 1: Audit & Identify ✅ COMPLETED

**Audit Results:**
- ✅ Identified 33+ section components using `SectionContainer`
- ✅ Identified 3 self-contained sections (CTASection, EmailCapture, CoreFeaturesTabs)
- ✅ Created list organized by page/feature area
- ✅ Confirmed `TemplateContainer` exists and is exported

**Component Categories:**
- **Templates to Migrate** (33+ components): All components currently using `SectionContainer`
- **Self-Contained Templates**: `CTASection`, `EmailCapture`, `CoreFeaturesTabs` (evaluate case-by-case)
- **True Organisms**: Navigation, Footer, and other composite components that aren't full sections
- **Already Migrated**: `HomeHero`, `WhatIsRbee` (in `/templates/`)

### Phase 2: Template Migration (Per Section)

For each section being converted to a template:

#### Step 1: Create Template Structure
**NAMING RULE**: If migrating a component ending with "Section", rename to "Template". Otherwise, keep the original name.

```bash
mkdir -p src/templates/[TemplateName]
touch src/templates/[TemplateName]/[TemplateName].tsx
touch src/templates/[TemplateName]/[TemplateName].stories.tsx
touch src/templates/[TemplateName]/index.ts
```

Examples:
```bash
# Component ending with "Section" → rename to "Template"
mkdir -p src/templates/ProblemTemplate
touch src/templates/ProblemTemplate/ProblemTemplate.tsx

# Component without "Section" → keep original name
mkdir -p src/templates/EmailCapture
touch src/templates/EmailCapture/EmailCapture.tsx
```

#### Step 2: Extract & Clean Component
1. **Copy component from `/organisms/` to `/templates/`**
2. **Remove `SectionContainer` wrapper**
   - Extract all SectionContainer props (eyebrow, title, bgVariant, etc.)
   - Remove the wrapper, keep only the inner content
3. **Define Props Interface**
   ```typescript
   // If renamed from Section → Template
   export interface ProblemTemplateProps {
     // All content-related props
   }
   export function ProblemTemplate({ ... }: ProblemTemplateProps) { }
   
   // If keeping original name
   export interface EmailCaptureProps {
     // All content-related props
   }
   export function EmailCapture({ ... }: EmailCaptureProps) { }
   ```
4. **Update imports**
   - Remove `SectionContainer` import
   - Keep only content-related imports

#### Step 3: Create Props Objects in Page
1. **In `HomePage.tsx` (or relevant page), add props objects in visual order**
   
   **IMPORTANT**: Props objects should be defined at the TOP of the file, BEFORE the page component, in the order they appear on the page.

   ```typescript
   // At top of HomePage.tsx, after imports
   
   // === Hero Section ===
   export const homeHeroProps: HomeHeroProps = {
     // ... hero content
   }
   
   // === What is rbee Section ===
   export const whatIsRbeeContainerProps: Omit<TemplateContainerProps, "children"> = {
     eyebrow: <Badge variant="secondary" className="uppercase tracking-wide">
       Open-source • Self-hosted
     </Badge>,
     title: "What is rbee?",
     bgVariant: "secondary",
     maxWidth: "5xl",
     paddingY: "xl",
     align: "center",
   }
   
   export const whatIsRbeeProps: WhatIsRbeeProps = {
     // ... template content props
   }
   
   // === Next Section ===
   // Continue pattern...
   ```

2. **Use `Omit<TemplateContainerProps, "children">` for container props**
   - This is required because `children` is passed separately when using the component
   - TypeScript will error if you use `TemplateContainerProps` directly

#### Step 4: Update Page Composition
```typescript
export default function HomePage() {
  return (
    <main>
      {/* Other templates */}
      
      <TemplateContainer {...[templateName]ContainerProps}>
        <[TemplateName] {...[templateName]Props} />
      </TemplateContainer>
      
      {/* More templates */}
    </main>
  )
}
```

#### Step 5: Create Storybook Story
**IMPORTANT**: Stories should import props from the page that uses the template. Do NOT duplicate props in stories.

**Story Purpose**: Show how each page uses the template with its specific props. NOT to demonstrate component variants or prop combinations.

```typescript
import type { Meta, StoryObj } from '@storybook/react'
import { [templateName]Props } from '@rbee/ui/pages/HomePage'  // Import from page!
import { [TemplateName] } from './[TemplateName]'

const meta = {
  title: 'Templates/[TemplateName]',
  component: [TemplateName],
  parameters: {
    layout: 'padded',  // or 'fullscreen' for hero templates
  },
  tags: ['autodocs'],
} satisfies Meta<typeof [TemplateName]>

export default meta
type Story = StoryObj<typeof meta>

// Use props from the page - single source of truth
export const OnHomePage: Story = {
  args: [templateName]Props,
}

// If used on multiple pages, create stories for each
export const OnDevelopersPage: Story = {
  args: developersPageProps,  // Import from DevelopersPage
}
```

**Key Points:**
- Props are defined ONCE in the page file
- Stories import and reuse those props
- Each page that uses the template gets its own story (e.g., `OnHomePage`, `OnDevelopersPage`)
- **NO variant stories** (e.g., `WithoutGradient`, `LargeSize`, etc.)
- The goal is to show real usage, not demonstrate all possible prop combinations
- Variants are how different pages use the template with different props

#### Step 6: Export from Template Index
```typescript
// src/templates/[TemplateName]/index.ts
export * from './[TemplateName]'

// src/templates/index.ts
export * from './[TemplateName]'
```

#### Step 7: Update Page Index
```typescript
// src/pages/HomePage/index.ts
export { 
  homeHeroProps, 
  whatIsRbeeContainerProps,
  whatIsRbeeProps,
  // Add new exports as you migrate
} from './HomePage'
```

**Note**: The old `HomeContent.tsx` file has been removed. All props are now in `HomePage.tsx`.

### Phase 3: Props Organization in Pages

**Order props objects in pages by visual flow:**

1. Above-the-fold hero section props first
2. Follow the page layout top-to-bottom
3. Group template props with their container props together

**Example structure:**
```typescript
// HomePage.tsx

// === Hero Section ===
export const homeHeroProps: HomeHeroProps = { /* ... */ }

// === What is rbee Section ===
export const whatIsRbeeContainerProps: Omit<TemplateContainerProps, "children"> = { /* ... */ }
export const whatIsRbeeProps: WhatIsRbeeProps = { /* ... */ }

// === Problem Section ===
export const problemContainerProps: Omit<TemplateContainerProps, "children"> = { /* ... */ }
export const problemProps: ProblemProps = { /* ... */ }

// === Solution Section ===
export const solutionContainerProps: Omit<TemplateContainerProps, "children"> = { /* ... */ }
export const solutionProps: SolutionProps = { /* ... */ }

// ... and so on

// === Page Component ===
export default function HomePage() {
  return (
    <main>
      <HomeHero {...homeHeroProps} />
      
      <TemplateContainer {...whatIsRbeeContainerProps}>
        <WhatIsRbee {...whatIsRbeeProps} />
      </TemplateContainer>
      
      <TemplateContainer {...problemContainerProps}>
        <Problem {...problemProps} />
      </TemplateContainer>
      
      {/* Continue pattern */}
    </main>
  )
}
```

### Phase 4: Special Cases

#### Templates Without SectionContainer (e.g., CTASection)
Some templates have their own section wrapper and don't use `SectionContainer`:

**Example: CTASection**
```typescript
// CTASection already has its own <section> wrapper with custom styling
export function CTASection({ title, primary, secondary, ... }: CTASectionProps) {
  return (
    <section className="border-b border-border bg-background py-24">
      {/* content */}
    </section>
  )
}
```

**Migration approach:**
1. Move to `/templates/` as-is
2. Create props object in page file
3. Use directly in page WITHOUT TemplateContainer wrapper
4. Keep the component's internal section wrapper

```typescript
// In HomePage.tsx
export const ctaSectionProps: CTASectionProps = { /* ... */ }

// In page component
<CTASection {...ctaSectionProps} />  // No TemplateContainer needed
```

#### Shared/Reusable Templates
Templates used across multiple pages (e.g., `HowItWorksSection`, `ProblemSection`):

**Pattern:**
- Create template once in `/templates/`
- Each page defines its own props object
- Same template component, different content per page

```typescript
// In HomePage.tsx
export const howItWorksProps: HowItWorksProps = {
  title: "Get started in 15 minutes",
  steps: [/* home-specific steps */]
}

// In DevelopersPage.tsx
export const howItWorksProps: HowItWorksProps = {
  title: "Integrate in minutes",
  steps: [/* developer-specific steps */]
}
```

#### Complex Templates with Nested Components
For templates with many sub-sections (e.g., `SolutionSection` with topology diagrams):

**Evaluation criteria:**
- If sub-sections are tightly coupled → Keep as one template
- If sub-sections could be reused independently → Split into multiple templates
- Prefer composition when it makes sense

**Example of keeping together:**
```typescript
export interface SolutionSectionProps {
  benefits: Benefit[]
  topology: TopologyConfig  // Complex nested structure
  features: Feature[]
}
```

**Example of splitting:**
```typescript
// Split into:
// - BenefitsTemplate
// - TopologyTemplate  
// - FeaturesTemplate
// Then compose in page
```

## Migration Checklist

### Per Template Migration:
- [ ] Identify section component in `/organisms/`
- [ ] Create template folder structure (`mkdir -p src/templates/[Name]`)
- [ ] Copy component from `/organisms/` to `/templates/`
- [ ] Remove `SectionContainer` wrapper from template
- [ ] Remove `SectionContainer` import
- [ ] Update Props interface (remove layout props like title, eyebrow, bgVariant)
- [ ] Create template props object in page file (at top, in visual order)
- [ ] Create container props object with `Omit<TemplateContainerProps, "children">`
- [ ] Update page composition with TemplateContainer wrapper
- [ ] Create Storybook story importing props from page
- [ ] Update `/templates/[Name]/index.ts` barrel export
- [ ] Update `/templates/index.ts` to export new template
- [ ] Update page index.ts to export new props
- [ ] Test in Storybook
- [ ] Test in dev server
- [ ] Delete old organism component (after confirming no other usage)
- [ ] Remove from `/organisms/index.ts` exports

### Page-Level Checklist:
- [ ] All section components converted to templates
- [ ] All props objects defined at TOP of page file (before component)
- [ ] Props ordered by visual page flow (hero first, then down the page)
- [ ] Container props and template props grouped together per section
- [ ] All templates wrapped with TemplateContainer (except self-contained ones)
- [ ] Page component is clean composition only (no JSX logic)
- [ ] All imports organized (atoms, molecules, templates, icons)
- [ ] Page exports all props for use in stories
- [ ] Page has Storybook story (optional but recommended)

## Benefits of This Pattern

### For Developers
- Clear separation of content and presentation
- Easy to find and edit copy
- Type-safe props
- Reusable templates across pages
- Testable in isolation

### For Content Editors
- Page file reads like a CMS editor
- All copy in one place (props objects)
- No need to dig through JSX
- Clear structure and organization

### For Design System
- Consistent section layouts via TemplateContainer
- Storybook stories for every template
- Easy to preview variants
- Maintainable and scalable

## Common Patterns

### Pattern 1: Simple Template with Container
```typescript
// In page file
export const myTemplateContainerProps = {
  title: "Section Title",
  bgVariant: "secondary",
}

export const myTemplateProps = {
  content: "...",
  cta: { label: "Click", href: "/path" },
}

// In page component
<TemplateContainer {...myTemplateContainerProps}>
  <MyTemplate {...myTemplateProps} />
</TemplateContainer>
```

### Pattern 2: Hero Template (No Container)
```typescript
// In page file
export const heroProps: HeroProps = {
  headline: "...",
  subheadline: "...",
  cta: { ... },
}

// In page component
<Hero {...heroProps} />
```

### Pattern 3: Multi-Section Template
```typescript
// If a template internally has multiple sub-sections
export const complexTemplateProps: ComplexTemplateProps = {
  sections: [
    { title: "...", items: [...] },
    { title: "...", items: [...] },
  ],
  footer: { ... },
}
```

## Anti-Patterns to Avoid

❌ **Don't**: Hardcode content in template components
```typescript
// BAD
export function Template() {
  return <h1>Hardcoded Title</h1>
}
```

✅ **Do**: Accept content as props
```typescript
// GOOD
export function Template({ title }: { title: string }) {
  return <h1>{title}</h1>
}
```

❌ **Don't**: Include SectionContainer in template
```typescript
// BAD
export function Template(props) {
  return (
    <SectionContainer title="...">
      {/* content */}
    </SectionContainer>
  )
}
```

✅ **Do**: Wrap template with TemplateContainer at page level
```typescript
// GOOD - in page
<TemplateContainer title="...">
  <Template {...props} />
</TemplateContainer>
```

❌ **Don't**: Put JSX logic in page component
```typescript
// BAD
export default function HomePage() {
  return (
    <main>
      <div className="...">
        <h1>{data.map(...)}</h1>
      </div>
    </main>
  )
}
```

✅ **Do**: Props objects + clean composition
```typescript
// GOOD
export const templateProps = {
  items: data.map(...), // Transform in props object
}

export default function HomePage() {
  return (
    <main>
      <Template {...templateProps} />
    </main>
  )
}
```

❌ **Don't**: Create variant stories in Storybook
```typescript
// BAD - Don't create noise stories
export const WithoutGradient: Story = {
  args: { ...props, showGradient: false }
}

export const LargeSize: Story = {
  args: { ...props, size: 'large' }
}
```

✅ **Do**: Only create stories for page usage
```typescript
// GOOD - Show how each page uses the template
export const OnHomePage: Story = {
  args: homePageAudienceSelectorProps,
}

export const OnDevelopersPage: Story = {
  args: developersPageAudienceSelectorProps,
}
```

## Migration Priority Order

### Phase A: HomePage Templates (High Priority)
These are used on the main landing page and should be migrated first:

1. ✅ **AudienceSelector** - COMPLETED - HomePage specific
2. ✅ **ProblemTemplate** - COMPLETED - Reusable across pages (renamed from ProblemSection)
3. ✅ **HowItWorks** - COMPLETED - Reusable across pages (renamed from HowItWorksSection)
4. ✅ **SolutionTemplate** - COMPLETED - Replaces HomeSolutionSection, BeeArchitecture optional (renamed from SolutionSection)
5. **UseCasesSection** - High content, reusable
6. **TestimonialsSection** - Content-heavy
7. **TechnicalSection** - HomePage specific
8. **ComparisonSection** - HomePage specific
9. **FeaturesSection** - HomePage specific
10. **CTASection** - Self-contained, used on HomePage
11. ✅ **EmailCapture** - COMPLETED - Self-contained, reusable

### Phase B: Shared Templates (Medium Priority)
These are used across multiple pages:

1. **PricingSection** - Used on multiple pages
2. **CoreFeaturesTabs** - Reusable feature display

### Phase C: Feature Page Templates (Lower Priority)
These are specific to the Features page:

1. **MultiBackendGpu**
2. **CrossNodeOrchestration**
3. **IntelligentModelManagement**
4. **RealTimeProgress**
5. **ErrorHandling**
6. **SecurityIsolation**
7. **AdditionalFeaturesGrid**

### Phase D: Use Case Templates (Lower Priority)
1. **UseCasesIndustry**
2. **UseCasesPrimary**

### Phase E: Cleanup
1. Remove duplicate `WhatIsRbee` from `/organisms/`
2. Update all imports across the codebase
3. Remove unused organism exports
4. Update documentation

## Next Steps

1. ✅ **Complete audit of `/organisms/` folder** - DONE
2. ✅ **Migrate AudienceSelector** - DONE
3. **Continue with remaining Phase A templates** (ProblemSection, HowItWorksSection, etc.)
   - Follow the 7-step migration process
   - Document any issues or learnings
4. **After HomePage is complete, update commercial app**
   - Import entire HomePage from `@rbee/ui/pages`
   - Do NOT import individual templates to commercial app
   - Commercial app will use the composed page, not individual templates
5. **Create other page files** (DevelopersPage, EnterprisePage, etc.)
6. **Migrate Phase B-D templates as needed**
7. **Final cleanup and documentation**

## Commercial App Integration

**IMPORTANT**: Do NOT update the commercial app until HomePage refactoring is complete.

When ready to integrate:
```typescript
// Commercial app should import the entire page
import { HomePage } from '@rbee/ui/pages'

// NOT individual templates like this:
// import { AudienceSelector } from '@rbee/ui/templates'  ❌ WRONG
```

The commercial app will use the composed HomePage from the UI library, not individual templates.

## Key Decisions & Answers

### ✅ Decided: Template Naming Convention
- **ONLY rename "Section" to "Template"** (e.g., `ProblemSection` → `ProblemTemplate`, `HowItWorksSection` → `HowItWorksTemplate`)
- **DO NOT add "Template" suffix** to components without "Section" (e.g., `HomeHero`, `EmailCapture`, `AudienceSelector` stay as-is)
- **Rationale**: Avoid polluting the namespace with obligatory affixes. Only replace the "Section" suffix since templates are not sections.
- Props interfaces follow the component name (e.g., `ProblemTemplateProps`, `EmailCaptureProps`)

### ✅ Decided: Container props are separate objects
- Use `Omit<TemplateContainerProps, "children">` for typing
- Keep container props and template props as separate exports
- Group them together per section in the page file
- Props naming: `[name]TemplateContainerProps` and `[name]TemplateProps` (or `[name]ContainerProps` and `[name]Props` for non-Section components)

### ✅ Decided: Templates don't include SectionContainer
- All layout/section concerns handled by TemplateContainer at page level
- Templates are pure content components
- Exception: Self-contained templates like CTASection keep their own wrappers

### ✅ Decided: Props objects live in page files
- No separate `HomeContent.tsx` or similar files
- All props at top of page file, before component
- Ordered by visual page flow

### 🤔 To Decide During Migration:
- Exact granularity for complex templates (split vs. keep together)
- Whether to create composition helpers for common patterns
- How to handle future CMS/API integration
- Whether to add runtime props validation (Zod, etc.)

## Success Criteria

- [ ] All HomePage sections migrated to templates
- [ ] All templates have TypeScript interfaces
- [ ] HomePage.tsx is primarily props objects + composition
- [ ] All templates have Storybook stories
- [ ] No hardcoded content in templates
- [ ] Consistent use of TemplateContainer (where appropriate)
- [ ] Pattern documented in this file
- [ ] At least one other page (Developers/Enterprise) follows pattern
- [ ] Old organism components removed after migration
- [ ] No breaking changes to existing pages during migration
