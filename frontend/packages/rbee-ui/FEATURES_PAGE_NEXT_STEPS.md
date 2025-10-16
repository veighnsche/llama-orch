# FeaturesPage - Required Fixes

## Issues Identified

1. **Props split across 3 files** - User wants ONE consolidated file
2. **Missing TemplateContainer wrappers** - Templates need to be wrapped like HomePage
3. **No stories created** - Need stories for all 8 templates

## Required Actions

### 1. Consolidate Props (HIGH PRIORITY)

**Create ONE file:** `FeaturesPageProps.tsx` with ALL props

**Structure:**
```typescript
import { Badge } from "@rbee/ui/atoms";
import { type TemplateContainerProps } from "@rbee/ui/molecules";
import type { /* all template types */ } from "@rbee/ui/templates";
import { /* all icons */ } from "lucide-react";
import { CodeBlock, GPUUtilizationBar, TerminalWindow } from "@rbee/ui/molecules";

// ===  Features Hero ===
export const featuresHeroProps: FeaturesHeroProps = { /* ... */ };

// === Features Tabs ===
export const featuresFeaturesTabsProps: FeaturesTabsProps = { /* ... */ };

// === Cross-Node Orchestration ===
export const crossNodeOrchestrationContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Cross-Pool Orchestration",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const crossNodeOrchestrationProps: CrossNodeOrchestrationTemplateProps = { /* ... */ };

// === Intelligent Model Management ===
export const intelligentModelManagementContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Intelligent Model Management",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "5xl",
  align: "center",
};
export const intelligentModelManagementProps: IntelligentModelManagementTemplateProps = { /* ... */ };

// === Multi-Backend GPU ===
export const multiBackendGpuContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Multi-Backend GPU Support",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const multiBackendGpuProps: MultiBackendGpuTemplateProps = { /* ... */ };

// === Error Handling ===
export const errorHandlingContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Comprehensive Error Handling",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const errorHandlingProps: ErrorHandlingTemplateProps = { /* ... */ };

// === Real-Time Progress ===
export const realTimeProgressContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Real‑time Progress Tracking",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const realTimeProgressProps: RealTimeProgressTemplateProps = { /* ... */ };

// === Security & Isolation ===
export const securityIsolationContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Security & Isolation",
  subtitle: "...",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const securityIsolationProps: SecurityIsolationTemplateProps = { /* ... */ };

// === Additional Features Grid ===
export const additionalFeaturesGridContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: "Everything You Need for AI Infrastructure",
  bgVariant: "background",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};
export const additionalFeaturesGridProps: AdditionalFeaturesGridTemplateProps = { /* ... */ };

// === Email Capture ===
export const featuresEmailCaptureProps: EmailCaptureProps = { /* ... */ };
```

**Note:** Templates that have their own section wrappers (title/subtitle) need those extracted to container props!

### 2. Update FeaturesPage.tsx

Already done - just needs the container props to exist.

### 3. Update index.ts

```typescript
export { default as FeaturesPage } from './FeaturesPage'
export * from './FeaturesPageProps'  // Single file export
```

### 4. Create Stories for All 8 Templates

**Pattern (from refactoring plan):**
```typescript
// src/templates/[TemplateName]/[TemplateName].stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import { [TemplateName] } from './[TemplateName]'
import { [templateName]Props } from '@rbee/ui/pages/FeaturesPage'

const meta: Meta<typeof [TemplateName]> = {
  title: 'Templates/[TemplateName]',
  component: [TemplateName],
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof [TemplateName]>

export const OnFeaturesPage: Story = {
  args: [templateName]Props,
}
```

**Create stories for:**
1. FeaturesHero
2. CrossNodeOrchestrationTemplate
3. IntelligentModelManagementTemplate
4. MultiBackendGpuTemplate
5. ErrorHandlingTemplate
6. RealTimeProgressTemplate
7. SecurityIsolationTemplate
8. AdditionalFeaturesGridTemplate

### 5. Remove Template Internal Wrappers

**Check each template** - if they have internal `<section>` with title/subtitle, those should be:
- Removed from template
- Moved to container props in FeaturesPageProps.tsx
- Template should accept pure content props only

**Example:**
```typescript
// ❌ WRONG - Template has its own wrapper
export function MyTemplate({ title, subtitle, content }) {
  return (
    <section className="py-16">
      <h2>{title}</h2>
      <p>{subtitle}</p>
      {content}
    </section>
  )
}

// ✅ CORRECT - Template is pure content
export function MyTemplate({ content }) {
  return <div>{content}</div>
}

// Container props handle the wrapper
export const myTemplateContainerProps = {
  title: "My Title",
  subtitle: "My Subtitle",
  bgVariant: "background",
  paddingY: "2xl",
}
```

## Checklist

- [ ] Delete 3 small props files
- [ ] Create ONE consolidated FeaturesPageProps.tsx
- [ ] Add container props for all 7 templates (not Hero, not EmailCapture)
- [ ] Extract any internal section wrappers from templates
- [ ] Update index.ts exports
- [ ] Create 8 story files
- [ ] Verify all templates follow HomePage pattern
- [ ] Test in Storybook

## Reference

See HomePage.tsx for the correct pattern - every template wrapped in TemplateContainer with separate container props.
