import { TemplateContainer } from '@rbee/ui/molecules'
import { coreFeatureTabsContainerProps, coreFeatureTabsProps } from '@rbee/ui/pages/DevelopersPage'
import { featuresFeaturesTabsContainerProps, featuresFeaturesTabsProps } from '@rbee/ui/pages/FeaturesPage'
import { featuresTabsContainerProps, featuresTabsProps } from '@rbee/ui/pages/HomePage'
import { providersFeaturesContainerProps, providersFeaturesProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { FeaturesTabs } from './FeaturesTabs'

const meta = {
  title: 'Templates/FeaturesTabs',
  component: FeaturesTabs,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The FeaturesTabs section presents rbee's four core capabilities through an interactive tabbed interface. Each tab provides detailed technical information, code examples, and visual demonstrations of key features: OpenAI-compatible API, Multi-GPU orchestration, Programmable scheduler (Rhai), and Real-time SSE streaming.

## Composition
This organism contains:
- **Section Title**: "Core capabilities"
- **Section Subtitle**: Brief overview of the four capabilities
- **Sticky Left Rail**: Tab list with icons and descriptions
- **Right Content Area**: Tab panels with detailed content
- **4 Tabs**:
  1. **OpenAI-Compatible API**: Drop-in replacement, code example, benefits
  2. **Multi-GPU Orchestration**: Device utilization bars, scaling benefits
  3. **Programmable Scheduler (Rhai)**: Custom routing logic, code example
  4. **Real-time SSE**: Task-based API with event streaming, log example

## Marketing Strategy

### Target Sub-Audience
**Primary**: Technical decision makers evaluating capabilities
**Secondary**: Developers and architects assessing technical depth

### Page-Specific Messaging
- **Features page**: Deep technical dive into each capability
- **Technical level**: Advanced
- **Focus**: Implementation details, proof of capability

### Copy Analysis
- **Technical level**: Advanced
- **Tab 1 - OpenAI-Compatible API**:
  - Benefit: "Drop-in replacement. Point to localhost."
  - Proof: Before/after environment variable example
  - Why it matters: No vendor lock-in, use your models, keep existing tooling
- **Tab 2 - Multi-GPU Orchestration**:
  - Benefit: "Higher throughput by saturating all devices."
  - Proof: Visual utilization bars (RTX 4090 #1: 92%, RTX 4090 #2: 88%, M2 Ultra: 76%, CPU: 34%)
  - Why it matters: Bigger models fit, lower latency, no single-machine bottleneck
- **Tab 3 - Programmable Scheduler**:
  - Benefit: "Optimize for cost, latency, or compliance—your rules."
  - Proof: Rhai code example showing custom routing logic
  - Why it matters: Deterministic routing, policy-ready, easy to evolve
- **Tab 4 - Real-time SSE**:
  - Benefit: "Full visibility for every inference job."
  - Proof: Event stream log (task.created, model.loading, token.generated)
  - Why it matters: Faster debugging, trustworthy UX, accurate cost tracking

### Conversion Elements
- **Interactive**: Tabs encourage exploration
- **Technical proof**: Code examples and visualizations build credibility
- **"Why it matters"**: Each tab explains business value
- **Progressive disclosure**: Start simple (API), build to advanced (scheduler)

## Tab Order Strategy

**Current Order**: API → GPU → Scheduler → SSE

**Rationale**:
1. **API** first: Most important (drop-in replacement)
2. **GPU** second: Core technical capability (multi-device orchestration)
3. **Scheduler** third: Advanced feature (programmability)
4. **SSE** fourth: Developer experience feature (observability)

**Alternative**: GPU → API → SSE → Scheduler (lead with visual proof)

## When to Use
- On the Features page after the hero
- To provide deep technical details
- To demonstrate each core capability
- To build technical credibility

## Content Requirements
- **Section Title**: Clear heading
- **Section Subtitle**: Brief overview
- **Tab List**: 4 tabs with icons and labels
- **Tab Panels**: Detailed content for each capability
- **Code Examples**: Where relevant (API, Scheduler, SSE)
- **Visual Aids**: Where relevant (GPU utilization bars)
- **"Why it matters"**: Business value for each capability

## Variants
- **Default**: All 4 tabs
- **Single Tab**: Focus on one capability (e.g., API only)
- **Interactive Demo**: Clickable tabs with animations

## Examples
\`\`\`tsx
import { FeaturesTabs } from '@rbee/ui/templates/FeaturesTabs'

// Simple usage - no props needed
<FeaturesTabs />
\`\`\`

## Used In
- Features page (\`/features\`)

## Technical Implementation
- Uses Radix UI Tabs for accessibility
- Sticky left rail on desktop (lg:sticky lg:top-24)
- Responsive: Horizontal tabs on mobile, vertical on desktop
- Animated tab transitions (fade-in slide-in-from-right-4)

## Related Components
- Tabs (Radix UI)
- Badge
- Check icon (Lucide)

## Accessibility
- **Keyboard Navigation**: Arrow keys navigate tabs, Tab key moves to panel
- **ARIA Labels**: TabsList has aria-label="Core features"
- **Semantic HTML**: Uses proper tab roles and relationships
- **Focus Management**: Focus moves to panel when tab is selected
- **Screen Readers**: Tab labels and panel content are properly associated
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FeaturesTabs>

export default meta
type Story = StoryObj<typeof meta>

/**
 * FeaturesTabs as used on the Home page
 * - Title: "Core capabilities"
 * - Description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time."
 * - 4 tabs: API, GPU, Scheduler, SSE
 * - Same content as Features page but positioned differently in page flow
 */
export const OnHomePage: Story = {
  render: (args) => (
    <TemplateContainer {...featuresTabsContainerProps}>
      <FeaturesTabs {...args} />
    </TemplateContainer>
  ),
  args: featuresTabsProps,
}

/**
 * FeaturesTabs as used on the Features page
 * - Title: "Core capabilities"
 * - Description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time."
 * - 4 tabs: API, GPU, Scheduler, SSE
 * - Appears after hero section for deep technical dive
 */
export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...featuresFeaturesTabsContainerProps}>
      <FeaturesTabs {...args} />
    </TemplateContainer>
  ),
  args: featuresFeaturesTabsProps,
}

/**
 * FeaturesTabs as used on the Developers page
 * - Title: "Core capabilities"
 * - Description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time."
 * - 4 tabs: API, GPU, Scheduler, SSE
 * - Developer-focused content and examples
 */
export const OnDevelopersPage: Story = {
  render: (args) => (
    <TemplateContainer {...coreFeatureTabsContainerProps}>
      <FeaturesTabs {...args} />
    </TemplateContainer>
  ),
  args: coreFeatureTabsProps,
}

/**
 * FeaturesTabs as used on the Providers page
 * - Title: "Everything You Need to Maximize Earnings"
 * - Description: "Professional-grade tools to manage your GPU fleet and optimize your passive income."
 * - Provider-focused tabs with earnings and marketplace features
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersFeaturesContainerProps}>
      <FeaturesTabs {...args} />
    </TemplateContainer>
  ),
  args: providersFeaturesProps,
}
