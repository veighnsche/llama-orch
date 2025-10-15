import type { Meta, StoryObj } from "@storybook/react";
import { CoreFeaturesTabs } from "./CoreFeaturesTabs";
import { defaultTabConfigs } from "./tabConfigs";

const meta = {
  title: "Organisms/CoreFeaturesTabs",
  component: CoreFeaturesTabs,
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component: `
## Overview
The CoreFeaturesTabs section presents rbee's four core capabilities through an interactive tabbed interface. Each tab provides detailed technical information, code examples, and visual demonstrations of key features: OpenAI-compatible API, Multi-GPU orchestration, Programmable scheduler (Rhai), and Real-time SSE streaming.

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
import { CoreFeaturesTabs } from '@rbee/ui/organisms/Features/CoreFeaturesTabs'

// Simple usage - no props needed
<CoreFeaturesTabs />
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
  tags: ["autodocs"],
} satisfies Meta<typeof CoreFeaturesTabs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const FeaturesPageDefault: Story = {
  args: {
    title: "Core capabilities",
    description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.",
    tabs: defaultTabConfigs,
    defaultTab: "api",
  },
  parameters: {
    docs: {
      description: {
        story:
          'Default core features tabs for the Features page. Shows 4 interactive tabs: OpenAI-compatible API, Multi-GPU orchestration, Programmable scheduler (Rhai), and Real-time SSE. Each tab includes technical details, code examples or visualizations, and business value ("Why it matters").',
      },
    },
  },
};

export const SingleTabFocus: Story = {
  args: {
    title: "Core capabilities",
    description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.",
    tabs: defaultTabConfigs,
    defaultTab: "api",
  },
  render: (args) => (
    <div className="space-y-8">
      <CoreFeaturesTabs {...args} />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">
          Alternative: Single Tab Deep Dive
        </h3>
        <div className="max-w-2xl mx-auto">
          <p className="text-sm text-muted-foreground mb-4">
            For landing pages targeting a specific capability (e.g.,
            "OpenAI-Compatible API"), consider showing just one tab with more
            detail, more code examples, and a stronger CTA.
          </p>
          <div className="bg-background p-4 rounded-lg">
            <strong className="block mb-2">
              Example: API-Only Landing Page
            </strong>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                • Show multiple API examples (chat, completion, embeddings)
              </li>
              <li>• Compare to OpenAI, Anthropic, Cohere APIs</li>
              <li>• Show migration guide (3 steps)</li>
              <li>• Strong CTA: "Try the API in 5 minutes"</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          "Alternative approach: Deep dive on a single capability for targeted landing pages or campaigns.",
      },
    },
  },
};

export const InteractiveDemo: Story = {
  args: {
    title: "Core capabilities",
    description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.",
    tabs: defaultTabConfigs,
    defaultTab: "api",
  },
  render: (args) => (
    <div className="space-y-8">
      <div className="bg-primary/10 p-6 text-center">
        <h3 className="text-xl font-bold">Interactive Tabs Demo</h3>
        <p className="text-muted-foreground">
          Click tabs to explore each capability. Animations show transitions.
        </p>
      </div>
      <CoreFeaturesTabs {...args} />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">
          Tab Order Strategy
        </h3>
        <div className="max-w-3xl mx-auto space-y-4 text-sm">
          <div>
            <strong>Current Order:</strong> API → GPU → Scheduler → SSE
            <br />
            <span className="text-muted-foreground">
              → Leads with most important (API), builds to advanced (Scheduler)
            </span>
          </div>
          <div>
            <strong>Alternative A:</strong> GPU → API → SSE → Scheduler
            <br />
            <span className="text-muted-foreground">
              → Leads with visual proof (GPU bars), ends with advanced feature
            </span>
          </div>
          <div>
            <strong>Alternative B:</strong> API → SSE → GPU → Scheduler
            <br />
            <span className="text-muted-foreground">
              → Groups developer experience features (API, SSE) before
              infrastructure features
            </span>
          </div>
          <div className="pt-2">
            <strong>Testing Recommendation:</strong> A/B test the order. Current
            order (API first) likely converts best for developers. Alternative A
            (GPU first) may convert better for infrastructure teams.
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          "Interactive demo showing tab transitions and animations. Includes analysis of tab order strategy for A/B testing.",
      },
    },
  },
};
