import type { Meta, StoryObj } from '@storybook/react'
import { Anchor, DollarSign, Laptop, Shield } from 'lucide-react'
import { SolutionSection } from './SolutionSection'

const meta = {
  title: 'Organisms/Home/SolutionSection',
  component: SolutionSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The SolutionSection presents the core value proposition through a combination of feature tiles, step-by-step process, and optional earnings/compliance metrics. Used across multiple pages with different contexts.

## Marketing Strategy

### Target Audience
Visitors who understand the problem and are evaluating the solution. They need:
- Clear articulation of benefits
- Understanding of how it works
- Proof that setup is achievable
- Context-specific value (home vs. providers vs. enterprise)

### Primary Message
**"Your hardware. Your models. Your control."** — Emphasizes ownership and autonomy.

### Copy Analysis
- **Headline tone**: Declarative, ownership-focused
- **Emotional appeal**: Control, independence, cost savings
- **Power words**: "Your", "Zero", "Complete", "Locked to your rules"
- **Social proof**: Topology diagram shows real multi-host setup

### Conversion Elements
- **Feature tiles**: Four benefits with icons (cost, privacy, stability, hardware utilization)
- **How It Works**: Three-step process reduces perceived complexity
- **Topology diagram**: Visual proof of multi-GPU capability
- **Optional CTAs**: Context-dependent (Get Started, View Docs, etc.)

### Objection Handling
- **"Is it expensive?"** → "Zero ongoing costs" + "Pay only for electricity"
- **"Is my data safe?"** → "Complete privacy" + "Code and data never leave your network"
- **"Will it break?"** → "Locked to your rules" + "Models update only when you approve"
- **"Can I use my hardware?"** → "Use all your hardware" + "CUDA, Metal, and CPU orchestrated as one pool"

### Variations to Test
- Alternative headline: "Stop Paying for AI. Start Owning It."
- Alternative benefit order: Lead with cost vs. privacy vs. control
- Alternative topology: Single-host vs. multi-host examples

## Composition
This organism contains:
- **Header**: Optional kicker, title, subtitle
- **Feature Tiles**: Grid of 4 features with icons, titles, bodies, optional badges
- **Steps Card**: Numbered timeline showing "How It Works"
- **Aside**: Optional custom content or earnings/compliance metrics
- **Topology Diagram**: Visual showing multi-host GPU orchestration
- **CTA Bar**: Optional primary and secondary buttons with caption

## When to Use
- Home page: Show core benefits after problem section
- Providers page: Show earning potential
- Enterprise page: Show compliance metrics
- Features page: Show technical capabilities

## Content Requirements
- **Title**: Clear value proposition
- **Subtitle**: Supporting context
- **Features**: 4 benefit tiles with icons
- **Steps**: 3-step process explanation
- **Aside**: Context-specific metrics or custom content
- **CTAs**: Clear next actions

## Usage in Commercial Site

### Home Page (/) - HomeSolutionSection
\`\`\`tsx
<HomeSolutionSection
  title="Your hardware. Your models. Your control."
  subtitle="rbee orchestrates inference across every GPU in your home network..."
  benefits={[
    { icon: DollarSign, title: 'Zero ongoing costs', body: 'Pay only for electricity...' },
    { icon: Shield, title: 'Complete privacy', body: 'Code and data never leave...' },
    { icon: Anchor, title: 'Locked to your rules', body: 'Models update only...' },
    { icon: Laptop, title: 'Use all your hardware', body: 'CUDA, Metal, and CPU...' }
  ]}
  topology={{ mode: 'multi-host', hosts: [...] }}
/>
\`\`\`

**Context**: Appears after ProblemSection, before HowItWorksSection  
**Purpose**: Present solution to problems raised earlier, show topology proof  
**Metrics**: Key decision point for technical visitors

## Examples
\`\`\`tsx
import { SolutionSection } from '@rbee/ui/organisms'
import { DollarSign, Shield } from 'lucide-react'

// Home page variant
<SolutionSection
  title="Your hardware. Your models. Your control."
  subtitle="rbee orchestrates inference across every GPU..."
  features={[...]}
  steps={[...]}
  topology={...}
/>

// Providers page variant with earnings
<SolutionSection
  title="Turn idle GPUs into income"
  features={[...]}
  steps={[...]}
  earnings={{
    title: "Earning Potential",
    rows: [...]
  }}
/>
\`\`\`

## Related Components
- Button
- IconBox
- TopologyDiagram

## Accessibility
- **Keyboard Navigation**: All buttons and interactive elements are keyboard accessible
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Proper heading hierarchy, ordered list for steps
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **ARIA**: Steps use aria-label for screen readers
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Section title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    subtitle: {
      control: 'text',
      description: 'Section subtitle',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    kicker: {
      control: 'text',
      description: 'Small text above title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
  },
} satisfies Meta<typeof SolutionSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
  args: {
    title: 'Your hardware. Your models. Your control.',
    subtitle:
      'rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform.',
    features: [
      {
        icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Zero ongoing costs',
        body: 'Pay only for electricity. No API bills, no per-token surprises.',
      },
      {
        icon: <Shield className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Complete privacy',
        body: 'Code and data never leave your network. Audit-ready by design.',
      },
      {
        icon: <Anchor className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Locked to your rules',
        body: 'Models update only when you approve. No breaking changes.',
      },
      {
        icon: <Laptop className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Use all your hardware',
        body: 'CUDA, Metal, and CPU orchestrated as one pool.',
      },
    ],
    steps: [
      {
        title: 'Install rbee-keeper',
        body: 'One command installs the orchestrator daemon on your primary machine.',
      },
      {
        title: 'Add your machines',
        body: 'Point rbee to other GPUs on your network via SSH or local discovery.',
      },
      {
        title: 'Start inferencing',
        body: 'Use the OpenAI-compatible API. rbee routes requests across your pool.',
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: `**Home page context** — Exact implementation from \`/\` route as HomeSolutionSection.

**Marketing Notes:**
- **Headline**: "Your hardware. Your models. Your control." — Triple ownership emphasis
- **Four benefits**: Address core concerns in priority order:
  1. Cost (Zero ongoing costs)
  2. Privacy (Complete privacy)
  3. Stability (Locked to your rules)
  4. Utilization (Use all your hardware)
- **Topology diagram**: Shows Gaming PC + MacBook Pro + Workstation with mixed backends (CUDA, Metal, CPU)
- **Three-step process**: Reduces perceived complexity of setup

**Conversion Strategy:**
- Positioned after ProblemSection to present solution
- Topology diagram provides visual proof of capability
- "How It Works" steps show achievable path
- No CTA at this level (appears later in flow)

**Tone**: Declarative, confident, technical but accessible`,
      },
    },
  },
}

export const WithoutTopology: Story = {
  args: {
    title: 'Your hardware. Your models. Your control.',
    subtitle: 'rbee orchestrates inference across every GPU in your home network.',
    features: [
      {
        icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Zero ongoing costs',
        body: 'Pay only for electricity. No API bills, no per-token surprises.',
      },
      {
        icon: <Shield className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Complete privacy',
        body: 'Code and data never leave your network. Audit-ready by design.',
      },
      {
        icon: <Anchor className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Locked to your rules',
        body: 'Models update only when you approve. No breaking changes.',
      },
      {
        icon: <Laptop className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Use all your hardware',
        body: 'CUDA, Metal, and CPU orchestrated as one pool.',
      },
    ],
    steps: [
      {
        title: 'Install rbee-keeper',
        body: 'One command installs the orchestrator daemon on your primary machine.',
      },
      {
        title: 'Add your machines',
        body: 'Point rbee to other GPUs on your network via SSH or local discovery.',
      },
      {
        title: 'Start inferencing',
        body: 'Use the OpenAI-compatible API. rbee routes requests across your pool.',
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story:
          'Benefits and steps only, without topology diagram. Useful for tighter layouts or when visual proof is not needed.',
      },
    },
  },
}

export const AlternativeBenefits: Story = {
  args: {
    title: 'Stop paying for AI. Start owning it.',
    subtitle: 'Run LLMs on hardware you already own. No monthly fees, no vendor lock-in, no rate limits.',
    features: [
      {
        icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: '$0 per month',
        body: 'Use hardware you already own. Electricity is your only cost.',
        badge: 'Lowest',
      },
      {
        icon: <Shield className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Private by default',
        body: 'Your code and prompts never leave your network. GDPR-ready.',
      },
      {
        icon: <Anchor className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'No rate limits',
        body: 'Inference speed limited only by your hardware, not arbitrary quotas.',
      },
      {
        icon: <Laptop className="h-6 w-6 text-primary" aria-hidden="true" />,
        title: 'Works with your tools',
        body: 'OpenAI-compatible API. Zed, Cursor, Continue—all work out of the box.',
      },
    ],
    steps: [
      {
        title: 'Install in 15 minutes',
        body: 'Single command installs rbee-keeper. No complex configuration.',
      },
      {
        title: 'Add your GPUs',
        body: 'Point to machines on your network. rbee discovers and pools them.',
      },
      {
        title: 'Use your existing code',
        body: 'Change OPENAI_API_BASE to localhost. Everything else stays the same.',
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: `Alternative benefit messaging for A/B testing. This variant:
- Leads with cost savings ("$0 per month" vs. "Zero ongoing costs")
- Emphasizes "No rate limits" instead of "Locked to your rules"
- Highlights tool compatibility more explicitly
- Uses more aggressive headline ("Stop paying... Start owning")
- Steps emphasize speed ("15 minutes") and simplicity

**Use case**: Test with cost-conscious audience or developers frustrated with API rate limits.`,
      },
    },
  },
}
