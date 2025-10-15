import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersHero } from './DevelopersHero'

const meta = {
  title: 'Organisms/Developers/DevelopersHero',
  component: DevelopersHero,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The DevelopersHero is the primary hero section for the Developers page (\`/developers\`). It targets developers specifically with technical messaging, code examples, and developer-focused value propositions. Unlike the home page hero, this version emphasizes technical control, OpenAI compatibility, and zero ongoing costs for developers building with AI.

## Composition
This organism contains:
- **Developer Badge**: "For developers who build with AI" announcement
- **Two-line Headline**: "Build with AI. Own your infrastructure."
- **Technical Value Proposition**: OpenAI-compatible API, home network hardware, zero costs
- **CTA Buttons**: "Get started free" (primary) and "View on GitHub" (secondary)
- **Trust Chips**: Open source, OpenAI-compatible, Zed/Cursor support, no cloud
- **TerminalWindow**: Live code generation demo with GPU utilization
- **Hardware Montage**: Visual showing homelab hardware (GPU tower, MacBook, workstation)

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers actively building with AI (using Cursor, Zed, Continue, or similar AI coding assistants)
**Secondary**: Developers concerned about vendor lock-in and API costs

### Developers Page vs. Home Page Messaging

**Home Page Hero:**
- Broader audience (developers, teams, enterprises)
- Focus: Cost savings, privacy, control
- Tone: Accessible to non-technical decision makers

**Developers Page Hero:**
- Narrow audience: Developers only
- Focus: Technical control, OpenAI compatibility, workflow integration
- Tone: Technical, assumes familiarity with AI coding tools
- Proof: Terminal demo showing actual code generation

### Copy Analysis
- **Technical level**: Intermediate to Advanced
- **Code examples**: Yes - TypeScript code generation in terminal
- **Proof points**: 
  - "OpenAI-compatible API" (drop-in replacement)
  - "Works with Zed & Cursor" (specific tools developers use)
  - "Zero ongoing costs" (economic benefit)
  - GPU utilization metrics (87%, 92%, $0.00 cost)

### Conversion Elements
- **Primary CTA**: "Get started free" - action-oriented, removes friction
- **Secondary CTA**: "View on GitHub" - credibility for developers (open source)
- **Tertiary**: "How it works" mobile link - educational path

## When to Use
- As the first section on the Developers page (\`/developers\`)
- To immediately communicate developer-specific value
- To show technical proof (terminal demo)
- To drive developers to GitHub or getting started

## Content Requirements
- **Badge**: Developer-specific announcement
- **Headline**: Benefit-focused, technical (2 lines max)
- **Subheadline**: Technical details (OpenAI-compatible, hardware orchestration, costs)
- **Trust Chips**: 4 technical proof points
- **Terminal Demo**: Realistic code generation with GPU metrics
- **Hardware Image**: Visual proof of homelab setup

## Variants
- **Default**: Full hero with terminal demo and hardware image
- **Alternative Headlines**: Different value prop emphasis
- **Simplified**: Minimal version without terminal demo

## Examples
\`\`\`tsx
import { DevelopersHero } from '@rbee/ui/organisms/Developers/DevelopersHero'

// Simple usage - no props needed
<DevelopersHero />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Comparison to Home Page

### Home Page HeroSection:
- Headline: "Run LLMs on Your Hardware. Pay Nothing."
- Audience: General (developers, teams, enterprises)
- Proof: Terminal showing GPU orchestration
- CTAs: "Get Started Free" + "View Documentation"

### Developers Page DevelopersHero:
- Headline: "Build with AI. Own your infrastructure."
- Audience: Developers specifically
- Proof: Terminal showing code generation (more relevant to developers)
- CTAs: "Get started free" + "View on GitHub" (GitHub more relevant)
- Additional: Trust chips emphasize Zed/Cursor compatibility

**Key Difference**: Developers hero shows AI *generating code* (the developer's use case), while home hero shows GPU orchestration (the technical capability).

## Related Components
- HeroSection (home page variant)
- TerminalWindow
- Badge
- Button

## Accessibility
- **Keyboard Navigation**: All buttons and links keyboard accessible
- **ARIA Labels**: Proper labels on interactive elements
- **Semantic HTML**: Uses \`<section>\` with proper heading hierarchy
- **Reduced Motion**: Respects prefers-reduced-motion for animations
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevelopersHero>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default hero section for the Developers page with exact copy from `/developers`. Shows terminal demo with code generation and GPU metrics. Use theme toggle to test dark mode.',
      },
    },
  },
}

export const AlternativeHeadlines: Story = {
  render: () => (
    <div className="space-y-8">
      <DevelopersHero />
      <div className="bg-muted p-8 text-center">
        <h3 className="text-xl font-bold mb-4">Alternative Headline Options (A/B Test)</h3>
        <div className="space-y-4 text-left max-w-2xl mx-auto">
          <div>
            <strong>Current:</strong> "Build with AI. Own your infrastructure."
            <br />
            <span className="text-muted-foreground">→ Emphasizes ownership and control (appeals to independence)</span>
          </div>
          <div>
            <strong>Alternative A:</strong> "Stop paying for AI. Start building."
            <br />
            <span className="text-muted-foreground">→ Emphasizes cost savings (appeals to budget-conscious)</span>
          </div>
          <div>
            <strong>Alternative B:</strong> "Your code. Your models. Your GPUs."
            <br />
            <span className="text-muted-foreground">→ Emphasizes complete ownership (appeals to privacy-focused)</span>
          </div>
          <div>
            <strong>Alternative C:</strong> "OpenAI-compatible. Zero vendor lock-in."
            <br />
            <span className="text-muted-foreground">→ Emphasizes compatibility (appeals to pragmatic developers)</span>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Comparison of current headline with A/B test alternatives. Each headline emphasizes different value propositions: ownership, cost, privacy, or compatibility.',
      },
    },
  },
}

export const ComparisonToHomePage: Story = {
  render: () => (
    <div className="space-y-8">
      <div className="bg-primary/10 p-6 text-center">
        <h3 className="text-xl font-bold">Developers Page Hero (Below)</h3>
        <p className="text-muted-foreground">Technical focus: "Build with AI. Own your infrastructure." + GitHub CTA</p>
      </div>
      <DevelopersHero />
      <div className="bg-muted p-8 text-center">
        <h3 className="text-xl font-bold mb-4">Key Differences from Home Page</h3>
        <div className="grid md:grid-cols-2 gap-6 text-left max-w-4xl mx-auto">
          <div>
            <h4 className="font-semibold mb-2">Home Page Hero</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Headline: "Run LLMs on Your Hardware. Pay Nothing."</li>
              <li>• Audience: General (all personas)</li>
              <li>• Badge: "Open source GPU orchestration"</li>
              <li>• CTAs: "Get Started" + "View Documentation"</li>
              <li>• Terminal: Shows GPU orchestration</li>
              <li>• Tone: Accessible, broad appeal</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Developers Page Hero</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Headline: "Build with AI. Own your infrastructure."</li>
              <li>• Audience: Developers specifically</li>
              <li>• Badge: "For developers who build with AI"</li>
              <li>• CTAs: "Get started free" + "View on GitHub"</li>
              <li>• Terminal: Shows code generation (developer use case)</li>
              <li>• Tone: Technical, assumes AI coding familiarity</li>
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
          'Side-by-side comparison of Developers hero vs. Home hero. Developers version is more technical, shows code generation instead of orchestration, and uses GitHub CTA instead of Documentation.',
      },
    },
  },
}
