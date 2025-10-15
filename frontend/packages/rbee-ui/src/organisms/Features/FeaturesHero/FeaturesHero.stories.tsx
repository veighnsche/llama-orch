import type { Meta, StoryObj } from '@storybook/react'
import { FeaturesHero } from './FeaturesHero'

const meta = {
  title: 'Organisms/Features/FeaturesHero',
  component: FeaturesHero,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The FeaturesHero is the primary hero section for the Features page (\`/features\`). It positions rbee as "Enterprise-grade AI. Homelab simple." and showcases key technical capabilities through a feature mosaic. Unlike the home and developers heroes, this version emphasizes technical depth and enterprise capabilities.

## Composition
This organism contains:
- **Headline**: "Enterprise-grade AI. Homelab simple."
- **Subheadline**: Technical positioning (orchestration, OpenAI-compatible, no cloud)
- **Micro-badges**: Quick technical proof points (GPUs, OpenAI-compatible, GDPR, backends)
- **CTAs**: "See all features" + "How it works"
- **Feature Mosaic**: 3 feature cards
  1. **Programmable Scheduler** (tall card, primary feature)
  2. **Model Catalog** (shorter card)
  3. **Cascading Shutdown** (shorter card)
- **Stat Strip**: BDD scenarios, zero cloud dependencies, multi-backend support
- **HoneycombPattern**: Background visual

## Marketing Strategy

### Target Sub-Audience
**Primary**: Technical decision makers (CTOs, lead engineers, architects)
**Secondary**: Developers evaluating technical capabilities

### Features Page vs. Home/Developers Messaging

**Home Page Hero:**
- Headline: "Run LLMs on Your Hardware. Pay Nothing."
- Audience: General (all personas)
- Focus: Cost savings, simplicity

**Developers Page Hero:**
- Headline: "Build with AI. Own your infrastructure."
- Audience: Developers
- Focus: Technical control, OpenAI compatibility

**Features Page Hero:**
- Headline: "Enterprise-grade AI. Homelab simple."
- Audience: Technical decision makers
- Focus: Enterprise capabilities + accessibility
- **Key positioning**: Bridges enterprise quality with homelab simplicity

### Copy Analysis
- **Technical level**: Advanced
- **Positioning**: "Enterprise-grade" (quality, reliability) + "Homelab simple" (accessibility, no complexity)
- **Proof points**:
  - "42/62 BDD scenarios passing" (quality, testing)
  - "Zero cloud dependencies" (independence)
  - "Multi-backend: CUDA · Metal · CPU" (flexibility)
- **Feature highlights**:
  - Programmable Scheduler (Rhai scripts, 40+ helpers)
  - Model Catalog (one-click loading)
  - Cascading Shutdown (clean teardown)

### Conversion Elements
- **Primary CTA**: "See all features" (exploration)
- **Secondary CTA**: "How it works" (education)
- **Feature cards**: Show technical depth without overwhelming
- **Stat strip**: Credibility through metrics

## When to Use
- As the first section on the Features page (\`/features\`)
- To position rbee as enterprise-grade but accessible
- To showcase key technical capabilities
- To drive exploration of detailed features

## Content Requirements
- **Headline**: Positioning statement (2 parts: quality + accessibility)
- **Subheadline**: Technical explanation (2-3 sentences)
- **Micro-badges**: 4 quick proof points
- **CTAs**: Dual action paths
- **Feature Mosaic**: 3 key features with icons and descriptions
- **Stat Strip**: 3 credibility metrics

## Variants
- **Default**: Full hero with feature mosaic and stat strip
- **Alternative Headlines**: Different positioning emphasis
- **Minimal**: Without stat strip or mosaic

## Examples
\`\`\`tsx
import { FeaturesHero } from '@rbee/ui/organisms/Features/FeaturesHero'

// Simple usage - no props needed
<FeaturesHero />
\`\`\`

## Used In
- Features page (\`/features\`)

## Comparison to Other Heroes

### Home Page HeroSection:
- Focus: Cost savings ("Pay Nothing")
- Audience: General
- Proof: Terminal showing GPU orchestration

### Developers Page DevelopersHero:
- Focus: Technical control ("Own your infrastructure")
- Audience: Developers
- Proof: Terminal showing code generation

### Features Page FeaturesHero:
- Focus: Enterprise quality + accessibility ("Enterprise-grade. Homelab simple.")
- Audience: Technical decision makers
- Proof: Feature mosaic + stat strip (BDD scenarios, zero cloud, multi-backend)

**Key Difference**: Features hero positions rbee as bridging enterprise quality with homelab simplicity—appealing to technical decision makers who want both.

## Related Components
- HeroSection (home page)
- DevelopersHero (developers page)
- HoneycombPattern
- Badge
- Button
- Card

## Accessibility
- **Keyboard Navigation**: All buttons keyboard accessible
- **ARIA Labels**: Icons marked as decorative with aria-hidden and focusable="false"
- **Semantic HTML**: Uses \`<section>\` with proper heading hierarchy
- **Reduced Motion**: Respects prefers-reduced-motion for animations
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FeaturesHero>

export default meta
type Story = StoryObj<typeof meta>

export const FeaturesPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default hero section for the Features page with exact copy from `/features`. Positions rbee as "Enterprise-grade AI. Homelab simple." Shows feature mosaic with Programmable Scheduler, Model Catalog, and Cascading Shutdown. Includes stat strip with BDD scenarios, zero cloud dependencies, and multi-backend support.',
      },
    },
  },
}

export const AlternativeHeadlines: Story = {
  render: () => (
    <div className="space-y-8">
      <FeaturesHero />
      <div className="bg-muted p-8 text-center">
        <h3 className="text-xl font-bold mb-4">Alternative Headline Options (A/B Test)</h3>
        <div className="space-y-4 text-left max-w-2xl mx-auto">
          <div>
            <strong>Current:</strong> "Enterprise-grade AI. Homelab simple."
            <br />
            <span className="text-muted-foreground">
              → Bridges quality and accessibility (appeals to technical decision makers)
            </span>
          </div>
          <div>
            <strong>Alternative A:</strong> "Production-ready AI. Zero cloud dependencies."
            <br />
            <span className="text-muted-foreground">
              → Emphasizes reliability and independence (appeals to autonomy-focused)
            </span>
          </div>
          <div>
            <strong>Alternative B:</strong> "Enterprise orchestration. Developer simplicity."
            <br />
            <span className="text-muted-foreground">
              → Emphasizes both audiences (appeals to teams with mixed roles)
            </span>
          </div>
          <div>
            <strong>Alternative C:</strong> "Advanced features. Simple setup."
            <br />
            <span className="text-muted-foreground">
              → Emphasizes capability vs. complexity (appeals to pragmatic engineers)
            </span>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Comparison of current headline with A/B test alternatives. Each headline emphasizes different aspects: quality+accessibility, reliability+independence, audiences, or capability+simplicity.',
      },
    },
  },
}

export const ComparisonToOtherHeroes: Story = {
  render: () => (
    <div className="space-y-8">
      <div className="bg-primary/10 p-6 text-center">
        <h3 className="text-xl font-bold">Features Page Hero (Below)</h3>
        <p className="text-muted-foreground">
          Technical focus: "Enterprise-grade AI. Homelab simple." + Feature mosaic
        </p>
      </div>
      <FeaturesHero />
      <div className="bg-muted p-8 text-center">
        <h3 className="text-xl font-bold mb-4">Hero Comparison Across Pages</h3>
        <div className="grid md:grid-cols-3 gap-6 text-left max-w-6xl mx-auto">
          <div>
            <h4 className="font-semibold mb-2">Home Page Hero</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Headline: "Run LLMs on Your Hardware. Pay Nothing."</li>
              <li>• Audience: General (all personas)</li>
              <li>• Focus: Cost savings, simplicity</li>
              <li>• Proof: Terminal (GPU orchestration)</li>
              <li>• Tone: Accessible, broad appeal</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Developers Page Hero</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Headline: "Build with AI. Own your infrastructure."</li>
              <li>• Audience: Developers</li>
              <li>• Focus: Technical control, OpenAI compatibility</li>
              <li>• Proof: Terminal (code generation)</li>
              <li>• Tone: Technical, workflow-focused</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Features Page Hero</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Headline: "Enterprise-grade AI. Homelab simple."</li>
              <li>• Audience: Technical decision makers</li>
              <li>• Focus: Enterprise quality + accessibility</li>
              <li>• Proof: Feature mosaic + stat strip</li>
              <li>• Tone: Advanced, credibility-focused</li>
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
          'Side-by-side comparison of all three hero sections. Features hero bridges enterprise quality with homelab simplicity, targeting technical decision makers.',
      },
    },
  },
}
