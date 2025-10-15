import type { Meta, StoryObj } from '@storybook/react'
import { ComparisonSection } from './ComparisonSection'

const meta = {
  title: 'Organisms/Home/ComparisonSection',
  component: ComparisonSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ComparisonSection presents a feature comparison table showing rbee vs. cloud APIs (OpenAI/Anthropic), Ollama, and cloud GPU providers (Runpod/Vast.ai). Uses visual indicators (checkmarks, X marks) and detailed notes to communicate differentiation.

## Marketing Strategy

### Target Audience
Visitors evaluating alternatives. They need:
- Clear differentiation (why rbee vs. competitors)
- Objective comparison (not just marketing claims)
- Understanding of trade-offs (cost vs. features vs. complexity)
- Confidence that rbee is the right choice for their needs

### Primary Message
**"Why Developers Choose rbee"** — Social proof positioning (others have chosen, you should too).

### Copy Analysis
- **Headline tone**: Social proof, confident
- **Emotional appeal**: Validation (others chose this), smart decision
- **Power words**: "$0", "Complete privacy", "Orchestrated", "None" (rate limits)
- **Social proof**: Implicit (developers choose this)

### Conversion Elements
- **Comparison table**: Six features across four options
- **Visual indicators**: Checkmarks (good), X marks (bad), text (neutral)
- **rbee column highlighted**: Visual emphasis on our solution
- **Footer CTA**: "See Quickstart" + "Architecture" buttons
- **Mobile-friendly**: Stacked cards on small screens

### Features Compared
1. **Total Cost**: rbee ($0) vs. OpenAI ($20-100/mo) vs. Ollama ($0) vs. Runpod ($0.50-2/hr)
2. **Privacy/Data Residency**: rbee (Complete) vs. OpenAI (Limited) vs. Ollama (Complete) vs. Runpod (Limited)
3. **Multi-GPU Utilization**: rbee (Orchestrated) vs. OpenAI (N/A) vs. Ollama (Limited) vs. Runpod (✓)
4. **OpenAI-Compatible API**: rbee (✓) vs. OpenAI (✓) vs. Ollama (Partial) vs. Runpod (✗)
5. **Custom Routing Policies**: rbee (Rhai-based) vs. others (✗)
6. **Rate Limits/Quotas**: rbee (None) vs. OpenAI (Yes) vs. Ollama (None) vs. Runpod (Yes)

### Objection Handling
- **"Why not just use OpenAI?"** → Cost ($20-100/mo), privacy (limited), rate limits (yes)
- **"Why not just use Ollama?"** → Multi-GPU (limited), API compatibility (partial), routing (none)
- **"Why not use Runpod?"** → Cost ($0.50-2/hr), privacy (limited), API (none), routing (none)
- **"What's the catch?"** → Table shows rbee wins on most dimensions

### Variations to Test
- Alternative competitors: Add/remove competitors based on audience
- Alternative feature order: Lead with most compelling differentiator
- Alternative visual style: More aggressive (red X's) vs. neutral

## Composition
This organism contains:
- **SectionContainer**: Wrapper with title and description
- **Legend**: Explains visual indicators
- **Desktop Table**: Full comparison table with sticky header
- **Mobile Cards**: Stacked comparison cards for small screens
- **CellContent**: Renders checkmarks, X marks, text, badges
- **Footer CTA**: Two buttons (Quickstart, Architecture)

## When to Use
- Home page: After use cases section
- Pricing page: Show value vs. alternatives
- Features page: Demonstrate differentiation
- Landing pages: Competitive positioning

## Content Requirements
- **Title**: Clear positioning statement
- **Description**: Supporting context
- **Features**: 6-8 comparison dimensions
- **Competitors**: 3-4 alternatives
- **Visual indicators**: Checkmarks, X marks, text
- **Footer CTA**: Clear next actions

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<ComparisonSection />
\`\`\`

**Context**: Appears after UseCasesSection, before PricingSection  
**Purpose**: Differentiate from alternatives, build confidence in choice  
**Metrics**: Engagement (do visitors scroll through table?)

## Examples
\`\`\`tsx
import { ComparisonSection } from '@rbee/ui/organisms/ComparisonSection'

// Default usage
<ComparisonSection />
\`\`\`

## Related Components
- SectionContainer
- Button

## Accessibility
- **Keyboard Navigation**: Table is keyboard navigable
- **Screen Readers**: Uses proper table semantics (caption, th, td)
- **Visual Indicators**: Checkmarks/X marks have sr-only text
- **Focus States**: Visible focus indicators on buttons
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Responsive**: Mobile cards maintain information hierarchy
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ComparisonSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Why Developers Choose rbee" — Social proof positioning
- **Subheadline**: "Local-first AI that's faster, private, and costs $0 on your hardware."
- **Six features compared**: Cost, Privacy, Multi-GPU, API, Routing, Rate Limits
- **Four competitors**: OpenAI/Anthropic, Ollama, Runpod/Vast.ai

**Feature 1: Total Cost**
- rbee: **$0** (runs on your hardware) — Badge: "Lowest"
- OpenAI: $20-100/mo per dev
- Ollama: $0
- Runpod: $0.50-2/hr
- **Winner**: rbee (tied with Ollama, but rbee has more features)

**Feature 2: Privacy/Data Residency**
- rbee: ✓ Complete
- OpenAI: ✗ Limited
- Ollama: ✓ Complete
- Runpod: ✗ Limited
- **Winner**: rbee (tied with Ollama)

**Feature 3: Multi-GPU Utilization**
- rbee: ✓ Orchestrated (unified pool across CUDA, Metal, CPU)
- OpenAI: N/A
- Ollama: Limited
- Runpod: ✓
- **Winner**: rbee (orchestrated vs. basic support)

**Feature 4: OpenAI-Compatible API**
- rbee: ✓
- OpenAI: ✓
- Ollama: Partial (some endpoints missing)
- Runpod: ✗
- **Winner**: rbee (tied with OpenAI)

**Feature 5: Custom Routing Policies**
- rbee: ✓ Rhai-based policies (script routing by model, region, cost)
- OpenAI: ✗
- Ollama: ✗
- Runpod: ✗
- **Winner**: rbee (unique differentiator)

**Feature 6: Rate Limits/Quotas**
- rbee: None (positive)
- OpenAI: Yes (negative)
- Ollama: None (positive)
- Runpod: Yes (negative)
- **Winner**: rbee (tied with Ollama)

**Conversion Strategy:**
- Table shows rbee wins on most dimensions
- rbee column highlighted visually
- Footer CTA drives to quickstart or architecture docs
- Mobile-friendly stacked cards maintain hierarchy

**Competitive Positioning:**
- **vs. OpenAI**: Cost, privacy, rate limits
- **vs. Ollama**: Multi-GPU orchestration, API compatibility, routing
- **vs. Runpod**: Cost, privacy, API, routing

**Tone**: Objective, fact-based, confident`,
      },
    },
  },
}

export const TwoWayComparison: Story = {
  render: () => (
    <div>
      <ComparisonSection />
      <div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
        <h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Two-Way Comparison Variants</h3>
        <div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
          <p>
            <strong>rbee vs. OpenAI:</strong> Emphasize cost savings, privacy, rate limits
          </p>
          <p>
            <strong>rbee vs. Ollama:</strong> Emphasize multi-GPU orchestration, API compatibility, routing policies
          </p>
          <p>
            <strong>rbee vs. Runpod:</strong> Emphasize cost (ongoing), privacy, API compatibility, routing
          </p>
          <p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
            <strong>A/B Test Recommendation:</strong> Test two-way comparison (rbee vs. one competitor) for landing
            pages targeting specific audiences.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Analysis of two-way comparison variants. Useful for targeted landing pages.',
      },
    },
  },
}

export const AlternativeCompetitors: Story = {
  render: () => (
    <div>
      <ComparisonSection />
      <div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
        <h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Alternative Competitor Sets</h3>
        <div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
          <p>
            <strong>Current:</strong> OpenAI/Anthropic, Ollama, Runpod/Vast.ai
          </p>
          <p>
            <strong>Alt 1:</strong> Add Together.ai, Replicate (more API providers)
          </p>
          <p>
            <strong>Alt 2:</strong> Add LocalAI, LM Studio (more local alternatives)
          </p>
          <p>
            <strong>Alt 3:</strong> Add Ray, Kubernetes (more orchestration alternatives)
          </p>
          <p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
            <strong>Recommendation:</strong> Current set covers main categories (cloud APIs, local inference, cloud
            GPUs). Add competitors based on audience feedback.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Alternative competitor sets for different positioning strategies.',
      },
    },
  },
}
