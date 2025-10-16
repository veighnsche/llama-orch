import type { Meta, StoryObj } from '@storybook/react'
import { problemSectionProps } from '@rbee/ui/pages/DevelopersPage'
import { AlertTriangle, Cloud, DollarSign, Lock, Shield, TrendingDown } from 'lucide-react'
import { ProblemSection } from './ProblemSection'

const meta = {
  title: 'Organisms/ProblemSection',
  component: ProblemSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProblemSection displays a grid of problem cards to highlight pain points and challenges. It uses visual hierarchy with icons, tone-based styling, optional loss tags, and a CTA banner to drive action.

## Marketing Strategy

### Target Audience
Visitors who landed on the homepage and need to understand the problem before evaluating the solution. They need:
- Clear articulation of pain points
- Validation that their problems are understood
- Urgency to act (quantified losses)
- Emotional connection to frustrations

### Primary Message
Implicit: **"We understand your pain"** — Empathy-driven problem framing.

### Copy Analysis
- **Headline tone**: Problem-focused, empathetic
- **Emotional appeal**: Frustration with status quo, urgency to change
- **Power words**: "Unpredictable", "lock-in", "risk", "concerns", "Loss"
- **Social proof**: Loss tags quantify impact ("Loss €50/mo")

### Conversion Elements
- **Three problem cards**: Cost, vendor lock-in, privacy
- **Icons**: Visual differentiation (DollarSign, Lock, Shield)
- **Tone-based styling**: Destructive (red) for critical, primary (blue) for important
- **Loss tags**: Quantify monthly cost impact
- **Optional CTA banner**: Drive to solution after problem framing

## Composition
This organism contains:
- **Header**: Optional kicker, title, and subtitle
- **Problem Cards**: Grid of 3 cards with:
  - Icon (component or ReactNode)
  - Title and body text
  - Optional tag (e.g., "Loss €50/mo")
  - Tone-based styling (primary, destructive, muted)
- **CTA Banner**: Optional call-to-action with primary and secondary buttons
- **Staggered Animations**: Cards animate in with delays

## When to Use
- To highlight problems that the product solves
- Before solution or features sections
- On landing pages to create urgency
- To establish pain points before presenting value

## Content Requirements
- **Title**: Clear problem statement (REQUIRED)
- **Subtitle**: Context or amplification (optional)
- **Problem Items**: Array of cards with titles, descriptions, icons (REQUIRED)
- **Tags**: Optional loss indicators or metrics
- **CTA**: Clear next action (optional)

## Variants
- **Custom Problems**: Override with custom items
- **With CTA**: Include call-to-action banner
- **Without CTA**: Problems only

## Examples
\`\`\`tsx
import { ProblemSection } from '@rbee/ui/organisms'
import { AlertTriangle, DollarSign, Lock } from 'lucide-react'

// Custom problems
<ProblemSection
  title="Why developers struggle with AI"
  subtitle="Common challenges when building with AI"
  items={[
    {
      title: 'Vendor lock-in',
      body: 'Your code depends on proprietary APIs',
      icon: Lock,
      tone: 'destructive',
      tag: 'High risk'
    },
    {
      title: 'Rising costs',
      body: 'API fees multiply with team size',
      icon: DollarSign,
      tone: 'primary',
      tag: 'Loss €200/mo'
    },
    {
      title: 'Model changes',
      body: 'Updates break your workflows',
      icon: AlertTriangle,
      tone: 'destructive'
    }
  ]}
  ctaPrimary={{ label: 'Start Free', href: '/signup' }}
  ctaSecondary={{ label: 'Learn More', href: '/docs' }}
  ctaCopy="Take control of your AI infrastructure"
/>
\`\`\`

## Used In
- Home page (/)
- Developers page (/developers)
- Enterprise page (/enterprise)
- GPU Providers page (/gpu-providers)

## Related Components
- Button
- IconBox

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Proper heading hierarchy
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **ARIA**: Icons marked as aria-hidden
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Section title (REQUIRED)',
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
    items: {
      control: 'object',
      description: 'Array of problem items (REQUIRED)',
      table: {
        type: { summary: 'ProblemItem[]' },
        category: 'Content',
      },
    },
  },
} satisfies Meta<typeof ProblemSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageContext: Story = {
  args: {
    title: 'The hidden risk of AI-assisted development',
    subtitle: "You're building complex codebases with AI assistance. What happens when the provider changes the rules?",
    items: [
      {
        title: 'The model changes',
        body: 'Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked.',
        icon: <AlertTriangle className="h-6 w-6" />,
        tone: 'destructive' as const,
      },
      {
        title: 'The price increases',
        body: '$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.',
        icon: <DollarSign className="h-6 w-6" />,
        tone: 'primary' as const,
      },
      {
        title: 'The provider shuts down',
        body: 'APIs get deprecated. Your AI-built code becomes unmaintainable overnight.',
        icon: <Lock className="h-6 w-6" />,
        tone: 'destructive' as const,
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Three problems**: Unpredictable costs, Vendor lock-in, Privacy concerns
- **Visual hierarchy**: Icons + tone-based styling
- **Optional CTA**: Drives to solution after problem framing

**Problem 1: The model changes**
- **Icon**: AlertTriangle (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked."
- **No tag**: Impact is immediate and qualitative
- **Target**: Developers who experienced breaking changes from AI provider updates
- **Why this pain point**: Addresses the #1 fear of AI-assisted development—loss of control. When Claude/GPT updates, code generation patterns change, breaking established workflows. This is a visceral, immediate pain that developers have experienced firsthand.

**Problem 2: The price increases**
- **Icon**: DollarSign (blue/primary)
- **Tone**: Primary (important, cost-focused)
- **Copy**: "$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral."
- **No tag**: Copy already quantifies the impact (10x increase)
- **Target**: Engineering managers, CTOs concerned about budget predictability
- **Why this pain point**: Addresses the economic reality of AI tooling. GitHub Copilot started at $10/mo, now $19-39/mo. Cursor is $20/mo per seat. For a 10-person team, that's $200-400/month, and prices keep rising. This creates budget anxiety and makes AI tooling a line-item risk.

**Problem 3: The provider shuts down**
- **Icon**: Lock (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "APIs get deprecated. Your AI-built code becomes unmaintainable overnight."
- **No tag**: Existential threat doesn't need quantification
- **Target**: Developers building long-term codebases with AI assistance
- **Why this pain point**: Addresses the existential risk of dependency. When you build complex codebases with AI assistance, you're creating technical debt that's tied to a specific provider. If that provider shuts down or changes their API, your codebase becomes unmaintainable. This is the "vendor lock-in" fear taken to its logical extreme.

**Conversion Strategy:**
- Three problems cover main pain points (workflow stability, cost predictability, long-term maintainability)
- Tone-based styling creates urgency (red = critical, blue = important)
- No CTA on home page—let the problem breathe before presenting solution
- Copy is developer-focused but accessible to non-technical decision-makers

**Tone**: Empathetic, urgent, problem-focused. Not fear-mongering, but realistic about the risks of AI dependency.`,
      },
    },
  },
}

export const WithoutCTA: Story = {
  args: {
    title: 'Why cloud AI is risky',
    subtitle: "Relying on external providers creates dependencies you can't control",
    items: [
      {
        title: 'Vendor lock-in',
        body: 'Your entire codebase depends on proprietary APIs. Switching providers means rewriting everything.',
        icon: <Lock className="h-6 w-6" />,
        tone: 'destructive' as const,
        tag: 'High risk',
      },
      {
        title: 'Unpredictable costs',
        body: 'Usage-based pricing scales exponentially. What starts at $20/month becomes $2000/month.',
        icon: <TrendingDown className="h-6 w-6" />,
        tone: 'primary' as const,
        tag: 'Loss €180/mo',
      },
      {
        title: 'Data privacy concerns',
        body: 'Your code and prompts are sent to external servers. Compliance becomes a nightmare.',
        icon: <Shield className="h-6 w-6" />,
        tone: 'destructive' as const,
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: 'Problem section without CTA banner, showing only the problem cards.',
      },
    },
  },
}

export const CustomProblems: Story = {
  args: {
    kicker: 'Common Challenges',
    title: 'Why cloud AI is risky',
    subtitle: "Relying on external providers creates dependencies you can't control",
    items: [
      {
        title: 'Vendor lock-in',
        body: 'Your entire codebase depends on proprietary APIs. Switching providers means rewriting everything.',
        icon: <Lock className="h-6 w-6" />,
        tone: 'destructive' as const,
        tag: 'High risk',
      },
      {
        title: 'Unpredictable costs',
        body: 'Usage-based pricing scales exponentially. What starts at $20/month becomes $2000/month.',
        icon: <TrendingDown className="h-6 w-6" />,
        tone: 'primary' as const,
        tag: 'Loss €180/mo',
      },
      {
        title: 'Data privacy concerns',
        body: 'Your code and prompts are sent to external servers. Compliance becomes a nightmare.',
        icon: <Shield className="h-6 w-6" />,
        tone: 'destructive' as const,
      },
    ],
    ctaPrimary: { label: 'Host Your Own', href: '/signup' },
    ctaSecondary: { label: 'Compare Options', href: '/pricing' },
    ctaCopy: 'Run AI on your own infrastructure and eliminate these risks',
  },
  parameters: {
    docs: {
      description: {
        story: 'Custom problem items with different icons, tones, and tags.',
      },
    },
  },
}

export const ToneVariations: Story = {
  args: {
    title: 'Problem tone variations',
    subtitle: 'Different visual treatments for different problem types',
    items: [
      {
        title: 'Destructive tone',
        body: 'Used for critical problems and risks. Red/destructive color scheme.',
        icon: <AlertTriangle className="h-6 w-6" />,
        tone: 'destructive' as const,
        tag: 'Critical',
      },
      {
        title: 'Primary tone',
        body: 'Used for important problems related to cost or efficiency. Primary color scheme.',
        icon: <DollarSign className="h-6 w-6" />,
        tone: 'primary' as const,
        tag: 'Important',
      },
      {
        title: 'Muted tone',
        body: 'Used for less urgent problems or informational items. Neutral color scheme.',
        icon: <Cloud className="h-6 w-6" />,
        tone: 'muted' as const,
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: 'Demonstrates the three tone options: destructive, primary, and muted.',
      },
    },
  },
}

export const OnDevelopersPage: Story = {
  args: problemSectionProps,
  parameters: {
    docs: {
      description: {
        story: '**Developers page context** — Problem section highlighting risks of AI-assisted development dependency.',
      },
    },
  },
}
