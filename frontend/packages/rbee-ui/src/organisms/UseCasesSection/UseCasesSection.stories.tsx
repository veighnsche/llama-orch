import type { Meta, StoryObj } from '@storybook/react'
import { Building, Home as HomeIcon, Laptop, Users } from 'lucide-react'
import { UseCasesSection } from './UseCasesSection'

const meta = {
	title: 'Organisms/UseCasesSection',
	component: UseCasesSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The UseCasesSection presents persona-based use cases in a card grid format. Each card follows a Scenario → Solution → Outcome structure, making the value proposition concrete for different audience segments.

## Marketing Strategy

### Target Audience
Visitors evaluating fit for their specific use case. They need:
- Relatable personas (solo dev, small team, homelab, enterprise)
- Concrete scenarios (not abstract benefits)
- Specific outcomes (cost savings, compliance, etc.)
- Proof that rbee works for "people like me"

### Primary Message
**"Built for those who value independence"** — Emphasizes autonomy and control across all personas.

### Copy Analysis
- **Headline tone**: Values-driven, inclusive
- **Emotional appeal**: Independence, control, cost savings, compliance
- **Power words**: "$0/month", "$6,000+ saved", "GDPR-friendly", "EU-only compliance"
- **Social proof**: Specific outcomes (quantified savings, compliance)

### Conversion Elements
- **Four personas**: Solo developer, Small team, Homelab enthusiast, Enterprise
- **Scenario → Solution → Outcome**: Clear narrative structure
- **Quantified outcomes**: Specific cost savings, compliance benefits
- **Icons**: Visual differentiation between personas
- **Highlighted outcomes**: Green callout boxes draw attention to results

### Objection Handling
- **"Is it for me?"** → Four personas cover most audience segments
- **"What will I save?"** → "$0/month", "$6,000+ saved per year"
- **"Is it production-ready?"** → Enterprise persona shows org-scale usage
- **"Will it work for my team?"** → Small team persona shows collaboration

### Variations to Test
- Alternative persona order: Lead with most common (solo dev) vs. highest value (enterprise)
- Alternative outcomes: Emphasize cost vs. compliance vs. control
- Alternative scenarios: Show different industries or use cases

## Composition
This organism contains:
- **Header**: Title and subtitle
- **Cards Grid**: 2-3 column responsive grid
- **IconPlate**: Persona icon in colored plate
- **Scenario**: Problem statement
- **Solution**: How rbee addresses it
- **Outcome**: Quantified result (highlighted)
- **Optional Tags**: Additional metadata
- **Optional CTA**: Per-card action link

## When to Use
- Home page: After features section
- Use Cases page: As primary content
- Landing pages: To show persona fit
- Pricing page: To show value by segment

## Content Requirements
- **Title**: Clear positioning statement
- **Subtitle**: Supporting context
- **Items**: 3-4 persona cards
- **Per-card**: Icon, title, scenario, solution, outcome
- **Outcomes**: Quantified when possible

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<UseCasesSection
  title="Built for those who value independence"
  subtitle="Run serious AI on your own hardware. Keep costs at zero, keep control at 100%."
  items={[
    {
      icon: Laptop,
      title: 'The solo developer',
      scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
      solution: 'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
      outcome: '$0/month AI costs. Full control. No rate limits.'
    },
    // ... 3 more personas
  ]}
/>
\`\`\`

**Context**: Appears after FeaturesSection, before ComparisonSection  
**Purpose**: Show persona fit, quantify value, build relatability  
**Metrics**: Which personas resonate most? (engagement by card)

## Examples
\`\`\`tsx
import { UseCasesSection } from '@rbee/ui/organisms/UseCasesSection'
import { Laptop, Users } from 'lucide-react'

// Default usage
<UseCasesSection
  title="Built for those who value independence"
  subtitle="Run serious AI on your own hardware."
  items={[...]}
/>

// Two-column layout
<UseCasesSection
  title="Who uses rbee?"
  items={[...]}
  columns={2}
/>
\`\`\`

## Related Components
- IconPlate
- Badge

## Accessibility
- **Keyboard Navigation**: Cards are focusable with tab key
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Uses article elements for cards
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Outcome Highlighting**: Color-coded for emphasis but not sole indicator
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
		columns: {
			control: 'select',
			options: [2, 3],
			description: 'Number of columns in grid',
			table: {
				type: { summary: 'number' },
				defaultValue: { summary: '3' },
				category: 'Layout',
			},
		},
	},
} satisfies Meta<typeof UseCasesSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
	args: {
		title: 'Built for those who value independence',
		subtitle: 'Run serious AI on your own hardware. Keep costs at zero, keep control at 100%.',
		items: [
			{
				icon: Laptop,
				title: 'The solo developer',
				scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
				solution:
					'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
				outcome: '$0/month AI costs. Full control. No rate limits.',
			},
			{
				icon: Users,
				title: 'The small team',
				scenario: '5-person startup burning $500/mo on APIs.',
				solution:
					'Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.',
				outcome: '$6,000+ saved per year. GDPR-friendly by design.',
			},
			{
				icon: HomeIcon,
				title: 'The homelab enthusiast',
				scenario: 'Four GPUs gathering dust.',
				solution: 'Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.',
				outcome: 'Idle GPUs → productive. Auto-download models, clean shutdowns.',
			},
			{
				icon: Building,
				title: 'The enterprise',
				scenario: '50-dev org. Code cannot leave the premises.',
				solution:
					'On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.',
				outcome: 'EU-only compliance. Zero external dependencies.',
			},
		],
	},
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Built for those who value independence" — Values-driven positioning
- **Four personas**: Cover primary audience segments
  1. **Solo developer**: Cost-conscious, wants control
  2. **Small team**: Burning money on APIs, needs collaboration
  3. **Homelab enthusiast**: Has hardware, wants to use it
  4. **Enterprise**: Compliance requirements, on-prem needs

**Persona 1: Solo Developer**
- **Scenario**: Shipping SaaS, wants control
- **Solution**: Gaming PC + workstation, Llama 70B + SD
- **Outcome**: "$0/month AI costs. Full control. No rate limits."
- **Target**: Most common persona, leads with cost

**Persona 2: Small Team**
- **Scenario**: 5-person startup, $500/mo burn
- **Solution**: Pool 3 workstations + 2 Macs
- **Outcome**: "$6,000+ saved per year. GDPR-friendly by design."
- **Target**: Teams with budget pressure, compliance needs

**Persona 3: Homelab Enthusiast**
- **Scenario**: Four GPUs gathering dust
- **Solution**: Spread workers across LAN, build agents
- **Outcome**: "Idle GPUs → productive. Auto-download models, clean shutdowns."
- **Target**: Hobbyists with hardware, want to experiment

**Persona 4: Enterprise**
- **Scenario**: 50-dev org, code cannot leave premises
- **Solution**: On-prem rbee, audit trails, policy routing
- **Outcome**: "EU-only compliance. Zero external dependencies."
- **Target**: Organizations with strict compliance requirements

**Conversion Strategy:**
- Four personas ensure most visitors see themselves
- Quantified outcomes build credibility (not vague promises)
- Scenario → Solution → Outcome structure is easy to scan
- Highlighted outcomes draw attention to results

**Tone**: Relatable, specific, outcome-focused`,
			},
		},
	},
}

export const SoloDeveloperOnly: Story = {
	args: {
		title: 'For developers who ship',
		subtitle: 'Build AI features without burning cash on APIs.',
		items: [
			{
				icon: Laptop,
				title: 'The solo developer',
				scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
				solution:
					'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
				outcome: '$0/month AI costs. Full control. No rate limits.',
			},
		],
		columns: 2,
	},
	parameters: {
		docs: {
			description: {
				story: `Single persona deep dive. This variant:
- Focuses on solo developer (most common persona)
- Removes other personas to reduce cognitive load
- Uses 2-column layout for larger card
- Headline targets "developers who ship" (action-oriented)

**Use case**: Landing page targeting solo developers, or A/B test against multi-persona version.`,
			},
		},
	},
}

export const AlternativePersonas: Story = {
	args: {
		title: 'Who uses rbee?',
		subtitle: 'From indie hackers to Fortune 500s.',
		items: [
			{
				icon: Laptop,
				title: 'Indie hacker',
				scenario: 'Building AI-powered SaaS on nights and weekends.',
				solution: 'Use your gaming rig. No cloud bills, no rate limits, no vendor lock-in.',
				outcome: 'Ship faster. Keep 100% of revenue.',
			},
			{
				icon: Users,
				title: 'Agency',
				scenario: 'Building AI features for clients. API costs eat margins.',
				solution: 'Pool agency hardware. Bill clients for AI features, keep costs at zero.',
				outcome: '80% margin improvement on AI projects.',
			},
			{
				icon: Building,
				title: 'Regulated industry',
				scenario: 'Healthcare/finance. Data cannot leave premises.',
				solution: 'On-prem rbee. Audit trails. Policy routing by data classification.',
				outcome: 'HIPAA/SOC2 compliant. Zero external dependencies.',
			},
		],
	},
	parameters: {
		docs: {
			description: {
				story: `Alternative persona set for A/B testing. This variant:
- **Indie hacker**: Emphasizes speed and revenue retention
- **Agency**: Highlights margin improvement (B2B angle)
- **Regulated industry**: Focuses on compliance (healthcare/finance)

**Use case**: Test with different audience segments. Agency persona may resonate with B2B visitors.`,
			},
		},
	},
}
