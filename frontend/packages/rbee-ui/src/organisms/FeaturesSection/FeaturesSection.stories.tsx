import type { Meta, StoryObj } from '@storybook/react'
import { FeaturesSection } from './FeaturesSection'

const meta = {
	title: 'Organisms/FeaturesSection',
	component: FeaturesSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The FeaturesSection uses a tabbed interface to showcase four key technical capabilities: OpenAI-Compatible API, Multi-GPU Orchestration, Programmable Rhai Scheduler, and Real-time SSE streaming. Each tab includes code examples, visual demonstrations, and benefit callouts.

## Marketing Strategy

### Target Audience
Technical decision-makers evaluating capabilities. They need:
- Proof of technical depth (not just marketing claims)
- Real code examples (not pseudocode)
- Visual demonstrations (GPU utilization, SSE events)
- Understanding of differentiation (vs. Ollama, cloud APIs)

### Primary Message
**"Enterprise-Grade Features. Homelab Simplicity."** — Positioning rbee as powerful yet accessible.

### Copy Analysis
- **Headline tone**: Contrast positioning (enterprise vs. homelab)
- **Emotional appeal**: Technical capability without complexity
- **Power words**: "OpenAI-Compatible", "Multi-GPU", "Programmable", "Real-time"
- **Social proof**: Code examples show real implementation

### Conversion Elements
- **Four tabs**: Segment features by use case (API, GPU, Scheduler, SSE)
- **Code examples**: Real bash/Rust/JSON showing actual usage
- **Visual demos**: GPU utilization bars, SSE event stream
- **Benefit callouts**: Summarize key advantage of each feature
- **Feature badges**: Quick-scan benefits (No API fees, Multi-node, Latency-aware, etc.)

### Objection Handling
- **"Is it compatible with my tools?"** → OpenAI-Compatible API tab
- **"Can it use all my GPUs?"** → Multi-GPU Orchestration tab with utilization bars
- **"Can I customize routing?"** → Programmable Rhai Scheduler tab with policy example
- **"Can I stream results?"** → Real-time SSE tab with event stream example

### Variations to Test
- Alternative tab order: Lead with most compelling feature (GPU vs. API)
- Alternative code examples: Show different languages (Python vs. TypeScript)
- Alternative visual demos: Show different GPU configurations

## Composition
This organism contains:
- **SectionContainer**: Wrapper with title and description
- **Tabs**: Four-tab interface (API, GPU, Scheduler, SSE)
- **TabsList**: Responsive grid (2 cols mobile, 4 cols desktop)
- **TabsContent**: Per-tab content with code blocks, visuals, callouts
- **CodeBlock**: Syntax-highlighted code with copy button
- **BenefitCallout**: Highlighted benefit statement
- **Feature Badges**: Quick-scan benefit pills

## When to Use
- Home page: After how-it-works section
- Features page: As primary content
- Documentation: Technical overview
- Comparison pages: Show differentiation

## Content Requirements
- **Title**: Clear feature positioning
- **Description**: Context for feature set
- **Four tabs**: API, GPU, Scheduler, SSE
- **Per-tab content**: Code example, visual demo, benefit callout
- **Feature badges**: 3 quick-scan benefits per tab

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<FeaturesSection />
\`\`\`

**Context**: Appears after HowItWorksSection, before UseCasesSection  
**Purpose**: Demonstrate technical depth, differentiate from competitors  
**Metrics**: Tab engagement (which features get most attention?)

## Examples
\`\`\`tsx
import { FeaturesSection } from '@rbee/ui/organisms/FeaturesSection'

// Default usage with all tabs
<FeaturesSection />
\`\`\`

## Related Components
- Tabs
- TabsList
- TabsTrigger
- TabsContent
- CodeBlock
- BenefitCallout
- SectionContainer

## Accessibility
- **Keyboard Navigation**: Tab list is keyboard accessible (arrow keys)
- **Focus States**: Visible focus indicators on tabs
- **Semantic HTML**: Uses ARIA tabs pattern
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Live Regions**: Tab content uses aria-live="polite"
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof FeaturesSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Enterprise-Grade Features. Homelab Simplicity." — Contrast positioning
- **Four tabs**: Segment features by technical capability
  1. **OpenAI-Compatible API**: Drop-in replacement, no code changes
  2. **Multi-GPU Orchestration**: Unified pool across CUDA, Metal, CPU
  3. **Programmable Rhai Scheduler**: Custom routing logic
  4. **Real-time SSE**: Stream job lifecycle into UI

**Tab 1: OpenAI-Compatible API**
- **Code example**: Shows before/after (OpenAI → rbee)
- **Benefit badges**: No API fees, Local tokens, Secure by default
- **Benefit callout**: "No code changes. Just point to localhost."
- **Target**: Developers using Zed, Cursor, Continue

**Tab 2: Multi-GPU Orchestration**
- **Visual demo**: Four GPU utilization bars (RTX 4090 #1/2, M2 Ultra, CPU Backend)
- **Benefit badges**: Multi-node, Backend-aware, Auto discovery
- **Benefit callout**: "10× throughput by using all your hardware."
- **Target**: Users with multiple GPUs or machines

**Tab 3: Programmable Rhai Scheduler**
- **Code example**: Rhai policy routing by model size, type, labels
- **Benefit badges**: Latency-aware, Cost caps, Compliance routes
- **Benefit callout**: "Optimize for cost, latency, or compliance—your rules."
- **Target**: Enterprise users with routing requirements

**Tab 4: Real-time SSE**
- **Code example**: JSON event stream (task.created, model.loading, token.generated, task.completed)
- **Benefit badges**: Real-time, Back-pressure safe, Cost visible
- **Benefit callout**: "Full visibility into every inference job."
- **Target**: Developers building UIs with live updates

**Conversion Strategy:**
- Tabs allow self-selection (users explore relevant features)
- Code examples build credibility (real implementation)
- Visual demos make abstract concepts tangible
- Benefit callouts summarize key advantage

**Tone**: Technical, confident, feature-focused`,
			},
		},
	},
}

export const CoreFeatures: Story = {
	render: () => (
		<div>
			<FeaturesSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Feature Prioritization</h3>
				<div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
					<p>
						<strong>Most compelling for solo developers:</strong> OpenAI-Compatible API (no code changes)
					</p>
					<p>
						<strong>Most compelling for multi-GPU users:</strong> Multi-GPU Orchestration (10× throughput)
					</p>
					<p>
						<strong>Most compelling for enterprise:</strong> Programmable Rhai Scheduler (compliance routing)
					</p>
					<p>
						<strong>Most compelling for UI developers:</strong> Real-time SSE (live job updates)
					</p>
					<p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
						<strong>A/B Test Recommendation:</strong> Reorder tabs based on audience segment. Solo devs see API
						first, enterprise sees Scheduler first.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Feature prioritization analysis. Different audience segments value different features.',
			},
		},
	},
}

export const AllFeatures: Story = {
	render: () => (
		<div>
			<FeaturesSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>Feature Highlights</h3>
				<div
					style={{
						maxWidth: '800px',
						margin: '0 auto',
						display: 'grid',
						gap: '1rem',
						gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
					}}
				>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>OpenAI-Compatible API</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Drop-in replacement for OpenAI. Change base URL, keep your code. Works with Zed, Cursor, Continue.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Multi-GPU Orchestration</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Pool CUDA, Metal, and CPU backends. Mixed nodes act as one. 10× throughput by using all hardware.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Programmable Rhai Scheduler</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Route by model size, task type, labels, or compliance rules. Your policy, your trade-offs.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Real-time SSE</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Stream job lifecycle events—model loads, token output, cost—right into your UI.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Overview of all four features with summaries. Useful for quick reference or comparison.',
			},
		},
	},
}
