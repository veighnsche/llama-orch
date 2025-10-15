import type { Meta, StoryObj } from '@storybook/react'
import { TechnicalSection } from './TechnicalSection'

const meta = {
	title: 'Organisms/TechnicalSection',
	component: TechnicalSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The TechnicalSection showcases the engineering principles and technology stack behind rbee. Uses a two-column layout with architecture highlights on the left and technology stack on the right, plus an optional architecture diagram.

## Marketing Strategy

### Target Audience
Technical decision-makers evaluating engineering quality. They need:
- Proof of technical rigor (not just marketing)
- Understanding of architecture principles
- Confidence in technology choices
- Evidence of production-readiness

### Primary Message
**"Built by Engineers, for Engineers"** — Peer-to-peer credibility positioning.

### Copy Analysis
- **Headline tone**: Peer-to-peer, technical credibility
- **Emotional appeal**: Professional respect, engineering quality
- **Power words**: "BDD-Driven", "Cascading Shutdown", "Process Isolation", "Rust-native"
- **Social proof**: "42/62 scenarios passing (68% complete)" — Transparency about progress

### Conversion Elements
- **Five architecture principles**: BDD, Cascading Shutdown, Process Isolation, Protocol-Aware, Smart/Dumb Separation
- **BDD coverage progress bar**: Visual proof of testing rigor (68% complete)
- **Five technology choices**: Rust, Candle ML, Rhai Scripting, SQLite, Axum + Vue.js
- **Architecture diagram**: Visual showing orchestrator, policy engine, worker pools
- **Open source CTA**: GitHub link with "100% Open Source" badge
- **Architecture docs link**: "Read Architecture →"

### Architecture Principles
1. **BDD-Driven Development**: 42/62 scenarios passing (68% complete), Live CI coverage
2. **Cascading Shutdown Guarantee**: No orphaned processes, Clean VRAM lifecycle
3. **Process Isolation**: Worker-level sandboxes, Zero cross-leak
4. **Protocol-Aware Orchestration**: SSE, JSON, binary protocols
5. **Smart/Dumb Separation**: Central brain, distributed execution

### Technology Stack
1. **Rust**: Performance + memory safety
2. **Candle ML**: Rust-native inference
3. **Rhai Scripting**: Embedded, sandboxed policies
4. **SQLite**: Embedded, zero-ops DB
5. **Axum + Vue.js**: Async backend + modern UI

### Objection Handling
- **"Is it production-ready?"** → BDD coverage (68% complete), live CI
- **"Will it leak memory?"** → Cascading shutdown guarantee, clean VRAM lifecycle
- **"Can I trust the architecture?"** → Five principles, architecture diagram
- **"What if I need to customize?"** → Rhai scripting, open source

### Variations to Test
- Alternative headline: "Engineering Excellence. Production Ready." vs. "Built by Engineers, for Engineers"
- Alternative principle order: Lead with most compelling (Cascading Shutdown vs. BDD)
- Alternative tech stack: Emphasize Rust vs. Candle ML vs. Rhai

## Composition
This organism contains:
- **SectionContainer**: Wrapper with title and description
- **Two-Column Layout**: Architecture highlights (left) + Technology stack (right)
- **Architecture Highlights**: Five principles with descriptions
- **BDD Coverage Progress Bar**: Visual showing 68% complete
- **Architecture Diagram**: SVG showing system architecture
- **Technology Stack**: Five tech cards with descriptions
- **Open Source CTA Card**: GitHub link with badge
- **Architecture Docs Link**: Text link to docs

## When to Use
- Home page: After social proof section
- About page: To showcase engineering
- Documentation: Technical overview
- Careers page: To attract engineers

## Content Requirements
- **Title**: Clear engineering positioning
- **Description**: Technical context
- **Architecture principles**: 5 key principles
- **Technology stack**: 5 technology choices
- **BDD coverage**: Current test coverage percentage
- **Architecture diagram**: Visual representation

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<TechnicalSection />
\`\`\`

**Context**: Appears after SocialProofSection, before FAQSection  
**Purpose**: Build technical credibility, attract engineers  
**Metrics**: Engagement (do visitors click GitHub or Architecture links?)

## Examples
\`\`\`tsx
import { TechnicalSection } from '@rbee/ui/organisms/TechnicalSection'

// Default usage
<TechnicalSection />
\`\`\`

## Related Components
- SectionContainer
- Button
- Card
- RbeeArch (architecture diagram)

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Proper heading hierarchy, lists for principles
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Progress Bar**: Includes text label for screen readers
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof TechnicalSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Built by Engineers, for Engineers" — Peer credibility
- **Subheadline**: "Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."
- **Five architecture principles**: BDD, Cascading Shutdown, Process Isolation, Protocol-Aware, Smart/Dumb
- **BDD coverage**: 42/62 scenarios (68%) — Transparency about progress
- **Five technology choices**: Rust, Candle ML, Rhai, SQLite, Axum + Vue.js

**Left Column: Architecture Highlights**

**Principle 1: BDD-Driven Development**
- 42/62 scenarios passing (68% complete)
- Live CI coverage
- **Message**: Test-driven, rigorous development

**Principle 2: Cascading Shutdown Guarantee**
- No orphaned processes
- Clean VRAM lifecycle
- **Message**: Operational reliability

**Principle 3: Process Isolation**
- Worker-level sandboxes
- Zero cross-leak
- **Message**: Security and stability

**Principle 4: Protocol-Aware Orchestration**
- SSE, JSON, binary protocols
- **Message**: Flexible integration

**Principle 5: Smart/Dumb Separation**
- Central brain, distributed execution
- **Message**: Scalable architecture

**BDD Coverage Progress Bar:**
- Visual: 68% filled green bar
- Text: "42/62 scenarios passing"
- **Message**: Transparency about development progress

**Right Column: Technology Stack**

**Tech 1: Rust**
- Performance + memory safety
- **Message**: Production-grade foundation

**Tech 2: Candle ML**
- Rust-native inference
- **Message**: No Python overhead

**Tech 3: Rhai Scripting**
- Embedded, sandboxed policies
- **Message**: Customizable without recompilation

**Tech 4: SQLite**
- Embedded, zero-ops DB
- **Message**: No database administration

**Tech 5: Axum + Vue.js**
- Async backend + modern UI
- **Message**: Modern web stack

**Open Source CTA:**
- Badge: "100% Open Source"
- License: "MIT License"
- Button: "View Source" → GitHub
- **Message**: Full transparency, community-driven

**Conversion Strategy:**
- Architecture principles build technical credibility
- BDD coverage shows transparency (not hiding issues)
- Technology stack demonstrates thoughtful choices
- Open source CTA invites contribution
- Architecture diagram makes abstract concrete

**Tone**: Technical, transparent, peer-to-peer`,
			},
		},
	},
}

export const SimplifiedTech: Story = {
	render: () => (
		<div>
			<TechnicalSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Simplified Technical Messaging</h3>
				<div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
					<p>
						<strong>Current:</strong> "Rust-native orchestrator with process isolation, protocol awareness, and policy
						routing via Rhai."
					</p>
					<p>
						<strong>Alt 1:</strong> "Built with Rust for speed and reliability. No Python overhead." (Simpler, focuses
						on Rust)
					</p>
					<p>
						<strong>Alt 2:</strong> "Production-grade orchestration. Test-driven development. Clean shutdowns."
						(Focuses on outcomes)
					</p>
					<p>
						<strong>Alt 3:</strong> "Enterprise architecture. Homelab simplicity." (Contrast positioning)
					</p>
					<p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
						<strong>Recommendation:</strong> Current messaging works for technical audience. Use Alt 2 for less
						technical visitors.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Alternative technical messaging for different audience segments.',
			},
		},
	},
}

export const DeepDive: Story = {
	render: () => (
		<div>
			<TechnicalSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
					Technical Deep Dive Topics
				</h3>
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
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>BDD Testing</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Cucumber-based scenarios. 42/62 passing. Live CI. Spec-driven development.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Cascading Shutdown</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							SIGTERM propagation. VRAM cleanup. No orphaned processes. Ctrl+C just works.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Process Isolation</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Worker-level sandboxes. Memory boundaries. Zero cross-contamination.
						</p>
					</div>
					<div>
						<h4 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Rhai Policies</h4>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Embedded scripting. Route by model, region, cost. No recompilation needed.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Deep dive into technical topics. Useful for documentation or technical blog posts.',
			},
		},
	},
}
