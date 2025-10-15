import type { Meta, StoryObj } from '@storybook/react'
import { SocialProofSection } from './SocialProofSection'

const meta = {
	title: 'Organisms/Home/SocialProofSection',
	component: SocialProofSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The SocialProofSection displays testimonials from users in a card grid format, along with key statistics showing adoption and impact. Uses persona-based testimonials (solo dev, CTO, DevOps) to build credibility across audience segments.

## Marketing Strategy

### Target Audience
Visitors evaluating trust and credibility. They need:
- Proof that real people use rbee
- Relatable testimonials (not generic)
- Quantified adoption metrics
- Confidence that rbee is production-ready

### Primary Message
**"Trusted by developers who value independence"** — Values-driven social proof.

### Copy Analysis
- **Headline tone**: Trust-building, values-driven
- **Emotional appeal**: Belonging (join others), validation (smart choice)
- **Power words**: "Trusted", "independence", "$0", "GDPR-friendly", "clean shutdowns"
- **Social proof**: Testimonials + stats (1,200+ stars, 500+ installations, 8,000+ GPUs, €0 cost)

### Conversion Elements
- **Three testimonials**: Solo developer, CTO, DevOps (cover main personas)
- **Four stats**: GitHub stars, Active installations, GPUs orchestrated, Avg. monthly cost
- **Emoji avatars**: Humanize testimonials without real photos
- **Persona labels**: Role context (Solo Developer, CTO, DevOps)
- **Specific outcomes**: "$80/mo → $0", "$500/mo → zero", "Ctrl+C and everything cleans up"

### Testimonial Strategy
- **Testimonial 1 (Alex K., Solo Developer)**: Cost savings angle
  - "Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost."
  - **Target**: Cost-conscious solo developers
  
- **Testimonial 2 (Sarah M., CTO)**: Team/collaboration angle
  - "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes."
  - **Target**: Small teams, startups with budget pressure
  
- **Testimonial 3 (Marcus T., DevOps)**: Technical reliability angle
  - "Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up."
  - **Target**: DevOps engineers, infrastructure-focused

### Stats Strategy
- **1,200+ GitHub stars**: Community validation
- **500+ Active installations**: Adoption proof
- **8,000+ GPUs orchestrated**: Scale demonstration
- **€0 Avg. monthly cost**: Value reinforcement (highlighted in primary color)

### Objection Handling
- **"Is anyone actually using this?"** → 500+ active installations, 8,000+ GPUs
- **"Will it save me money?"** → "$80/mo → $0", "$500/mo → zero"
- **"Is it reliable?"** → DevOps testimonial about clean shutdowns
- **"Will it work for my team?"** → CTO testimonial about team usage

### Variations to Test
- Alternative testimonial order: Lead with CTO (authority) vs. solo dev (relatability)
- Alternative stats: Emphasize community (stars) vs. adoption (installations) vs. scale (GPUs)
- Alternative personas: Add enterprise testimonial, remove homelab

## Composition
This organism contains:
- **Header**: Title and optional subtitle
- **Optional Logo Strip**: Company logos (if testimonials have companyLogoSrc)
- **Testimonials Grid**: 3-column responsive grid (2 cols tablet, 1 col mobile)
- **Testimonial Cards**: Quote, author, role, optional avatar/logo
- **Stats Row**: 4-column grid with key metrics
- **Staggered Animations**: Cards and stats animate in with delays

## When to Use
- Home page: After pricing section
- About page: To build credibility
- Landing pages: To reduce skepticism
- Case studies page: As introduction

## Content Requirements
- **Title**: Clear trust statement
- **Subtitle**: Optional context
- **Testimonials**: 3-6 quotes with author, role, optional avatar
- **Stats**: 3-4 key metrics
- **Avatars**: Emoji, initials, or image URLs

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<SocialProofSection />
\`\`\`

**Context**: Appears after PricingSection, before TechnicalSection  
**Purpose**: Build trust through testimonials and stats  
**Metrics**: Engagement (do visitors read testimonials?)

## Examples
\`\`\`tsx
import { SocialProofSection } from '@rbee/ui/organisms/SocialProofSection'

// Default usage (uses built-in testimonials and stats)
<SocialProofSection />

// Custom testimonials
import { TestimonialsSection } from '@rbee/ui/organisms/SocialProofSection'

<TestimonialsSection
  title="Trusted by developers who value independence"
  testimonials={[...]}
  stats={[...]}
/>
\`\`\`

## Related Components
- None (self-contained)

## Accessibility
- **Keyboard Navigation**: Cards are focusable with tab key
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Uses article elements for testimonials, blockquote for quotes
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Avatar Alt Text**: Descriptive alt text for image avatars
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof SocialProofSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Trusted by developers who value independence" — Values-driven trust
- **Three testimonials**: Cover main personas (solo dev, CTO, DevOps)
- **Four stats**: GitHub stars, installations, GPUs, cost

**Testimonial 1: Alex K., Solo Developer**
- **Avatar**: 👨‍💻 (emoji)
- **Quote**: "Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost."
- **Target**: Cost-conscious solo developers
- **Key message**: Cost savings without quality loss
- **Outcome**: $80/mo → $0

**Testimonial 2: Sarah M., CTO**
- **Avatar**: 👩‍💼 (emoji)
- **Quote**: "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes."
- **Target**: Small teams, startups
- **Key message**: Team collaboration, no code changes
- **Outcome**: $500/mo → $0

**Testimonial 3: Marcus T., DevOps**
- **Avatar**: 👨‍🔧 (emoji)
- **Quote**: "Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up."
- **Target**: DevOps engineers, infrastructure-focused
- **Key message**: Technical reliability, clean operations
- **Outcome**: Operational excellence

**Stats:**
1. **1,200+ GitHub stars**: Community validation
2. **500+ Active installations**: Adoption proof
3. **8,000+ GPUs orchestrated**: Scale demonstration
4. **€0 Avg. monthly cost**: Value reinforcement (primary color)

**Conversion Strategy:**
- Three personas ensure most visitors see themselves
- Specific outcomes build credibility ($80 → $0, not "saves money")
- Stats quantify adoption and impact
- Emoji avatars humanize without requiring real photos

**Tone**: Authentic, specific, outcome-focused`,
			},
		},
	},
}

export const WithoutLogos: Story = {
	render: () => (
		<div>
			<SocialProofSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Testimonial Variations</h3>
				<div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
					<p>
						<strong>Current:</strong> Emoji avatars (👨‍💻, 👩‍💼, 👨‍🔧)
					</p>
					<p>
						<strong>Alt 1:</strong> Real photos (requires permission, higher trust)
					</p>
					<p>
						<strong>Alt 2:</strong> Initials (AK, SM, MT) (more professional)
					</p>
					<p>
						<strong>Alt 3:</strong> No avatars (cleaner, focus on quote)
					</p>
					<p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
						<strong>Recommendation:</strong> Emoji avatars work well for developer audience. Consider real photos if
						you can get permission from actual users.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Current implementation uses emoji avatars. No company logos in default version.',
			},
		},
	},
}

export const MetricsOnly: Story = {
	render: () => (
		<div>
			<div style={{ padding: '4rem 2rem', textAlign: 'center' }}>
				<h2 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '1rem' }}>
					Trusted by developers who value independence
				</h2>
				<div
					style={{
						maxWidth: '800px',
						margin: '2rem auto',
						display: 'grid',
						gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
						gap: '2rem',
					}}
				>
					<div>
						<div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--primary)' }}>1,200+</div>
						<div style={{ fontSize: '0.875rem', color: '#666' }}>GitHub stars</div>
					</div>
					<div>
						<div style={{ fontSize: '2.5rem', fontWeight: 'bold' }}>500+</div>
						<div style={{ fontSize: '0.875rem', color: '#666' }}>Active installations</div>
					</div>
					<div>
						<div style={{ fontSize: '2.5rem', fontWeight: 'bold' }}>8,000+</div>
						<div style={{ fontSize: '0.875rem', color: '#666' }}>GPUs orchestrated</div>
					</div>
					<div>
						<div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--primary)' }}>€0</div>
						<div style={{ fontSize: '0.875rem', color: '#666' }}>Avg. monthly cost</div>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Stats-only variant without testimonials. Useful for tighter layouts or when testimonials are shown elsewhere.',
			},
		},
	},
}
