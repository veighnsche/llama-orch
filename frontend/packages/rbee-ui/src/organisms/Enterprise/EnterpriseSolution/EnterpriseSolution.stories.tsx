import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseSolution } from './EnterpriseSolution'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseSolution',
	component: EnterpriseSolution,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseSolution section presents the rbee solution to enterprise compliance challenges. It uses a wrapper around SolutionSection with enterprise-specific features, deployment steps, and compliance metrics. Emphasizes EU data sovereignty, audit retention, and zero US cloud dependencies.

## Composition
This organism is a wrapper around SolutionSection with enterprise-specific defaults:
- **Kicker**: "How rbee Works"
- **Title**: "EU-Native AI Infrastructure That Meets Compliance by Design"
- **Features (4)**: 100% Data Sovereignty, 7-Year Audit Retention, 32 Audit Event Types, Zero US Cloud Dependencies
- **Steps (4)**: Deploy On-Premises, Configure Compliance Policies, Enable Audit Logging, Run Compliant AI
- **Compliance Metrics Table**: Data Sovereignty (100% EU-only), Audit Retention (7 years immutable), Security Layers (5 layers zero-trust)
- **CTAs**: "Request Demo" (primary), "View Compliance Docs" (secondary)

## When to Use
- On the /enterprise page after the problem section
- To present the solution to compliance challenges
- To demonstrate deployment workflow
- To show quantifiable compliance metrics

## Content Requirements
- **Features**: Enterprise-specific benefits (compliance, audit, sovereignty)
- **Steps**: Deployment workflow (on-prem, policies, audit, run)
- **Metrics**: Quantifiable compliance metrics with regulatory references
- **CTAs**: Enterprise-appropriate actions (demo, docs)

## Variants
- **Default**: Full compliance solution with all features
- **Security First**: Lead with security and audit capabilities
- **ROI First**: Lead with cost savings and compliance risk reduction

## Examples
\`\`\`tsx
import { EnterpriseSolution } from '@rbee/ui/organisms/Enterprise/EnterpriseSolution'

// Simple usage - no props needed
<EnterpriseSolution />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- SolutionSection (parent component)
- Card
- Badge

## Accessibility
- **Keyboard Navigation**: All CTAs are keyboard accessible
- **Semantic HTML**: Proper heading hierarchy
- **ARIA Labels**: Icons marked as aria-hidden
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, VP Engineering, Compliance Officer, DPO
- **Decision Criteria**: Compliance guarantees, deployment complexity, time-to-value, support quality

### Enterprise Messaging
- **Primary value prop**: Compliance by design (not bolted on)
- **Proof points**: Specific regulations (GDPR Article 44, Article 30), quantifiable metrics (7 years, 32 event types)
- **Differentiation**: EU-native (no US cloud dependencies), self-hosted (full control), immutable audit trails

### Solution Positioning
- **vs. Cloud APIs**: On-prem = full control, EU residency, no Schrems II risk
- **vs. DIY**: Compliance features out-of-box, professional support, faster time-to-value
- **vs. Azure/AWS**: True EU residency (not just "EU region"), no vendor lock-in, open-source foundation

### Deployment Workflow
1. **Deploy On-Premises**: Install on EU infrastructure, air-gap support
2. **Configure Compliance Policies**: Rhai-based policies for data residency, audit retention, access controls
3. **Enable Audit Logging**: Immutable audit trail with 32 event types
4. **Run Compliant AI**: Your models, your data, your infrastructure

### Compliance Metrics
- **Data Sovereignty**: 100% EU-only (GDPR Article 44)
- **Audit Retention**: 7 years immutable (GDPR Article 30)
- **Security Layers**: 5 layers zero-trust (defense-in-depth)

### Conversion Strategy
- **Primary CTA**: "Request Demo" (sales-assisted evaluation)
- **Secondary CTA**: "View Compliance Docs" (self-serve education)
- **Lead qualification**: Capture compliance requirements, deployment timeline, infrastructure details
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseSolution>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise solution section showing the full compliance workflow: Deploy On-Premises, Configure Compliance Policies, Enable Audit Logging, Run Compliant AI. Includes compliance metrics table with Data Sovereignty (100% EU-only), Audit Retention (7 years immutable), and Security Layers (5 layers zero-trust).',
			},
		},
	},
}

export const SecurityFirst: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant leading with security and audit capabilities. This version would emphasize: Immutable audit trails (7-year retention, tamper-evident), 32 audit event types (auth, data access, policy changes), zero-trust architecture (5 security layers), and defense-in-depth approach. Ideal for security-conscious enterprises (financial services, healthcare, government).',
			},
		},
	},
}

export const ROIFirst: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing cost savings and ROI. This version would lead with: Avoid GDPR fines (€20M or 4% of revenue), eliminate unpredictable cloud API costs, reduce vendor dependency risk, and faster compliance certification. Includes cost comparison vs. cloud APIs showing 60-80% cost savings at scale. Ideal for CFOs or cost-conscious enterprises.',
			},
		},
	},
}
