import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseProblem } from './EnterpriseProblem'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseProblem',
	component: EnterpriseProblem,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseProblem section articulates enterprise-specific pain points related to compliance, audit trails, regulatory fines, and lack of control when using external AI providers. It uses a 4-column grid with destructive-tone cards to emphasize the severity of compliance risks.

## Composition
This organism is a wrapper around ProblemSection with enterprise-specific defaults:
- **Kicker**: "The Compliance Risk"
- **Title**: "The Compliance Challenge of Cloud AI"
- **Subtitle**: Compliance risk messaging
- **Problem Cards (4)**: Data Sovereignty Violations, Missing Audit Trails, Regulatory Fines, Zero Control
- **CTAs**: "Request Demo" (primary), "Compliance Overview" (secondary)
- **Quote**: CTO/DPO testimonial about GDPR compliance requirements

## When to Use
- On the /enterprise page after the hero section
- To establish the severity of compliance risks
- To create urgency around regulatory requirements
- To differentiate from consumer/developer pain points

## Content Requirements
- **Problem Cards**: Enterprise-specific issues (compliance, audit, fines, control)
- **Tone**: Destructive (red/warning) to emphasize risk severity
- **Tags**: Specific regulations (GDPR Art. 44, HIPAA, PCI-DSS)
- **Quantifiable Risks**: Specific fine amounts (€20M, $50K per record)
- **Quote**: From enterprise decision-maker (CTO, DPO)

## Variants
- **Default**: All four compliance problems
- **Compliance Focus**: Emphasize GDPR and data sovereignty
- **Cost Focus**: Lead with regulatory fines and financial risk
- **Control Focus**: Emphasize lack of control and vendor dependency

## Examples
\`\`\`tsx
import { EnterpriseProblem } from '@rbee/ui/organisms/Enterprise/EnterpriseProblem'

// Simple usage - no props needed
<EnterpriseProblem />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- ProblemSection (parent component)
- Card
- Badge

## Accessibility
- **Keyboard Navigation**: All CTAs are keyboard accessible
- **Semantic HTML**: Proper heading hierarchy
- **Color Contrast**: Destructive tone meets WCAG AA standards
- **Screen Readers**: Problem cards have descriptive content

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, Compliance Officer, Data Protection Officer (DPO), Legal Counsel
- **Pain Points**: Regulatory compliance, audit requirements, data sovereignty, vendor risk
- **Fear Factor**: GDPR fines (€20M or 4% of revenue), HIPAA violations ($50K per record), reputation damage

### Enterprise Messaging
- **Primary concern**: Compliance violations and regulatory fines
- **Proof points needed**: Specific regulations (GDPR Article 44, Schrems II, HIPAA, PCI-DSS)
- **Objections to address**: "Cloud providers have compliance certifications" (but data still crosses borders)
- **Tone**: Serious, risk-focused, regulatory-aware (not fear-mongering, but realistic)

### Problem Differentiation

**vs. Home Page ProblemSection:**
- Home: General cost/privacy concerns ("$200/month API bills", "data sent to US")
- Enterprise: Specific compliance violations ("GDPR Article 44 violations", "€20M fines")

**vs. Developers ProblemSection:**
- Developers: Workflow issues ("slow iteration", "API rate limits", "no local testing")
- Enterprise: Governance issues ("no audit trails", "cannot prove compliance", "DPO cannot sleep")

### Conversion Strategy
- **Primary CTA**: "Request Demo" (not "Get Started" - enterprise needs compliance walkthrough)
- **Secondary CTA**: "Compliance Overview" (link to detailed compliance documentation)
- **Lead qualification**: Capture industry (healthcare, finance, government), compliance requirements, timeline
- **Sales-assisted**: Yes - all enterprise leads need custom compliance demos

### Key Pain Points Addressed
1. **Data Sovereignty Violations**: GDPR Article 44, Schrems II, cross-border data transfers
2. **Missing Audit Trails**: GDPR Article 30, SOC2, ISO 27001 requirements
3. **Regulatory Fines**: Specific amounts (€20M GDPR, $50K/record HIPAA)
4. **Zero Control**: Vendor changes terms, DPAs worthless, no residency guarantees
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseProblem>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise problem section showing all four compliance risks: Data Sovereignty Violations (GDPR Article 44), Missing Audit Trails (GDPR Article 30, SOC2), Regulatory Fines (€20M GDPR, $50K/record HIPAA), and Zero Control (vendor dependency). Uses destructive tone (red/warning) to emphasize severity.',
			},
		},
	},
}

export const ComplianceFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing compliance and data sovereignty issues. This version would prioritize: GDPR Article 44 violations (cross-border data transfers), Schrems II non-compliance, Data Protection Authority scrutiny, and missing audit trails. Ideal for EU-based enterprises or those in highly regulated industries (healthcare, finance, government).',
			},
		},
	},
}

export const CostFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant leading with financial risk and regulatory fines. This version would emphasize: GDPR fines (€20M or 4% of global revenue), HIPAA violations ($50K per record), PCI-DSS breaches (reputation damage), and unpredictable cloud API costs. Includes real-world examples of companies fined for GDPR violations. Ideal for CFOs, finance teams, or risk-averse enterprises.',
			},
		},
	},
}

export const ControlFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing lack of control and vendor dependency. This version would lead with: Provider changes terms unilaterally, Data Processing Agreements (DPAs) provide no real guarantees, cannot guarantee data residency, cannot prove compliance to auditors, and DPO/legal counsel liability. Ideal for enterprises burned by vendor lock-in or those with strong governance requirements.',
			},
		},
	},
}
