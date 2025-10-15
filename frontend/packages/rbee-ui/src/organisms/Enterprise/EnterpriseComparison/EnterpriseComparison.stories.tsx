import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseComparison } from './EnterpriseComparison'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseComparison',
	component: EnterpriseComparison,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseComparison section presents a feature matrix comparing rbee to external AI providers (OpenAI, Anthropic, Azure OpenAI, AWS Bedrock) across compliance and security dimensions. Uses a responsive table (desktop) and card switcher (mobile).

## Composition
This organism contains:
- **Header**: "Why Enterprises Choose rbee" with subtitle
- **Legend**: Visual key for feature availability (Full Support, Partial Support, Not Available)
- **Desktop Table**: Full comparison matrix with all providers and features
- **Mobile Card Switcher**: Provider selector with single-provider card view
- **Comparison Data**: Features grouped by compliance, security, and control categories
- **Footnote**: Disclaimer about public information accuracy

## When to Use
- On the /enterprise page after the use cases section
- To demonstrate competitive advantages
- To show compliance and security superiority
- To address "why not use cloud APIs?" objection

## Content Requirements
- **Providers**: rbee, OpenAI, Anthropic, Azure OpenAI, AWS Bedrock
- **Features**: Compliance (GDPR, SOC2, ISO 27001, audit trails), Security (zero-trust, encryption, isolation), Control (data residency, no vendor lock-in)
- **Legend**: Clear visual indicators for feature availability
- **Disclaimer**: Accuracy disclaimer for public information

## Variants
- **Default**: Full comparison matrix with all providers
- **vs. Azure OpenAI**: Direct comparison with Azure OpenAI
- **vs. AWS Bedrock**: Direct comparison with AWS Bedrock

## Examples
\`\`\`tsx
import { EnterpriseComparison } from '@rbee/ui/organisms/Enterprise/EnterpriseComparison'

// Simple usage - no props needed
<EnterpriseComparison />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- MatrixTable
- MatrixCard
- Legend

## Accessibility
- **Keyboard Navigation**: Table and card switcher are keyboard accessible
- **ARIA Labels**: Provider buttons have aria-pressed state
- **Semantic HTML**: Uses <section> with aria-labelledby, <table> with <caption>
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards
- **Screen Reader**: Skip link for screen readers to jump to desktop table

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, Compliance Officer, Procurement Manager
- **Pain Points**: Vendor evaluation, compliance verification, feature comparison, cost comparison
- **Decision Criteria**: Compliance guarantees, security features, data residency, vendor lock-in risk

### Competitive Positioning

**rbee vs. Cloud APIs (OpenAI, Anthropic)**
- **Data Residency**: rbee = EU-only (guaranteed), Cloud APIs = US-based (GDPR Article 44 violation)
- **Audit Trails**: rbee = Immutable 7-year retention, Cloud APIs = Limited or none
- **Vendor Lock-in**: rbee = Open-source foundation (no lock-in), Cloud APIs = Proprietary (lock-in)
- **Cost Predictability**: rbee = Fixed infrastructure cost, Cloud APIs = Variable per-token cost (unpredictable)

**rbee vs. Azure OpenAI**
- **True EU Residency**: rbee = EU-only (no US parent company access), Azure = "EU region" but Microsoft is US company (Schrems II risk)
- **Audit Trails**: rbee = 32 event types with 7-year retention, Azure = Limited audit logs
- **Vendor Lock-in**: rbee = Open-source foundation, Azure = Microsoft ecosystem lock-in
- **Compliance**: rbee = GDPR Article 44 compliant (no US dependencies), Azure = GDPR Article 44 risk (US parent company)

**rbee vs. AWS Bedrock**
- **True EU Residency**: rbee = EU-only (no US parent company access), Bedrock = "EU region" but AWS is US company (Schrems II risk)
- **Audit Trails**: rbee = Immutable audit trail with tamper detection, Bedrock = CloudTrail (limited)
- **Vendor Lock-in**: rbee = Open-source foundation, Bedrock = AWS ecosystem lock-in
- **Compliance**: rbee = GDPR Article 44 compliant (no US dependencies), Bedrock = GDPR Article 44 risk (US parent company)

### Feature Matrix Categories

**Compliance Features**
- EU Data Residency (GDPR Article 44)
- Immutable Audit Trails (GDPR Article 30)
- 7-Year Retention (GDPR requirement)
- SOC2 Type II Ready
- ISO 27001 Aligned
- HIPAA-Aligned Architecture

**Security Features**
- Zero-Trust Authentication
- Tamper-Evident Hash Chains
- Memory Zeroization (secrets)
- Input Validation (injection prevention)
- Deadline Propagation (SLO protection)
- 32 Audit Event Types

**Control Features**
- Self-Hosted (full control)
- No Vendor Lock-in (open-source)
- No US Cloud Dependencies
- Programmable Policies (Rhai)
- OpenAI-Compatible API
- Unlimited GPUs

### Conversion Strategy
- **Primary CTA**: "Download Comparison Guide" (detailed PDF comparison)
- **Secondary CTA**: "Request Demo" (sales-assisted evaluation)
- **Lead qualification**: Capture current provider, compliance requirements, migration timeline
- **Proof points**: Feature matrix, compliance guarantees, customer testimonials

### Disclaimer
- **Accuracy**: "Based on publicly available information as of October 2025"
- **Verification**: "Verify requirements with your legal team"
- **No Liability**: Comparison is for informational purposes only
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseComparison>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise comparison section showing full feature matrix comparing rbee to OpenAI, Anthropic, Azure OpenAI, and AWS Bedrock. Features are grouped by compliance (GDPR, SOC2, ISO 27001, audit trails), security (zero-trust, encryption, isolation), and control (data residency, no vendor lock-in). Desktop shows full table, mobile shows provider switcher with single-provider card view.',
			},
		},
	},
}

export const VsAzureOpenAI: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant focusing on direct comparison with Azure OpenAI. This version would emphasize key differentiators: True EU Residency (rbee = EU-only with no US parent company access vs. Azure = "EU region" but Microsoft is US company with Schrems II risk), Immutable Audit Trails (rbee = 32 event types with 7-year retention vs. Azure = limited audit logs), No Vendor Lock-in (rbee = open-source foundation vs. Azure = Microsoft ecosystem lock-in), and GDPR Article 44 Compliance (rbee = no US dependencies vs. Azure = US parent company risk). Ideal for enterprises currently using or evaluating Azure OpenAI.',
			},
		},
	},
}

export const VsAWSBedrock: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant focusing on direct comparison with AWS Bedrock. This version would emphasize key differentiators: True EU Residency (rbee = EU-only with no US parent company access vs. Bedrock = "EU region" but AWS is US company with Schrems II risk), Immutable Audit Trails (rbee = tamper-evident hash chains with 7-year retention vs. Bedrock = CloudTrail with limited retention), No Vendor Lock-in (rbee = open-source foundation vs. Bedrock = AWS ecosystem lock-in), and GDPR Article 44 Compliance (rbee = no US dependencies vs. Bedrock = US parent company risk). Ideal for enterprises currently using or evaluating AWS Bedrock.',
			},
		},
	},
}
