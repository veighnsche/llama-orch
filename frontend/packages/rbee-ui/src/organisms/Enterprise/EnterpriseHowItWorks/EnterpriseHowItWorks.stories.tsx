import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseHowItWorks } from './EnterpriseHowItWorks'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseHowItWorks',
	component: EnterpriseHowItWorks,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseHowItWorks section presents the enterprise deployment process in four stages: Compliance Assessment, On-Premises Deployment, Compliance Validation, and Production Launch. Includes a sticky timeline panel showing typical deployment duration (7 weeks).

## Composition
This organism contains:
- **Header**: "Enterprise Deployment Process" with subtitle
- **Four Deployment Steps**: Each with icon, title, intro, and checklist
- **Step 1 - Compliance Assessment**: Gap analysis, data flow mapping, risk assessment, architecture proposal
- **Step 2 - On-Premises Deployment**: EU data centers, on-premises, private cloud, hybrid options
- **Step 3 - Compliance Validation**: Documentation package, auditor access, security review, penetration testing
- **Step 4 - Production Launch**: 99.9% uptime SLA, 24/7 support, dedicated account manager, quarterly compliance reviews
- **Sticky Timeline Panel**: Week-by-week breakdown (Week 1-2: Assessment, Week 3-4: Deployment, Week 5-6: Validation, Week 7: Launch)

## When to Use
- On the /enterprise page after the security section
- To demonstrate deployment workflow and timeline
- To set expectations for implementation complexity
- To show enterprise support and SLAs

## Content Requirements
- **Deployment Steps**: Detailed workflow from assessment to production
- **Timeline**: Realistic week-by-week breakdown
- **Support Details**: SLAs, response times, account management
- **Deployment Options**: On-prem, private cloud, hybrid

## Variants
- **Default**: Full 4-step deployment process with 7-week timeline
- **On-Prem Deployment**: Focus on on-premises deployment options
- **Cloud Deployment**: Focus on private cloud deployment (AWS EU, Azure EU, GCP EU)

## Examples
\`\`\`tsx
import { EnterpriseHowItWorks } from '@rbee/ui/organisms/Enterprise/EnterpriseHowItWorks'

// Simple usage - no props needed
<EnterpriseHowItWorks />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- StepCard
- Card

## Accessibility
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Semantic HTML**: Uses <section> with aria-labelledby, <ol> for steps
- **ARIA Labels**: Progress bar marked as aria-hidden
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, VP Engineering, Project Manager, Implementation Lead
- **Pain Points**: Implementation complexity, timeline uncertainty, resource requirements, support quality
- **Decision Criteria**: Time-to-value, implementation support, deployment options, SLAs

### Deployment Complexity
- **Not a "5-minute setup"**: Enterprise deployments require planning, compliance validation, and support
- **Realistic timeline**: 7 weeks from assessment to production (not "instant")
- **Professional services**: Compliance assessment, deployment support, architecture review, penetration testing
- **Ongoing support**: 24/7 support, dedicated account manager, quarterly compliance reviews

### Deployment Options

**On-Premises (Your Servers)**
- Full control, air-gap support, no external dependencies
- Ideal for: Government, defense, highly regulated industries
- Timeline: 3-4 weeks (after assessment)

**EU Data Centers (Frankfurt, Amsterdam, Paris)**
- Managed infrastructure, EU residency guaranteed, professional support
- Ideal for: Enterprises without data center capacity
- Timeline: 2-3 weeks (after assessment)

**Private Cloud (AWS EU, Azure EU, GCP EU)**
- Cloud convenience with EU residency, VPC isolation, no US parent company access
- Ideal for: Cloud-first enterprises with EU compliance requirements
- Timeline: 2-3 weeks (after assessment)

**Hybrid (On-Prem + Marketplace)**
- On-prem for sensitive workloads, marketplace for burst capacity
- Ideal for: Enterprises with variable workloads
- Timeline: 3-4 weeks (after assessment)

### Deployment Timeline (7 Weeks)
- **Week 1-2**: Compliance Assessment (gap analysis, data flow mapping, risk assessment, architecture proposal)
- **Week 3-4**: Deployment & Configuration (install, configure policies, enable audit logging)
- **Week 5-6**: Compliance Validation (auditor access, security review, penetration testing)
- **Week 7**: Production Launch (go live with SLAs, monitoring, support)

### Enterprise Support & SLAs
- **99.9% uptime SLA**: Guaranteed availability (8.76 hours downtime/year max)
- **24/7 support**: 1-hour response time for critical issues
- **Dedicated account manager**: Single point of contact
- **Quarterly compliance reviews**: Ongoing compliance validation

### Conversion Strategy
- **Primary CTA**: "Request Deployment Brief" (sales-assisted evaluation)
- **Secondary CTA**: "View Deployment Options" (self-serve education)
- **Lead qualification**: Capture deployment timeline, infrastructure details, compliance requirements
- **Proof points**: 7-week timeline, 99.9% uptime SLA, 24/7 support, dedicated account manager
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseHowItWorks>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise deployment process showing all four stages: Compliance Assessment (Week 1-2: gap analysis, data flow mapping, risk assessment, architecture proposal), On-Premises Deployment (Week 3-4: EU data centers, on-premises, private cloud, hybrid), Compliance Validation (Week 5-6: documentation package, auditor access, security review, penetration testing), and Production Launch (Week 7: 99.9% uptime SLA, 24/7 support, dedicated account manager, quarterly compliance reviews). Includes sticky timeline panel showing 7-week deployment duration.',
			},
		},
	},
}

export const OnPremDeployment: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant focusing on on-premises deployment options. This version would emphasize: Deploy on your own servers (full control, air-gap support, no external dependencies), EU data center options (Frankfurt, Amsterdam, Paris), private cloud options (AWS EU, Azure EU, GCP EU with VPC isolation), and hybrid deployment (on-prem for sensitive workloads, marketplace for burst capacity). Ideal for government, defense, or highly regulated industries requiring full control.',
			},
		},
	},
}

export const CloudDeployment: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant focusing on private cloud deployment options. This version would emphasize: AWS EU (Frankfurt, Ireland regions with VPC isolation), Azure EU (Netherlands, Germany regions with private endpoints), GCP EU (Belgium, Finland regions with private service connect), and EU data residency guarantees (no US parent company access, GDPR Article 44 compliant). Ideal for cloud-first enterprises with EU compliance requirements.',
			},
		},
	},
}
