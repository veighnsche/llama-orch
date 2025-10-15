import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseFeatures } from './EnterpriseFeatures'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseFeatures',
	component: EnterpriseFeatures,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseFeatures section presents four enterprise-specific capabilities: Enterprise SLAs, White-Label Option, Professional Services, and Multi-Region Support. Includes an outcomes band showing key metrics (99.9% uptime SLA, <1 hour support response, EU-only data residency).

## Composition
This organism contains:
- **Header**: "Enterprise Features" with subtitle
- **Four Feature Cards**: Enterprise SLAs, White-Label Option, Professional Services, Multi-Region Support
- **Enterprise SLAs**: 99.9% uptime, 24/7 support (1-hour response), dedicated account manager, quarterly reviews
- **White-Label Option**: Custom branding/logo, custom domain (ai.yourcompany.com), UI customization, API endpoint customization
- **Professional Services**: Deployment consulting, integration support, custom development, team training
- **Multi-Region Support**: EU multi-region, automatic failover, load balancing, geo-redundancy
- **Outcomes Band**: 99.9% uptime SLA, <1 hour support response, EU-only data residency

## When to Use
- On the /enterprise page after the comparison section
- To demonstrate enterprise-specific features
- To show SLAs and support commitments
- To differentiate from open-source/free tier

## Content Requirements
- **Features**: Enterprise-specific capabilities (SLAs, white-label, professional services, multi-region)
- **Metrics**: Quantifiable outcomes (99.9% uptime, <1 hour response, EU-only)
- **Details**: Specific feature bullets for each capability

## Variants
- **Default**: All four enterprise features
- **Authentication Focus**: Emphasize SSO/SAML and RBAC
- **Governance Focus**: Emphasize policies and audit

## Examples
\`\`\`tsx
import { EnterpriseFeatures } from '@rbee/ui/organisms/Enterprise/EnterpriseFeatures'

// Simple usage - no props needed
<EnterpriseFeatures />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- FeatureCard
- Card

## Accessibility
- **Keyboard Navigation**: All links are keyboard accessible
- **ARIA Labels**: Proper labels on all interactive elements
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, VP Engineering, Procurement Manager
- **Pain Points**: SLA requirements, support quality, customization needs, multi-region redundancy
- **Decision Criteria**: SLA commitments, support response times, white-label options, professional services

### Enterprise Features vs. OSS Version

**Enterprise SLAs (Enterprise Only)**
- **99.9% uptime SLA**: Guaranteed availability (8.76 hours downtime/year max)
- **24/7 support**: 1-hour response time for critical issues (vs. community support for OSS)
- **Dedicated account manager**: Single point of contact (vs. self-service for OSS)
- **Quarterly compliance reviews**: Ongoing compliance validation (vs. self-managed for OSS)

**White-Label Option (Enterprise Only)**
- **Custom branding/logo**: Run rbee as your brand (vs. rbee branding for OSS)
- **Custom domain**: ai.yourcompany.com (vs. default domain for OSS)
- **UI customization**: Custom colors, logos, messaging (vs. standard UI for OSS)
- **API endpoint customization**: Custom API endpoints (vs. standard endpoints for OSS)

**Professional Services (Enterprise Only)**
- **Deployment consulting**: Expert guidance for deployment (vs. self-deployment for OSS)
- **Integration support**: Help integrating with existing systems (vs. self-integration for OSS)
- **Custom development**: Custom features for your use case (vs. standard features for OSS)
- **Team training**: On-site or remote training (vs. self-learning for OSS)

**Multi-Region Support (Enterprise Only)**
- **EU multi-region**: Deploy across multiple EU regions (Frankfurt, Amsterdam, Paris)
- **Automatic failover**: Automatic failover to backup region if primary fails
- **Load balancing**: Distribute load across regions for performance
- **Geo-redundancy**: Data replicated across regions for disaster recovery

### Feature vs. Benefit

**Feature**: 99.9% uptime SLA  
**Benefit**: Your AI infrastructure is always available when you need it (8.76 hours downtime/year max)

**Feature**: 24/7 support with 1-hour response  
**Benefit**: Critical issues resolved quickly, minimizing business impact

**Feature**: White-label option  
**Benefit**: Run rbee as your brand, no vendor attribution, full control over user experience

**Feature**: Professional services  
**Benefit**: Faster time-to-value, expert guidance, reduced implementation risk

**Feature**: Multi-region support  
**Benefit**: High availability, disaster recovery, compliance with data residency requirements

### Conversion Strategy
- **Primary CTA**: "Request Enterprise Quote" (sales-assisted evaluation)
- **Secondary CTA**: "View Enterprise Docs" (self-serve education)
- **Lead qualification**: Capture SLA requirements, support needs, customization needs, multi-region requirements
- **Proof points**: 99.9% uptime SLA, <1 hour support response, EU-only data residency
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseFeatures>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise features section showing all four enterprise capabilities: Enterprise SLAs (99.9% uptime, 24/7 support with 1-hour response, dedicated account manager, quarterly reviews), White-Label Option (custom branding/logo, custom domain, UI customization, API endpoint customization), Professional Services (deployment consulting, integration support, custom development, team training), and Multi-Region Support (EU multi-region, automatic failover, load balancing, geo-redundancy). Includes outcomes band showing 99.9% uptime SLA, <1 hour support response, and EU-only data residency.',
			},
		},
	},
}

export const AuthenticationFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing authentication and access control features. This version would focus on: SSO/SAML integration (Okta, Azure AD, Google Workspace), advanced RBAC (role-based access control with fine-grained permissions), audit logging (all authentication and access attempts logged), and JWT lifecycle management (RS256/ES256 validation, revocation lists, short-lived refresh tokens). Ideal for enterprises with strong authentication requirements or those using SSO providers.',
			},
		},
	},
}

export const GovernanceFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing governance and policy features. This version would focus on: Custom policies (Rhai-based programmable routing and access control), audit logging (32 event types with 7-year retention), compliance reporting (automated compliance reports for GDPR, SOC2, ISO 27001), and data residency controls (enforce EU-only data processing). Ideal for enterprises with strong governance requirements or those in highly regulated industries.',
			},
		},
	},
}
