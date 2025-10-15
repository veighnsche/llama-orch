import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseUseCases } from './EnterpriseUseCases'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Enterprise/EnterpriseUseCases',
  component: EnterpriseUseCases,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The EnterpriseUseCases section presents industry-specific use cases for rbee in highly regulated sectors: Financial Services, Healthcare, Legal Services, and Government. Each use case includes compliance badges, challenges, and solutions tailored to that industry.

## Composition
This organism contains:
- **Header**: "Built for Regulated Industries" with subtitle
- **Four Industry Case Cards**: Financial Services, Healthcare, Legal Services, Government
- **Financial Services**: PCI-DSS, GDPR, SOC2 compliance (banks, insurance, fintech)
- **Healthcare**: HIPAA, GDPR Article 9 compliance (hospitals, medtech, pharma)
- **Legal Services**: GDPR, Legal Hold compliance (law firms, legaltech)
- **Government**: ISO 27001, Data Sovereignty compliance (public sector, defense)
- **CTA Rail**: "Request Industry Brief" and "Talk to a Solutions Architect" buttons
- **Industry Links**: Quick links to Finance, Healthcare, Legal, Government pages

## When to Use
- On the /enterprise page after the deployment section
- To demonstrate industry-specific compliance requirements
- To show real-world use cases in regulated sectors
- To drive industry-specific lead generation

## Content Requirements
- **Industry Cases**: Specific compliance requirements for each sector
- **Challenges**: Industry-specific pain points (PCI-DSS, HIPAA, attorney-client privilege, data sovereignty)
- **Solutions**: How rbee addresses each challenge
- **Compliance Badges**: Specific regulations (PCI-DSS, HIPAA, GDPR Art. 9, Legal Hold, ISO 27001)

## Variants
- **Default**: All four industry cases
- **Financial Services**: Focus on PCI-DSS and SOC2 compliance
- **Healthcare**: Focus on HIPAA and GDPR Article 9 compliance

## Examples
\`\`\`tsx
import { EnterpriseUseCases } from '@rbee/ui/organisms/Enterprise/EnterpriseUseCases'

// Simple usage - no props needed
<EnterpriseUseCases />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- IndustryCaseCard
- Button
- Badge

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: Industry-specific decision makers (CFO/CTO for finance, CIO/CISO for healthcare, Managing Partner for legal, CIO for government)
- **Pain Points**: Industry-specific compliance requirements, regulatory fines, audit failures, data breaches
- **Decision Criteria**: Industry-specific compliance, proven track record, industry expertise

### Industry-Specific Messaging

**Financial Services (Banks, Insurance, FinTech)**
- **Compliance**: PCI-DSS (no external APIs), GDPR (EU data residency), SOC2 (audit trail)
- **Challenges**: Cannot use external APIs (PCI-DSS), need complete audit trail (SOC2), EU data residency (GDPR), 7-year retention
- **Solutions**: On-prem (EU data center), immutable audit logs (7-year), zero external dependencies, SOC2 Type II ready
- **Use Case**: EU bank needed internal code-gen but PCI-DSS/GDPR blocked external AI
- **Proof Points**: PCI-DSS compliant architecture, SOC2 Type II ready, 7-year audit retention

**Healthcare (Hospitals, MedTech, Pharma)**
- **Compliance**: HIPAA (PHI protection), GDPR Article 9 (health data)
- **Challenges**: HIPAA/PHI protection, GDPR Article 9 (special category data), no US clouds, breach notifications
- **Solutions**: Self-hosted (hospital data center), EU-only deployment, full audit trail (breach detection), HIPAA-aligned architecture
- **Use Case**: AI-assisted patient tooling with HIPAA + GDPR Article 9 constraints
- **Proof Points**: HIPAA-aligned architecture, GDPR Article 9 compliant, breach detection, EU-only deployment

**Legal Services (Law Firms, LegalTech)**
- **Compliance**: GDPR, Legal Hold (attorney-client privilege)
- **Challenges**: Attorney-client privilege (cannot risk disclosure), no external uploads, legal-hold audit trail, EU residency
- **Solutions**: On-prem (firm servers), zero data transfer, immutable legal-hold logs, full confidentiality
- **Use Case**: Document analysis without risking privilege
- **Proof Points**: Attorney-client privilege protected, zero data transfer, immutable legal-hold logs

**Government (Public Sector, Defense)**
- **Compliance**: ISO 27001, Data Sovereignty
- **Challenges**: Data sovereignty (no foreign clouds), no foreign clouds, transparent audit trail, ISO 27001 required
- **Solutions**: Gov data center deployment, EU-only infrastructure, ISO 27001 aligned, complete sovereignty
- **Use Case**: Citizen services with strict sovereignty + security controls
- **Proof Points**: Data sovereignty guaranteed, ISO 27001 aligned, transparent audit trail, EU-only infrastructure

### Conversion Strategy
- **Primary CTA**: "Request Industry Brief" (industry-specific sales materials)
- **Secondary CTA**: "Talk to a Solutions Architect" (sales-assisted evaluation)
- **Lead qualification**: Capture industry, compliance requirements, timeline, infrastructure details
- **Proof points**: Industry-specific compliance, proven track record, industry expertise

### Industry Playbooks
- **Finance**: PCI-DSS compliance guide, SOC2 Type II readiness, 7-year audit retention
- **Healthcare**: HIPAA compliance guide, GDPR Article 9 compliance, breach detection
- **Legal**: Attorney-client privilege protection, legal-hold audit trail, zero data transfer
- **Government**: Data sovereignty guide, ISO 27001 alignment, transparent audit trail
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseUseCases>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default enterprise use cases section showing all four regulated industries: Financial Services (PCI-DSS, GDPR, SOC2 - banks, insurance, fintech), Healthcare (HIPAA, GDPR Art. 9 - hospitals, medtech, pharma), Legal Services (GDPR, Legal Hold - law firms, legaltech), and Government (ISO 27001, Sovereignty - public sector, defense). Each case includes industry-specific challenges and solutions.',
      },
    },
  },
}

export const FinancialServices: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant focusing on financial services use case. This version would emphasize: PCI-DSS compliance (no external APIs allowed), SOC2 Type II readiness (complete audit trail with 7-year retention), GDPR compliance (EU data residency), and zero external dependencies. Use case: EU bank needed internal code-gen for developer productivity but PCI-DSS and GDPR blocked external AI providers. Solution: On-prem deployment in EU data center with immutable audit logs and SOC2 Type II ready architecture. Ideal for banks, insurance companies, and fintech startups.',
      },
    },
  },
}

export const Healthcare: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant focusing on healthcare use case. This version would emphasize: HIPAA compliance (PHI protection, breach notifications), GDPR Article 9 compliance (special category health data), no US cloud dependencies, and full audit trail for breach detection. Use case: AI-assisted patient tooling (clinical decision support, medical coding, patient communication) with HIPAA + GDPR Article 9 constraints. Solution: Self-hosted in hospital data center with EU-only deployment and HIPAA-aligned architecture. Ideal for hospitals, medtech companies, and pharmaceutical companies.',
      },
    },
  },
}
