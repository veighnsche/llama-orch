import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseCompliance } from './EnterpriseCompliance'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Enterprise/EnterpriseCompliance',
  component: EnterpriseCompliance,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The EnterpriseCompliance section presents three compliance pillars (GDPR, SOC2, ISO 27001) with detailed checklists, compliance endpoints, and audit readiness messaging. Each pillar includes specific regulatory requirements, implementation details, and proof points.

## Composition
This organism contains:
- **Header**: "Compliance by Design" with subtitle
- **Three Compliance Pillars**: GDPR (EU Regulation), SOC2 (US Standard), ISO 27001 (International Standard)
- **GDPR Pillar**: 6-item checklist (Art. 30, Art. 15, Art. 17, Art. 7, Art. 44, Art. 33) + compliance endpoints
- **SOC2 Pillar**: 6-item checklist (auditor API, 32 event types, 7-year retention, hash chains, access logging, encryption) + Trust Service Criteria
- **ISO 27001 Pillar**: 6-item checklist (incident records, 3-year retention, access logging, crypto controls, ops security, policies) + ISMS controls
- **Audit Readiness Band**: CTAs to download compliance pack or talk to compliance team

## When to Use
- On the /enterprise page after the solution section
- To demonstrate specific compliance standards
- To provide detailed regulatory requirements
- To offer compliance documentation downloads

## Content Requirements
- **Compliance Standards**: GDPR, SOC2, ISO 27001 with specific article/control references
- **Checklists**: Detailed requirements for each standard
- **Proof Points**: Compliance endpoints, Trust Service Criteria, ISMS controls
- **CTAs**: Download compliance pack, talk to compliance team

## Variants
- **Default**: All three compliance pillars
- **GDPR Focus**: Emphasize EU compliance only
- **Audit Trail Focus**: Emphasize audit logging and retention

## Examples
\`\`\`tsx
import { EnterpriseCompliance } from '@rbee/ui/organisms/Enterprise/EnterpriseCompliance'

// Simple usage - no props needed
<EnterpriseCompliance />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- CompliancePillar
- Button
- Card

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **ARIA Labels**: Proper labels on compliance endpoints and controls
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators on all interactive elements
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: Compliance Officer, Data Protection Officer (DPO), Legal Counsel, CTO, IT Director
- **Pain Points**: Regulatory compliance, audit requirements, certification costs, legal liability
- **Decision Criteria**: Specific compliance standards, audit readiness, legal defensibility

### Compliance as Selling Point
- **Not just a checkbox**: Compliance is engineered in, not bolted on
- **Specific standards**: GDPR (EU), SOC2 (US), ISO 27001 (International)
- **Audit readiness**: Compliance pack includes endpoints, retention policy, audit-logging design
- **Legal defensibility**: Immutable audit trails, tamper-evident hash chains, 7-year retention

### GDPR Compliance (EU Regulation)
- **Article 30**: 7-year audit retention (record of processing activities)
- **Article 15**: Data access records (right of access)
- **Article 17**: Erasure tracking (right to be forgotten)
- **Article 7**: Consent management (conditions for consent)
- **Article 44**: Data residency controls (transfers to third countries)
- **Article 33**: Breach notification (notification of personal data breach)
- **Compliance Endpoints**: GET /v2/compliance/data-access, POST /v2/compliance/data-export, POST /v2/compliance/data-deletion, GET /v2/compliance/audit-trail

### SOC2 Compliance (US Standard)
- **Auditor query API**: Programmatic access for auditors
- **32 audit event types**: Comprehensive event coverage
- **7-year retention (Type II)**: Long-term audit trail
- **Tamper-evident hash chains**: Cryptographic proof of integrity
- **Access control logging**: All access attempts logged
- **Encryption at rest**: Data protection
- **Trust Service Criteria**: Security (CC1-CC9), Availability (A1.1-A1.3), Confidentiality (C1.1-C1.2)

### ISO 27001 Compliance (International Standard)
- **Incident records (A.16)**: Information security incident management
- **3-year minimum retention**: Audit trail retention
- **Access logging (A.9)**: Access control
- **Crypto controls (A.10)**: Cryptography
- **Ops security (A.12)**: Operations security
- **Security policies (A.5)**: Information security policies
- **ISMS Controls**: 114 controls implemented, risk assessment framework, continuous monitoring

### Conversion Strategy
- **Primary CTA**: "Download Compliance Pack" (self-serve education)
- **Secondary CTA**: "Talk to Compliance Team" (sales-assisted evaluation)
- **Lead qualification**: Capture industry (healthcare, finance, government), compliance requirements, audit timeline
- **Proof points**: Compliance pack includes endpoints, retention policy, audit-logging design
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseCompliance>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default enterprise compliance section showing all three compliance pillars: GDPR (EU Regulation with 6 requirements + compliance endpoints), SOC2 (US Standard with 6 requirements + Trust Service Criteria), and ISO 27001 (International Standard with 6 requirements + ISMS controls). Includes audit readiness band with CTAs to download compliance pack or talk to compliance team.',
      },
    },
  },
}

export const GDPRFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing GDPR compliance only. This version would focus exclusively on EU regulations: Article 30 (7-year audit retention), Article 15 (data access records), Article 17 (erasure tracking), Article 7 (consent management), Article 44 (data residency controls), and Article 33 (breach notification). Includes detailed compliance endpoints (GET /v2/compliance/data-access, POST /v2/compliance/data-export, POST /v2/compliance/data-deletion, GET /v2/compliance/audit-trail). Ideal for EU-based enterprises or those with strict GDPR requirements.',
      },
    },
  },
}

export const AuditTrailFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing audit trail and logging capabilities across all three standards. This version would highlight: 7-year retention (GDPR Article 30, SOC2 Type II), 32 audit event types (SOC2), tamper-evident hash chains (SOC2), immutable audit trail (GDPR), and incident records (ISO 27001 A.16). Includes detailed audit logging design and cryptographic proof of integrity. Ideal for enterprises with strong audit requirements (financial services, healthcare, government).',
      },
    },
  },
}
