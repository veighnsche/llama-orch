import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseHero } from './EnterpriseHero'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Enterprise/EnterpriseHero',
  component: EnterpriseHero,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The EnterpriseHero is the primary landing section for the Enterprise page, emphasizing compliance, security, and EU data residency. It features a split layout with messaging on the left and an interactive audit trail console on the right, demonstrating enterprise-grade compliance features.

## Composition
This organism contains:
- **Badge**: EU-Native AI Infrastructure eyebrow with shield icon
- **H1 Headline**: Compliance-focused value proposition
- **Support Copy**: GDPR, SOC2, ISO 27001 compliance messaging
- **StatsGrid**: Proof tiles showing compliance metrics (100% GDPR, 7 Years Audit Retention, Zero US Cloud Dependencies)
- **CTA Buttons**: "Schedule Demo" (primary) and "View Compliance Details" (secondary)
- **ComplianceChip Bar**: Visual proof badges (GDPR, SOC2, ISO 27001)
- **Audit Console Visual**: Interactive UI showing immutable audit trail with real-time events

## When to Use
- As the first section on the /enterprise page
- To immediately establish compliance and security credibility
- To demonstrate audit trail capabilities visually
- To drive enterprise buyers to schedule demos

## Content Requirements
- **Headline**: Compliance-focused, addresses regulatory requirements
- **Support Copy**: Specific compliance standards (GDPR, SOC2, ISO 27001)
- **Proof Points**: Quantifiable compliance metrics
- **Audit Events**: Realistic event log entries with timestamps
- **CTAs**: Enterprise-appropriate actions (demo, not "sign up free")

## Variants
- **Default**: Full compliance focus with all proof points
- **GDPR Focus**: Emphasize EU data residency and GDPR compliance
- **Security Focus**: Lead with security and audit trail capabilities

## Examples
\`\`\`tsx
import { EnterpriseHero } from '@rbee/ui/organisms/Enterprise/EnterpriseHero'

// Simple usage - no props needed
<EnterpriseHero />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- Badge
- Button
- Card
- StatsGrid
- ComplianceChip

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **ARIA Labels**: Proper labels on interactive elements and audit events
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Live Regions**: Compliance proof bar uses aria-live="polite"
- **Focus States**: Visible focus indicators on all interactive elements
- **Color Contrast**: Meets WCAG AA standards in both themes

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, VP Engineering, Compliance Officer, Data Protection Officer (DPO)
- **Company size**: 50-5000+ employees
- **Budget authority**: €10K-€500K+ annual budget
- **Decision process**: Committee-based, 3-12 month sales cycle, requires legal/compliance sign-off

### Enterprise Messaging
- **Primary concern**: Compliance risk (GDPR fines up to €20M or 4% of global revenue)
- **Proof points needed**: Specific compliance standards (GDPR Article 44, SOC2, ISO 27001), immutable audit trails, EU data residency guarantees
- **Objections to address**: "Can we trust a smaller vendor?", "What about implementation complexity?", "Do you have enterprise support?"
- **Tone**: Professional, compliance-focused, ROI-driven (not casual developer tone)

### Conversion Strategy
- **Primary CTA**: "Schedule Demo" (not "Get Started Free" - enterprise buyers need sales-assisted evaluation)
- **Lead qualification**: Demo request form should capture company size, compliance requirements, timeline
- **Sales-assisted**: Yes - all enterprise leads go to sales team for qualification and custom demos
- **Proof required**: Compliance certifications, audit trail demo, data residency guarantees

### Competitive Positioning
- **vs. Cloud APIs (OpenAI, Anthropic)**: On-prem = full control, EU data residency, no US cloud dependencies, predictable costs
- **vs. Azure OpenAI/AWS Bedrock**: True EU residency (not just "EU region" with US parent company), no vendor lock-in, open-source foundation
- **vs. DIY (Ollama, vLLM)**: Enterprise features (audit trails, RBAC, SLAs), professional support, faster time-to-value, compliance out-of-box

### Key Differentiators
- **EU-Native**: No US cloud dependencies (critical for Schrems II compliance)
- **Immutable Audit Trails**: 7-year retention, tamper-evident logs (GDPR Article 30 compliance)
- **Compliance by Design**: GDPR, SOC2, ISO 27001 aligned from day one
- **Zero Data Sovereignty Risk**: All data stays in EU, no cross-border transfers
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseHero>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default enterprise hero as seen on /enterprise page. Emphasizes compliance (GDPR, SOC2, ISO 27001), EU data residency, and immutable audit trails. The audit console visual demonstrates real-time compliance monitoring. Use the theme toggle to test light/dark modes.',
      },
    },
  },
}

export const GDPRFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing GDPR compliance and EU data residency. This version would highlight: GDPR Article 44 compliance (no US cloud dependencies), Schrems II compliance, Data Protection Authority approval, and EU-only data processing. Ideal for EU-based enterprises with strict data sovereignty requirements.',
      },
    },
  },
}

export const SecurityAuditFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant leading with security and audit trail capabilities. This version would emphasize: Immutable audit logs (tamper-evident, 7-year retention), SOC2 and ISO 27001 alignment, zero-trust architecture, and comprehensive event tracking (32 audit event types). Ideal for enterprises with strong security/audit requirements (financial services, healthcare).',
      },
    },
  },
}

export const ROIFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing cost savings and ROI. This version would lead with: Avoid GDPR fines (up to €20M or 4% of revenue), eliminate unpredictable cloud API costs, reduce vendor dependency risk, and faster compliance certification. Includes cost comparison vs. cloud APIs and DIY approaches. Ideal for cost-conscious enterprises or those burned by cloud API bills.',
      },
    },
  },
}
