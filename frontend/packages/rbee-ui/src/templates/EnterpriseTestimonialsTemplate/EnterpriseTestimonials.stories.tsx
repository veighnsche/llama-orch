import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseTestimonials } from './EnterpriseTestimonials'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Enterprise/EnterpriseTestimonials',
  component: EnterpriseTestimonials,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The EnterpriseTestimonials section presents testimonials from regulated industries (finance, healthcare, legal) using the TestimonialsRail component. Filtered to show only enterprise-relevant testimonials with a grid layout and stats.

## Composition
This organism is a wrapper around TestimonialsRail with enterprise-specific configuration:
- **Header**: "Trusted by Regulated Industries" with subtitle
- **TestimonialsRail**: Filtered by sector (finance, healthcare, legal), grid layout, with stats
- **Sector Filter**: Only shows testimonials from finance, healthcare, and legal sectors
- **Layout**: Grid layout (vs. carousel for other pages)
- **Stats**: Shows testimonial stats (number of testimonials, average rating, etc.)

## When to Use
- On the /enterprise page after the features section
- To demonstrate social proof from regulated industries
- To show real-world enterprise customers
- To build credibility with enterprise buyers

## Content Requirements
- **Testimonials**: From enterprise decision-makers (CTOs, CIOs, Compliance Officers, DPOs)
- **Industries**: Finance, healthcare, legal, government
- **Proof Points**: Specific outcomes (compliance achieved, fines avoided, time saved)
- **Company Details**: Company name, industry, company size

## Variants
- **Default**: All enterprise testimonials (finance, healthcare, legal)
- **Compliance Testimonials**: Focus on compliance-focused quotes
- **ROI Testimonials**: Focus on cost savings and ROI quotes

## Examples
\`\`\`tsx
import { EnterpriseTestimonials } from '@rbee/ui/organisms/Enterprise/EnterpriseTestimonials'

// Simple usage - no props needed
<EnterpriseTestimonials />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- TestimonialsRail
- Card

## Accessibility
- **Keyboard Navigation**: All testimonials are keyboard accessible
- **Semantic HTML**: Uses <section> with proper heading hierarchy
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, Compliance Officer, DPO, Procurement Manager
- **Pain Points**: Vendor credibility, proof of compliance, real-world outcomes, industry-specific experience
- **Decision Criteria**: Social proof from similar companies, industry-specific testimonials, quantifiable outcomes

### Testimonial Strategy

**Enterprise buyers need social proof from similar companies:**
- Same industry (finance, healthcare, legal, government)
- Similar company size (50-5000+ employees)
- Similar compliance requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS)
- Specific outcomes (compliance achieved, fines avoided, time saved, cost savings)

**Testimonial Types:**

**Compliance Testimonials**
- Quote focus: "We achieved GDPR compliance in 6 weeks"
- Proof point: Specific compliance standard achieved
- Outcome: Compliance certification, audit passed, fines avoided
- Ideal for: Compliance officers, DPOs, legal counsel

**ROI Testimonials**
- Quote focus: "We saved €200K/year vs. cloud APIs"
- Proof point: Specific cost savings or ROI
- Outcome: Cost reduction, budget predictability, faster time-to-value
- Ideal for: CFOs, procurement managers, budget-conscious buyers

**Technical Testimonials**
- Quote focus: "Deployment took 3 weeks with full support"
- Proof point: Implementation timeline, technical capabilities
- Outcome: Faster deployment, technical success, integration success
- Ideal for: CTOs, IT directors, engineering managers

**Industry-Specific Testimonials**

**Finance (Banks, Insurance, FinTech)**
- Compliance: PCI-DSS, GDPR, SOC2
- Proof points: "No external APIs (PCI-DSS compliant)", "7-year audit retention (SOC2 Type II)", "EU data residency (GDPR)"
- Outcome: Compliance achieved, audit passed, fines avoided

**Healthcare (Hospitals, MedTech, Pharma)**
- Compliance: HIPAA, GDPR Article 9
- Proof points: "HIPAA-aligned architecture", "GDPR Article 9 compliant (health data)", "Breach detection with full audit trail"
- Outcome: HIPAA compliance, GDPR Article 9 compliance, breach prevention

**Legal (Law Firms, LegalTech)**
- Compliance: GDPR, Legal Hold, Attorney-Client Privilege
- Proof points: "Attorney-client privilege protected", "Immutable legal-hold logs", "Zero data transfer"
- Outcome: Privilege protected, legal-hold compliance, confidentiality maintained

### Credibility Signals
- **Company name**: Real company names (not "Anonymous" or "Confidential")
- **Role**: Specific role (CTO, CIO, Compliance Officer, DPO)
- **Company size**: Specific company size (50-500 employees, 500-5000 employees)
- **Industry**: Specific industry (finance, healthcare, legal, government)
- **Outcome**: Specific outcome (compliance achieved, fines avoided, time saved, cost savings)

### Conversion Strategy
- **Primary CTA**: "Read Case Studies" (detailed customer stories)
- **Secondary CTA**: "Request Demo" (sales-assisted evaluation)
- **Lead qualification**: Capture industry, company size, compliance requirements
- **Proof points**: Industry-specific testimonials, quantifiable outcomes, credibility signals
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseTestimonials>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default enterprise testimonials section showing testimonials from regulated industries: finance (banks, insurance, fintech), healthcare (hospitals, medtech, pharma), and legal (law firms, legaltech). Uses grid layout with stats. Testimonials are filtered to show only enterprise-relevant quotes from CTOs, CIOs, Compliance Officers, and DPOs.',
      },
    },
  },
}

export const ComplianceTestimonials: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant focusing on compliance-focused testimonials. This version would emphasize quotes about: GDPR compliance achieved ("We achieved GDPR compliance in 6 weeks"), SOC2 Type II readiness ("Passed SOC2 Type II audit on first try"), HIPAA compliance ("HIPAA-aligned architecture made compliance straightforward"), and audit trail success ("Immutable audit logs saved us during compliance audit"). Ideal for compliance officers, DPOs, and legal counsel.',
      },
    },
  },
}

export const ROITestimonials: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant focusing on cost savings and ROI testimonials. This version would emphasize quotes about: Cost savings vs. cloud APIs ("We saved €200K/year vs. OpenAI"), budget predictability ("No more surprise API bills"), faster time-to-value ("Deployed in 3 weeks with full support"), and compliance cost avoidance ("Avoided €20M GDPR fine risk"). Ideal for CFOs, procurement managers, and budget-conscious buyers.',
      },
    },
  },
}
