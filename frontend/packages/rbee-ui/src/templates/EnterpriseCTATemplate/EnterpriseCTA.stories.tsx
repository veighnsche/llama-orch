import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseCTA } from './EnterpriseCTA'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Enterprise/EnterpriseCTA',
  component: EnterpriseCTA,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The EnterpriseCTA section presents three conversion options for enterprise buyers: Schedule Demo (primary), Compliance Pack (secondary), and Talk to Sales (tertiary). Includes trust strip with testimonial stats and deployment timeline information.

## Composition
This organism contains:
- **Header**: "Ready to Meet Your Compliance Requirements?" with subtitle
- **Trust Strip**: Testimonial stats (4 metrics from TESTIMONIAL_STATS)
- **Three CTA Option Cards**: Schedule Demo, Compliance Pack, Talk to Sales
- **Schedule Demo**: Primary CTA with Calendar icon, "30-minute demo with our compliance team"
- **Compliance Pack**: Secondary CTA with FileText icon, "Download GDPR, SOC2, and ISO 27001 documentation"
- **Talk to Sales**: Tertiary CTA with MessageSquare icon, "Discuss your specific compliance requirements"
- **Footer Caption**: "Enterprise support 24/7 • Typical deployment: 6–8 weeks from consultation to production"

## When to Use
- On the /enterprise page as the final section
- To drive enterprise conversions
- To offer multiple conversion paths (demo, docs, sales)
- To set expectations for deployment timeline

## Content Requirements
- **CTAs**: Multiple conversion options (demo, docs, sales)
- **Trust Signals**: Testimonial stats, deployment timeline
- **Messaging**: Compliance-focused, enterprise-appropriate
- **Timeline**: Realistic deployment expectations (6-8 weeks)

## Variants
- **Default**: All three CTA options
- **Demo Focus**: Emphasize "Schedule Demo" as primary CTA
- **Contact Sales Focus**: Emphasize "Talk to Sales" as primary CTA

## Examples
\`\`\`tsx
import { EnterpriseCTA } from '@rbee/ui/organisms/Enterprise/EnterpriseCTA'

// Simple usage - no props needed
<EnterpriseCTA />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- CTAOptionCard
- Button
- Card

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **ARIA Labels**: Proper labels on all CTAs
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CTO, IT Director, Compliance Officer, DPO, Procurement Manager
- **Pain Points**: Vendor evaluation, compliance verification, deployment timeline, support quality
- **Decision Criteria**: Demo quality, documentation completeness, sales responsiveness, deployment timeline

### Enterprise Conversion Strategy

**Multiple Conversion Paths:**
- **Schedule Demo**: For buyers ready to evaluate (high intent)
- **Compliance Pack**: For buyers in research phase (medium intent)
- **Talk to Sales**: For buyers with specific questions (variable intent)

**CTA Hierarchy:**
1. **Primary**: Schedule Demo (highest intent, highest value)
2. **Secondary**: Compliance Pack (medium intent, lead nurture)
3. **Tertiary**: Talk to Sales (variable intent, qualification needed)

### CTA Option Details

**Schedule Demo (Primary)**
- **Icon**: Calendar
- **Title**: "Schedule Demo"
- **Body**: "30-minute demo with our compliance team. See rbee in action."
- **Note**: "30-minute session • live environment"
- **Button**: "Book Demo" (primary button)
- **Target**: Enterprise buyers ready to evaluate (high intent)
- **Outcome**: Qualified demo lead, sales team engagement

**Compliance Pack (Secondary)**
- **Icon**: FileText
- **Title**: "Compliance Pack"
- **Body**: "Download GDPR, SOC2, and ISO 27001 documentation."
- **Note**: "GDPR, SOC2, ISO 27001 summaries"
- **Button**: "Download Docs" (outline button)
- **Target**: Enterprise buyers in research phase (medium intent)
- **Outcome**: Lead capture, email nurture, self-serve education

**Talk to Sales (Tertiary)**
- **Icon**: MessageSquare
- **Title**: "Talk to Sales"
- **Body**: "Discuss your specific compliance requirements."
- **Note**: "Share requirements & timelines"
- **Button**: "Contact Sales" (outline button)
- **Target**: Enterprise buyers with specific questions (variable intent)
- **Outcome**: Sales qualification, custom proposal, specific questions answered

### Trust Signals

**Testimonial Stats (Trust Strip)**
- Shows 4 metrics from TESTIMONIAL_STATS
- Examples: "50+ enterprises", "99.9% uptime", "24/7 support", "EU-only"
- Purpose: Build credibility, reduce risk perception

**Deployment Timeline (Footer Caption)**
- "Typical deployment: 6–8 weeks from consultation to production"
- Purpose: Set realistic expectations, reduce uncertainty
- Context: Enterprise support 24/7 (reassurance)

### Conversion Messaging

**Headline**: "Ready to Meet Your Compliance Requirements?"
- Compliance-focused (not "Ready to get started?")
- Addresses primary enterprise concern (compliance)
- Question format (engages buyer)

**Subtitle**: "Book a demo with our compliance team, or download the documentation pack."
- Two clear options (demo or docs)
- Compliance team (not "sales team" - less pushy)
- Documentation pack (self-serve option)

**Footer Caption**: "Enterprise support 24/7 • Typical deployment: 6–8 weeks from consultation to production."
- Support commitment (24/7)
- Realistic timeline (6-8 weeks, not "instant")
- Professional tone (consultation to production)

### Lead Qualification

**Demo Leads (High Intent)**
- Capture: Company name, role, compliance requirements, timeline
- Follow-up: Sales team within 24 hours
- Outcome: Custom demo, proposal, proof-of-concept

**Compliance Pack Leads (Medium Intent)**
- Capture: Email, company name, role, industry
- Follow-up: Email nurture sequence, compliance resources
- Outcome: Lead nurture, eventual demo request

**Sales Leads (Variable Intent)**
- Capture: Company name, role, specific questions, timeline
- Follow-up: Sales team qualification call
- Outcome: Custom proposal, specific questions answered, demo scheduled
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseCTA>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default enterprise CTA section showing all three conversion options: Schedule Demo (primary CTA with Calendar icon, 30-minute demo with compliance team), Compliance Pack (secondary CTA with FileText icon, download GDPR/SOC2/ISO 27001 documentation), and Talk to Sales (tertiary CTA with MessageSquare icon, discuss specific compliance requirements). Includes trust strip with testimonial stats and footer caption showing deployment timeline (6-8 weeks from consultation to production).',
      },
    },
  },
}

export const DemoFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing "Schedule Demo" as the primary CTA. This version would make the demo option more prominent: Larger card, more compelling copy ("See compliance in action - live demo with our team"), additional trust signals ("Join 50+ enterprises using rbee"), and stronger call-to-action ("Book Your Demo Now"). Ideal for high-intent enterprise buyers ready to evaluate or those responding to outbound sales campaigns.',
      },
    },
  },
}

export const ContactSalesFocus: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing "Talk to Sales" as the primary CTA. This version would make the sales option more prominent: Larger card, more compelling copy ("Custom proposal for your compliance requirements"), additional trust signals ("Trusted by 50+ regulated enterprises"), and stronger call-to-action ("Contact Sales Now"). Ideal for enterprise buyers with specific questions, custom requirements, or those needing custom proposals (RFP responses, custom pricing, etc.).',
      },
    },
  },
}
