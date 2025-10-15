import type { Meta, StoryObj } from '@storybook/react'
import { PricingComparison } from './PricingComparison'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Pricing/PricingComparison',
  component: PricingComparison,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The PricingComparison section presents a detailed feature comparison table across three pricing tiers: Home/Lab (free/OSS), Team (paid), and Enterprise (custom pricing). Features are grouped by Core Platform, Productivity, and Support & Services categories.

## Composition
This organism contains:
- **Header**: "Detailed Feature Comparison" with subtitle
- **Legend**: Visual key for feature availability (Included, Not available, or specific values)
- **Decisions Panel**: Key differences summary (Team adds Web UI + collaboration, Enterprise adds SLA + white-label + services, All plans support unlimited GPUs)
- **Desktop Table**: Full comparison table with all features and tiers
- **Mobile Card List**: Mobile-friendly card view of features
- **CTA Strip**: "Start with Team" and "Talk to Sales" buttons
- **Feature Groups**: Core Platform (GPUs, API, orchestration, scheduler, CLI), Productivity (Web UI, team collaboration), Support & Services (support levels, SLA, white-label, professional services)

## When to Use
- On the /pricing page after the hero section
- To show detailed feature differences between tiers
- To help buyers choose the right tier
- To justify pricing differences

## Content Requirements
- **Features**: Detailed feature list grouped by category
- **Tiers**: Home/Lab (free), Team (paid), Enterprise (custom)
- **Legend**: Clear visual indicators for feature availability
- **Key Differences**: Summary of main differences between tiers

## Variants
- **Default**: All three tiers with all features
- **Free vs. Pro**: Compare two tiers (Home/Lab vs. Team)
- **Pro vs. Enterprise**: Compare two tiers (Team vs. Enterprise)

## Examples
\`\`\`tsx
import { PricingComparison } from '@rbee/ui/organisms/Pricing/PricingComparison'

// Simple usage - no props needed
<PricingComparison />

// With custom last updated date
<PricingComparison lastUpdated="January 2025" />
\`\`\`

## Used In
- Pricing page (/pricing)

## Related Components
- Card
- Badge
- Button
- Tooltip

## Accessibility
- **Keyboard Navigation**: Table and tooltips are keyboard accessible
- **ARIA Labels**: Table has caption, tooltips have proper ARIA
- **Semantic HTML**: Uses <table> with <caption>, <thead>, <tbody>
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards
- **Responsive**: Mobile card list for small screens

## Pricing Strategy

### Tier Strategy (Good/Better/Best)

**Home/Lab (Free/OSS) - "Good"**
- **Target**: Solo developers, hobbyists, homelab enthusiasts
- **Value Prop**: Full orchestrator, unlimited GPUs, OpenAI-compatible API
- **Limitations**: No web UI, no team collaboration, community support only
- **Purpose**: Adoption, community building, open-source credibility

**Team (Paid) - "Better"**
- **Target**: Small teams (2-10 people)
- **Value Prop**: Everything in Free + Web UI, team collaboration, priority support
- **Limitations**: No SLA, no white-label, no professional services
- **Purpose**: Revenue generation, team adoption, upgrade path

**Enterprise (Custom) - "Best"**
- **Target**: Large enterprises (50-5000+ employees)
- **Value Prop**: Everything in Team + SLA, 24/7 support, white-label, professional services, multi-region
- **Limitations**: None (all features included)
- **Purpose**: High-value customers, enterprise revenue, custom deals

### Feature Gating Strategy

**What's NOT gated (available on all tiers):**
- Number of GPUs (unlimited)
- OpenAI-compatible API
- Multi-GPU orchestration (one or many nodes)
- Programmable routing (Rhai scheduler)
- CLI access

**What IS gated (Team and above):**
- Web UI (manage nodes, models, jobs)
- Team collaboration

**What IS gated (Enterprise only):**
- Support: Dedicated (SLA-backed) vs. Priority email (Team) vs. Community (Free)
- SLA: Response and uptime commitments
- White-label: Custom branding
- Professional services: Deployment, integration, training

### Upgrade Triggers

**Free → Team:**
- Need web UI (manage nodes, models, jobs visually)
- Need team collaboration (multiple users, shared resources)
- Need priority support (email support during business hours)

**Team → Enterprise:**
- Need SLA (99.9% uptime, 1-hour response time)
- Need 24/7 support (critical issues outside business hours)
- Need white-label (custom branding, custom domain)
- Need professional services (deployment consulting, custom development, team training)
- Need multi-region support (EU multi-region, automatic failover, load balancing)

### Feature vs. Benefit

**Feature**: Web UI  
**Benefit**: Manage nodes, models, and jobs visually (no CLI required)

**Feature**: Team collaboration  
**Benefit**: Multiple users can share resources and collaborate on projects

**Feature**: Priority email support  
**Benefit**: Get help faster (business hours response) vs. community support

**Feature**: 99.9% uptime SLA  
**Benefit**: Your AI infrastructure is always available (8.76 hours downtime/year max)

**Feature**: 24/7 support with 1-hour response  
**Benefit**: Critical issues resolved quickly, even outside business hours

**Feature**: White-label  
**Benefit**: Run rbee as your brand (no vendor attribution)

**Feature**: Professional services  
**Benefit**: Expert help with deployment, integration, and training

**Feature**: Multi-region support  
**Benefit**: High availability, disaster recovery, compliance with data residency

### Conversion Strategy
- **Primary CTA**: "Start with Team" (recommended tier for most teams)
- **Secondary CTA**: "Talk to Sales" (enterprise tier, custom pricing)
- **Lead qualification**: Capture team size, use case, infrastructure details
- **Proof points**: Unlimited GPUs on all tiers, no feature gates on core platform, OpenAI-compatible API
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    lastUpdated: {
      control: 'text',
      description: 'Last updated date for the comparison table',
      table: {
        type: { summary: 'string' },
        defaultValue: { summary: 'This month' },
        category: 'Content',
      },
    },
  },
} satisfies Meta<typeof PricingComparison>

export default meta
type Story = StoryObj<typeof meta>

export const PricingPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default pricing comparison table showing all three tiers: Home/Lab (free/OSS with unlimited GPUs, API, orchestration, CLI), Team (paid with Web UI, team collaboration, priority email support), and Enterprise (custom pricing with SLA, 24/7 support, white-label, professional services, multi-region). Features are grouped by Core Platform, Productivity, and Support & Services. Desktop shows full table, mobile shows card list.',
      },
    },
  },
}

export const FreeVsPro: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant comparing two tiers: Home/Lab (free) vs. Team (paid). This version would emphasize the key differences: Web UI (Team only - manage nodes, models, jobs visually), Team collaboration (Team only - multiple users, shared resources), and Priority email support (Team only - business hours response vs. community support). Ideal for solo developers or small teams evaluating whether to upgrade from free tier.',
      },
    },
  },
}

export const ProVsEnterprise: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant comparing two tiers: Team (paid) vs. Enterprise (custom). This version would emphasize the key differences: SLA (Enterprise only - 99.9% uptime, 1-hour response), 24/7 support (Enterprise only - critical issues outside business hours), White-label (Enterprise only - custom branding, custom domain), Professional services (Enterprise only - deployment consulting, custom development, team training), and Multi-region support (Enterprise only - EU multi-region, automatic failover, load balancing). Ideal for teams evaluating whether to upgrade to enterprise tier.',
      },
    },
  },
}
