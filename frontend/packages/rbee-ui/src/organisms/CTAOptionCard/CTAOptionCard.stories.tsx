import { Button } from '@rbee/ui/atoms/Button'
import type { Meta, StoryObj } from '@storybook/react'
import { Building2, Users } from 'lucide-react'
import { CTAOptionCard } from './CTAOptionCard'

const meta: Meta<typeof CTAOptionCard> = {
  title: 'Organisms/CTAOptionCard',
  component: CTAOptionCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The CTAOptionCard molecule is an enterprise-grade, persuasive CTA component with strong visual hierarchy, motion, and accessibility. It presents a call-to-action option with an icon chip, eyebrow label, title, body text, and action button.

## Composition
This molecule is composed of:
- **Header**: Icon chip with subtle halo + optional eyebrow badge
- **Content**: Title (with primary tone color option) + body text
- **Footer**: Primary action button + optional trust note
- **Motion**: Entrance animation, hover depth, and icon bounce
- **Accessibility**: ARIA region with proper labeling and keyboard focus

## Key Features
- **Three-part vertical composition**: Header → Content → Footer
- **Interactive depth**: Hover elevates border and shadow
- **Motion design**: Entrance fade-in/zoom, icon bounce on hover, button micro-interactions
- **Primary tone**: Radial highlight, primary text color, enhanced border
- **Accessibility**: role="region", aria-describedby, focus-visible ring

## When to Use
- Presenting multiple CTA options (e.g., "Contact Sales" vs "Self-Service")
- Enterprise vs Developer paths
- Different onboarding flows
- Pricing tier selection

## Used In
- **EnterpriseCTA**: Presents enterprise contact vs self-service options
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Icon element to display',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    title: {
      control: 'text',
      description: 'Card title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    body: {
      control: 'text',
      description: 'Description text',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    action: {
      control: false,
      description: 'Action button or link',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    tone: {
      control: 'select',
      options: ['primary', 'outline'],
      description: 'Visual emphasis level',
      table: {
        type: { summary: "'primary' | 'outline'" },
        defaultValue: { summary: 'outline' },
        category: 'Appearance',
      },
    },
    note: {
      control: 'text',
      description: 'Optional fine print below action',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    eyebrow: {
      control: 'text',
      description: 'Optional eyebrow label above title (e.g., "For large teams")',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof CTAOptionCard>

export const Default: Story = {
  args: {
    icon: <Building2 className="h-6 w-6" />,
    title: 'Enterprise',
    body: 'Custom deployment, SSO, and priority support—delivered with SLAs tailored to your risk profile.',
    action: (
      <Button variant="default" className="hover:translate-y-0.5 active:translate-y-[1px] transition-transform">
        Contact Sales
      </Button>
    ),
    eyebrow: 'For large teams',
  },
}

export const WithIcon: Story = {
  args: {
    icon: <Users className="h-6 w-6" />,
    title: 'Self-Service',
    body: 'Get started immediately with our developer-friendly platform and scale as you grow.',
    action: (
      <Button variant="outline" className="hover:translate-y-0.5 active:translate-y-[1px] transition-transform">
        Start Free Trial
      </Button>
    ),
    eyebrow: 'For developers',
  },
}

export const Highlighted: Story = {
  args: {
    icon: <Building2 className="h-6 w-6" />,
    title: 'Enterprise',
    body: 'Custom deployment, SSO, and priority support—delivered with SLAs tailored to your risk profile.',
    action: (
      <Button variant="default" className="hover:translate-y-0.5 active:translate-y-[1px] transition-transform">
        Contact Sales
      </Button>
    ),
    tone: 'primary',
    note: 'We respond within one business day.',
    eyebrow: 'For large teams',
  },
}

export const InCTAContext: Story = {
  render: () => (
    <div className="w-full max-w-5xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: CTAOptionCard in EnterpriseCTA organism</div>
      <div className="grid gap-6 md:grid-cols-2">
        <CTAOptionCard
          icon={<Building2 className="h-6 w-6" />}
          title="Enterprise"
          body="Custom deployment, SSO, and priority support—delivered with SLAs tailored to your risk profile."
          action={
            <Button
              variant="default"
              className="w-full hover:translate-y-0.5 active:translate-y-[1px] transition-transform"
            >
              Contact Sales
            </Button>
          }
          tone="primary"
          note="We respond within one business day."
          eyebrow="For large teams"
        />
        <CTAOptionCard
          icon={<Users className="h-6 w-6" />}
          title="Self-Service"
          body="Get started immediately with our developer-friendly platform and scale as you grow."
          action={
            <Button
              variant="outline"
              className="w-full hover:translate-y-0.5 active:translate-y-[1px] transition-transform"
            >
              Start Free Trial
            </Button>
          }
          note="No credit card required"
          eyebrow="For developers"
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'CTAOptionCard as used in the EnterpriseCTA organism, presenting two paths: enterprise contact and self-service signup.',
      },
    },
  },
}

export const CompactVariant: Story = {
  args: {
    icon: <Building2 className="h-6 w-6" />,
    title: 'Enterprise',
    body: 'Custom deployment, SSO, and priority support—delivered with SLAs tailored to your risk profile.',
    action: (
      <Button
        variant="default"
        size="sm"
        className="hover:translate-y-0.5 active:translate-y-[1px] transition-transform"
      >
        Contact Sales
      </Button>
    ),
    tone: 'primary',
    note: 'We respond within one business day.',
    eyebrow: 'For large teams',
    className: 'p-5',
  },
  parameters: {
    docs: {
      description: {
        story:
          'Compact variant with reduced padding (p-5) for tighter layouts. Pass `className="p-5"` to override default spacing.',
      },
    },
  },
}
