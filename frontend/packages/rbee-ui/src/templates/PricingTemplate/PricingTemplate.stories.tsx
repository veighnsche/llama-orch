import type { Meta, StoryObj } from '@storybook/react'
import { Layers, Shield, Unlock, Zap } from 'lucide-react'
import { pricingHero } from '@rbee/ui/assets'
import { PricingTemplate } from './PricingTemplate'

const meta = {
  title: 'Templates/PricingTemplate',
  component: PricingTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    kickerBadges: [
      {
        icon: <Unlock className="h-3.5 w-3.5" aria-hidden="true" />,
        label: 'Open source',
      },
      {
        icon: <Zap className="h-3.5 w-3.5" aria-hidden="true" />,
        label: 'OpenAI-compatible',
      },
      {
        icon: <Layers className="h-3.5 w-3.5" aria-hidden="true" />,
        label: 'Multi-GPU',
      },
      {
        icon: <Shield className="h-3.5 w-3.5" aria-hidden="true" />,
        label: 'No feature gates',
      },
    ],
    tiers: [
      {
        title: 'Home/Lab',
        price: '€0',
        period: 'forever',
        features: [
          'Unlimited GPUs on your hardware',
          'OpenAI-compatible API',
          'Multi-modal models',
          'Active community support',
          'Open source core',
        ],
        ctaText: 'Download rbee',
        ctaHref: '/download',
        ctaVariant: 'outline',
        footnote: 'Local use. No feature gates.',
        className:
          'col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500',
      },
      {
        title: 'Team',
        price: '€99',
        priceYearly: '€990',
        period: '/month',
        features: [
          'Everything in Home/Lab',
          'Web UI for cluster & models',
          'Shared workspaces & quotas',
          'Priority support (business hours)',
          'Rhai policy templates (rate/data)',
        ],
        ctaText: 'Start 30-Day Trial',
        ctaHref: '/signup?plan=team',
        highlighted: true,
        badge: 'Most Popular',
        footnote: 'Cancel anytime during trial.',
        saveBadge: '2 months free',
        className:
          'col-span-12 md:col-span-4 order-first md:order-none motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-100',
      },
      {
        title: 'Enterprise',
        price: 'Custom',
        features: [
          'Everything in Team',
          'Dedicated, isolated instances',
          'Custom SLAs & onboarding',
          'White-label & SSO options',
          'Enterprise security & support',
        ],
        ctaText: 'Contact Sales',
        ctaHref: '/contact?type=enterprise',
        ctaVariant: 'outline',
        footnote: "We'll reply within 1 business day.",
        className:
          'col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-200',
      },
    ],
    editorialImage: {
      src: pricingHero,
      alt: 'Detailed isometric 3D illustration in dark mode showing a progression from left to right: a compact single-GPU homelab server rack (glowing neon teal accents) seamlessly transforming into a large-scale multi-node GPU cluster with interconnected nodes (amber and teal lighting). Clean editorial photography style with dramatic cinematic lighting, sharp focus on hardware details, floating UI panels showing metrics, dark navy background with subtle grid, professional tech marketing aesthetic, 4K quality, Octane render look',
    },
    footer: {
      mainText: 'Every plan includes the full rbee orchestrator. No feature gates. No artificial limits.',
      subText: 'Prices exclude VAT. OSS license applies to Home/Lab.',
    },
  },
}
