'use client'

import { PricingScaleVisual } from '@rbee/ui/icons'
import type { Provider, Row, RowGroup, TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EmailCaptureProps,
  FAQItem,
  FAQTemplateProps,
  PricingComparisonTemplateProps,
  PricingHeroTemplateProps,
  PricingTemplateProps,
} from '@rbee/ui/templates'
import { Sparkles } from 'lucide-react'

// ============================================================================
// Props Objects
// ============================================================================
// All props for the Pricing page in visual order
// ============================================================================

// === PricingHero Template ===

/** Hero section - Start free, scale when ready message */
export const pricingHeroProps: PricingHeroTemplateProps = {
  badgeText: 'Honest Pricing',
  heading: (
    <>
      Start Free.
      <br />
      <span className="text-primary">Scale When Ready.</span>
    </>
  ),
  description:
    'Every tier ships the full rbee orchestrator—no feature gates, no artificial limits. OpenAI-compatible API, same power on day one. Pay only when you grow.',
  primaryCta: {
    text: 'View Plans',
  },
  secondaryCta: {
    text: 'Talk to Sales',
  },
  assuranceItems: [
    { text: 'Full orchestrator on every tier', icon: <Sparkles className="size-6" /> },
    { text: 'No feature gates or limits', icon: <Sparkles className="size-6" /> },
    { text: 'OpenAI-compatible API', icon: <Sparkles className="size-6" /> },
    { text: 'Cancel anytime', icon: <Sparkles className="size-6" /> },
  ],
  visual: (
    <PricingScaleVisual
      size="100%"
      className="rounded-md opacity-70"
      aria-label="Illustration showing rbee pricing scales from single-GPU homelab to multi-node server setups with progressive cost tiers"
    />
  ),
  visualAriaLabel:
    'Illustration showing rbee pricing scales from single-GPU homelab to multi-node server setups with progressive cost tiers',
}

// === PricingTemplate ===

/** Pricing template container - wraps the pricing tiers section */
export const pricingTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Simple, honest pricing.',
  description:
    "Every plan includes the full rbee orchestrator—no feature gates, no artificial limits. Start free and grow when you're ready.",
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/** Pricing template data - three pricing tiers with monthly/yearly toggle */
export const pricingTemplateProps: PricingTemplateProps = {
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
  footer: {
    mainText: 'Cancel anytime • No feature gates • Full orchestrator on every tier.',
    subText: 'Prices exclude taxes. OSS license applies to Home/Lab.',
  },
}

// === PricingComparison Template ===

/** Pricing comparison container - wraps the comparison table */
export const pricingComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: '',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
}

/** Pricing comparison data - feature matrix table */
const pricingComparisonColumns: Provider[] = [
  {
    key: 'h',
    label: 'Home/Lab',
    subtitle: 'Solo / Homelab',
  },
  {
    key: 't',
    label: 'Team',
    subtitle: 'Small teams',
    badge: 'Best for most teams',
    accent: true,
  },
  {
    key: 'e',
    label: 'Enterprise',
    subtitle: 'Security & SLA',
  },
]

const pricingComparisonGroups: RowGroup[] = [
  { id: 'core', label: 'Core Platform' },
  { id: 'productivity', label: 'Productivity' },
  { id: 'support', label: 'Support & Services' },
]

const pricingComparisonRows: Row[] = [
  // Core Platform
  {
    feature: 'Number of GPUs',
    group: 'core',
    values: { h: 'Unlimited', t: 'Unlimited', e: 'Unlimited' },
  },
  {
    feature: 'OpenAI-compatible API',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'Multi-GPU orchestration (one or many nodes)',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'Programmable routing (Rhai scheduler)',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'CLI access',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  // Productivity
  {
    feature: 'Web UI (manage nodes, models, jobs)',
    group: 'productivity',
    values: { h: false, t: true, e: true },
  },
  {
    feature: 'Team collaboration',
    group: 'productivity',
    values: { h: false, t: true, e: true },
  },
  // Support & Services
  {
    feature: 'Support',
    group: 'support',
    values: {
      h: 'Community',
      t: 'Priority email (business hours)',
      e: 'Dedicated (SLA-backed)',
    },
  },
  {
    feature: 'SLA',
    group: 'support',
    values: { h: false, t: false, e: true },
    note: 'Response and uptime commitments (Enterprise only).',
  },
  {
    feature: 'White-label',
    group: 'support',
    values: { h: false, t: false, e: true },
  },
  {
    feature: 'Professional services',
    group: 'support',
    values: { h: false, t: false, e: true },
  },
]

export const pricingComparisonProps: PricingComparisonTemplateProps = {
  title: 'Detailed Feature Comparison',
  subtitle: 'What changes across Home/Lab, Team, and Enterprise.',
  lastUpdated: 'This month',
  legend: {
    includedText: 'Included',
    notAvailableText: 'Not available',
  },
  keyDifferences: [
    'Team adds Web UI + collaboration',
    'Enterprise adds SLA + white-label + services',
    'All plans support unlimited GPUs',
  ],
  columns: pricingComparisonColumns,
  rows: pricingComparisonRows,
  groups: pricingComparisonGroups,
  tableCaption: 'Feature availability comparison across Home/Lab, Team, and Enterprise plans.',
  cta: {
    text: 'Ready to get started?',
    buttons: [
      { text: 'Start with Team', href: '/signup' },
      { text: 'Talk to Sales', href: '/contact', variant: 'outline' },
    ],
  },
}

// === FAQ Template ===

/** FAQ container - wraps the pricing FAQ section */
export const pricingFaqContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Pricing FAQs',
  description: 'Answers on licensing, upgrades, trials, and payments.',
  background: {
    variant: 'background',
  },
}

/** Pricing FAQ data */
const pricingFaqCategories = ['Licensing', 'Plans', 'Billing', 'Trials']

const pricingFaqItemsData: FAQItem[] = [
  {
    value: 'free-tier',
    question: 'Is the free tier really free forever?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>
          Yes. rbee is GPL open source. The Home/Lab tier is free with no time limits or feature gates. You only cover
          your own compute and electricity.
        </p>
      </div>
    ),
    category: 'Licensing',
  },
  {
    value: 'tier-difference',
    question: "What's the difference between tiers?",
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>
          All plans run the full rbee orchestrator. Paid tiers add the Web UI, team collaboration, priority support, and
          enterprise options (SLA, white-label, services).
        </p>
      </div>
    ),
    category: 'Plans',
  },
  {
    value: 'upgrade-downgrade',
    question: 'Can I upgrade or downgrade anytime?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>Yes. Move between tiers at any time. Your data and configuration stay intact when you change plans.</p>
      </div>
    ),
    category: 'Plans',
  },
  {
    value: 'payment-methods',
    question: 'What payment methods do you accept?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>
          Credit cards, bank transfers, and purchase orders (Enterprise). Payments are processed securely via Stripe.
        </p>
      </div>
    ),
    category: 'Billing',
  },
  {
    value: 'nonprofit-discount',
    question: 'Do you offer discounts for non-profits?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>Yes—50% for registered non-profits, education, and open-source projects. Contact sales for eligibility.</p>
      </div>
    ),
    category: 'Billing',
  },
  {
    value: 'trial-period',
    question: 'Is there a trial period?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>Team includes a 30-day free trial (no credit card). Enterprise trials are available on request.</p>
      </div>
    ),
    category: 'Trials',
  },
]

export const pricingFaqProps: FAQTemplateProps = {
  badgeText: 'Pricing • Plans & Billing',
  categories: pricingFaqCategories,
  faqItems: pricingFaqItemsData,
  jsonLdEnabled: true,
}

// === EmailCapture Template ===

/** Email capture section - newsletter signup CTA */
export const pricingEmailCaptureProps: EmailCaptureProps = {
  headline: 'Stay Updated on Pricing',
  subheadline: 'Get notified about new plans, discounts, and billing updates.',
  emailInput: {
    placeholder: 'your@email.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Subscribe',
  },
  trustMessage: 'We respect your privacy. Unsubscribe at any time.',
  successMessage: 'Thanks for subscribing! Check your inbox.',
}

/**
 * Email capture container - Background wrapper
 */
export const pricingEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}
