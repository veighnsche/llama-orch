import { FAQItem } from './FaqSection'

// Created by: TEAM-086
// Pricing-specific FAQ data for use with FAQSection component

export const pricingCategories = ['Licensing', 'Plans', 'Billing', 'Trials']

export const pricingFaqItems: FAQItem[] = [
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
          All plans run the full rbee orchestrator. Paid tiers add the Web UI, team collaboration, priority support,
          and enterprise options (SLA, white-label, services).
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
