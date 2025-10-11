<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-002: Implemented PricingFAQ organism -->
<script setup lang="ts">
interface FAQ {
  question: string
  answer: string
}

interface Props {
  title?: string
  faqs?: FAQ[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Pricing FAQs',
  faqs: () => [
    {
      question: 'Is the free tier really free forever?',
      answer:
        'Yes. rbee is GPL open source. The Home/Lab tier is completely free with no time limits, no feature restrictions, and no hidden costs. You only pay for electricity to run your hardware.',
    },
    {
      question: "What's the difference between tiers?",
      answer:
        'All tiers include the full rbee orchestrator with no feature gates. Paid tiers add convenience features like Web UI, team collaboration, priority support, and enterprise services. The core AI orchestration is identical across all tiers.',
    },
    {
      question: 'Can I upgrade or downgrade anytime?',
      answer:
        'Yes. You can upgrade to a paid tier anytime to access additional features. You can also downgrade back to the free tier without losing your data or configuration.',
    },
    {
      question: 'Do you offer discounts for non-profits?',
      answer:
        'Yes. We offer 50% discounts for registered non-profits, educational institutions, and open source projects. Contact sales for details.',
    },
    {
      question: 'What payment methods do you accept?',
      answer:
        'We accept credit cards, bank transfers, and purchase orders for enterprise customers. All payments are processed securely through Stripe.',
    },
    {
      question: 'Is there a trial period?',
      answer:
        'The Team tier includes a 30-day free trial with full access to all features. No credit card required to start. Enterprise customers can request a custom trial period.',
    },
  ],
})
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="max-w-3xl mx-auto space-y-6">
        <div
          v-for="faq in faqs"
          :key="faq.question"
          class="bg-card border border-border rounded-lg p-6"
        >
          <h3 class="text-lg font-bold text-foreground mb-2">
            {{ faq.question }}
          </h3>
          <p class="text-muted-foreground leading-relaxed">
            {{ faq.answer }}
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
