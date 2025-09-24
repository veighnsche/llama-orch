<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'FAQs | Orchyra',
  description: 'Answers about pricing, OSS models, prepaid credits, serving efficiency, and EU AI Act alignment.',
  keywords: ['faq', 'pricing', 'credits', 'models', 'EU AI Act', 'vLLM'],
})
</script>

<template>
  <main class="page">
    <h1>Frequently Asked Questions</h1>

    <section>
      <h2>Is this cheaper than OpenAI?</h2>
      <p>
        No. If you want the absolute lowest token price, use OpenAI. If you want transparency, EU-friendly deployments,
        model choice, and dedicated GPUs, consider a Private Tap. For testing, start with the Public Tap.
      </p>
    </section>

    <section>
      <h2>Can I bring any OSS model?</h2>
      <p>
        Yes, for Private Taps you can choose models from Hugging Face or other sources; VRAM is validated during setup.
      </p>
    </section>

    <section>
      <h2>How do prepaid credits work?</h2>
      <p>
        You buy a pack, receive a token balance, and consume until zero. Credits are non-refundable and valid for 12
        months. Your balance is always visible via dashboard/API (descriptive).
      </p>
    </section>

    <section>
      <h2>What makes your serving stack efficient?</h2>
      <p>
        We use open-source, high-throughput serving engines (e.g., vLLM with PagedAttention and continuous batching)
        and expose metrics for real SLOs.
      </p>
    </section>

    <section>
      <h2>Are you EU AI Act ready?</h2>
      <p>
        We operate with the Act's documentation and transparency spirit in mind (risk-aware, logged, documented
        deployments). The artifacts (logs, metrics, SSE transcripts) support your internal governance.
      </p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
</style>
