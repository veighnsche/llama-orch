<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'Contact & Legal | Orchyra',
  description:
    'Contact Orchyra and review key legal notes. Public Tap terms: prepaid, non-refundable, 12-month validity. Marketing page only.',
  keywords: ['contact', 'legal', 'terms', 'credits'],
})
</script>

<template>
  <main class="page">
    <h1>Contact & Legal</h1>

    <section>
      <h2>Contact</h2>
      <ul>
        <li>Email: <a href="mailto:info@example.com">info@example.com</a></li>
        <li>LinkedIn: <a href="#" rel="nofollow noopener">(add link)</a></li>
        <li>GitHub: <a href="#" rel="nofollow noopener">(add link)</a></li>
      </ul>
    </section>

    <section>
      <h2>Public Tap Terms</h2>
      <ul>
        <li>Prepaid credits are non-refundable</li>
        <li>12-month validity from purchase</li>
        <li>Service halts at zero balance until recharge</li>
      </ul>
      <p class="note">Legal copy to be finalized; this page is informational only.</p>
    </section>

    <section>
      <h2>Data & Logs</h2>
      <p>
        Deployment reports, metrics, and relevant logs are made available to customers as part of the proof-first
        approach.
      </p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.note { color: #6b7280; }
</style>
