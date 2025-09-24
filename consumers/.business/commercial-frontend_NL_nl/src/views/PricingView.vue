<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'Pricing — Credits & GPU-hour | Orchyra',
  description:
    'Draft pricing snapshots for Public Tap credit packs and Private Tap GPU-hour rates. Marketing page only.',
  keywords: [
    'pricing',
    'prepaid credits',
    'GPU-hour',
    'A100',
    'H100',
  ],
})
</script>

<template>
  <main class="page">
    <h1>Pricing (Draft)</h1>

    <section>
      <h2>Public Tap — Credits</h2>
      <ul>
        <li>Baseline: €1.20 per 1M tokens (input + output combined)</li>
        <li>Starter: €50 → ~41M tokens</li>
        <li>Builder: €200 → ~166M tokens</li>
        <li>Pro: €500 → ~416M tokens</li>
      </ul>
      <p class="note">Draft — subject to change after benchmarking.</p>
    </section>

    <section>
      <h2>Private Tap — GPU-hour</h2>
      <ul>
        <li>A100 80GB — €1.80 / GPU-hour + €250 / month base fee</li>
        <li>H100 80GB — €3.80 / GPU-hour + €400 / month base fee</li>
        <li>Scaling: 1× / 2× / 4× / 8× GPUs</li>
      </ul>
      <p class="note">Draft — provider rates vary; examples only.</p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.note { color: #6b7280; }
</style>
