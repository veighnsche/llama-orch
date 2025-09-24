<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'Private LLM Hosting on Dedicated GPUs | Orchyra',
  description:
    'Your own AI API on dedicated GPUs with OSS models, observability, and optional OpenAI-compatible gateway. Marketing page only.',
  keywords: [
    'private LLM hosting',
    'dedicated GPUs',
    'open-source models',
    'SLA',
    'EU AI Act',
  ],
})
</script>

<template>
  <main class="page">
    <h1>Private Tap — Dedicated GPUs</h1>

    <section>
      <h2>What it is</h2>
      <p>
        A dedicated, isolated agentic API box provisioned per client. Any OSS model (e.g., Hugging Face), with
        GPU scaling options and an optional OpenAI-compatible gateway for drop-in usage.
      </p>
    </section>

    <section>
      <h2>Value</h2>
      <ul>
        <li>Privacy & control with your own endpoint and quotas</li>
        <li>Observability with metrics and logs for real SLOs</li>
        <li>High-throughput serving (e.g., vLLM) engineered for production</li>
      </ul>
    </section>

    <section>
      <h2>Draft pricing (snapshot)</h2>
      <ul>
        <li>A100 80GB — €1.80 / GPU-hour + €250 / month base fee</li>
        <li>H100 80GB — €3.80 / GPU-hour + €400 / month base fee</li>
        <li>Scale 1×/2×/4×/8× GPUs</li>
      </ul>
      <p class="note">Draft — examples based on public provider rates; subject to change.</p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.note { color: #6b7280; }
</style>
