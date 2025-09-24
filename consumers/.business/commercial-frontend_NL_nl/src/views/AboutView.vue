<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'About | Orchyra',
  description: 'Independent AI plumber — OSS transparency, proof-first operations, prepaid simplicity.',
  keywords: ['about', 'independent', 'OSS', 'proof-first', 'AI plumbing'],
})
</script>

<template>
  <main class="page">
    <h1>About Orchyra</h1>

    <section>
      <h2>Identity</h2>
      <p>
        Orchyra is led by an independent tradesman — your AI plumber. The focus is on robust engineering, transparent
        tooling, and proof-first operations so teams can run agentic AI as reliably as water or electricity.
      </p>
    </section>

    <section>
      <h2>USP pillars</h2>
      <ul>
        <li><strong>Open-source plumbing</strong> — orchestrator, SDK, and utils under GPL.</li>
        <li><strong>Proof-first operations</strong> — logs, Prometheus metrics, SSE transcripts, version pinning.</li>
        <li><strong>Independent, local</strong> — personal, accountable, practical.</li>
        <li><strong>Prepaid simplicity</strong> — predictable costs via credits and GPU-hour packs.</li>
      </ul>
    </section>

    <section>
      <h2>Approach</h2>
      <p>
        The trade is “AI plumbing” — installing and maintaining pipelines, applying observability and guardrails,
        and delivering documentation bundles that align with the EU AI Act’s transparency spirit.
      </p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
</style>
