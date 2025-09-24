<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'

useMeta({
  title: 'Proof — Logs, Metrics, SSE | Orchyra',
  description:
    'Proof-first operations with logs, Prometheus metrics, SSE transcripts, and version pinning. Static placeholders; marketing page only.',
  keywords: ['proof', 'prometheus', 'sse', 'metrics', 'observability'],
})
</script>

<template>
  <main class="page">
    <h1>Proof — Logs, Metrics, SSE</h1>

    <section>
      <h2>Artifacts (described)</h2>
      <ul>
        <li>Deployment report with SSE transcripts and performance metrics</li>
        <li>Prometheus dashboard snapshots and alert thresholds</li>
        <li>Version pinning and roll-back plan</li>
        <li>Documentation bundle aligned with EU AI Act transparency spirit</li>
      </ul>
      <p class="microcopy">This is infrastructure—like water or electricity. You shouldn’t have to “trust” it. You should see it.</p>
    </section>

    <section>
      <h2>Coming visuals</h2>
      <ul>
        <li>Grafana/Prometheus screenshots</li>
        <li>Pipeline diagram with “certified” stamp</li>
      </ul>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.microcopy { color: #374151; }
</style>
