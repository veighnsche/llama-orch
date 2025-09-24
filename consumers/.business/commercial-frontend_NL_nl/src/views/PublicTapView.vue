<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

useMeta({
  title: `${t('publicTap.h1')} | Orchyra`,
  description: t('publicTap.whatP'),
  keywords: ['public tap', 'prepaid credits', 'agentic API', 'open-source models', 'private LLM hosting'],
})
</script>

<template>
  <main class="page">
    <h1>{{ $t('publicTap.h1') }}</h1>

    <section>
      <h2>{{ $t('publicTap.what') }}</h2>
      <p>{{ $t('publicTap.whatP') }}</p>
    </section>

    <section>
      <h2>{{ $t('publicTap.who') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bWho1') }}</li>
        <li>{{ $t('publicTap.bWho2') }}</li>
        <li>{{ $t('publicTap.bWho3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.why') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bWhy1') }}</li>
        <li>{{ $t('publicTap.bWhy2') }}</li>
        <li>{{ $t('publicTap.bWhy3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.terms') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bTerms1') }}</li>
        <li>{{ $t('publicTap.bTerms2') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.pricing') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bPrice1') }}</li>
        <li>{{ $t('publicTap.bPrice2') }}</li>
        <li>{{ $t('publicTap.bPrice3') }}</li>
        <li>{{ $t('publicTap.bPrice4') }}</li>
      </ul>
      <p class="note">{{ $t('publicTap.note') }}</p>
    </section>
  </main>
</template>

<style scoped>
.page {
  display: grid;
  gap: 1rem;
}
.note { color: #6b7280; }
</style>
